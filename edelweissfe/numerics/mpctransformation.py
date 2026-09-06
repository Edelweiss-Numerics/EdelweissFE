#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  ---------------------------------------------------------------------
#
#  _____    _      _              _         _____ _____
# | ____|__| | ___| |_      _____(_)___ ___|  ___| ____|
# |  _| / _` |/ _ \ \ \ /\ / / _ \ / __/ __| |_  |  _|
# | |__| (_| |  __/ |\ V  V /  __/ \__ \__ \  _| | |___
# |_____\__,_|\___|_| \_/\_/ \___|_|___/___/_|   |_____|
#
#
#  Unit of Strength of Materials and Structural Analysis
#  University of Innsbruck,
#  2017 - today
#
#  Matthias Neuner matthias.neuner@uibk.ac.at
#
#  This file is part of EdelweissFE.
#
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation; either
#  version 2.1 of the License, or (at your option) any later version.
#
#  The full text of the license can be found in the file LICENSE.md at
#  the top level directory of EdelweissFE.
#  ---------------------------------------------------------------------

import os

import numpy as np
from scipy.sparse import csr_matrix

from edelweissfe.utils import performancetiming

#: Set to enable a per-call cross-check of :meth:`MultiPointConstraintTransformation.transformSystemMatrix`'s
#: AMGCL-threaded expression (``useAmgclSpgemm``) against the plain SciPy ``T^T @ K @ T + C``
#: expression it is an alternative to. Expensive (recomputes the plain expression every call it fires
#: on) -- development/CI use only.
_ASSERT_EXACT_ENV_VAR = "EDELWEISS_MPC_ASSERT_EXACT"

#: Set to a directory to dump the raw K plus this transformation's own T/D/S/C matrices for the
#: first few calls to :meth:`MultiPointConstraintTransformation.transformSystemMatrix` -- an offline
#: harness for investigating the SpGEMM-threading behaviour below: unlike the linsolve-level
#: ``linsolveDumps`` harness (whose dumps are already-condensed ``Kt``), this needs the
#: *pre*-condensation K and T/D/S/C themselves. Capped (see ``_DUMP_MPC_MAX_CALLS``) -- `K` on a
#: large coupled model is hundreds of MB uncompressed, and this is a diagnostic tool, not a feature.
_DUMP_MPC_ENV_VAR = "EDELWEISS_DUMP_MPC"
_DUMP_MPC_MAX_CALLS = 3
_dumpMpcCallsWritten = 0

"""
Master-slave condensation (Abaqus-style DOF elimination) of linear multi-point constraints

.. math::
    u_s = \\sum_a N_a \\, u_{m_a}

expressed as a full-size square transformation, so the equation system keeps its size ``nDof`` and
every consumer of the DOF vector layout (Dirichlet indices, convergence checks, node-field
write-back, field outputs) remains untouched.

With the transformation matrix :math:`T` (identity on independent DOFs, row :math:`s` carrying the
weights :math:`N_a` in the master columns, slave columns entirely zero) and the constraint-row
matrix :math:`C` (rows :math:`s` only: :math:`C_{ss} = 1`, :math:`C_{s m_a} = -N_a`), the
condensed implicit system reads

.. math::
    (T^T K \\, T + C) \\; \\delta U = T^T R,

where the slave rows of :math:`T^T K T` are structurally zero and :math:`C` re-inserts the
constraint equations, so the solution satisfies
:math:`\\delta U_s = \\sum_a N_a \\, \\delta U_{m_a}` exactly. The row replacement breaks symmetry
in exactly the same benign way the existing Dirichlet row treatment does.

For explicit dynamics with a lumped (diagonal) mass vector, the consistent row-sum lumping of
:math:`T^T M T` stays diagonal: each master receives its own mass plus the :math:`N_a`-weighted
mass of the slaves glued to it (total mass is conserved when :math:`\\sum_a N_a = 1`, which holds
for interpolation-type constraints). Forces are folded as :math:`\\tilde{P} = T^T P`, and the slave
kinematics are assigned directly from the masters, adding zero stiffness and hence leaving the
critical time step untouched.
"""


def _flattenChainedRecords(
    records: list[tuple[int, list[tuple[int, float]]]],
) -> list[tuple[int, list[tuple[int, float]]]]:
    """Resolve a slave DOF's masters that are themselves slave DOFs of another (or the same) record,
    substituting them recursively until every master is an independent DOF.

    Distinct MPC instances are free to compose this way -- e.g. a tie constraint's projected facet
    can legitimately reference a hanging-node MPC's slave node as one of its own interpolation
    nodes. :class:`~edelweissfe.adaptivity.refinement.AdaptiveMesh` already flattens chains *within*
    the hanging-node MPC's own records; this generalizes the same substitution *across* all of a
    model's multi-point constraints, in whatever order they were collected.

    Parameters
    ----------
    records
        The raw per-constraint records, one per slave DOF.

    Returns
    -------
    list of (int, list of (int, float))
        The same slave DOFs, with every master substituted down to independent DOFs and duplicate
        ultimate masters (reached via more than one path) coalesced by summing their coefficients.
    """
    recordOf = dict(records)
    resolved = {}

    def resolve(slaveDof, visiting):
        if slaveDof in resolved:
            return resolved[slaveDof]
        if slaveDof in visiting:
            raise ValueError(
                "Multi-point constraints: circular master/slave dependency detected at DOF {:}.".format(slaveDof)
            )
        visiting = visiting | {slaveDof}
        flat = {}
        for masterDof, coefficient in recordOf[slaveDof]:
            if masterDof in recordOf:
                for mm, cc in resolve(masterDof, visiting).items():
                    flat[mm] = flat.get(mm, 0.0) + coefficient * cc
            else:
                flat[masterDof] = flat.get(masterDof, 0.0) + coefficient
        resolved[slaveDof] = flat
        return flat

    return [(slaveDof, list(resolve(slaveDof, frozenset()).items())) for slaveDof in recordOf]


class MultiPointConstraintTransformation:
    """The assembled master-slave condensation operator for all linear multi-point constraints of
    an equation system.

    Parameters
    ----------
    records
        The linear dependency records, one per slave DOF:
        ``(slaveDofIndex, [(masterDofIndex, coefficient), ...])``, as collected from all
        :class:`~edelweissfe.constraints.base.multipointconstraintbase.MultiPointConstraintBase`
        instances of a model.
    nDof
        The total size of the equation system.
    useAmgclSpgemm
        Compute ``transformSystemMatrix`` as the direct ``Tᵀ K T + C`` expression, but via AMGCL's
        own OpenMP-threaded ``product()``/``sum()`` instead of SciPy's single-threaded CSR sparse
        routines -- SciPy's sparse module carries no OpenMP parallelism at all, the same gap this
        codebase also closes for its outer GMRES matvec and AMG relaxation kernels elsewhere.
        Offline-measured on a reference 280k-dof model: **~2.4–2.6x faster** than the direct
        expression (``~1.4–1.5 s/call`` vs ``~2.4–2.7 s/call``), correctness-verified to
        floating-point precision.

        AMGCL's ``product()``/``sum()`` do not eliminate exact-cancellation zeros the way SciPy's
        own sparse routines do -- confirmed directly that the extra entries are >99.9999% bit-exact
        zero (20,102,955 of 20,102,975 on the reference model, the rest floating-point noise at
        ~1e-19 relative scale), so this method leaves noticeably more raw nnz (~1.6x) than the plain
        expression. **Not** pruned here: :meth:`~edelweissfe.solvers.nonlinearimplicitstatic.NIST.
        applyDirichletK` already calls ``eliminate_zeros()`` immediately after this, gated by the
        existing ``pruneCondensedMatrixZeros`` option (default ``True``, uniformly for both
        expressions) -- that gate exists precisely because PARDISO's reordering on these
        path-dependent condensed systems is known to drift with unpruned explicit-zero structural
        entries, and because ``blockamg``'s hierarchy-reuse gates on raw ``nnz``. Pruning here too
        would double that cost under the default config; setting ``pruneCondensedMatrixZeros=False``
        together with this option carries the same, now-larger, unpruned pattern into the solve,
        same as it already does for the plain expression -- verify the load path if combining the
        two. Default ``False`` pending a live gate (offline validation only so far).
    """

    def __init__(
        self,
        records: list[tuple[int, list[tuple[int, float]]]],
        nDof: int,
        useAmgclSpgemm: bool = False,
    ):
        slaveDofs = [slaveDof for slaveDof, _ in records]

        if len(set(slaveDofs)) != len(slaveDofs):
            raise ValueError("Multi-point constraints: a DOF is claimed as slave by more than one constraint record.")

        for slaveDof, masters in records:
            if not masters:
                raise ValueError("Multi-point constraints: slave DOF {:} has no master DOFs.".format(slaveDof))

        # a master referenced by one constraint may itself be a slave DOF of another (or the same)
        # constraint -- e.g. a tie facet referencing a hanging-node MPC's slave node. Substitute those
        # down to independent DOFs rather than rejecting the composition.
        records = _flattenChainedRecords(records)

        self.nDof = nDof
        self.slaveDofIndices = np.array(sorted(slaveDofs), dtype=int)

        recordOfSlaveDof = {slaveDof: masters for slaveDof, masters in records}

        # W: (nSlaves x nDof) weight matrix, row k carrying the master weights of the k-th
        # (sorted) slave DOF. Serves the drift correction, the slave-kinematics assignment,
        # and the lumped-mass folding.
        wRows, wCols, wVals = [], [], []
        for k, slaveDof in enumerate(self.slaveDofIndices):
            for masterDof, coefficient in recordOfSlaveDof[slaveDof]:
                wRows.append(k)
                wCols.append(masterDof)
                wVals.append(coefficient)
        self._W = csr_matrix((wVals, (wRows, wCols)), shape=(len(self.slaveDofIndices), nDof))

        # T: identity on independent DOFs, slave rows carrying the master weights, slave columns
        # entirely zero.
        independentDofs = np.setdiff1d(np.arange(nDof, dtype=int), self.slaveDofIndices, assume_unique=True)
        tRows = np.concatenate(
            [independentDofs, np.repeat(self.slaveDofIndices, [len(recordOfSlaveDof[s]) for s in self.slaveDofIndices])]
        )
        tCols = np.concatenate([independentDofs, np.array(wCols, dtype=int)])
        tVals = np.concatenate([np.ones(len(independentDofs)), np.array(wVals)])
        self._T = csr_matrix((tVals, (tRows, tCols)), shape=(nDof, nDof))

        # C: the constraint equations themselves, re-inserted into the (structurally zero) slave
        # rows of T^T K T.
        cRows = np.concatenate(
            [
                self.slaveDofIndices,
                np.repeat(self.slaveDofIndices, [len(recordOfSlaveDof[s]) for s in self.slaveDofIndices]),
            ]
        )
        cCols = np.concatenate([self.slaveDofIndices, np.array(wCols, dtype=int)])
        cVals = np.concatenate([np.ones(len(self.slaveDofIndices)), -np.array(wVals)])
        self._C = csr_matrix((cVals, (cRows, cCols)), shape=(nDof, nDof))

        self._useAmgclSpgemm = useAmgclSpgemm
        # Lazily populated by _transformSystemMatrixAmgcl on first use: self._T is fixed for the
        # lifetime of this object (assigned once above, never reassigned), so both the
        # transpose-to-CSR conversion and the AMGCL SpGEMM helper are safe to build once and reuse
        # across every condensation call rather than repeating the work every Newton iteration.
        self._TTranspose = None
        self._amgclSpgemmHelper = None

    @property
    def nEliminatedDof(self) -> int:
        """The number of slave DOFs eliminated from the equation system."""

        return len(self.slaveDofIndices)

    def checkDirichletConflicts(self, dirichletDofIndices: np.ndarray):
        """Raise if any Dirichlet-constrained DOF is a slave DOF of a multi-point constraint --
        a slave DOF's motion is fully determined by its masters and cannot be prescribed.

        Parameters
        ----------
        dirichletDofIndices
            The global DOF indices constrained by Dirichlet boundary conditions.
        """

        conflicts = np.intersect1d(dirichletDofIndices, self.slaveDofIndices)
        if len(conflicts):
            raise ValueError(
                "{:} Dirichlet-constrained DOF(s) are slave DOFs of a multi-point constraint. "
                "Prescribe the masters instead.".format(len(conflicts))
            )

    @performancetiming.timeit("mpc transform system matrix")
    def _transformSystemMatrixLegacy(self, K: csr_matrix) -> csr_matrix:
        """The plain ``T^T @ K @ T + C`` expression via SciPy's own (single-threaded) sparse
        routines -- the default condensation strategy."""
        KT = self._T.T @ K
        Kt = KT @ self._T
        Kt = (Kt + self._C).tocsr()
        Kt.sort_indices()
        return Kt

    def _assertExact(self, K: csr_matrix, result: csr_matrix) -> None:
        """:envvar:`EDELWEISS_MPC_ASSERT_EXACT` cross-check: the AMGCL-threaded result must match
        the plain ``T^T K T + C`` expression to within a matrix-norm-relative tolerance
        (entry-relative fails on cancellation-tiny entries)."""
        legacy = self._transformSystemMatrixLegacy(K)
        diff = (result - legacy).tocsr()
        maxDiff = np.max(np.abs(diff.data)) if diff.nnz else 0.0
        scale = max(
            np.max(np.abs(result.data)) if result.nnz else 0.0,
            np.max(np.abs(legacy.data)) if legacy.nnz else 0.0,
            1e-300,
        )
        if maxDiff > 1e-9 * scale:
            raise AssertionError(
                "mpctransformation: EDELWEISS_MPC_ASSERT_EXACT caught a real mismatch between the "
                "AMGCL-threaded result and the plain T^T K T + C expression: max|delta|={:.3e}, "
                "tolerance={:.3e} (1e-9 x max|data|={:.3e}).".format(maxDiff, 1e-9 * scale, scale)
            )

    def _dumpForOfflineProbe(self, K: csr_matrix) -> None:
        """Write this call's raw `K` plus `T`/`C` to :envvar:`EDELWEISS_DUMP_MPC`'s directory,
        capped at `_DUMP_MPC_MAX_CALLS` process-wide -- see that env var's own comment.
        """
        global _dumpMpcCallsWritten
        directory = os.environ.get(_DUMP_MPC_ENV_VAR)
        if not directory or _dumpMpcCallsWritten >= _DUMP_MPC_MAX_CALLS:
            return
        from scipy.sparse import save_npz

        os.makedirs(directory, exist_ok=True)
        stem = "{:03d}".format(_dumpMpcCallsWritten)
        save_npz(os.path.join(directory, "K_{:}.npz".format(stem)), K.tocsr(), compressed=False)
        save_npz(os.path.join(directory, "T_{:}.npz".format(stem)), self._T, compressed=False)
        save_npz(os.path.join(directory, "C_{:}.npz".format(stem)), self._C, compressed=False)
        _dumpMpcCallsWritten += 1
        print(
            "mpctransformation: dumped call {:} ({:} rows, {:} nnz K) to {:} ({:}/{:} used)".format(
                stem, K.shape[0], K.nnz, directory, _dumpMpcCallsWritten, _DUMP_MPC_MAX_CALLS
            ),
            flush=True,
        )

    def transformSystemMatrix(self, K: csr_matrix) -> csr_matrix:
        """Condense the system matrix: :math:`\\tilde{K} = T^T K \\, T + C`.

        Dispatches to the AMGCL-threaded expression (``useAmgclSpgemm``) or the plain SciPy
        expression (default) -- see the constructor's ``useAmgclSpgemm`` docstring for why the
        AMGCL path is not yet the default despite being faster (pending a live gate).
        """
        self._dumpForOfflineProbe(K)
        if self._useAmgclSpgemm:
            result = self._transformSystemMatrixAmgcl(K)
            if os.environ.get(_ASSERT_EXACT_ENV_VAR):
                self._assertExact(K, result)
            return result
        return self._transformSystemMatrixLegacy(K)

    @performancetiming.timeit("mpc transform system matrix")
    def _transformSystemMatrixAmgcl(self, K: csr_matrix) -> csr_matrix:
        """Condense the system matrix via the same direct :math:`T^T K \\, T + C` expression
        :meth:`_transformSystemMatrixLegacy` computes, but through AMGCL's own OpenMP-threaded
        ``product()``/``sum()`` instead of SciPy's single-threaded CSR sparse routines -- see
        ``useAmgclSpgemm``'s own constructor-argument docstring for the measured speedup and the
        ``eliminate_zeros()`` rationale.
        """
        if self._amgclSpgemmHelper is None:
            from edelweissfe.linsolve.amgcl.amgcl import PyAMGCLSpGEMM

            self._amgclSpgemmHelper = PyAMGCLSpGEMM()
            self._TTranspose = self._T.T.tocsr()

        KT = self._amgclSpgemmHelper.product(K, self._T)
        KtNoC = self._amgclSpgemmHelper.product(self._TTranspose, KT)
        Kt = self._amgclSpgemmHelper.sum(1.0, KtNoC, 1.0, self._C)
        # No eliminate_zeros() here, deliberately -- not because it is unneeded (AMGCL's
        # product()/sum() leave far more exact-cancellation zeros unpruned than SciPy's own SpGEMM
        # does, confirmed directly: 54.3M nnz here vs. the plain expression's 34.2M on a reference
        # model, with >99.9999% of that gap being bit-exact zero, not numerical
        # approximation), but because NISTSolver.applyDirichletK already does this immediately
        # after, gated by pruneCondensedMatrixZeros (default True), uniformly for both condensation
        # strategies -- that gate exists precisely because PARDISO's reordering on these
        # path-dependent condensed systems is sensitive to extra explicit-zero structural entries
        # (nonlinearimplicitstatic.py's own documented, previously-observed drift), and because
        # blockamg's hierarchy-reuse gates on raw nnz (an unpruned, possibly call-to-call-unstable
        # zero count would risk spurious hierarchy rebuilds independent of any PARDISO concern).
        # Pruning here too would only double that work under the default config.
        Kt.sort_indices()
        return Kt

    @performancetiming.timeit("mpc transform residual")
    def transformResidual(self, R: np.ndarray, dU: np.ndarray) -> np.ndarray:
        """Condense the residual: :math:`\\tilde{R} = T^T R`, with the slave rows replaced by the
        current constraint violation :math:`-(dU_s - \\sum_a N_a \\, dU_{m_a})` (exactly zero for
        any consistently accumulated increment; written explicitly as a drift correction).

        Parameters
        ----------
        R
            The assembled residual.
        dU
            The current displacement increment.

        Returns
        -------
        np.ndarray
            The condensed residual.
        """

        Rt = self._T.T @ R
        Rt[self.slaveDofIndices] = -(dU[self.slaveDofIndices] - self._W @ dU)
        return Rt

    def foldLumpedMass(self, M: np.ndarray):
        """Fold the slave masses onto their masters (in place): :math:`M_{m_a} \\mathrel{+}= N_a
        M_s`, then :math:`M_s = 0` -- the row-sum lumping of :math:`T^T M T`, which keeps the mass
        vector diagonal and conserves total mass for interpolation-type constraints
        (:math:`\\sum_a N_a = 1`).

        Parameters
        ----------
        M
            The lumped (diagonal) mass vector, modified in place.
        """

        foldedSlaveMasses = self._W.T @ M[self.slaveDofIndices]
        M[self.slaveDofIndices] = 0.0
        M += foldedSlaveMasses

    def foldExplicitForce(self, P: np.ndarray) -> np.ndarray:
        """Fold the nodal forces acting on slave DOFs onto their masters:
        :math:`\\tilde{P} = T^T P` (slave rows zero) -- the action-reaction transfer through the
        rigid interpolation link.

        Parameters
        ----------
        P
            The assembled force vector.

        Returns
        -------
        np.ndarray
            The folded force vector.
        """

        return self._T.T @ P

    def applySlaveKinematics(self, V: np.ndarray):
        """Assign the slave DOFs their master-interpolated values (in place):
        :math:`V_s = \\sum_a N_a \\, V_{m_a}`. Used on the velocity vector in explicit dynamics;
        displacements then follow automatically from the time integration.

        Parameters
        ----------
        V
            The vector to slave, modified in place.
        """

        V[self.slaveDofIndices] = self._W @ V
