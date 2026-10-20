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
#: threaded condensation (``useThreadedCondensation``) against the plain SciPy ``T^T @ K @ T + C``
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
weights :math:`N_a` in the master columns, slave columns entirely zero), both the slave rows and the
slave columns of :math:`T^T K T` are structurally zero. They are filled with the identity
:math:`I_s` (ones on the slave diagonal, nothing else), so the condensed implicit system reads

.. math::
    (T^T K \\, T + I_s) \\; \\delta U = \\tilde{R},
    \\qquad \\tilde{R}_m = (T^T R)_m, \\qquad
    \\tilde{R}_s = -\\Big(dU_s - \\sum_a N_a \\, dU_{m_a}\\Big),

and the slave DOFs are recovered from the masters after the linear solve
(:meth:`MultiPointConstraintTransformation.recoverSlaveSolution`):

.. math::
    \\delta U_s \\leftarrow \\tilde{R}_s + \\sum_a N_a \\, \\delta U_{m_a},

which satisfies the constraint :math:`dU_s + \\delta U_s = \\sum_a N_a (dU_{m_a} + \\delta U_{m_a})`
exactly. The masters decouple from the slaves completely, so the condensed operator is symmetric
whenever :math:`K` is, and its slave rows carry no scale of their own -- in particular no master
coupling that a symmetric diagonal equilibration :math:`D = \\mathrm{diag}(|A_{ii}|^{-1/2})` would
blow up to :math:`N_a / \\sqrt{K_{mm}}` for a small-valued operator such as a consistent mass matrix.

The Dirichlet elimination of a prescribed master's column moves nothing into a slave row (the slave
rows hold no master columns), and the linear solve returns the prescribed increment at that master,
so the recovery :math:`\\delta U_s \\mathrel{+}= \\sum_a N_a \\, \\delta U_{m_a}` with the full weights
is exact for prescribed and free masters alike. (Writing the raw constraint equation into the slave
rows instead, :math:`C_{ss} = 1, C_{s m_a} = -N_a`, would yield the same solution, but a
block-triangular, non-symmetric operator.)

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
    nodes. The substitution works across all of a model's multi-point constraints, in whatever order
    they were collected; :meth:`~edelweissfe.adaptivity.refinement.AdaptiveMesh.hanging_mpc_records`
    uses it for the chains within the hanging-node constraint's own records.

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
    useThreadedCondensation
        Compute ``transformSystemMatrix`` with the OpenMP-threaded kernel of
        :class:`~edelweissfe.numerics.mpccondensation.MPCCondensation` instead of SciPy's
        single-threaded CSR routines: every row of ``Tᵀ K T + C`` is computed directly from the rows
        of ``K`` it combines, with no intermediate matrix, and rows of ``K`` that no constraint
        touches are copied unchanged. The result does not depend on the number of threads.

        Explicit zeros are kept (the pattern is the structural pattern of ``Tᵀ K T + C``), unlike
        SciPy's products, which drop exact-cancellation zeros, so this leaves more raw nnz than the
        plain expression. **Not** pruned here: :meth:`~edelweissfe.solvers.nonlinearimplicitstatic.NIST.
        applyDirichletToStiffness` calls ``eliminate_zeros()`` immediately after this, gated by the
        ``pruneCondensedMatrixZeros`` option (default auto: pruned for PARDISO, kept for
        ``blockamg``, whose hierarchy reuse needs a pattern that is stable across iterations).
    """

    def __init__(
        self,
        records: list[tuple[int, list[tuple[int, float]]]],
        nDof: int,
        useThreadedCondensation: bool = False,
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

        # C: the identity on the slave DOFs only, filling the (structurally zero) slave rows and columns
        # of T^T K T. Deliberately NOT the raw constraint equation (C_sm = -N_a): that would couple the
        # slave rows to the masters one-sidedly -- a non-symmetric operator whose slave-to-master
        # couplings blow up under a symmetric diagonal equilibration of a small-valued K (e.g. a
        # consistent mass matrix). The slaves are instead recovered after the solve, see
        # recoverSlaveSolution.
        nSlaves = len(self.slaveDofIndices)
        self._C = csr_matrix((np.ones(nSlaves), (self.slaveDofIndices, self.slaveDofIndices)), shape=(nDof, nDof))

        self._useThreadedCondensation = useThreadedCondensation
        # Built by _transformSystemMatrixThreaded on first use, from T, T^T and C, which are fixed for the
        # lifetime of this object.
        self._threadedCondensation = None

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
        Kt.setdiag(Kt.diagonal())
        Kt.sort_indices()
        return Kt

    def _assertExact(self, K: csr_matrix, result: csr_matrix) -> None:
        """:envvar:`EDELWEISS_MPC_ASSERT_EXACT` cross-check: the threaded result must match
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
                "threaded result and the plain T^T K T + C expression: max|delta|={:.3e}, "
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
        """Condense the system matrix: :math:`\\tilde{K} = T^T K \\, T + C`, with :math:`C` the identity on
        the slave DOFs. The slave DOFs of the solution must be completed with
        :meth:`recoverSlaveSolution` after the linear solve.

        Dispatches to the threaded kernel (``useThreadedCondensation``) or the plain SciPy
        expression (default) -- see the constructor's ``useThreadedCondensation`` docstring.
        """
        self._dumpForOfflineProbe(K)
        if self._useThreadedCondensation:
            result = self._transformSystemMatrixThreaded(K)
            if os.environ.get(_ASSERT_EXACT_ENV_VAR):
                self._assertExact(K, result)
            return result
        return self._transformSystemMatrixLegacy(K)

    @performancetiming.timeit("mpc transform system matrix")
    def _transformSystemMatrixThreaded(self, K: csr_matrix) -> csr_matrix:
        """Condense the system matrix, :math:`T^T K \\, T + C`, with the OpenMP-threaded kernel of
        :class:`~edelweissfe.numerics.mpccondensation.MPCCondensation` -- see the constructor's
        ``useThreadedCondensation`` docstring. The result has sorted column indices and keeps explicit
        zeros; :meth:`~edelweissfe.solvers.nonlinearimplicitstatic.NIST.applyDirichletToStiffness` prunes them
        if ``pruneCondensedMatrixZeros`` says so.
        """
        if self._threadedCondensation is None:
            from edelweissfe.numerics.mpccondensation import MPCCondensation

            self._threadedCondensation = MPCCondensation(self._T, self._T.T.tocsr(), self._C)

        return self._threadedCondensation.condense(K)

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

    def recoverSlaveSolution(self, x: np.ndarray) -> np.ndarray:
        """Complete the solution of the condensed system (in place): add the master-interpolated
        part to the slave DOFs, :math:`x_s \\mathrel{+}= \\sum_a N_a \\, x_{m_a}`.

        Must be called right after every linear solve of a system condensed with
        :meth:`transformSystemMatrix` and :meth:`transformResidual`, before the solution is used: the
        solve itself only returns the drift correction (the condensed residual's slave rows) at the
        slave DOFs. The weights read master entries only (masters are independent DOFs), so the
        order of the update does not matter. Prescribed (Dirichlet) masters are included: the solve
        returns their prescribed increment, and the Dirichlet column elimination has moved nothing
        into the slave rows.

        Parameters
        ----------
        x
            The solution of the condensed system, of shape ``(nDof,)`` or ``(nDof, nRhs)``; modified
            in place.

        Returns
        -------
        np.ndarray
            The completed solution (the same object as ``x``).
        """

        x[self.slaveDofIndices] += self._W @ x
        return x

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
