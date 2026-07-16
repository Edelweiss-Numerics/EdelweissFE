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

import numpy as np
from scipy.sparse import csr_matrix

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
    """

    def __init__(self, records: list[tuple[int, list[tuple[int, float]]]], nDof: int):
        slaveDofs = [slaveDof for slaveDof, _ in records]

        if len(set(slaveDofs)) != len(slaveDofs):
            raise ValueError("Multi-point constraints: a DOF is claimed as slave by more than one constraint record.")

        allMasterDofs = {masterDof for _, masters in records for masterDof, _ in masters}
        chained = allMasterDofs.intersection(slaveDofs)
        if chained:
            raise ValueError(
                "Multi-point constraints: {:} slave DOF(s) appear as master DOFs of another "
                "constraint -- chained multi-point constraints are not supported.".format(len(chained))
            )

        for slaveDof, masters in records:
            if not masters:
                raise ValueError("Multi-point constraints: slave DOF {:} has no master DOFs.".format(slaveDof))

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

    def transformSystemMatrix(self, K: csr_matrix) -> csr_matrix:
        """Condense the system matrix: :math:`\\tilde{K} = T^T K \\, T + C`.

        Parameters
        ----------
        K
            The assembled system matrix.

        Returns
        -------
        csr_matrix
            The condensed system matrix, same size, with the constraint equations in the slave
            rows.
        """

        Kt = (self._T.T @ K @ self._T + self._C).tocsr()
        Kt.sort_indices()
        return Kt

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
