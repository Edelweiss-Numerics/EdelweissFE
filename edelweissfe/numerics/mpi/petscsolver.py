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

"""
Distributed PETSc linear solver for MPI-parallel finite element analysis.

This module provides a linear solver that uses PETSc's distributed
Mat/Vec/KSP infrastructure for solving linear systems across multiple
MPI ranks. Unlike the existing ``petsclu`` solver which constructs a
PETSc matrix from a local SciPy CSR matrix, this solver:

- Creates a truly distributed PETSc matrix with local ownership ranges.
- Assembles element contributions directly into the distributed matrix.
- Uses PETSc KSP with configurable preconditioners and solvers.
- Supports parallel direct solvers (MUMPS, SuperLU_DIST) and iterative
  solvers (GMRES, CG) with PETSc preconditioners.

Usage
-----
This solver is intended to be called from the MPI-aware NIST solver,
which provides assembled element contributions in global DOF indices.
"""

from __future__ import annotations

import numpy as np

from edelweissfe.numerics.mpi.distributeddofmanager import DistributedDofManager
from edelweissfe.numerics.mpi.mpiutils import getComm, getRank, getSize

try:
    from petsc4py import PETSc

    _PETSC_AVAILABLE = True
except ImportError:
    _PETSC_AVAILABLE = False


class DistributedPETScSolver:
    """A distributed PETSc-based linear solver for MPI parallelism.

    This solver creates and manages PETSc Mat/Vec objects that are
    distributed across MPI ranks according to the
    :class:`DistributedDofManager` layout.

    Parameters
    ----------
    distDofManager
        The distributed DOF manager with ownership information.
    solverType
        PETSc KSP solver type (e.g., 'preonly', 'gmres', 'cg').
    preconditionerType
        PETSc PC type (e.g., 'lu', 'ilu', 'gamg', 'jacobi').
    options
        Additional PETSc options as a dict.
    comm
        MPI communicator.
    """

    def __init__(
        self,
        distDofManager: DistributedDofManager,
        solverType: str = "preonly",
        preconditionerType: str = "lu",
        options: dict = None,
        comm=None,
    ):
        if not _PETSC_AVAILABLE:
            raise ImportError("petsc4py is required for DistributedPETScSolver")

        self.distDofManager = distDofManager
        self.comm = comm if comm is not None else getComm()

        # Create PETSc communicator
        if self.comm is not None:
            self.petscComm = PETSc.Comm(self.comm)
        else:
            self.petscComm = PETSc.COMM_WORLD

        nGlobal = distDofManager.nGlobal
        nLocalOwned = distDofManager.nLocalOwned

        # Create distributed matrix
        self.A = PETSc.Mat().create(comm=self.petscComm)
        self.A.setSizes([(nLocalOwned, nGlobal), (nLocalOwned, nGlobal)])
        self.A.setType(PETSc.Mat.Type.AIJ)
        self.A.setFromOptions()
        self.A.setUp()

        # Create distributed vectors
        self.x = self.A.createVecLeft()
        self.b = self.A.createVecRight()

        # Create KSP solver
        self.ksp = PETSc.KSP().create(comm=self.petscComm)
        self.ksp.setType(solverType)
        self.ksp.getPC().setType(preconditionerType)

        # Apply additional options
        if options:
            opts = PETSc.Options()
            for key, val in options.items():
                opts[key] = val
            self.ksp.setFromOptions()

        self.ksp.setConvergenceHistory()

    def assembleMatrix(self, I: np.ndarray, J: np.ndarray, V: np.ndarray):
        """Assemble element contributions into the distributed PETSc matrix.

        Parameters
        ----------
        I : np.ndarray
            Row indices (global DOF numbers).
        J : np.ndarray
            Column indices (global DOF numbers).
        V : np.ndarray
            Values for each (I, J) entry.
        """
        self.A.zeroEntries()

        # Map global DOFs to PETSc global numbering
        # For now, assume global DOF indices map directly to PETSc indices
        # (contiguous ownership is handled by PETSc internally)
        for i, j, v in zip(I, J, V):
            self.A.setValue(int(i), int(j), float(v), addv=PETSc.InsertMode.ADD_VALUES)

        self.A.assemblyBegin()
        self.A.assemblyEnd()

    def assembleMatrixCSR(self, csrMatrix):
        """Assemble from a local SciPy CSR matrix contribution.

        Each rank provides its local contribution to the global matrix.
        Entries are added (not inserted) so overlapping contributions
        from different ranks are summed correctly.

        Parameters
        ----------
        csrMatrix : scipy.sparse.csr_matrix
            Local CSR matrix contribution in global indices.
        """
        self.A.zeroEntries()

        indptr = csrMatrix.indptr
        indices = csrMatrix.indices
        data = csrMatrix.data

        ownedDofs = self.distDofManager.ownedGlobalDofs

        for localRow in range(len(ownedDofs)):
            globalRow = int(ownedDofs[localRow])
            start = indptr[localRow]
            end = indptr[localRow + 1]
            cols = indices[start:end].astype(np.int32)
            vals = data[start:end]
            self.A.setValues(globalRow, cols, vals, addv=PETSc.InsertMode.ADD_VALUES)

        self.A.assemblyBegin()
        self.A.assemblyEnd()

    def assembleRHS(self, rhs: np.ndarray):
        """Assemble the right-hand side vector.

        Parameters
        ----------
        rhs : np.ndarray
            The local portion of the RHS vector (owned DOFs only).
        """
        ownedDofs = self.distDofManager.ownedGlobalDofs

        self.b.zeroEntries()
        for localIdx in range(len(ownedDofs)):
            self.b.setValue(int(ownedDofs[localIdx]), float(rhs[localIdx]), addv=PETSc.InsertMode.ADD_VALUES)

        self.b.assemblyBegin()
        self.b.assemblyEnd()

    def solve(self, rhs: np.ndarray = None) -> np.ndarray:
        """Solve the linear system Ax = b.

        Parameters
        ----------
        rhs : np.ndarray, optional
            If provided, assembles this as the RHS before solving.
            Should contain values for owned DOFs only.

        Returns
        -------
        np.ndarray
            The solution vector (owned DOFs only on this rank).
        """
        if rhs is not None:
            self.assembleRHS(rhs)

        self.ksp.setOperators(self.A)
        self.ksp.solve(self.b, self.x)

        # Extract solution for owned DOFs
        solution = self.x.getArray().copy()
        return solution

    def destroy(self):
        """Release PETSc resources."""
        self.ksp.destroy()
        self.A.destroy()
        self.x.destroy()
        self.b.destroy()

    def __del__(self):
        try:
            self.destroy()
        except Exception:
            pass
