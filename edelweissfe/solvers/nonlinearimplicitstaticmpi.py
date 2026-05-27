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
MPI-parallel Nonlinear Implicit Static (NIST) solver.

This solver extends the serial NIST solver with MPI-based distributed
memory parallelization using PETSc for the linear algebra backend.

The solver operates in the following mode:
- All ranks have a replicated model (full mesh, all elements).
- Elements are partitioned across ranks for parallel assembly.
- Each rank computes element contributions for its owned elements only.
- Assembly into PETSc distributed Mat/Vec uses global DOF indices.
- Convergence checks use MPI collective reductions.
- Linear solve is performed by PETSc KSP (MUMPS, GAMG, etc.).

This approach provides:
- Parallel element computation (the expensive part).
- Distributed linear solve for large systems.
- Minimal changes to the existing solver infrastructure.
"""

import numpy as np
from scipy.sparse import csr_matrix

import edelweissfe.utils.performancetiming as performancetiming
from edelweissfe.models.femodel import FEModel
from edelweissfe.numerics.dofmanager import DofManager, DofVector, VIJSystemMatrix
from edelweissfe.numerics.mpi.mpiutils import (
    allreduce_array_sum,
    allreduce_max,
    allreduce_sum,
    barrier,
    getComm,
    getRank,
    getSize,
    isMPIAvailable,
    isRootRank,
)
from edelweissfe.numerics.mpi.partitioner import MeshPartition, partitionMesh
from edelweissfe.solvers.nonlinearimplicitstatic import NIST
from edelweissfe.timesteppers.timestep import TimeStep


class NISTMPI(NIST):
    """MPI-parallel Nonlinear Implicit Static solver.

    Extends the serial NIST solver with:
    - Element partitioning across MPI ranks.
    - Parallel element computation (each rank handles its subset).
    - MPI collective operations for convergence checks.
    - PETSc distributed solver backend for the linear system.

    Parameters
    ----------
    jobInfo
        A dictionary containing the job information.
    journal
        The journal instance for logging.
    """

    identification = "NISTMPISolver"

    SolverSpecificOptions = {
        **NIST.SolverSpecificOptions,
        "linsolver": "petsclu",  # Default to PETSc for MPI
    }

    def __init__(self, jobInfo, journal, **kwargs):
        super().__init__(jobInfo, journal, **kwargs)

        self.comm = getComm()
        self.rank = getRank(self.comm)
        self.nRanks = getSize(self.comm)
        self.partition = None

    def solveStep(self, step, model: FEModel, fieldOutputController, outputmanagers):
        """Solve a step with MPI parallelization.

        Partitions the mesh across ranks, then delegates to the
        parent solver with overridden element computation.

        Parameters
        ----------
        step
            The step definition.
        model
            The FE model.
        fieldOutputController
            The field output controller.
        outputmanagers
            The output managers.
        """
        if isRootRank(self.comm):
            self.journal.message(
                "Using MPI with {:} ranks".format(self.nRanks),
                self.identification,
            )

        # Partition the mesh
        self.partition = partitionMesh(model, self.comm)

        if isRootRank(self.comm):
            self.journal.message(
                "Rank 0 owns {:} elements, {:} nodes ({:} ghost)".format(
                    len(self.partition.ownedElementIds),
                    len(self.partition.ownedNodeIds),
                    len(self.partition.ghostNodeIds),
                ),
                self.identification,
            )

        barrier(self.comm)

        # Proceed with the standard solve step
        return super().solveStep(step, model, fieldOutputController, outputmanagers)

    @performancetiming.timeit("elements (MPI)")
    def computeElements(
        self,
        elements: list,
        U_np: DofVector,
        dU: DofVector,
        P: DofVector,
        K: VIJSystemMatrix,
        F: DofVector,
        timeStep: TimeStep,
    ) -> tuple[DofVector, VIJSystemMatrix, DofVector]:
        """Compute element contributions in parallel across MPI ranks.

        Each rank only computes its owned elements, then results are
        combined using MPI Allreduce.

        Parameters
        ----------
        elements
            The dictionary of all elements.
        U_np
            The current solution vector.
        dU
            The current solution increment.
        P
            The reaction vector.
        K
            The system matrix.
        F
            The accumulated flux vector.
        timeStep
            The current time step.

        Returns
        -------
        tuple[DofVector, VIJSystemMatrix, DofVector]
            The assembled reaction vector, system matrix, and flux vector.
        """

        time = np.array([timeStep.stepTime, timeStep.totalTime])
        dT = timeStep.timeIncrement

        # Each rank computes only its owned elements
        if self.partition is not None:
            ownedElementIds = self.partition.ownedElementIds
        else:
            # Fallback: all elements on all ranks (redundant computation)
            ownedElementIds = list(elements.keys())

        for elId in ownedElementIds:
            el = elements[elId]
            Ke = K[el]
            Pe = np.zeros(el.nDof)

            el.computeYourself(Ke, Pe, U_np[el], dU[el], time, dT)

            P[el] += Pe
            F[el] += abs(Pe)

        # Allreduce the vectors to combine contributions from all ranks
        # P and F are DofVectors (numpy arrays), K is VIJSystemMatrix (numpy array)
        P_global = allreduce_array_sum(np.asarray(P), self.comm)
        K_global = allreduce_array_sum(np.asarray(K), self.comm)
        F_global = allreduce_array_sum(np.asarray(F), self.comm)

        P[:] = P_global
        K[:] = K_global
        F[:] = F_global

        return P, K, F

    def checkConvergence(self, R, ddU, F, iterationCounter, incrementResidualHistory):
        """MPI-aware convergence check.

        The convergence norms are already global since we Allreduce
        the assembled vectors. The parent class checkConvergence will
        thus work correctly on all ranks with identical data.

        Parameters
        ----------
        R
            Residual vector.
        ddU
            Correction vector.
        F
            Accumulated flux vector.
        iterationCounter
            Current iteration count.
        incrementResidualHistory
            History of residuals.

        Returns
        -------
        tuple[bool, dict]
            Convergence status and outlier nodes.
        """
        # Since P, K, F are already globally reduced in computeElements,
        # and loads/constraints are also computed on all ranks identically,
        # the residual R is the same on all ranks.
        # The parent checkConvergence will give identical results on each rank.
        return super().checkConvergence(R, ddU, F, iterationCounter, incrementResidualHistory)
