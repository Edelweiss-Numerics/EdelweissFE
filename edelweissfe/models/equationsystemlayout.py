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

"""What the global equation system of a model is made of: the answer a model gives a solver that (re)builds its
equation system, see :meth:`~edelweissfe.models.femodel.FEModel.buildEquationSystem` and
:ref:`model-equation-system`."""

from collections.abc import Callable
from typing import TYPE_CHECKING

from edelweissfe.numerics.dofmanager import DofManager, DofVector
from edelweissfe.timesteppers.timestep import TimeStep

if TYPE_CHECKING:
    from edelweissfe.models.femodel import FEModel


class EquationSystemLayout:
    """The layout of the global equation system of a model: which degrees of freedom it has, which elements contribute
    to it, and how the solution passes between the model and the vectors a solver iterates on.

    A solver asks the model for it whenever it (re)builds its equation system
    (:meth:`~edelweissfe.models.femodel.FEModel.buildEquationSystem`), and keeps it as long as the numbering of the
    degrees of freedom it holds: until the next rebuild. The solver itself only solves; what the model is made of, the
    layout says.

    This class is the layout of a finite element model. A model made of other entities -- EdelweissMeshfree's material
    point and RKPM models, with their background cells and particles -- returns a layout derived from it, which changes
    what its methods do; the solver calls them the same way.

    Parameters
    ----------
    model
        The model.
    dofManager
        The numbering of the degrees of freedom.
    elements
        The elements, in the order in which the solver computes them.
    """

    def __init__(self, model: "FEModel", dofManager: DofManager, elements: list):
        #: The model this equation system belongs to.
        self.model = model
        #: The numbering of the degrees of freedom of the equation system; see
        #: :class:`~edelweissfe.numerics.dofmanager.DofManager`.
        self.dofManager = dofManager
        #: Everything the solver computes like an element in every evaluation of the model, see
        #: :meth:`~edelweissfe.solvers.nonlinearimplicitstatic.NIST.computeElements`: the finite elements of the model,
        #: and in a meshfree model its cells and particles as well. A list, which a parallel solver hands out in slices
        #: to its threads.
        self.elements = elements
        #: The model the linear solver is set up for (see :meth:`~edelweissfe.linsolve.base.LinearSolver.setModel`),
        #: whose node fields are the ones the DOF manager numbers: here, the model itself.
        self.modelForLinearSolver = model

    def writeSolutionToDofVector(self, U: DofVector) -> DofVector:
        """Write the solution the model carries from increment to increment into the solution vector of a (re)built
        equation system, from which the next increment starts. In a finite element model, every node carries its total
        solution: the entry ``U`` of every node field is written, and the values of the scalar variables (e.g. the
        Lagrange multipliers of constraints).

        Parameters
        ----------
        U
            The solution vector of the equation system, overwritten.

        Returns
        -------
        DofVector
            The solution vector.
        """

        for nodeField in self.model.nodeFields.values():
            U = self.dofManager.writeNodeFieldToDofVector(U, nodeField, "U")

        return self.dofManager.writeScalarVariablesToDofVector(U)

    def writeResultsToModel(self, U: DofVector, dU: DofVector, P: DofVector):
        """Write the results of a converged increment to the model: the solution ``U``, its increment ``dU`` and the
        reaction ``P`` to the entries of the same names of the node fields, and the solution to the scalar variables.

        Parameters
        ----------
        U
            The solution.
        dU
            The increment of the solution.
        P
            The reaction, the sum of the elements' internal forces.
        """

        for nodeField in self.model.nodeFields.values():
            self.dofManager.writeDofVectorToNodeField(U, nodeField, "U")
            self.dofManager.writeDofVectorToNodeField(P, nodeField, "P")
            self.dofManager.writeDofVectorToNodeField(dU, nodeField, "dU")

        self.dofManager.writeDofVectorToScalarVariables(U)

    def updateMaterialPoints(self, dU: DofVector, timeStep: TimeStep, forEachChunk: Callable[[list, Callable], None]):
        """Bring the material points that belong to no element up to date with the trial solution increment, right
        before the solver computes the elements. Called whenever a solver evaluates the elements (see
        :meth:`~edelweissfe.solvers.nonlinearimplicitstatic.NIST.assembleInternalForces`), e.g. in every Newton
        iteration and for every trial state of a line search.

        The loops over the entities belong to the solver, as the loop over the elements does: ``forEachChunk(entities,
        work)`` calls ``work(chunk)`` for chunks that together hold all ``entities``, in the calling thread or on the
        threads of a parallel solver (see :meth:`~edelweissfe.solvers.nonlinearimplicitstatic.NIST.forEachChunk`). A
        layout loops through it over entities that are independent of each other.

        A finite element model has no such material points: every element evaluates the material at its own quadrature
        points, in :meth:`~edelweissfe.elements.base.baseelement.BaseElement.computeKernels`, so there is nothing to do
        here. In the material point method, the material points move through a fixed background grid, and belong to the
        cell they currently lie in: here, each cell interpolates the solution increment to its material points, and the
        material points evaluate their material, before the solver computes the cells, which integrate the result.

        Parameters
        ----------
        dU
            The trial solution increment.
        timeStep
            The time step.
        forEachChunk
            The solver's loop over chunks of independent entities.
        """
