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

import numpy as np

from edelweissfe.numerics.dofmanager import DofManager, DofVector
from edelweissfe.timesteppers.timestep import TimeStep

if TYPE_CHECKING:
    from edelweissfe.fields.nodefield import NodeField
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
        #: The material points that belong to no element, which :meth:`updateMaterialPoints` updates: none in a
        #: finite element model, whose elements evaluate the material at their own quadrature points.
        self.materialPoints = []

    def readStateFromModel(self, U: DofVector, V: DofVector = None, A: DofVector = None, dynamicFields: list = None):
        """Read the state the model carries from increment to increment into the vectors of a (re)built equation
        system, from which the next increment starts (see the documentation page *Node field results*): the solution
        ``U``, and, for a solver that integrates in time, the velocity ``V`` and the acceleration ``A``.

        The entry ``U`` of every node field is read into ``U``, with the values of the scalar variables (e.g. the
        Lagrange multipliers of constraints), and the entries ``V`` and ``A`` into ``V`` and ``A``, if given; every
        entry through :meth:`readEntryOfNodeField`, which says on which nodes. In a finite element model, every node
        carries its total solution and its velocity, so every entry is read on every node.

        Parameters
        ----------
        U
            The solution vector of the equation system, zero, overwritten.
        V
            The velocity vector of the equation system, zero, overwritten; None if the solver has none.
        A
            The acceleration vector of the equation system, zero, overwritten; None if the solver has none.
        dynamicFields
            The names of the fields whose velocity and acceleration are read, e.g. those the solver integrates in time;
            None for every field.
        """

        for nodeField in self.model.nodeFields.values():
            self.readEntryOfNodeField(U, nodeField, "U")
        self.dofManager.writeScalarVariablesToDofVector(U)

        for nodeField in self._nodeFieldsWithVelocity(dynamicFields):
            if V is not None:
                self.readEntryOfNodeField(V, nodeField, "V")
            if A is not None:
                self.readEntryOfNodeField(A, nodeField, "A")

    def readEntryOfNodeField(self, dofVector: DofVector, nodeField: "NodeField", entry: str):
        """Read one entry of one node field of the model into a vector of the equation system; every read of
        :meth:`readStateFromModel` goes through here, as every write goes through :meth:`writeEntryOfNodeField`.

        In a finite element model, the equation system numbers every node of the field, and every node carries its
        total solution and its velocity: the whole entry is read. A model whose equation system covers a part of its
        nodes reads the entry on the nodes that carry it.

        Parameters
        ----------
        dofVector
            The vector of the equation system, overwritten on the degrees of freedom of the field.
        nodeField
            The node field of the model.
        entry
            The entry, one of :attr:`NodeField.ENTRIES <edelweissfe.fields.nodefield.NodeField.ENTRIES>`.
        """

        self.dofManager.writeNodeFieldToDofVector(dofVector, nodeField, entry)

    def zeroIncrementalCoefficients(self, U: DofVector):
        """Set the degrees of freedom of the nodes that carry no total solution to zero in the solution vector of a
        converged increment, before its results are written to the model: these *incremental* degrees of freedom are
        the coefficients of one increment, and every increment starts them at zero (see
        the documentation page *Node field results*).

        In a finite element model, every node carries its total solution, so there is nothing to do here. A meshfree
        model, whose background grid and kernel functions are incremental, sets their degrees of freedom to zero.

        Parameters
        ----------
        U
            The solution vector of the converged increment, modified in place.
        """

    def writeResultsToModel(self, U: DofVector, dU: DofVector | None, P: DofVector | None):
        """Write the results of an accepted increment to the model, once (see the documentation page *Node field
        results*): the solution ``U``, its increment ``dU`` and the force ``P`` to the entries of the same
        names of the node fields, and the solution to the scalar variables. The velocity and the acceleration, the
        state of a solver that integrates in time, go through :meth:`writeVelocityAndAccelerationToModel`.

        A solver whose increment ends on another equation system than it started on writes in two parts, each through
        the layout of the system that holds the values: EdelweissMeshfree's explicit solver takes the increment on the
        kernel functions the particles reached at its start, and evaluates the force on those they reach at its end.
        It passes None for the entries the other part writes.

        Parameters
        ----------
        U
            The solution.
        dU
            The increment of the solution; None to leave the entry ``dU`` of the node fields as it is.
        P
            The internal force, the sum of the elements' residual contributions, with the inertial and damping forces
            of a solver that integrates in time; None to leave the entry ``P`` of the node fields as it is.
        """

        for nodeField in self.model.nodeFields.values():
            self.writeEntryOfNodeField(U, nodeField, "U")
            if P is not None:
                self.writeEntryOfNodeField(P, nodeField, "P")
            if dU is not None:
                self.writeEntryOfNodeField(dU, nodeField, "dU")
        self.dofManager.writeDofVectorToScalarVariables(U)

    def writeVelocityAndAccelerationToModel(self, V: DofVector | None, A: DofVector | None):
        """Write the velocity and the acceleration to the entries ``V`` and ``A`` of every node field, the one way a
        solver that integrates in time passes them to the model: at acceptance, with :meth:`writeResultsToModel`, and
        when it completes the state an increment starts from, e.g. the acceleration the implicit dynamic solver
        computes from equilibrium at the start of a step.

        Parameters
        ----------
        V
            The velocity; None to leave the entry ``V`` as it is.
        A
            The acceleration; None to leave the entry ``A`` as it is.
        """

        for nodeField in self.model.nodeFields.values():
            if V is not None:
                self.writeEntryOfNodeField(V, nodeField, "V")
            if A is not None:
                self.writeEntryOfNodeField(A, nodeField, "A")

    def writeZeroVelocityAndAccelerationToModel(self):
        """Write zero to the entries ``V`` and ``A`` of every node field of the model, on every node: the state a
        quasi-static solver leaves at acceptance. It does not integrate in time, so the state it accepts is at rest
        (see the documentation page *Node field results*), and a dynamic step that follows starts from rest.

        A zero velocity and a zero acceleration of the equation system, written as every velocity and acceleration
        is, by :meth:`writeVelocityAndAccelerationToModel`. A layout covering a part of the nodes sets the entries to
        zero on the others (see :meth:`writeEntryOfNodeField`), so the model is at rest on every node, inside the
        equation system or not.
        """

        zero = self.dofManager.constructDofVector()
        self.writeVelocityAndAccelerationToModel(zero, zero)

    def writeEntryOfNodeField(self, dofVector: DofVector, nodeField: "NodeField", entry: str):
        """Write a vector of the equation system to one entry of one node field of the model; every write of
        :meth:`writeResultsToModel`, :meth:`writeVelocityAndAccelerationToModel` and
        :meth:`writeZeroVelocityAndAccelerationToModel` goes through here, as every read goes through
        :meth:`readEntryOfNodeField`.

        In a finite element model, the equation system numbers every node of the field, so the whole entry is written.
        A model whose equation system covers a part of its nodes writes the entry on these nodes, and sets it to zero
        on the others.

        Parameters
        ----------
        dofVector
            The vector of the equation system.
        nodeField
            The node field of the model.
        entry
            The entry, one of :attr:`NodeField.ENTRIES <edelweissfe.fields.nodefield.NodeField.ENTRIES>`.
        """

        self.dofManager.writeDofVectorToNodeField(dofVector, nodeField, entry)

    def _nodeFieldsWithVelocity(self, dynamicFields: list | None) -> list:
        """The node fields of the model whose velocity and acceleration a solver reads.

        Parameters
        ----------
        dynamicFields
            Their names, or None for every node field.

        Returns
        -------
        list
            The node fields.
        """

        if dynamicFields is None:
            return list(self.model.nodeFields.values())

        return [self.model.nodeFields[fieldName] for fieldName in dynamicFields]

    def dofsWithNodalInertia(self, fieldNames: list) -> np.ndarray:
        """The degrees of freedom of the given fields whose inertia is that of the nodes, i.e. the mass the elements
        assemble: those a dynamic solver integrates in time with its own Newmark (or central difference) scheme.

        In a finite element model, every node carries its total solution and every element its mass, so these are all
        degrees of freedom of the fields. A meshfree model leaves out the nodes of its kernel functions: their values
        are the coefficients of one increment, while the particles carry the state, and with it the inertia (see
        :meth:`assembleLumpedMassOfParticles`).

        Parameters
        ----------
        fieldNames
            The names of the fields, e.g. those whose inertia is a mass.

        Returns
        -------
        np.ndarray
            The indices of the degrees of freedom in the vectors of the equation system, sorted.
        """

        isWithNodalInertia = np.zeros(self.dofManager.nDof, dtype=bool)
        for fieldName in fieldNames:
            isWithNodalInertia[self.dofManager.idcsOfFieldsInDofVector[fieldName]] = True

        return np.flatnonzero(isWithNodalInertia)

    def assembleLumpedMassOfParticles(self, forEachChunk: Callable[[list, Callable], None]) -> DofVector | None:
        """The lumped (diagonal) mass of the meshfree particles at the nodes of their kernel functions,
        :math:`m_A = \\sum_p \\rho_0\\, V_{0,p}\\, T_A(\\boldsymbol{x}_p)`: what the explicit meshfree solver divides
        the net force by, and what the implicit dynamic solver adds to the mass of the nodes when it integrates the
        particles' inertia at the nodes (option ``meshfreeInertia=nodes``). The particles move with every increment,
        so this mass is one of the current configuration.

        A finite element model has no particles: None.

        Parameters
        ----------
        forEachChunk
            The solver's loop over chunks of independent entities, see :meth:`updateMaterialPoints`.

        Returns
        -------
        DofVector | None
            The lumped mass, one value per degree of freedom; None if the model has no particles.
        """

        return None

    def updateMaterialPoints(self, dU: DofVector, timeStep: TimeStep, forEachChunk: Callable[[list, Callable], None]):
        """Bring the material points that belong to no element up to date with the trial solution increment, right
        before the solver computes the elements. Called whenever ``NIST``, or a solver derived from it (``NEST``,
        ``NID``), evaluates the elements (see
        :meth:`~edelweissfe.solvers.nonlinearimplicitstatic.NIST.assembleInternalForces`), e.g. in every Newton
        iteration and for every trial state of a line search. The explicit dynamic solver ``NED`` evaluates the
        elements of a finite element model itself, and does not call it.

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
