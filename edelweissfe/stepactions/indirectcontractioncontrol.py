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
# Created on Thu May 12 18:35:44 2022

# @author: Matthias Neuner

import numpy as np

from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.stepactions.base.stepactionbase import StepActionBase
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.inputlanguage import InputLanguage

"""
Indirect (displacement) controller for the NISTArcLength solver
uses a ring to control the contraction, e.g., for tunneling simulations.

Currently 2D only!

The center is autotically computed from the bounding node coordinates.
"""


inputLanguage = InputLanguage()

# Register this step action for all available step types. This requires the step type
# modules to be imported before the step actions, as done in the input file parser.
modules = inputLanguage["step"].modules if "step" in inputLanguage else []

documentation = []

for module in modules:
    kw = module.addOptionalKeyword(
        "indirectcontractioncontrol",
        "Indirect (displacement) controller for the NISTArcLength solver using a ring to control the contraction, e.g., for tunneling simulations.",
    )
    kw.addRequiredArg("name", "Name of the step action.", str)
    kw.addRequiredArg("contractionNSet", "The node set defining the contraction ring", str)
    kw.addRequiredArg("L", "Final distance (e.g. crack opening)", float)
    kw.addOptionalArg("exportCVector", "File to export the computed c vector", str, None)
    kw.addOptionalArg("absolute", "Use absolute formulation", bool, True)

    documentation.append(kw)


class StepAction(StepActionBase):
    """Indirect (displacement) controller for the NISTPArcLength solver, controlling the average
    radial contraction of a ring of nodes, e.g. for tunneling simulations.

    The controlled quantity is the mean of the inward radial displacements of the nodes in
    ``contractionNSet``; the ring's center is computed from the bounding coordinates of that set.
    Currently 2D only.

    The constructor is typed: it takes the node set itself rather than its name, so the c vector is
    derived from real coordinates. Nothing here parses an input file -- resolving
    ``contractionNSet=innerRing`` against the model and honouring ``exportCVector`` is the job of
    :meth:`fromStepActionDefinition` / :meth:`updateStepActionFromDefinition` below, which is the
    only part of this module the ``.inp`` front-end needs.

    Parameters
    ----------
    name
        The name of this step action.
    contractionNSet
        The node set defining the contraction ring.
    L
        The final value of the mean contraction to be reached at the end of the step. Interpreted as
        an increment on top of the contraction already reached in previous steps unless ``absolute``
        is set.
    model
        The model tree.
    journal
        The journal object for logging.
    absolute
        If True, ``L`` is the absolute target contraction, i.e. the contraction reached in previous
        steps is subtracted from it. Fixed at construction time: a later step re-declaring this
        action cannot change the formulation.
    """

    identification = "IndirectControl"

    def __init__(
        self,
        name: str,
        contractionNSet: NodeSet,
        L: float,
        model: FEModel,
        journal: Journal,
        absolute: bool = True,
    ):
        self.name = name
        self.journal = journal
        self.model = model

        self.currentL0 = 0.0
        self._currentL = 0.0

        self.L = L

        self.generateCVector(contractionNSet)

        self.absolute = absolute

    @classmethod
    def fromStepActionDefinition(cls, name, definition, jobInfo, model, fieldOutputController, journal):
        """Build this controller from a parsed ``>>indirectcontractioncontrol`` definition. See
        :class:`StepActionBase` for why this is separate from ``__init__``.

        The ``exportCVector`` dump is written here rather than in ``__init__`` because appending
        ``.csv`` to a user-supplied file name is input-file shaping, and a typed constructor should
        not write files as a side effect -- a programmatic caller that wants the dump can
        ``np.savetxt`` the public ``cVector`` itself."""

        stepAction = cls(
            name,
            model.nodeSets[definition["contractionNSet"]],
            definition["L"],
            model,
            journal,
            absolute=definition["absolute"],
        )

        if definition["exportCVector"] is not None:
            np.savetxt(definition["exportCVector"] + ".csv", stepAction.cVector)

        return stepAction

    def updateStepActionFromDefinition(self, definition, jobInfo, model, fieldOutputController, journal):
        """Update from a parsed ``>>indirectcontractioncontrol`` definition re-declared in a later
        step.

        Re-declaring the control target ``L`` per step is the normal way this action is used, so
        this path is not a corner case. Note that neither ``definition["absolute"]`` nor
        ``definition["exportCVector"]`` is re-read here: the formulation has always been fixed by
        the first declaration, and the c vector dump has always been written once, at creation."""

        self.updateStepAction(model.nodeSets[definition["contractionNSet"]], definition["L"])

    def _getIdcsInDofVector(self, dofManager) -> np.ndarray:
        """Determine the indices of the contraction ring displacements in the dof vector.

        Parameters
        ----------
        dofManager
            The dof manager of the current equation system.

        Returns
        -------
        np.ndarray
            The indices in the dof vector.
        """

        return np.hstack(
            [dofManager.idcsOfFieldVariablesInDofVector[n.fields["displacement"]][:2] for n in self.contractionNSet]
        )

    def computeDDLambda(self, dU, ddU_0, ddU_f, timeStep: TimeStep, dofManager):
        """Compute the increment of the arc length load parameter for the current iteration.

        Parameters
        ----------
        dU
            The current increment of the solution vector.
        ddU_0
            The correction of the solution vector due to the dead and the current reference load.
        ddU_f
            The correction of the solution vector due to the unit reference load.
        timeStep
            The current time step.
        dofManager
            The dof manager of the current equation system.

        Returns
        -------
        float
            The increment of the load parameter.
        """

        idcs = self._getIdcsInDofVector(dofManager)

        dL = timeStep.stepProgressIncrement * self.L

        ddLambda = (dL - self.cVector.dot(dU[idcs] + ddU_0[idcs])) / self.cVector.dot(ddU_f[idcs])
        return ddLambda

    def finishIncrement(self, U, dU, dLambda, timeStep: TimeStep, dofManager):
        """Store the contraction reached at the end of a converged increment.

        Parameters
        ----------
        U
            The current solution vector.
        dU
            The current increment of the solution vector.
        dLambda
            The increment of the load parameter.
        timeStep
            The current time step.
        dofManager
            The dof manager of the current equation system.
        """

        idcs = self._getIdcsInDofVector(dofManager)
        self._currentL = self.cVector.dot(U[idcs] + dU[idcs])

    def applyAtStepEnd(self, model):
        """Remember the contraction reached in this step, so that the absolute formulation of a
        subsequent step accounts for it.

        Parameters
        ----------
        model
            The current state of the model.
        """

        self.currentL0 = self._currentL

    def updateStepAction(self, contractionNSet: NodeSet, L: float):
        """Control a new contraction ring and target contraction.

        Parameters
        ----------
        contractionNSet
            The node set defining the contraction ring.
        L
            The target contraction for this step.
        """

        if self.absolute:
            self.L = L - self.currentL0
        else:
            self.L = L

        self.generateCVector(contractionNSet)

    def generateCVector(self, contractionNSet: NodeSet):
        """Derive the c vector of the inward radial unit vectors of the ring's nodes, scaled such
        that the controlled quantity is the ring's mean contraction.

        The ring's center is computed from the bounding coordinates of the node set.

        Parameters
        ----------
        contractionNSet
            The node set defining the contraction ring.
        """

        nNodes = len(contractionNSet)

        allCoordinates = np.array([n.coordinates for n in contractionNSet])

        x_min = np.min(allCoordinates[:, 0])
        x_max = np.max(allCoordinates[:, 0])
        y_min = np.min(allCoordinates[:, 1])
        y_max = np.max(allCoordinates[:, 1])

        x_center = 0.5 * (x_max + x_min)
        y_center = 0.5 * (y_max + y_min)

        cVector = []

        for n in contractionNSet:
            vec_n_to_center = np.array([x_center - n.coordinates[0], y_center - n.coordinates[1]])
            norm_vec_n_to_center = np.linalg.norm(vec_n_to_center)

            vec_n_to_center_normalized = vec_n_to_center / norm_vec_n_to_center

            cVector.append(vec_n_to_center_normalized)

        self.cVector = np.hstack(cVector)

        # dividing c vector to make 'average' contraction of ring:
        self.cVector *= 1.0 / nNodes

        self.contractionNSet = contractionNSet
