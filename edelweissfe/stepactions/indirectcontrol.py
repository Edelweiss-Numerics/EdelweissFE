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
# Created on Thu Nov  2 18:35:44 2017

# @author: Matthias Neuner

import numpy as np

from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.numerics.dofmanager import DofManager
from edelweissfe.stepactions.base.stepactionbase import StepActionBase
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.inputlanguage import InputLanguage
from edelweissfe.utils.math import evalModelAccessibleExpression
from edelweissfe.variables.fieldvariable import FieldVariable

"""
Indirect (displacement) controller for the NISTArcLength solver
"""


inputLanguage = InputLanguage()

# Register this step action for all available step types. This requires the step type
# modules to be imported before the step actions, as done in the input file parser.
modules = inputLanguage["step"].modules if "step" in inputLanguage else []

documentation = []

for module in modules:
    kw = module.addOptionalKeyword(
        "indirectcontrol",
        "Indirect (displacement) controller for the NISTArcLength solver using a ring to control the contraction, e.g., for tunneling simulations.",
    )
    # kw.addRequiredArg("name", "Name of the step action.", str)
    kw.addRequiredArg("dof1", "Degree of freedom for the constraint (model access expression).", str)
    kw.addRequiredArg("dof2", "Degree of freedom for the constraint (model access expression).", str)
    kw.addRequiredArg("cVector1", "c vector.", str)
    kw.addRequiredArg("cVector2", "c vector.", str)
    kw.addRequiredArg("L", "Final distance (e.g. crack opening)", float)
    kw.addOptionalArg("exportCVector", "File to export the computed c vector", str, "")
    kw.addOptionalArg("absolute", "Use absolute formulation", bool, True)

    documentation.append(kw)


class StepAction(StepActionBase):
    """Indirect (displacement) controller for the NISTPArcLength solver, based on two field
    variables and the pair of c vectors weighting their components.

    The controlled quantity is ``c1·dof1 + c2·dof2``, e.g. the opening between two nodes if the
    c vectors are opposite unit vectors, and the solver's load parameter increment is chosen such
    that this quantity follows ``L`` over the step.

    The constructor is typed: it takes the two field variables themselves and the two c vectors as
    arrays. Nothing here parses an input file -- turning a model access expression such as
    ``dof1='model.nodes[18].fields["displacement"]'`` and a c vector expression such as
    ``cVector1='0, -1'`` into those arguments is the job of :meth:`fromStepActionDefinition` /
    :meth:`updateStepActionFromDefinition` below, which is the only part of this module the ``.inp``
    front-end needs.

    Parameters
    ----------
    name
        The name of this step action.
    dof1
        The first controlled field variable.
    dof2
        The second controlled field variable.
    cVector1
        The weights of ``dof1``'s components in the controlled quantity.
    cVector2
        The weights of ``dof2``'s components in the controlled quantity.
    L
        The final value of the controlled quantity (e.g. a crack opening) to be reached at the end
        of the step. Interpreted as an increment on top of the value already reached in previous
        steps unless ``absolute`` is set.
    model
        The model tree.
    journal
        The journal object for logging.
    absolute
        If True, ``L`` is the absolute target value, i.e. the value reached in previous steps is
        subtracted from it. Fixed at construction time: a later step re-declaring this action
        cannot change the formulation.
    """

    identification = "IndirectControl"

    def __init__(
        self,
        name: str,
        dof1: FieldVariable,
        dof2: FieldVariable,
        cVector1: np.ndarray,
        cVector2: np.ndarray,
        L: float,
        model: FEModel,
        journal: Journal,
        absolute: bool = True,
    ):
        self.name = name
        self.journal = journal
        self.model = model
        self.currentL0 = 0.0

        self.absolute = absolute

        self.updateStepAction(dof1, dof2, cVector1, cVector2, L)

    @classmethod
    def fromStepActionDefinition(cls, name, definition, jobInfo, model, fieldOutputController, journal):
        """Build this controller from a parsed ``>>indirectcontrol`` definition. See
        :class:`StepActionBase` for why this is separate from ``__init__``."""

        return cls(
            name,
            cls._dofFromDefinition(definition, "dof1", model),
            cls._dofFromDefinition(definition, "dof2", model),
            cls._cVectorFromDefinition(definition, "cVector1"),
            cls._cVectorFromDefinition(definition, "cVector2"),
            definition["L"],
            model,
            journal,
            absolute=definition["absolute"],
        )

    def updateStepActionFromDefinition(self, definition, jobInfo, model, fieldOutputController, journal):
        """Update from a parsed ``>>indirectcontrol`` definition re-declared in a later step.

        **Unreachable from an input file today**, which is worth knowing before relying on it. This
        module declares no ``name`` arg (it is commented out above), so
        ``helpers/inputfilehelpers.py`` gives every declaration an auto-generated unique name
        (``indirectcontrol-0``, ``indirectcontrol-1``, ...). ``StepManager`` therefore matches no
        existing action and takes the *create* branch every time, and the arc-length solver then
        picks ``[...][0]`` out of the accumulated collection, i.e. the first controller ever declared.
        So a per-step re-declared ``L`` has never taken effect, and ``currentL0`` together with the
        whole ``absolute`` formulation is inert via the ``.inp`` front-end. The hook is implemented
        regardless, because it *is* reachable programmatically -- and because whether to fix the
        reachability (declare a ``name``, or have the solver honour the ``arcLengthController``
        option's value) is a product decision, not part of a behaviour-neutral port. Its sibling
        ``indirectcontractioncontrol`` does declare ``name`` and is not affected.

        ``definition["absolute"]`` is deliberately not re-read: the formulation has always been fixed
        by the first declaration (see the ``absolute`` entry in the class docstring), and re-reading
        it would alter the ``L - currentL0`` bookkeeping."""

        self.updateStepAction(
            self._dofFromDefinition(definition, "dof1", model),
            self._dofFromDefinition(definition, "dof2", model),
            self._cVectorFromDefinition(definition, "cVector1"),
            self._cVectorFromDefinition(definition, "cVector2"),
            definition["L"],
        )

    def updateStepAction(
        self,
        dof1: FieldVariable,
        dof2: FieldVariable,
        cVector1: np.ndarray,
        cVector2: np.ndarray,
        L: float,
    ):
        """Control a new pair of field variables, c vectors and target value.

        Parameters
        ----------
        dof1
            The first controlled field variable.
        dof2
            The second controlled field variable.
        cVector1
            The weights of ``dof1``'s components in the controlled quantity.
        cVector2
            The weights of ``dof2``'s components in the controlled quantity.
        L
            The target value of the controlled quantity for this step.
        """

        if self.absolute:
            self.L = L - self.currentL0
        else:
            self.L = L

        self.dof1 = dof1
        self.dof2 = dof2

        self.c1 = np.asarray(cVector1, dtype=float)
        self.c2 = np.asarray(cVector2, dtype=float)

        self.c = np.hstack([self.c1, self.c2])

    def computeDDLambda(self, dU, ddU_0, ddU_f, timeStep: TimeStep, dofManager: DofManager):
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

        idcs = np.hstack(
            [
                dofManager.idcsOfFieldVariablesInDofVector[self.dof1],
                dofManager.idcsOfFieldVariablesInDofVector[self.dof2],
            ]
        )

        dL = timeStep.stepProgressIncrement * self.L

        ddLambda = (dL - self.c.dot(dU[idcs] + ddU_0[idcs])) / self.c.dot(ddU_f[idcs])
        return ddLambda

    def finishIncrement(self, U, dU, dLambda, timeStep: TimeStep, dofManager):
        """Report the currently controlled quantity at the end of a converged increment.

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

        self.journal.message(
            f"C1·DOF1: {self.c1.dot(self.dof1.values)}, C2·DOF2: {self.c2.dot(self.dof2.values)}",
            self.identification,
        )

    def applyAtStepEnd(self, model):
        """Remember the control target reached in this step, so that the absolute formulation of a
        subsequent step accounts for it.

        Parameters
        ----------
        model
            The current state of the model.
        """

        # self.currentL0 = self.c1.dot(self.dof1.values) + self.c2.dot(self.dof2.values)
        self.currentL0 = self.L

    @staticmethod
    def _dofFromDefinition(definition: dict, key: str, model: FEModel) -> FieldVariable:
        """Resolve one of a parsed definition's model access expressions into a field variable.

        Parameters
        ----------
        definition
            The parsed option mapping defining this step action.
        key
            The option holding the model access expression, i.e. ``"dof1"`` or ``"dof2"``.
        model
            The model tree.

        Returns
        -------
        FieldVariable
            The field variable the expression evaluates to.
        """

        return evalModelAccessibleExpression(definition[key], model)

    @staticmethod
    def _cVectorFromDefinition(definition: dict, key: str) -> np.ndarray:
        """Evaluate one of a parsed definition's c vector expressions.

        Parameters
        ----------
        definition
            The parsed option mapping defining this step action.
        key
            The option holding the expression, i.e. ``"cVector1"`` or ``"cVector2"``.

        Returns
        -------
        np.ndarray
            The c vector.
        """

        # An entry of 'x' marks a component as not participating in the controlled quantity, which
        # is expressed as a zero weight.
        return np.asarray(eval(definition[key].replace("x", "0")), dtype=float)
