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

from dataclasses import dataclass

import numpy as np

from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.numerics.dofmanager import DofManager
from edelweissfe.stepactions.base.indirectdisplacementcontrolbase import (
    IndirectDisplacementControlBase,
)
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.caseinsensitivedict import CaseInsensitiveDict
from edelweissfe.utils.math import evalModelAccessibleExpression
from edelweissfe.utils.misc import withoutParserBookkeepingKeys
from edelweissfe.utils.schema import buildSchemaFromOptions, schemaField
from edelweissfe.variables.fieldvariable import FieldVariable

"""
Indirect (displacement) controller for the NISTPArcLength solver
"""


@dataclass(frozen=True)
class IndirectControlSchema:
    """The scalar options of the ``indirectcontrol`` keyword, owned by this module and never
    mutated from outside it.

    ``name`` is a ``structuralOnly`` field: it is popped by ``helpers/inputfilehelpers.py`` before
    the schema is built, and declared here so that the rendered grammar surface documents it; the
    solver option ``arcLengthController`` selects the controller by it. None of
    ``dof1``/``dof2``/``cVector1``/``cVector2`` is structural in the usual sense (a model dict lookup
    by a fixed key): each is a model-access *expression*, evaluated by
    :func:`~edelweissfe.utils.math.evalModelAccessibleExpression`/``eval`` in
    :meth:`~StepAction._dofFromDefinition`/:meth:`~StepAction._cVectorFromDefinition`, so all four
    stay plain string schema fields, exactly like ``modelupdate``'s ``update``.
    """

    name: str | None = schemaField(
        description="Name of the step action, by which the solver option arcLengthController selects it.",
        dtype=str,
        default=None,
        required=True,
        structuralOnly=True,
    )
    dof1: str | None = schemaField(
        description="Degree of freedom for the constraint (model access expression).",
        dtype=str,
        default=None,
        required=True,
    )
    dof2: str | None = schemaField(
        description="Degree of freedom for the constraint (model access expression).",
        dtype=str,
        default=None,
        required=True,
    )
    cVector1: str | None = schemaField(description="c vector.", dtype=str, default=None, required=True)
    cVector2: str | None = schemaField(description="c vector.", dtype=str, default=None, required=True)
    L: float | None = schemaField(
        description="Final distance (e.g. crack opening)", dtype=float, default=None, required=True
    )
    exportCVector: str | None = schemaField(description="File to export the computed c vector", dtype=str, default="")
    absolute: bool | None = schemaField(description="Use absolute formulation", dtype=bool, default=True)
    cLambda: float | None = schemaField(
        description="Weight of the load parameter in the controlled quantity", dtype=float, default=0.0
    )


class StepAction(IndirectDisplacementControlBase):
    """Indirect (displacement) controller for the NISTPArcLength solver, based on two field
    variables and the pair of c vectors weighting their components.

    The controlled quantity is ``c1·dof1 + c2·dof2``, e.g. the opening between two nodes if the
    c vectors are opposite unit vectors, and the solver's load parameter increment is chosen such
    that this quantity follows ``L`` over the step; see
    :class:`~edelweissfe.stepactions.base.indirectdisplacementcontrolbase.IndirectDisplacementControlBase`.

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
        The target of the step: with ``absolute``, the value of the controlled quantity (e.g. a
        crack opening) at the end of the step, counted from its value in the model now; else its
        change over the step.
    model
        The model tree.
    journal
        The journal object for logging.
    absolute
        Whether ``L`` is the value of the controlled quantity at the end of the step, or its change
        over the step. Fixed at construction time: a later step re-declaring this action cannot
        change the formulation.
    cLambda
        The weight of the load parameter in the controlled quantity, see the base class. Fixed at
        construction time.
    """

    identification = "IndirectControl"

    #: Option schema for this step action, consumed by OptionSchemaProvider's registry.
    schema = IndirectControlSchema

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
        cLambda: float = 0.0,
    ):
        super().__init__(name, absolute, cLambda)
        self.journal = journal
        self.model = model

        self.updateStepAction(dof1, dof2, cVector1, cVector2, L)

    @classmethod
    def fromStepActionDefinition(cls, name, definition, jobInfo, model, fieldOutputController, journal):
        """Build this controller from a parsed ``>>indirectcontrol`` definition. See
        :class:`~edelweissfe.stepactions.base.stepactionbase.StepActionBase` for why this is separate from
        ``__init__``.

        ``name`` and the parser's bookkeeping keys are stripped before the remaining options are
        validated against :class:`IndirectControlSchema`; there is nothing structural to pop, since
        every option here is an expression string rather than a model dict lookup key."""

        definition = CaseInsensitiveDict(withoutParserBookkeepingKeys(definition))
        definition.pop("name", None)
        configuration = buildSchemaFromOptions(cls.schema, definition)

        return cls(
            name,
            cls._dofFromExpression(configuration.dof1, model),
            cls._dofFromExpression(configuration.dof2, model),
            cls._cVectorFromExpression(configuration.cVector1),
            cls._cVectorFromExpression(configuration.cVector2),
            configuration.L,
            model,
            journal,
            absolute=configuration.absolute,
            cLambda=configuration.cLambda,
        )

    def updateStepActionFromDefinition(self, definition, jobInfo, model, fieldOutputController, journal):
        """Update from a parsed ``>>indirectcontrol`` definition re-declared, under the same ``name``,
        in a later step: the normal way to give the controller a new target ``L`` per step.

        ``absolute`` is deliberately not re-read: the formulation is fixed by the first declaration
        (see the ``absolute`` entry in the class docstring). There is no ``update<keyword>`` grammar
        for this module, so a re-declaration is always validated against the full ``indirectcontrol``
        keyword and ``buildSchemaFromOptions`` is safe here too."""

        definition = CaseInsensitiveDict(withoutParserBookkeepingKeys(definition))
        definition.pop("name", None)
        configuration = buildSchemaFromOptions(self.schema, definition)

        self.updateStepAction(
            self._dofFromExpression(configuration.dof1, model),
            self._dofFromExpression(configuration.dof2, model),
            self._cVectorFromExpression(configuration.cVector1),
            self._cVectorFromExpression(configuration.cVector2),
            configuration.L,
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
            The target of the step, see the class docstring.
        """

        self.dof1 = dof1
        self.dof2 = dof2

        self.c1 = np.asarray(cVector1, dtype=float)
        self.c2 = np.asarray(cVector2, dtype=float)

        self.cVector = np.hstack([self.c1, self.c2])

        self.declareTarget(L)

    def controlledDofIndices(self, dofManager: DofManager) -> np.ndarray:
        """The indices of the components of both field variables in the dof vector.

        Parameters
        ----------
        dofManager
            The dof manager of the equation system.

        Returns
        -------
        np.ndarray
            The indices, those of ``dof1`` first.
        """

        return np.hstack(
            [
                dofManager.idcsOfFieldVariablesInDofVector[self.dof1],
                dofManager.idcsOfFieldVariablesInDofVector[self.dof2],
            ]
        )

    def controlledValuesOfModel(self) -> np.ndarray:
        """The values of both field variables in the model.

        Returns
        -------
        np.ndarray
            The values, those of ``dof1`` first.
        """

        return np.hstack([self.dof1.values, self.dof2.values])

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

    @staticmethod
    def _dofFromExpression(expression: str, model: FEModel) -> FieldVariable:
        """Resolve a validated schema's model access expression (``dof1`` or ``dof2``) into a field
        variable.

        Parameters
        ----------
        expression
            The model access expression, e.g. ``'model.nodes[18].fields["displacement"]'``.
        model
            The model tree.

        Returns
        -------
        FieldVariable
            The field variable the expression evaluates to.
        """

        return evalModelAccessibleExpression(expression, model)

    @staticmethod
    def _cVectorFromExpression(expression: str) -> np.ndarray:
        """Evaluate a validated schema's c vector expression (``cVector1`` or ``cVector2``).

        Parameters
        ----------
        expression
            The c vector expression, e.g. ``'0, -1'``.

        Returns
        -------
        np.ndarray
            The c vector.
        """

        # An entry of 'x' marks a component as not participating in the controlled quantity, which
        # is expressed as a zero weight.
        return np.asarray(eval(expression.replace("x", "0")), dtype=float)
