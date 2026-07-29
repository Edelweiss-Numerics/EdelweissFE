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
# Created on Thu Nov 15 13:15:14 2018

# @author: Matthias Neuner

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from edelweissfe.stepactions.base.amplitude import (
    amplitudeFromExpression,
    linearAmplitude,
)
from edelweissfe.stepactions.base.bodyloadbase import BodyLoadBase
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.caseinsensitivedict import CaseInsensitiveDict
from edelweissfe.utils.inputlanguage import InputLanguage
from edelweissfe.utils.schema import buildSchemaFromOptions, schemaField

"""
Simple body force load.
If not modified in subsequent steps, the load held constant.
"""


inputLanguage = InputLanguage()
# Register this step action for all available step types. This requires the step type
# modules to be imported before the step actions, as done in the input file parser.
modules = inputLanguage["step"].modules if "step" in inputLanguage else []

documentation = []

for module in modules:
    kw = module.addOptionalKeyword("bodyforce", "Apply body forces on element sets.")
    kw.addRequiredArg("name", "Name of the step action.", str)
    kw.addRequiredArg("elSet", "The element set for application of the boundary condition.", str)
    kw.addRequiredArg("forceVector", "The force vector.", str)
    kw.addOptionalArg("f(t)", "Define an amplitude in the step progress interval [0...1]", str, None)
    kw.addOptionalArg("delta", "In subsequent steps only: define the updated force vector incrementally", str, 0)

    documentation.append(kw)


@dataclass(frozen=True)
class BodyForceSchema:
    """L2: the scalar options of the ``bodyforce`` keyword, owned by this module and never mutated
    from outside it.

    Mirrors the ``module.addRequiredArg``/``addOptionalArg(...)`` declarations above one-for-one.
    The two declarations coexist while the migration is in progress; the ``Module`` one goes away
    with the ``InputLanguage`` singleton in P5.

    ``elSet`` is *not* a schema field: it names an existing model object, resolved by
    :meth:`fromStepActionDefinition` before the schema is even built, exactly like every other
    category's structural names. ``forceVector`` is declared ``required=True`` explicitly, but is
    still given a ``default=None`` so the schema remains constructible for the L1 constructor's
    default argument.
    """

    forceVector: str | None = schemaField(description="The force vector.", dtype=str, default=None, required=True)
    f_t: str | None = schemaField(
        description="Define an amplitude in the step progress interval [0...1]",
        dtype=str,
        default=None,
        optionName="f(t)",
    )
    delta: str | None = schemaField(
        description="In subsequent steps only: define the updated force vector incrementally",
        dtype=str,
        default=None,
    )


class StepAction(BodyLoadBase):
    """Body force load, based on an element set.

    The constructor is typed: it takes the element set itself, the force vector as an
    ``np.ndarray`` and the amplitude as a callable. Nothing here parses an input file -- turning
    ``elSet=all``, ``forceVector='0.0, 10.0'`` and ``f(t)='t**2'`` into those arguments is the job of
    :meth:`fromStepActionDefinition` below, which is the only part of this module the ``.inp``
    front-end needs.

    The load accumulates *over steps*: the force reached at the end of a step is remembered, and the
    next step's declaration prescribes either a new total (``forceVector``) or an increment on top of
    that total (``delta``). See :meth:`updateStepAction` for the exact convention.

    Parameters
    ----------
    name
        The name of this step action.
    elSet
        The element set the body force is applied to.
    forceVector
        The force vector, with one entry per spatial dimension.
    model
        The model tree.
    journal
        The journal object for logging.
    f_t
        The amplitude over the step progress interval ``[0...1]``. Defaults to the identity, i.e. the
        force vector is reached linearly at the end of the step.
    """

    #: L2 schema declared for the L3 registry, per OptionSchemaProvider.
    schema = BodyForceSchema

    def __init__(
        self,
        name,
        elSet,
        forceVector: np.ndarray,
        model,
        journal,
        f_t: Callable[[float], float] = None,
    ):
        self._name = name
        self._elSet = elSet

        self._journal = journal
        self._model = model

        if len(forceVector) < model.domainSize:
            raise Exception("BodyForce {:}: force vector has wrong dimension!".format(self._name))

        self._forceAtStepStart = 0.0

        self.updateStepAction(forceVector=forceVector, f_t=f_t)

    @classmethod
    def fromStepActionDefinition(cls, name, definition, jobInfo, model, fieldOutputController, journal):
        """Build this body force from a parsed ``>>bodyforce`` definition. See
        :class:`StepActionBase` for why this is separate from ``__init__``.

        ``elSet`` is structural (it names a model object), so it is resolved directly and popped
        before the remaining options are validated against :class:`BodyForceSchema`."""

        definition = CaseInsensitiveDict(definition)
        elSetName = definition.pop("elSet")
        configuration = buildSchemaFromOptions(cls.schema, definition)

        return cls(
            name,
            model.elementSets[elSetName],
            np.fromstring(configuration.forceVector, sep=",", dtype=np.double),
            model,
            journal,
            f_t=amplitudeFromExpression(configuration.f_t),
        )

    def updateStepActionFromDefinition(self, definition, jobInfo, model, fieldOutputController, journal):
        """Update from a parsed ``>>bodyforce`` definition re-declared in a later step.

        The two magnitude options are mutually exclusive and ``forceVector`` wins, exactly as before:
        ``forceVector`` is a new *total*, ``delta`` an *increment*.

        Unlike ``nodeforces``/``distributedload``, this module's ``delta`` really is **unreachable**.
        A *partial* re-declaration would have to be validated against an ``updatebodyforce`` keyword
        (that is how the parser handles a re-declaration missing required args, see
        ``utils/inputfileparser.py``, ``parseModuleKeywordLine``) and no such keyword is declared, so
        a partial ``>>bodyforce`` fails parsing outright. What remains is the *full* re-declaration,
        which always carries ``forceVector`` -- a required arg -- so the first branch always wins.
        The ``elif`` is carried across unchanged rather than simplified away, so that the day
        ``bodyforce`` grows an update keyword the intended semantics are still written down here.
        """

        definition = CaseInsensitiveDict(definition)
        definition.pop("elSet")
        configuration = buildSchemaFromOptions(self.schema, definition)

        forceVector = None
        delta = None

        if configuration.forceVector is not None:
            forceVector = np.fromstring(configuration.forceVector, sep=",", dtype=np.double)
        elif configuration.delta is not None:
            delta = np.fromstring(configuration.delta, sep=",", dtype=np.double)

        self.updateStepAction(
            forceVector=forceVector,
            delta=delta,
            f_t=amplitudeFromExpression(configuration.f_t),
        )

    def updateStepAction(
        self,
        forceVector: np.ndarray = None,
        delta: np.ndarray = None,
        f_t: Callable[[float], float] = None,
    ):
        """Prescribe a new force vector and amplitude on the same element set.

        The load accumulated up to the start of this step stays untouched; what is prescribed here is
        the increment applied on top of it during this step. ``forceVector`` and ``delta`` are two
        ways of expressing that increment and ``forceVector`` takes precedence; supplying neither
        leaves the increment as it is, which after a completed step means zero -- i.e. the load is
        held constant at its accumulated level.

        Parameters
        ----------
        forceVector
            The new *total* force vector, i.e. the value to be reached at the end of this step. The
            increment applied during the step is the difference to the accumulated force.
        delta
            The *increment* of the force vector to be applied during this step. Only consulted if
            ``forceVector`` is omitted.
        f_t
            The amplitude over the step progress interval ``[0...1]``; the identity if omitted.
        """

        if forceVector is not None:
            self._delta = forceVector - self._forceAtStepStart
        elif delta is not None:
            self._delta = delta

        self._amplitude = f_t if f_t is not None else linearAmplitude

        self._idle = False

    def applyAtStepEnd(self, model, stepMagnitude=None):
        """Fold this step's increment into the accumulated force and go idle.

        Idle means "no increment pending": until a later step re-declares this load, it stays at the
        accumulated level.

        Parameters
        ----------
        model
            The current state of the model.
        stepMagnitude
            The fraction of the increment that was actually applied. None means the full increment,
            i.e. the amplitude evaluated at the end of the step; the arc length solvers pass their
            load parameter here instead.
        """

        if not self._idle:
            if stepMagnitude is None:
                # standard case
                self._forceAtStepStart += self._delta * self._amplitude(1.0)
            else:
                # set the 'actual' increment manually, e.g. for arc length method
                self._forceAtStepStart += self._delta * stepMagnitude

            self._delta = 0
            self._idle = True

    def getCurrentLoad(self, timeStep: TimeStep) -> np.ndarray:
        """The force vector at the current point of the step.

        Parameters
        ----------
        timeStep
            The current time step.

        Returns
        -------
        np.ndarray
            The accumulated force plus the amplitude-scaled increment of this step.
        """

        if self._idle is True:
            t = 1.0
        else:
            t = timeStep.stepProgress

        return self._forceAtStepStart + self._delta * self._amplitude(t)

    @property
    def elementSet(self) -> list:
        """The elements this body force is acting on.

        Returns
        -------
        list
            The element set.
        """

        return self._elSet
