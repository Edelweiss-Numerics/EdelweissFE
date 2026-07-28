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
# Created on Wed May 10 13:12:40 2017

# @author: Matthias Neuner

from collections.abc import Callable

import numpy as np

from edelweissfe.stepactions.base.amplitude import (
    amplitudeFromExpression,
    linearAmplitude,
)
from edelweissfe.stepactions.base.distributedloadbase import DistributedLoadBase
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.inputlanguage import InputLanguage

"""
Standard distributed load, applied on a surface set.
If not modified in subsequent steps, the load held constant.
"""

inputLanguage = InputLanguage()

# Register this step action for all available step types. This requires the step type
# modules to be imported before the step actions, as done in the input file parser.
modules = inputLanguage["step"].modules if "step" in inputLanguage else []


documentation = []

for module in modules:
    kw = module.addOptionalKeyword("distributedload", "Standard distributed load, applied on a surface set.")
    kw.addRequiredArg("name", "Name of the step action.", str)
    kw.addRequiredArg("surface", "Surface for application of the distributed load", str)
    # kw.addRequiredArg("field", "Field for which the boundary condition is active.", str)
    kw.addOptionalArg("field", "Field for which the boundary condition is active.", str, "displacement")
    kw.addRequiredArg("magnitude", "Magnitude of the distributed load", str)
    # kw.addOptionalArg("delta", "In subsequent steps only: define the new magnitude incrementally", str, 0)
    kw.addOptionalArg("f(t)", "Define an amplitude in the step progress interval [0...1]", str, None)
    kw.addRequiredArg(
        "type", "The load type, e.g., pressure or surface traction; Must be supported by the element type", str
    )

    documentation.append(kw)

    kw = module.addOptionalKeyword("updatedistributedload", "Update a previously defined distributedload definition.")
    kw.addRequiredArg("name", "Name of the step action to update.", str)
    # kw.addRequiredArg("surface", "Surface for application of the distributed load", str)
    # kw.addRequiredArg("field", "Field for which the boundary condition is active.", str)
    # kw.addOptionalArg("field", "Field for which the boundary condition is active.", str, "displacement")
    kw.addOptionalArg("magnitude", "Magnitude of the distributed load", str, None)
    kw.addOptionalArg("delta", "In subsequent steps only: define the new magnitude incrementally", str, None)
    kw.addOptionalArg("f(t)", "Define an amplitude in the step progress interval [0...1]", str, None)
    # kw.addRequiredArg(
    #     "type", "The load type, e.g., pressure or surface traction; Must be supported by the element type", str
    # )

    documentation.append(kw)


class StepAction(DistributedLoadBase):
    """Distributed load, defined on an element-based surface.

    The constructor is typed: it takes the surface itself, the magnitude as an ``np.ndarray`` and the
    amplitude as a callable. Nothing here parses an input file -- turning ``surface=gen_top``,
    ``magnitude=0.15`` and ``f(t)='t'`` into those arguments is the job of
    :meth:`fromStepActionDefinition` below, which is the only part of this module the ``.inp``
    front-end needs.

    The load accumulates *over steps*: the magnitude reached at the end of a step is remembered, and
    the next step's declaration prescribes either a new total (``magnitude``) or an increment on top
    of that total (``delta``). See :meth:`updateStepAction` for the exact convention.

    Parameters
    ----------
    name
        The name of this step action.
    surface
        The surface the distributed load is applied to.
    magnitude
        The magnitude of the distributed load.
    loadType
        The load type, e.g. ``"pressure"``; must be supported by the element type.
    model
        The model tree.
    journal
        The journal object for logging.
    f_t
        The amplitude over the step progress interval ``[0...1]``. Defaults to the identity, i.e. the
        magnitude is reached linearly at the end of the step.
    """

    def __init__(
        self,
        name,
        surface,
        magnitude: np.ndarray,
        loadType: str,
        model,
        journal,
        f_t: Callable[[float], float] = None,
    ):
        self._name = name
        self._surface = surface
        self._loadType = loadType

        self._journal = journal
        self._model = model

        self._magnitudeAtStepStart = 0.0

        self.updateStepAction(magnitude=magnitude, f_t=f_t)

    @classmethod
    def fromStepActionDefinition(cls, name, definition, jobInfo, model, fieldOutputController, journal):
        """Build this distributed load from a parsed ``>>distributedload`` definition. See
        :class:`StepActionBase` for why this is separate from ``__init__``.

        The keyword's ``field`` option is deliberately not passed on: this class never consumed it,
        and inventing a meaning for it here would not be a behaviour-neutral port."""

        return cls(
            name,
            model.surfaces[definition["surface"]],
            np.fromstring(definition["magnitude"], sep=","),
            definition["type"],
            model,
            journal,
            f_t=amplitudeFromExpression(definition["f(t)"]),
        )

    def updateStepActionFromDefinition(self, definition, jobInfo, model, fieldOutputController, journal):
        """Update from a parsed ``>>distributedload`` definition re-declared in a later step.

        The two magnitude options are mutually exclusive and ``magnitude`` wins, exactly as before:
        ``magnitude`` is a new *total*, ``delta`` an *increment*.

        Which options the definition carries depends on how the re-declaration was written, and the
        ``elif`` below is load-bearing for that reason. A *full* re-declaration (all required args
        repeated) is validated against the ``distributedload`` keyword, which declares no ``delta``
        at all -- reading ``definition["delta"]`` unconditionally would raise a ``KeyError``; it is
        only safe here because ``magnitude`` is a required arg of that keyword and therefore never
        None. A *partial* re-declaration (``>>distributedload, name=dlTop, delta=-10.0, f(t)=t``,
        as in ``testfiles/marmot/GeoStatic/test.inp``) fails that validation, whereupon the parser
        re-validates it against the ``update`` + keyword-name schema, i.e. ``updatedistributedload``
        (``utils/inputfileparser.py``, ``parseModuleKeywordLine``); that definition carries
        ``magnitude=None`` plus ``delta``, so this is how ``delta`` is reached. Note that only the
        *schema* is live: a ``>>updatedistributedload`` keyword *line* is unroutable, because the
        parser would look for a step action module of that name (see PLAN_INPUT_SYSTEM.md's P3 row)."""

        magnitude = None
        delta = None

        if definition["magnitude"] is not None:
            magnitude = np.fromstring(definition["magnitude"], sep=",")
        elif definition["delta"] is not None:
            delta = np.fromstring(definition["delta"], sep=",")

        self.updateStepAction(
            magnitude=magnitude,
            delta=delta,
            f_t=amplitudeFromExpression(definition["f(t)"]),
        )

    def updateStepAction(
        self,
        magnitude: np.ndarray = None,
        delta: np.ndarray = None,
        f_t: Callable[[float], float] = None,
    ):
        """Prescribe a new magnitude and amplitude on the same surface.

        The load accumulated up to the start of this step stays untouched; what is prescribed here is
        the increment applied on top of it during this step. ``magnitude`` and ``delta`` are two ways
        of expressing that increment and ``magnitude`` takes precedence; supplying neither leaves the
        increment as it is, which after a completed step means zero -- i.e. the load is held constant
        at its accumulated level.

        Parameters
        ----------
        magnitude
            The new *total* magnitude, i.e. the value to be reached at the end of this step. The
            increment applied during the step is the difference to the accumulated magnitude.
        delta
            The *increment* of the magnitude to be applied during this step. Only consulted if
            ``magnitude`` is omitted.
        f_t
            The amplitude over the step progress interval ``[0...1]``; the identity if omitted.
        """

        if magnitude is not None:
            self.delta = magnitude - self._magnitudeAtStepStart
        elif delta is not None:
            self.delta = delta

        self.amplitude = f_t if f_t is not None else linearAmplitude

        self.idle = False

    @property
    def surface(self) -> str:
        """The surface this distributed load is acting on.

        Returns
        -------
        str
            The surface definition.
        """

        return self._surface

    @property
    def loadType(self) -> str:
        """The type of this distributed load, e.g. pressure or surface traction.

        Returns
        -------
        str
            The load type.
        """

        return self._loadType

    def getCurrentLoad(self, timeStep: TimeStep) -> np.ndarray:
        """The magnitude at the current point of the step.

        Parameters
        ----------
        timeStep
            The current time step.

        Returns
        -------
        np.ndarray
            The accumulated magnitude plus the amplitude-scaled increment of this step.
        """

        if self.idle is True:
            t = 1.0
        else:
            t = timeStep.stepProgress

        return self._magnitudeAtStepStart + self.delta * self.amplitude(t)

    def applyAtStepEnd(self, model, stepMagnitude=None):
        """Fold this step's increment into the accumulated magnitude and go idle.

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

        if not self.idle:
            if stepMagnitude is None:
                # standard case
                self._magnitudeAtStepStart += self.delta * self.amplitude(1.0)
            else:
                # set the 'actual' increment manually, e.g. for arc length method
                self._magnitudeAtStepStart += self.delta * stepMagnitude

            self.delta = 0
            self.idle = True
