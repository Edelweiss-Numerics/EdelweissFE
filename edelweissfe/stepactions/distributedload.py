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

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from edelweissfe.stepactions.base.amplitude import (
    amplitudeFromExpression,
    linearAmplitude,
)
from edelweissfe.stepactions.base.distributedloadbase import DistributedLoadBase
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.caseinsensitivedict import CaseInsensitiveDict
from edelweissfe.utils.misc import withoutParserBookkeepingKeys
from edelweissfe.utils.schema import (
    buildSchemaFromOptions,
    coercePresentOptions,
    schemaField,
)

"""
Standard distributed load, applied on a surface set.
If not modified in subsequent steps, the load held constant.
"""


@dataclass(frozen=True)
class DistributedLoadSchema:
    """L2: the scalar options of the ``distributedload`` keyword, owned by this module and never
    mutated from outside it.

    ``name`` and ``surface`` are ``structuralOnly`` fields: ``surface`` names an existing model
    object, resolved by :meth:`fromStepActionDefinition` before the schema is even built, exactly
    like every other category's structural names, and ``name`` is popped even earlier, by
    ``helpers/inputfilehelpers.py``. Both are declared here purely so the rendered grammar surface
    documents them; neither is ever actually seen by
    :func:`~edelweissfe.utils.schema.buildSchemaFromOptions`. ``delta`` belongs only to the
    ``updatedistributedload`` grammar (a full ``distributedload`` declaration never carries it), so
    it is ``updateOnly`` -- rendered as part of :class:`UpdateDistributedloadSchema` below, not of
    this schema's own ``[distributedload] ...``/``< distributedload >`` block, even though it stays
    a real, validated field of this one shared runtime schema. ``field`` is accepted purely for
    backward compatibility -- ``DistributedLoadBase`` has no notion of a field, so a load is applied
    to whatever field the element's load type implies, and ``configuration.field`` is never read.
    """

    name: str | None = schemaField(
        description="Name of the step action.", dtype=str, default=None, required=True, structuralOnly=True
    )
    surface: str | None = schemaField(
        description="Surface for application of the distributed load",
        dtype=str,
        default=None,
        required=True,
        structuralOnly=True,
    )
    field: str | None = schemaField(
        description="Field for which the boundary condition is active.", dtype=str, default="displacement"
    )
    magnitude: str | None = schemaField(
        description="Magnitude of the distributed load", dtype=str, default=None, required=True
    )
    loadType: str | None = schemaField(
        description="The load type, e.g., pressure or surface traction; Must be supported by the element type",
        dtype=str,
        default=None,
        required=True,
        optionName="type",
    )
    f_t: str | None = schemaField(
        description="Define an amplitude in the step progress interval [0...1]",
        dtype=str,
        default=None,
        optionName="f(t)",
    )
    delta: str | None = schemaField(
        description="In subsequent steps only: define the new magnitude incrementally",
        dtype=str,
        default=None,
        updateOnly=True,
    )


@dataclass(frozen=True)
class UpdateDistributedloadSchema:
    """L2, documentation-only: the ``updatedistributedload`` keyword's own grammar.

    ``updatedistributedload`` is a genuinely different keyword from ``distributedload`` -- a
    partial re-declaration that restates only ``name`` (to identify which instance to update), and
    treats ``magnitude`` as optional (unlike the initial declaration, where it is required), adding
    ``delta`` as an alternative way to express it. This class is **not** referenced by any runtime
    code: :meth:`StepAction.updateStepActionFromDefinition` validates a re-declaration via
    :func:`~edelweissfe.utils.schema.coercePresentOptions` against :class:`DistributedLoadSchema`
    itself, which does not enforce required-ness at all (an override is by definition partial). It
    exists solely so :func:`~edelweissfe.utils.schemasurface.renderSchemaSurface` can reproduce the
    golden grammar surface's ``< updatedistributedload >`` block, which has its own distinct
    required-arg set and therefore cannot be rendered from :class:`DistributedLoadSchema` as-is.
    """

    name: str | None = schemaField(
        description="Name of the step action to update.", dtype=str, default=None, required=True, structuralOnly=True
    )
    magnitude: str | None = schemaField(description="Magnitude of the distributed load", dtype=str, default=None)
    delta: str | None = schemaField(
        description="In subsequent steps only: define the new magnitude incrementally", dtype=str, default=None
    )
    f_t: str | None = schemaField(
        description="Define an amplitude in the step progress interval [0...1]",
        dtype=str,
        default=None,
        optionName="f(t)",
    )


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

    #: L2 schema declared for the L3 registry, per OptionSchemaProvider.
    schema = DistributedLoadSchema

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

        ``name`` and the parser's bookkeeping keys are stripped, and ``surface`` is structural (it
        names a model object), so both are popped before the remaining options are validated
        against :class:`DistributedLoadSchema`. The keyword's ``field`` option is deliberately not
        passed on: this class never consumed it, and inventing a meaning for it here would not be a
        behaviour-neutral port."""

        definition = CaseInsensitiveDict(withoutParserBookkeepingKeys(definition))
        definition.pop("name", None)
        surfaceName = definition.pop("surface")
        configuration = buildSchemaFromOptions(cls.schema, definition)

        return cls(
            name,
            model.surfaces[surfaceName],
            np.fromstring(configuration.magnitude, sep=","),
            configuration.loadType,
            model,
            journal,
            f_t=amplitudeFromExpression(configuration.f_t),
        )

    def updateStepActionFromDefinition(self, definition, jobInfo, model, fieldOutputController, journal):
        """Update from a parsed ``>>distributedload`` definition re-declared in a later step.

        The two magnitude options are mutually exclusive and ``magnitude`` wins, exactly as before:
        ``magnitude`` is a new *total*, ``delta`` an *increment*.

        A re-declaration is validated either against the full ``distributedload`` keyword (restating
        every required arg, including ``surface``) or, if it omits them, against the
        ``updatedistributedload`` schema instead (``utils/inputfileparser.py``,
        ``parseModuleKeywordLine``) -- which declares no ``surface``/``type`` of its own at all. So
        this uses :func:`~edelweissfe.utils.schema.coercePresentOptions`, not
        :func:`~edelweissfe.utils.schema.buildSchemaFromOptions`: only whatever keys are actually
        present are validated, which is what makes ``configuration.delta`` safe to read regardless
        of which of the two the parser matched -- a full re-declaration never carries it, so it
        stays ``None`` and the ``magnitude`` branch wins, exactly as before."""

        definition = CaseInsensitiveDict(withoutParserBookkeepingKeys(definition))
        definition.pop("name", None)
        configuration = dataclasses.replace(self.schema(), **coercePresentOptions(self.schema, definition))

        magnitude = None
        delta = None

        if configuration.magnitude is not None:
            magnitude = np.fromstring(configuration.magnitude, sep=",")
        elif configuration.delta is not None:
            delta = np.fromstring(configuration.delta, sep=",")

        self.updateStepAction(
            magnitude=magnitude,
            delta=delta,
            f_t=amplitudeFromExpression(configuration.f_t),
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
