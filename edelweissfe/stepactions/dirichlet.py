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
# Created on Mon Jan 23 13:03:09 2017

# @author: Matthias Neuner

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from edelweissfe.config.phenomena import getFieldSize
from edelweissfe.stepactions.base.amplitude import amplitudeFromExpression
from edelweissfe.stepactions.base.dirichletbase import DirichletBase
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.caseinsensitivedict import CaseInsensitiveDict
from edelweissfe.utils.misc import withoutParserBookkeepingKeys
from edelweissfe.utils.schema import (
    buildSchemaFromOptions,
    coercePresentOptions,
    schemaField,
)

"""
Standard Dirichlet boundary condition.
If not modified in subsequent steps, the BC is held constant.
"""


@dataclass(frozen=True)
class DirichletSchema:
    """L2: the scalar options of the ``dirichlet`` keyword, owned by this module and never mutated
    from outside it.

    ``name``, ``nSet`` and ``analyticalField`` are ``structuralOnly`` fields: ``nSet`` and
    ``analyticalField`` each name an existing model object (a node set, an analytical field)
    resolved by :meth:`fromStepActionDefinition` before the schema is even built, exactly like
    every other category's structural names, and ``name`` is popped even earlier, by
    ``helpers/inputfilehelpers.py``. All three are declared here purely so the rendered grammar
    surface documents them -- :func:`~edelweissfe.utils.schema.buildSchemaFromOptions` never
    actually sees any of the three keys; see
    :attr:`~edelweissfe.utils.schema.SchemaFieldMeta.structuralOnly`. ``field`` stays an ordinary
    schema field -- it is used as a plain string tag (to compute the field size), never looked up
    in a model dict. The numbered components ``1``..``6`` are not valid Python identifiers, hence
    the ``optionName`` indirection on ``component1``..``component6``.
    """

    name: str | None = schemaField(
        description="Name of the step action.", dtype=str, default=None, required=True, structuralOnly=True
    )
    nSet: str | None = schemaField(
        description="The node set for application of the boundary condition.",
        dtype=str,
        default=None,
        required=True,
        structuralOnly=True,
    )
    field: str | None = schemaField(
        description="Field for which the boundary condition is active.", dtype=str, default=None, required=True
    )
    component1: float | None = schemaField(
        description="Prescribe first component of field.", dtype=float, default=None, optionName="1"
    )
    component2: float | None = schemaField(
        description="Prescribe second component of field.", dtype=float, default=None, optionName="2"
    )
    component3: float | None = schemaField(
        description="Prescribe third component of field.", dtype=float, default=None, optionName="3"
    )
    component4: float | None = schemaField(
        description="Prescribe fourth component of field.", dtype=float, default=None, optionName="4"
    )
    component5: float | None = schemaField(
        description="Prescribe fifth component of field.", dtype=float, default=None, optionName="5"
    )
    component6: float | None = schemaField(
        description="Prescribe sixth component of field.", dtype=float, default=None, optionName="6"
    )
    components: str | None = schemaField(
        description="Prescribe values using a numpy ndarray for representation; use 'x' for ignored values.",
        dtype=str,
        default=None,
    )
    analyticalField: str | None = schemaField(
        description="Scales the defined boundary condition", dtype=str, default=None, structuralOnly=True
    )
    f_t: str | None = schemaField(
        description="Define an amplitude in the step progress interval [0...1]",
        dtype=str,
        default=None,
        optionName="f(t)",
    )


@dataclass(frozen=True)
class UpdateDirichletSchema:
    """L2, documentation-only: the ``updateDirichlet`` keyword's own grammar.

    ``updateDirichlet`` is a genuinely different keyword from ``dirichlet`` -- a partial
    re-declaration that restates only ``name`` (to identify which instance to update) plus the
    same prescription arguments, dropping the ``nSet``/``field`` required on the initial
    declaration (``_addPrescriptionArgs`` in the ``Module`` block above). This class is **not**
    referenced by any runtime code: :meth:`StepAction.updateStepActionFromDefinition` validates a
    re-declaration via :func:`~edelweissfe.utils.schema.coercePresentOptions` against
    :class:`DirichletSchema` itself, which does not enforce required-ness at all (an override is by
    definition partial). It exists solely so :func:`~edelweissfe.utils.schemasurface.renderSchemaSurface`
    can reproduce the golden grammar surface's ``< updateDirichlet >`` block, which has its own
    distinct required-arg set and therefore cannot be rendered from :class:`DirichletSchema` as-is.
    """

    name: str | None = schemaField(
        description="Name of the step action to update.", dtype=str, default=None, required=True, structuralOnly=True
    )
    component1: float | None = schemaField(
        description="Prescribe first component of field.", dtype=float, default=None, optionName="1"
    )
    component2: float | None = schemaField(
        description="Prescribe second component of field.", dtype=float, default=None, optionName="2"
    )
    component3: float | None = schemaField(
        description="Prescribe third component of field.", dtype=float, default=None, optionName="3"
    )
    component4: float | None = schemaField(
        description="Prescribe fourth component of field.", dtype=float, default=None, optionName="4"
    )
    component5: float | None = schemaField(
        description="Prescribe fifth component of field.", dtype=float, default=None, optionName="5"
    )
    component6: float | None = schemaField(
        description="Prescribe sixth component of field.", dtype=float, default=None, optionName="6"
    )
    components: str | None = schemaField(
        description="Prescribe values using a numpy ndarray for representation; use 'x' for ignored values.",
        dtype=str,
        default=None,
    )
    analyticalField: str | None = schemaField(
        description="Scales the defined boundary condition", dtype=str, default=None, structuralOnly=True
    )
    f_t: str | None = schemaField(
        description="Define an amplitude in the step progress interval [0...1]",
        dtype=str,
        default=None,
        optionName="f(t)",
    )


class StepAction(DirichletBase):
    """Dirichlet boundary condition, based on a node set.

    The constructor is typed: it takes the node set itself, the prescribed values as a mapping, and
    the amplitude as a callable. Nothing here parses an input file -- turning ``nSet=bottom``,
    ``2=0.5`` and ``f(t)='t**2'`` into those arguments is the job of
    :meth:`fromStepActionDefinition` below, which is the only part of this module the ``.inp``
    front-end needs. That split is what lets an external caller (EdelweissMeshfree, a script) use
    this class directly; the signature deliberately matches EdelweissMeshfree's own
    ``stepactions/dirichlet.py``, which exists only because this one could not be constructed
    without a parser-shaped dict.

    Parameters
    ----------
    name
        The name of this step action.
    nSet
        The node set the boundary condition is applied to.
    field
        The field the boundary condition acts on, e.g. ``"displacement"``.
    prescribedComponents
        Maps the zero-based component index of ``field`` to the value prescribed for it. Components
        absent from the mapping are left free.
    model
        The model tree.
    journal
        The journal object for logging.
    f_t
        The amplitude over the step progress interval ``[0...1]``. Defaults to the identity, i.e. the
        prescribed values are reached linearly at the end of the step.
    analyticalField
        Scales the prescribed values per node, evaluated at each node's coordinates.
    """

    #: L2 schema declared for the L3 registry, per OptionSchemaProvider.
    schema = DirichletSchema

    def __init__(
        self,
        name,
        nSet,
        field,
        prescribedComponents: dict,
        model,
        journal,
        f_t: Callable[[float], float] = None,
        analyticalField=None,
    ):
        self.name = name

        self.field = field
        self.nSet = nSet
        self.fieldSize = getFieldSize(self.field, model.domainSize)

        self._components = None
        self._journal = journal
        self._model = model

        self.updateStepAction(prescribedComponents, f_t, analyticalField)

    @classmethod
    def fromStepActionDefinition(cls, name, definition, jobInfo, model, fieldOutputController, journal):
        """Build this boundary condition from a parsed ``>>dirichlet`` definition. See
        :class:`StepActionBase` for why this is separate from ``__init__``.

        ``name`` (already available as this method's own argument) and the parser's bookkeeping
        keys are stripped, and ``nSet``/``analyticalField`` are structural (they name a model
        object), so all four are popped before the remaining options are validated against
        :class:`DirichletSchema`."""

        definition = CaseInsensitiveDict(withoutParserBookkeepingKeys(definition))
        definition.pop("name", None)
        nSetName = definition.pop("nSet")
        analyticalFieldName = definition.pop("analyticalField")
        configuration = buildSchemaFromOptions(cls.schema, definition)

        return cls(
            name,
            model.nodeSets[nSetName],
            configuration.field,
            cls._prescribedComponentsFromDefinition(configuration, getFieldSize(configuration.field, model.domainSize)),
            model,
            journal,
            f_t=amplitudeFromExpression(configuration.f_t),
            analyticalField=model.analyticalFields[analyticalFieldName] if analyticalFieldName is not None else None,
        )

    def updateStepActionFromDefinition(self, definition, jobInfo, model, fieldOutputController, journal):
        """Update from a parsed ``>>dirichlet`` definition re-declared in a later step.

        A re-declaration is validated either against the full ``dirichlet`` keyword (restating
        every required arg, including ``nSet``/``field``) or, if it omits them, against the
        ``updateDirichlet`` schema instead (``utils/inputfileparser.py``,
        ``parseModuleKeywordLine``) -- which declares no ``nSet``/``field`` of its own at all. So
        this uses :func:`~edelweissfe.utils.schema.coercePresentOptions`, not
        :func:`~edelweissfe.utils.schema.buildSchemaFromOptions`: only whatever keys are actually
        present are validated, and ``field`` is never required here regardless of which of the two
        the parser matched -- its value is not read either way, since the node set and field are
        fixed at construction. ``analyticalField``, like ``f(t)``, resets to its default (no
        scaling / the identity amplitude) if omitted on this re-declaration -- it is not carried
        over from the previous declaration."""

        definition = CaseInsensitiveDict(withoutParserBookkeepingKeys(definition))
        definition.pop("name", None)
        definition.pop("nSet", None)
        analyticalFieldName = definition.pop("analyticalField", None)
        configuration = dataclasses.replace(self.schema(), **coercePresentOptions(self.schema, definition))

        self.updateStepAction(
            self._prescribedComponentsFromDefinition(configuration, self.fieldSize),
            amplitudeFromExpression(configuration.f_t),
            model.analyticalFields[analyticalFieldName] if analyticalFieldName is not None else None,
        )

    def _reconcileIfSetChanged(self):
        """Re-size the prescribed values if the node set was mutated in-place (e.g. AMR adding new
        boundary nodes) since the last check. Preserve the active/inactive flag: updateStepAction
        unconditionally activates, but a BC deactivated at a prior step end must stay inactive --
        otherwise a refinement in a later step would silently revive it. The node set itself needs
        no re-fetch: it has stable identity (mutated in place), so ``self.nSet`` is already current.

        Replays the *typed* state rather than a stashed definition dict, which is why
        ``__init__`` keeps ``prescribedComponents``/``f_t``/``analyticalField`` as attributes: the
        replay used to depend on the dict having been mutated in place by the ``components=``
        handling, an interaction that only worked because both lived in the same dict."""
        if self._checkSetChanged(self.nSet):
            wasActive = self.active
            self.updateStepAction(self._prescribedComponents, self._f_t, self._analyticalField)
            self.active = wasActive

    @property
    def components(
        self,
    ):
        return self._components

    def applyAtStepEnd(self, model):
        self.active = False

    def updateStepAction(
        self,
        prescribedComponents: dict,
        f_t: Callable[[float], float] = None,
        analyticalField=None,
    ):
        """Prescribe a new set of values, amplitude and analytical field on the same node set.

        Parameters
        ----------
        prescribedComponents
            Maps the zero-based component index of the field to the value prescribed for it.
        f_t
            The amplitude over the step progress interval ``[0...1]``; the identity if omitted.
        analyticalField
            Scales the prescribed values per node.
        """
        self.active = True

        self._checkSetChanged(self.nSet)

        outOfRange = [index for index in prescribedComponents if not 0 <= index < self.fieldSize]
        if outOfRange:
            raise ValueError(
                f"Dirichlet '{self.name}': component index/indices {sorted(outOfRange)} do not exist on "
                f"field '{self.field}', which has {self.fieldSize} component(s)."
            )

        self._prescribedComponents = dict(prescribedComponents)
        self._f_t = f_t
        self._analyticalField = analyticalField

        self._components = sorted(self._prescribedComponents)

        self.delta = np.tile([self._prescribedComponents[i] for i in self._components], (len(self.nSet), 1))

        if analyticalField is not None:
            self.analyticalField = analyticalField
            for i, node in enumerate(self.nSet):
                self.delta[i, :] *= analyticalField.evaluateAtCoordinates(node.coordinates)[0][0]

        self.amplitude = f_t if f_t is not None else lambda x: x

    def getPrescribedIncrement(self, timeStep: TimeStep):
        self._reconcileIfSetChanged()
        if self.active:
            return self.delta * (
                self.amplitude(timeStep.stepProgress)
                - (self.amplitude(timeStep.stepProgress - timeStep.stepProgressIncrement))
            )
        else:
            return self.delta * 0.0

    @staticmethod
    def _prescribedComponentsFromDefinition(configuration: "DirichletSchema", fieldSize: int) -> dict:
        """Collect the prescribed values from a validated ``DirichletSchema``'s ``component1``..
        ``component6`` and ``components`` fields.

        Parameters
        ----------
        configuration
            The validated options of this step action.
        fieldSize
            The number of components the field has.

        Returns
        -------
        dict
            Maps the zero-based component index to its prescribed value.
        """

        numberedComponents = (
            configuration.component1,
            configuration.component2,
            configuration.component3,
            configuration.component4,
            configuration.component5,
            configuration.component6,
        )

        prescribed = {
            index: numberedComponents[index] for index in range(fieldSize) if numberedComponents[index] is not None
        }

        if configuration.components is not None:
            # An entry of `x` marks a component as free; anything else overrides a numbered option
            # for the same component, which is the precedence the in-place dict mutation this
            # replaces happened to produce.
            values = np.array(eval(configuration.components.replace("x", "np.nan")), dtype=float)
            prescribed.update({index: value for index, value in enumerate(values) if not np.isnan(value)})

        return prescribed
