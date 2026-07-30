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
# Created on Tue May  9 19:52:53 2017

# @author: Matthias Neuner

# @ Magdalena
"""
Initialize materials to an geostatic stress state
"""

from dataclasses import dataclass

import numpy as np

from edelweissfe.stepactions.base.stepactionbase import StepActionBase
from edelweissfe.utils.caseinsensitivedict import CaseInsensitiveDict
from edelweissfe.utils.inputlanguage import InputLanguage
from edelweissfe.utils.misc import withoutParserBookkeepingKeys
from edelweissfe.utils.schema import buildSchemaFromOptions, schemaField

inputLanguage = InputLanguage()

# Register this step action for all available step types. This requires the step type
# modules to be imported before the step actions, as done in the input file parser.
modules = inputLanguage["step"].modules if "step" in inputLanguage else []

documentation = []

for module in modules:
    kw = module.addOptionalKeyword("geostatic", "Initialize materials to an geostatic stress state.")
    kw.addRequiredArg("name", "Name of the step action.", str)
    kw.addRequiredArg("p1", "sig_x=sig_y=sig_z in first point.", float)
    kw.addOptionalArg("p2", "sig_x=sig_y=sig_z in second point.", float, None)
    kw.addOptionalArg("h1", "y coordinate of first point", float, 1.0)
    kw.addOptionalArg("h2", "y coordinate of second point", float, -1.0)
    kw.addOptionalArg("xLateral", "ratio of sig_x/sig_y, default=1.0", float, 1.0)
    kw.addOptionalArg("zLateral", "ratio of sig_z/sig_y, default=1.0", float, 1.0)
    kw.addOptionalArg("elSet", "The element set for which the initaliziation is performed", str, "all")

    documentation.append(kw)


@dataclass(frozen=True)
class GeostaticSchema:
    """L2: the scalar options of the ``geostatic`` keyword, owned by this module and never mutated
    from outside it.

    ``name`` and ``elSet`` are ``structuralOnly`` fields: ``elSet`` names an existing model object,
    resolved by :meth:`fromStepActionDefinition` before the schema is even built, exactly like
    every other category's structural names, and ``name`` is popped even earlier, by
    ``helpers/inputfilehelpers.py``. Both are declared here purely so the rendered grammar surface
    documents them -- :func:`~edelweissfe.utils.schema.buildSchemaFromOptions` never actually sees
    either key; see :attr:`~edelweissfe.utils.schema.SchemaFieldMeta.structuralOnly`.
    """

    name: str | None = schemaField(
        description="Name of the step action.", dtype=str, default=None, required=True, structuralOnly=True
    )
    p1: float | None = schemaField(
        description="sig_x=sig_y=sig_z in first point.", dtype=float, default=None, required=True
    )
    p2: float | None = schemaField(description="sig_x=sig_y=sig_z in second point.", dtype=float, default=None)
    h1: float | None = schemaField(description="y coordinate of first point", dtype=float, default=1.0)
    h2: float | None = schemaField(description="y coordinate of second point", dtype=float, default=-1.0)
    xLateral: float | None = schemaField(description="ratio of sig_x/sig_y, default=1.0", dtype=float, default=1.0)
    zLateral: float | None = schemaField(description="ratio of sig_z/sig_y, default=1.0", dtype=float, default=1.0)
    elSet: str | None = schemaField(
        description="The element set for which the initaliziation is performed",
        dtype=str,
        default="all",
        structuralOnly=True,
    )


class StepAction(StepActionBase):
    """Initializes elements of set with an Abaqus-like geostatic stress state.
    Is automatically deactivated at the end of the step.

    The constructor is typed: it takes the element set itself and the geostatic stress state as
    plain floats. Nothing here parses an input file -- resolving ``elSet=all`` against the model,
    and defaulting an omitted ``p2`` to ``p1``, is the job of :meth:`fromStepActionDefinition`
    below, which is the only part of this module the ``.inp`` front-end needs.

    Parameters
    ----------
    name
        The name of this step action.
    elementSet
        The element set the geostatic stress state is applied to.
    p1
        sig_x=sig_y=sig_z in the first point.
    p2
        sig_x=sig_y=sig_z in the second point.
    h1
        y coordinate of the first point.
    h2
        y coordinate of the second point.
    xLateral
        Ratio of sig_x/sig_y.
    zLateral
        Ratio of sig_z/sig_y.
    journal
        The journal object for logging.
    """

    #: L2 schema declared for the L3 registry, per OptionSchemaProvider.
    schema = GeostaticSchema

    def __init__(
        self,
        name: str,
        elementSet,
        p1: float,
        p2: float,
        h1: float,
        h2: float,
        xLateral: float,
        zLateral: float,
        journal,
    ):
        self.name = name

        self.geostaticElements = elementSet
        self.p1 = p1
        self.p2 = p2
        self.level1 = h1
        self.level2 = h2
        self.xLateral = xLateral
        self.zLateral = zLateral

        self.geostaticDefinition = np.array(
            [
                self.p1,
                self.level1,
                self.p2,
                self.level2,
                self.xLateral,
                self.zLateral,
            ]
        )

        self.journal = journal

        self.active = True

    @classmethod
    def fromStepActionDefinition(cls, name, definition, jobInfo, model, fieldOutputController, journal):
        """Build this step action from a parsed ``>>geostatic`` definition. See
        :class:`StepActionBase` for why this is separate from ``__init__``.

        ``name`` and the parser's bookkeeping keys are stripped, and ``elSet`` is structural (it
        names a model object), so both are popped before the remaining options are validated
        against :class:`GeostaticSchema`. An omitted ``p2`` defaults to ``p1`` -- this is
        input-file convenience, so the defaulting happens here rather than in ``__init__``, which
        requires both to be passed explicitly."""

        definition = CaseInsensitiveDict(withoutParserBookkeepingKeys(definition))
        definition.pop("name", None)
        elSetName = definition.pop("elSet")
        configuration = buildSchemaFromOptions(cls.schema, definition)

        p2 = configuration.p2 if configuration.p2 is not None else configuration.p1

        return cls(
            name,
            model.elementSets[elSetName],
            configuration.p1,
            p2,
            configuration.h1,
            configuration.h2,
            configuration.xLateral,
            configuration.zLateral,
            journal,
        )

    def applyAtStepEnd(self, model, stepMagnitude=None):
        if not self.active:
            return

        self.journal.printSeperationLine()
        self.journal.message("End of geostatic step -- displacements are reset", self.name)
        self.journal.printSeperationLine()

        model.nodeFields["displacement"]["U"][:] = 0
        # U[self.theDofManager.indicesOfFieldsInDofVector["displacement"]] = 0.0

        self.active = False

    def applyAtIterationStart(
        self,
    ):
        if not self.active:
            return

        for el in self.geostaticElements:
            el.setInitialCondition("geostatic stress", self.geostaticDefinition)
