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
"""
P0.3 focused regression tests (see PLAN_INPUT_SYSTEM.md) for two standing bugs in
``edelweissfe/outputmanagers/ensight.py``:

Bug 1
    ``self.intermediateSaveInterval`` used to be assigned from
    ``module.getKeyword("configuration")["overwrite"].default`` instead of
    ``["intermediateSaveInterval"].default``. This has *already been fixed* upstream (found while
    verifying this branch's diagnosis against the current code -- see PLAN_INPUT_SYSTEM.md), but
    the test below still pins the correct behavior down so a regression is caught immediately.

Bug 2
    ``strtobool()`` (``edelweissfe/utils/misc.py``) calls ``.lower()`` on its argument, so a
    programmatic caller (e.g. an EdelweissMeshfree script) passing a real Python ``bool`` for
    ``overwrite``/``transient`` raised ``AttributeError``. Fixed by routing both call sites in
    ``OutputManager.updateDefinition`` through the new ``edelweissfe.utils.misc.asBool`` helper,
    which passes a real ``bool`` through unchanged and otherwise delegates to ``strtobool``.
"""

import numpy as np
import pytest

from edelweissfe.config.elementlibrary import getElementClass
from edelweissfe.config.materiallibrary import getMaterialClass
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.outputmanagers.ensight import OutputManager
from edelweissfe.outputmanagers.ensight import module as ensightModule
from edelweissfe.points.node import Node
from edelweissfe.sections.plane import PlaneSectionSchema, Section
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.utils.misc import asBool, strtobool


def _buildMinimalGeometryModel() -> FEModel:
    """A single CPE4 element model -- just enough geometry for the ensight OutputManager to
    build its Ensight parts from, no field outputs required."""
    n1 = Node(1, np.array([0.0, 0.0]))
    n2 = Node(2, np.array([1.0, 0.0]))
    n3 = Node(3, np.array([1.0, 1.0]))
    n4 = Node(4, np.array([0.0, 1.0]))

    ElementClass = getElementClass("CPE4", "edelweiss")
    element = ElementClass("CPE4", 1)
    element.setNodes([n1, n2, n3, n4])

    material = getMaterialClass("linearelastic", "edelweiss")(np.array([1000.0, 0.3]))

    model = FEModel(2)
    for n in (n1, n2, n3, n4):
        model.nodes[n.label] = n
    model.elements[element.elNumber] = element
    model.elementSets["all"] = ElementSet("all", [element])
    model.nodeSets["all"] = NodeSet("all", [n1, n2, n3, n4])

    section = Section(
        "section1",
        model,
        material,
        [model.elementSets["all"]],
        configuration=PlaneSectionSchema(thickness=1.0),
    )
    section.assignSectionPropertiesToElement(element)

    return model


def _buildEnsightOutputManager() -> OutputManager:
    model = _buildMinimalGeometryModel()
    journal = Journal(verbose=False)
    return OutputManager("myEnsight", model, None, journal, None)


def test_intermediateSaveInterval_reads_its_own_default_not_overwrites():
    """Bug 1: intermediateSaveInterval (schema default 10) must not silently pick up
    overwrite's default (False -> 0)."""
    intermediateSaveIntervalArg = ensightModule.getKeyword("configuration")["intermediateSaveInterval"]
    overwriteArg = ensightModule.getKeyword("configuration")["overwrite"]
    assert intermediateSaveIntervalArg.default == 10
    assert overwriteArg.default is False

    manager = _buildEnsightOutputManager()

    assert manager.intermediateSaveInterval == 10
    assert manager.overwrite is False  # its own (correct) default


def test_updateDefinition_configuration_accepts_real_bool_overwrite():
    """Bug 2 regression: passing a real bool (as a programmatic caller, e.g. EdelweissMeshfree,
    would) used to raise AttributeError inside strtobool()."""
    manager = _buildEnsightOutputManager()

    manager.updateDefinition(configuration=True, overwrite=False)
    assert manager.overwrite is False

    manager.updateDefinition(configuration=True, overwrite=True)
    assert manager.overwrite is True

    # strings must keep working exactly as before (this is how the input file parser calls it)
    manager.updateDefinition(configuration=True, overwrite="False")
    assert manager.overwrite is False


def test_asBool_accepts_bool_and_string_where_strtobool_would_crash_on_bool():
    assert asBool(True) is True
    assert asBool(False) is False
    assert asBool("True") is True
    assert asBool("False") is False
    assert asBool("yes") is True
    assert asBool("no") is False

    # demonstrate the exact regression asBool fixes: strtobool() itself still raises on a real
    # bool (it is untouched -- other callers depend on its string-only contract)
    with pytest.raises(AttributeError):
        strtobool(True)
