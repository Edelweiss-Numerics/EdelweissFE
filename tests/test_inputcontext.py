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
"""Tests for ``edelweissfe/utils/inputcontext.py``."""

import dataclasses

import pytest

from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.utils.inputcontext import InputContext


def _buildContext(journal: Journal) -> InputContext:
    return InputContext(model=FEModel(2), journal=journal)


def test_construction_with_only_required_fields_defaults_the_rest_to_none():
    journal = Journal(verbose=False)
    model = FEModel(2)

    ctx = InputContext(model=model, journal=journal)

    assert ctx.model is model
    assert ctx.journal is journal
    assert ctx.plotter is None
    assert ctx.fieldOutputController is None


def test_construction_with_all_fields():
    journal = Journal(verbose=False)
    model = FEModel(2)
    sentinelPlotter = object()
    sentinelFieldOutputController = object()

    ctx = InputContext(
        model=model,
        journal=journal,
        plotter=sentinelPlotter,
        fieldOutputController=sentinelFieldOutputController,
    )

    assert ctx.plotter is sentinelPlotter
    assert ctx.fieldOutputController is sentinelFieldOutputController


def test_input_context_is_frozen():
    ctx = _buildContext(Journal(verbose=False))
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.model = FEModel(2)


def test_journal_is_carried_explicitly_not_shared_globally():
    """Two InputContext instances built with two distinct Journal instances must each keep their
    own -- InputContext must never reach for a global/singleton Journal (see this codebase's
    explicit convention that Journal is always passed explicitly)."""
    journalA = Journal(verbose=False)
    journalB = Journal(verbose=False)

    ctxA = _buildContext(journalA)
    ctxB = _buildContext(journalB)

    assert ctxA.journal is journalA
    assert ctxB.journal is journalB
    assert ctxA.journal is not ctxB.journal
