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
#  ---------------------------------------------------------------------
"""Re-declaring a ``distributedload`` in a later step
(:meth:`edelweissfe.stepactions.distributedload.StepAction.updateStepActionFromDefinition`)."""

from types import SimpleNamespace

import numpy as np
import pytest

from edelweissfe.stepactions.distributedload import StepAction


def _loadOn(surface, surfaces):
    model = SimpleNamespace(surfaces=surfaces)
    return StepAction("theLoad", surface, np.array([1.0]), "pressure", model, None), model


def test_a_full_redeclaration_restating_the_surface_updates_the_magnitude():
    top = object()
    load, model = _loadOn(top, {"top": top})
    definition = {"name": "theLoad", "surface": "top", "type": "pressure", "magnitude": "3.0"}

    load.updateStepActionFromDefinition(definition, None, model, None, None)

    np.testing.assert_array_equal(load.delta, [3.0])


def test_a_partial_redeclaration_without_the_surface_still_works():
    top = object()
    load, model = _loadOn(top, {"top": top})

    load.updateStepActionFromDefinition({"name": "theLoad", "delta": "0.5"}, None, model, None, None)

    np.testing.assert_array_equal(load.delta, [0.5])


def test_a_redeclaration_naming_another_surface_is_refused():
    top, bottom = object(), object()
    load, model = _loadOn(top, {"top": top, "bottom": bottom})
    definition = {"name": "theLoad", "surface": "bottom", "type": "pressure", "magnitude": "3.0"}

    with pytest.raises(ValueError, match="cannot move the load to surface bottom"):
        load.updateStepActionFromDefinition(definition, None, model, None, None)
