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
"""Tests for blockamg's configuration factory (createSolver): every documented key reaches the
solver, and an unknown key is an error instead of being silently dropped."""

import json
from pathlib import Path

import pytest

from edelweissfe.linsolve.blockamg import createSolver

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_the_sparsification_and_gap_keys_reach_the_solver():
    solver = createSolver({"hierarchyDropTol": 1e-3, "hierarchyDropLumping": True, "gapMaxFactor": 50.0})
    assert solver._hierarchyDropTol == 1e-3
    assert solver._hierarchyDropLumping is True
    assert solver._gapMaxFactor == 50.0


def test_an_unknown_key_raises_and_names_it():
    with pytest.raises(ValueError, match="hierarchyDropTolerance"):
        createSolver({"hierarchyDropTolerance": 1e-3})


def test_a_non_mapping_config_still_means_all_defaults():
    solver = createSolver("")
    assert solver._hierarchyDropTol == 0.0
    assert solver._symmetric is True


@pytest.mark.parametrize(
    "configFile",
    [
        "examples/WinklerL/blockamg.json",
        "examples/AnchorPryOut/blockamg.json",
        "testfiles/edelweiss-only/CantileverBeamQuad4BlockAMG/blockamg.json",
    ],
)
def test_the_shipped_configurations_are_accepted(configFile):
    createSolver(json.load(open(_REPO_ROOT / configFile)))
