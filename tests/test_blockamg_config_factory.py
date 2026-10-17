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


def test_the_former_lgmres_recycling_keys_are_accepted_only_at_their_non_recycling_values(capsys):
    createSolver({"lgmresAlwaysReset": True, "lgmresResetOnNewIncrement": False})
    assert "deprecated" in capsys.readouterr().out
    with pytest.raises(ValueError, match="recycling"):
        createSolver({"lgmresAlwaysReset": False})
    with pytest.raises(ValueError, match="recycling"):
        createSolver({"lgmresResetOnNewIncrement": True})


def test_a_quoted_false_is_false_and_anything_else_raises():
    assert createSolver({"symmetric": "false"})._symmetric is False
    assert createSolver({"symmetric": "True"})._symmetric is True
    with pytest.raises(ValueError):
        createSolver({"symmetric": "no"})
    with pytest.raises(ValueError):
        createSolver({"gapCompensatedTolerance": 0})


def test_a_single_field_name_is_not_split_into_characters():
    assert createSolver({"p1FieldNames": "displacement"})._p1FieldNamesRequested == {"displacement"}
    with pytest.raises(ValueError):
        createSolver({"p1FieldNames": 3})


def test_null_field_names_and_trees_are_unset():
    solver = createSolver({"p1FieldNames": None, "p1CoarsePrecond": None, "fieldPreconds": None})
    assert solver._p1FieldNamesRequested == set()
    assert solver._p1CoarsePrecond is None
    assert solver._fieldPreconds == {}


@pytest.mark.parametrize(
    "options",
    [
        {"p1FineDegree": 0},
        {"p1NuPre": -1},
        {"p1FineBlockSize": 2},
        {"p1CoarsePrecond": []},
        {"fieldPreconds": {"displacement": {"backendPrecision": "half"}}},
        {"sweeps": 0},
    ],
    ids=lambda options: next(iter(options)),
)
def test_a_setting_the_run_could_not_use_raises_at_construction(options):
    with pytest.raises(ValueError, match=next(iter(options))):
        createSolver(options)
