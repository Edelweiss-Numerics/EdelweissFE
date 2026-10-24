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

import inspect
import json
import re
from pathlib import Path

import numpy as np
import pytest
from _blockamgsystems import displacementDamageSystem, hierarchyLevels

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
    assert solver._symmetric is False


@pytest.mark.parametrize(
    "configFile",
    [
        "examples/WinklerL/blockamg.json",
        "examples/AnchorPryOut/blockamg.json",
        "testfiles/edelweiss-only/CantileverBeamQuad4BlockAMG/blockamg.json",
        "testfiles/edelweiss-only/CantileverBeamHexa20BlockAMGRigidBody/blockamg.json",
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


@pytest.mark.parametrize("spelledOut", [False, True], ids=["default tree", "tree without blockTranslationNullspace"])
def test_the_removed_key_for_a_hierarchy_on_a_itself_names_what_replaces_it_when_it_was_true(spelledOut):
    # A fieldPreconds tree goes on to AMGCL, which would only warn about the unknown key on stderr: blockamg refuses
    # it, and names what it was -- the translations of the default trees plus a Jacobi-scaled smoother. A tree that
    # is spelled out replaces the default one and has to set the key itself, else it builds AMGCL's identity blocks.
    from edelweissfe.linsolve.blockamg.blockamg import _defaultFieldPrecond

    tree = {**_defaultFieldPrecond(3), "hierarchyOnUnscaledBlock": True}
    if spelledOut:
        del tree["blockTranslationNullspace"]
    with pytest.raises(
        ValueError,
        match=r'was removed.*set "blockTranslationNullspace": true.*put "scale": true inside the existing relax dict',
    ):
        createSolver({"fieldPreconds": {"displacement": tree}})


def test_the_removed_key_for_a_hierarchy_on_a_itself_was_a_no_op_when_it_was_false():
    from edelweissfe.linsolve.blockamg.blockamg import _defaultFieldPrecond

    tree = {**_defaultFieldPrecond(3), "hierarchyOnUnscaledBlock": False}
    with pytest.raises(
        ValueError, match="was removed; false was its default and did nothing, so drop the key"
    ) as error:
        createSolver({"fieldPreconds": {"displacement": tree}})
    # nothing to replace: no advice to switch anything on
    assert '"scale"' not in str(error.value)
    assert "blockTranslationNullspace" not in str(error.value)


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


def test_an_uncreatable_dump_directory_raises_and_is_never_created(tmp_path):
    # a plain file where the directory should go: os.makedirs() would raise a bare, unnamed OSError
    # if construction tried to create it before validating -- the settings check must reject this
    # first, with a ValueError naming the key, and must not touch the filesystem at all.
    blocked = tmp_path / "not_a_directory"
    blocked.write_text("")
    with pytest.raises(ValueError, match="dumpOnDegradationDir"):
        createSolver({"dumpOnDegradationDir": str(blocked)})
    assert blocked.is_file()


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


def test_the_defaults_solve_a_displacement_damage_system():
    A, b, blocks = displacementDamageSystem(12, coupling="symmetric", rowScaling=True)
    solver = createSolver({"verbosity": "silent"})
    solver.setFieldStructure(blocks)
    x = solver(A, b)
    # the displacement field runs on a multi-level float-stored 3x3 block hierarchy
    displacement = solver._preconditioners[0]
    assert displacement.blockSize == 3 and displacement.isFloat
    assert hierarchyLevels(displacement) > 1
    # the first solve asks for etaMax = 1e-3 on the unscaled system: the true residual itself
    assert np.linalg.norm(b - A @ x) <= 1.0001e-3 * np.linalg.norm(b)


def test_null_is_unset_for_every_key_whose_default_is_unset():
    solver = createSolver(
        {
            "dumpOnDegradationDir": None,
            "hotReloadConfigFile": None,
            "dumpOnDegradationThreshold": None,
            "p1Nu": None,
            "p1NuPre": None,
            "p1NuPost": None,
            "p1FineDegree": None,
        }
    )
    assert solver._dumpOnDegradationDir is None and solver._hotReloadConfigFile is None
    assert solver._p1Nu is None and solver._p1FineDegree is None


@pytest.mark.parametrize(
    "options",
    [{"etaMin": 1e-2, "etaMax": 1e-3}, {"residualGrowthFactor": 1.0}, {"hierarchyStalenessFactor": 0.5}],
    ids=["etaMin above etaMax", "residualGrowthFactor", "hierarchyStalenessFactor"],
)
def test_a_value_out_of_range_raises_at_construction(options):
    with pytest.raises(ValueError):
        createSolver(options)


@pytest.mark.parametrize(
    "options",
    [
        {"outerSolver": "scipy"},
        {"outerTol": 0.0},
        {"outerTol": 1.0},
        {"etaMin": 0.0},
        {"ewGamma": 0.0},
        {"ewGamma": 1.5},
        {"ewAlpha": 1.0},
        {"ewAlpha": 2.5},
        {"gapMaxFactor": 0.5},
        {"warnOuterIterationsThreshold": -1},
        {"trueResidualMaxContinuations": -1},
        {"dumpOnDegradationMaxDumps": -1},
        {"dumpOnDegradationThreshold": -1},
    ],
    ids=lambda options: "{:}={:}".format(*next(iter(options.items()))),
)
def test_a_value_out_of_its_range_raises_and_names_its_key(options):
    with pytest.raises(ValueError, match=next(iter(options))):
        createSolver(options)


def test_the_default_trees_on_the_theory_page_are_the_code_defaults():
    from edelweissfe.linsolve.blockamg.blockamg import _defaultFieldPrecond

    page = (_REPO_ROOT / "doc/source/documentation/blockamgtheory.rst").read_text()
    block = page.split("and change only what is intended. The defaults are", 1)[1].split(".. code-block:: json", 1)[1]
    trees = json.JSONDecoder().raw_decode(block.strip())[0]
    assert trees == {
        "3D vector field (e.g. displacement)": _defaultFieldPrecond(3),
        "2D vector field": _defaultFieldPrecond(2),
        "vector field of a dimension other than 2 or 3": _defaultFieldPrecond(4),
        "scalar field (e.g. nonlocal damage)": _defaultFieldPrecond(1),
    }


def test_a_block_size_that_does_not_fit_its_field_raises_at_the_first_build():
    A, b, blocks = displacementDamageSystem(6)
    solver = createSolver({"verbosity": "silent", "fieldPreconds": {"nonlocal damage": {"backendBlockSize": 3}}})
    solver.setFieldStructure(blocks)
    with pytest.raises(ValueError, match="do not fit the fields"):
        solver(A, b)


def test_every_configuration_key_is_documented_on_the_theory_page():
    from edelweissfe.linsolve.blockamg.blockamg import CONFIG_CASTS

    page = (_REPO_ROOT / "doc/source/documentation/blockamgtheory.rst").read_text()
    assert [key for key in CONFIG_CASTS if "``{:}``".format(key) not in page] == []


def test_the_default_column_of_the_key_tables_is_the_code_default():
    from edelweissfe.linsolve.blockamg.blockamg import _P1_OPTIONS, BlockAMGSolver
    from edelweissfe.linsolve.blockamg.ptwogrid import PTwoGridPreconditioner

    codeDefaults = {key: p.default for key, p in inspect.signature(BlockAMGSolver.__init__).parameters.items()}
    # a p-two-grid key left unset takes PTwoGridPreconditioner's own default
    p1Defaults = inspect.signature(PTwoGridPreconditioner.__init__).parameters
    for key, option in _P1_OPTIONS.items():
        if codeDefaults[key] is None:
            codeDefaults[key] = p1Defaults[option].default

    page = (_REPO_ROOT / "doc/source/documentation/blockamgtheory.rst").read_text()
    unset = re.findall(r"^    \* - ``(\w+)``\n      - unset$", page, re.M)
    assert [key for key in unset if codeDefaults[key] is not None] == []

    documented = re.findall(r"^    \* - ``(\w+)``\n      - ``([^`]+)``", page, re.M)
    assert len(documented) > 30
    mismatches = []
    for key, cell in documented:
        try:
            value = json.loads(cell)
        except json.JSONDecodeError:
            # a reference to another key's value, resolved at run time from None
            if cell in codeDefaults and codeDefaults[key] is None:
                continue
            mismatches.append((key, cell))
            continue
        if codeDefaults[key] is None and value in ({}, [], None):
            continue  # None stands for empty in the code
        # True == 1 in Python: a flag must also be documented as a flag
        if value != codeDefaults[key] or isinstance(value, bool) != isinstance(codeDefaults[key], bool):
            mismatches.append((key, cell))
    assert mismatches == []
