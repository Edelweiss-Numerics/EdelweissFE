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
"""Regression tests for BlockAMGSolver's hot reload of its own JSON config.

Written alongside the feature, whose whole point is to be used on a run that was expensive to
reach: a degraded AMG hierarchy typically only appears tens of hours into a nonlinear analysis, so
settings get tried in place rather than by rebuilding that state. That makes the failure modes the
interesting part -- editing a file a running process reads is racy by construction, and no bad edit
may be able to end the run. The malformed-config cases below are the reason this file exists.
"""

import inspect
import json

import numpy as np
import pytest

from edelweissfe.linsolve.base import FieldBlock
from edelweissfe.linsolve.blockamg.blockamg import BlockAMGSolver


def _write(path, options):
    path.write_text(json.dumps(options))


def _reload(solver):
    """One solve's worth of the hot-reload check, without needing a matrix."""
    solver._solveCount += 1
    return solver._maybeHotReloadConfig()


@pytest.fixture
def configuredSolver(tmp_path):
    configPath = tmp_path / "blockamg.json"
    _write(configPath, {"sweeps": 1, "hotReloadConfigFile": str(configPath)})
    solver = BlockAMGSolver(sweeps=1, hotReloadConfigFile=str(configPath))
    return solver, configPath


def test_hot_reload_is_off_by_default():
    solver = BlockAMGSolver()
    assert solver._maybeHotReloadConfig() is False


def test_identical_settings_do_not_force_a_refresh(configuredSolver):
    solver, _ = configuredSolver

    assert _reload(solver) is False
    assert solver._refreshNext is False

    # the second call short-circuits on the unchanged stamp
    assert _reload(solver) is False


def test_a_changed_setting_is_applied_and_forces_a_refresh(configuredSolver):
    solver, configPath = configuredSolver
    _reload(solver)

    _write(configPath, {"sweeps": 3, "etaMax": 1e-2, "hotReloadConfigFile": str(configPath)})

    assert _reload(solver) is True
    assert solver._sweeps == 3
    assert solver._etaMax == 1e-2
    # a hierarchy built under the old settings must not be reused under the new ones
    assert solver._refreshNext is True


def test_a_malformed_config_changes_nothing_and_is_retried(configuredSolver):
    solver, configPath = configuredSolver
    _reload(solver)

    # exactly what an editor saving in place leaves behind for a few milliseconds
    configPath.write_text('{"sweeps": 9, "etaMax":')

    assert _reload(solver) is False
    assert solver._sweeps == 1, "a malformed config must not apply anything"
    assert solver._refreshNext is False

    # and it must NOT be recorded as seen, or the good file that follows with the same size
    # could be skipped
    stampAfterBadParse = solver._hotReloadStamp
    assert _reload(solver) is False
    assert solver._hotReloadStamp == stampAfterBadParse

    _write(configPath, {"sweeps": 4, "hotReloadConfigFile": str(configPath)})
    assert _reload(solver) is True
    assert solver._sweeps == 4


def test_values_are_converted_as_at_construction(configuredSolver):
    """A JSON value is converted exactly as the factory converts it, not applied raw.

    ``"outerTol": "adaptive"`` -- in the shipped configs -- is ``None`` at construction; applied raw,
    the string ended a live run at its first solve (LEO5 job 8021070, TypeError in the forcing term).
    """
    solver, configPath = configuredSolver
    _reload(solver)

    _write(configPath, {"sweeps": 1, "outerTol": "adaptive", "hotReloadConfigFile": str(configPath)})
    assert _reload(solver) is False, "'adaptive' is the setting in force, not a change"
    assert solver._outerTol is None

    _write(configPath, {"sweeps": "2", "outerTol": "1e-6", "hotReloadConfigFile": str(configPath)})
    assert _reload(solver) is True
    assert solver._sweeps == 2
    assert solver._outerTol == 1e-6


def test_an_unconvertible_value_is_rejected_and_changes_nothing(configuredSolver):
    solver, configPath = configuredSolver
    _reload(solver)

    _write(configPath, {"sweeps": 1, "etaMax": "loose", "hotReloadConfigFile": str(configPath)})

    assert _reload(solver) is False
    assert solver._etaMax == BlockAMGSolver()._etaMax


def test_a_json_scalar_is_rejected_like_a_parse_error(configuredSolver):
    solver, configPath = configuredSolver
    _reload(solver)

    configPath.write_text("42")

    assert _reload(solver) is False
    assert solver._sweeps == 1


def test_a_missing_config_does_not_raise(configuredSolver):
    solver, configPath = configuredSolver
    _reload(solver)

    configPath.unlink()

    assert _reload(solver) is False
    assert solver._sweeps == 1


def test_nested_and_aliased_settings_are_diffed_by_value(configuredSolver):
    solver, configPath = configuredSolver
    _reload(solver)

    _write(
        configPath,
        {
            "fieldPreconds": {"displacement": {"backendBlockSize": 3}},
            "p1FieldNames": ["displacement"],
            "hotReloadConfigFile": str(configPath),
        },
    )

    assert _reload(solver) is True
    assert solver._fieldPreconds == {"displacement": {"backendBlockSize": 3}}
    # p1FieldNames is stored as a set under a different attribute name
    assert solver._p1FieldNamesRequested == {"displacement"}

    # re-reading the same content is not a change, even though it went through a dict/set conversion
    configPath.touch()
    assert _reload(solver) is False


def test_an_unknown_key_alone_does_not_force_a_refresh(configuredSolver):
    solver, configPath = configuredSolver
    _reload(solver)

    _write(configPath, {"sweeps": 1, "nonsenseKnob": 1, "hotReloadConfigFile": str(configPath)})

    assert _reload(solver) is False
    assert solver._refreshNext is False


def test_a_reload_drops_caches_derived_from_the_old_settings(configuredSolver):
    solver, configPath = configuredSolver
    _reload(solver)

    solver._nodeCoordinateCache["displacement"] = "stale"
    solver._p1FineSpectralRadiusCache["displacement"] = (1.0, None)
    solver._p1Maps["displacement"] = "stale"
    solver._lazyP1MapNames.add("displacement")

    _write(configPath, {"sweeps": 7, "hotReloadConfigFile": str(configPath)})
    assert _reload(solver) is True

    assert solver._nodeCoordinateCache == {}
    assert solver._p1FineSpectralRadiusCache == {}
    # a lazily built map belongs to the old settings; an explicitly supplied one would not
    assert "displacement" not in solver._p1Maps
    assert solver._lazyP1MapNames == set()


def test_every_constructor_setting_is_reloadable_or_explicitly_ignored(configuredSolver):
    """Guards the self._<name> convention the reload relies on.

    A setting whose stored attribute is named differently, and which is neither aliased nor
    ignored, would be silently applied to nothing -- so a new constructor parameter has to be
    classified here rather than quietly becoming un-reloadable.
    """
    solver, _ = configuredSolver

    settingNames = set(inspect.signature(BlockAMGSolver.__init__).parameters) - {"self"}
    unclassified = [
        name
        for name in settingNames
        if name not in BlockAMGSolver._HOT_RELOAD_IGNORED
        and BlockAMGSolver._HOT_RELOAD_ALIASES.get(name, "_" + name) not in vars(solver)
    ]

    assert unclassified == []


def test_the_factory_forwards_the_hot_reload_path(tmp_path):
    """The factory builds its kwargs from an explicit whitelist, so a setting that is only added
    to ``BlockAMGSolver.__init__`` is silently DROPPED on the path a deck actually uses.

    That is not hypothetical: it is how this feature first shipped, and the symptom was a run that
    read its config once and never again, with nothing in the log to say so -- because an
    unrecognized key in the JSON is dropped by the whitelist rather than rejected.
    """
    from edelweissfe.linsolve.blockamg import createSolver

    configPath = tmp_path / "blockamg.json"
    _write(configPath, {"sweeps": 1, "hotReloadConfigFile": str(configPath)})

    solver = createSolver({"sweeps": 1, "hotReloadConfigFile": str(configPath)})

    assert solver._hotReloadConfigFile == str(configPath)

    # and it is live, not merely stored
    _write(configPath, {"sweeps": 5, "hotReloadConfigFile": str(configPath)})
    assert _reload(solver) is True
    assert solver._sweeps == 5


def test_the_factory_defaults_the_hot_reload_path_to_off():
    from edelweissfe.linsolve.blockamg import createSolver

    solver = createSolver({"sweeps": 1})

    assert solver._hotReloadConfigFile is None
    assert solver._maybeHotReloadConfig() is False


def test_reloaded_lgmres_settings_rebuild_the_outer_solver(configuredSolver):
    solver, configPath = configuredSolver
    _reload(solver)
    solver._lgmresSolver = "standing"

    _write(configPath, {"lgmresM": 50, "hotReloadConfigFile": str(configPath)})

    assert _reload(solver) is True
    assert solver._lgmresM == 50
    assert solver._lgmresSolver is None


@pytest.mark.parametrize(
    "name, value",
    [
        ("outerSolver", "amgcl-lgmres"),
        ("outerSolver", "scipy"),
        ("lgmresM", 0),
        ("p1FineBlockSize", 2),
        ("p1FineValueType", "half"),
        ("p1FineDegree", 0),
        ("p1CoarsePrecond", 3),
        ("fieldPreconds", {"displacement": 3}),
        ("fieldPreconds", {"displacement": {"backendBlockSize": 2, "backendPrecision": "float"}}),
        ("lgmresAlwaysReset", False),
    ],
)
def test_a_value_the_run_could_not_use_is_rejected(configuredSolver, name, value):
    solver, configPath = configuredSolver
    _reload(solver)
    before = dict(vars(solver))

    _write(configPath, {name: value, "hotReloadConfigFile": str(configPath)})

    assert _reload(solver) is False
    assert solver._refreshNext is False
    assert solver._outerSolver == before["_outerSolver"]
    assert solver._lgmresM == before["_lgmresM"]


def test_a_reloaded_dump_directory_is_created_and_the_context_window_resized(configuredSolver, tmp_path):
    solver, configPath = configuredSolver
    _reload(solver)
    dumpDir = tmp_path / "dumps"

    _write(
        configPath,
        {
            "dumpOnDegradationDir": str(dumpDir),
            "dumpOnDegradationContextSolves": 4,
            "hotReloadConfigFile": str(configPath),
        },
    )

    assert _reload(solver) is True
    assert dumpDir.is_dir()
    assert solver._recentSolveHistory.maxlen == 4


def _withFields(solver):
    """The solver after a solve on a 3D displacement and a scalar nonlocal damage field."""
    solver._blocks = [
        FieldBlock("displacement", 0, 30, 3),
        FieldBlock("nonlocal damage", 30, 40, 1),
    ]
    return solver


@pytest.mark.parametrize(
    "options",
    [
        {"fieldPreconds": {"nonlocal damage": {"backendBlockSize": 3}}},
        {"p1FieldNames": ["nonlocal damage"], "p1FineBlockSize": 3},
    ],
    ids=["block size", "p1 block size"],
)
def test_a_block_size_that_does_not_match_the_field_is_rejected(configuredSolver, options):
    solver, configPath = configuredSolver
    _reload(_withFields(solver))

    # p1FieldNames and p1FineBlockSize in one edit: each fits the settings in force, only together they do not
    _write(configPath, {**options, "hotReloadConfigFile": str(configPath)})

    assert _reload(solver) is False
    assert solver._fieldPreconds == {}
    assert solver._p1FieldNamesRequested == set()


def test_a_field_precond_for_a_field_the_system_lacks_is_accepted(configuredSolver):
    solver, configPath = configuredSolver
    _reload(_withFields(solver))
    fieldPreconds = {"temperature": {"backendBlockSize": 1}, "displacement": {"backendBlockSize": 3}}

    _write(configPath, {"fieldPreconds": fieldPreconds, "hotReloadConfigFile": str(configPath)})

    assert _reload(solver) is True
    assert solver._fieldPreconds == fieldPreconds


@pytest.mark.parametrize(
    "alongside",
    [{}, {"lgmresM": 40}, {"outerMaxiter": 5}, {"krylovOnScaledSystem": False}],
    ids=["alone", "with lgmresM", "with outerMaxiter", "with krylovOnScaledSystem"],
)
def test_settings_the_build_fails_with_are_rolled_back(configuredSolver, alongside):
    from _blockamgsystems import displacementDamageSystem

    solver, configPath = configuredSolver
    # large enough for a multi-level hierarchy: a block AMGCL solves directly never constructs its smoother
    A, b, blocks = displacementDamageSystem(12)
    solver.setFieldStructure(blocks)
    messages = []
    solver._log = lambda level, message: messages.append(message)
    solver(A, b)

    # valid as far as the checks see, but AMGCL knows no smoother of that name
    badTree = {"displacement": {"relax": {"type": "chebychev"}}}
    _write(configPath, {"fieldPreconds": badTree, **alongside, "hotReloadConfigFile": str(configPath)})
    x = solver(A, b)

    assert np.linalg.norm(b - A @ x) <= 1e-2 * np.linalg.norm(b)
    assert solver._fieldPreconds == {} and solver._lgmresM == 30 and solver._outerSolver == "amgcl_lgmres"
    assert solver._krylovOnScaledSystem is True
    assert any("ROLLED BACK" in message for message in messages)
    # the file is recorded as seen: the next solve does not try it again
    messages.clear()
    solver(A, b)
    assert not any("hot reload" in message for message in messages)


def test_a_null_dump_threshold_reloaded_means_the_warning_threshold(tmp_path):
    configPath = tmp_path / "blockamg.json"
    options = {"dumpOnDegradationDir": str(tmp_path / "dumps"), "dumpOnDegradationThreshold": None}
    _write(configPath, {**options, "hotReloadConfigFile": str(configPath)})
    solver = BlockAMGSolver(**options, hotReloadConfigFile=str(configPath))
    _reload(solver)
    assert solver._degradationThreshold() == solver._warnOuterIterationsThreshold
