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

import pytest

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
    solver._pNodeCache["displacement"] = "stale"
    solver._p1Maps["displacement"] = "stale"
    solver._lazyP1MapNames.add("displacement")

    _write(configPath, {"sweeps": 7, "hotReloadConfigFile": str(configPath)})
    assert _reload(solver) is True

    assert solver._nodeCoordinateCache == {}
    assert solver._pNodeCache == {}
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
        and not hasattr(solver, BlockAMGSolver._HOT_RELOAD_ALIASES.get(name, "_" + name))
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
