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
"""The solvers find the loads of a step by what they are, not by where they are kept: every step action that is a
``LoadBase``, whatever step action module holds it (``NonlinearSolverBase.collectLoads``). A new kind of load,
registered as a step action module of its own, therefore needs no change to any solver, as ``LoadBase`` promises.
"""

import numpy as np

from edelweissfe.config import registry
from edelweissfe.config.configurator import loadConfiguration
from edelweissfe.helpers.inputfilehelpers import fillFEModelFromInputFile
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.solvers.base.nonlinearsolverbase import NonlinearSolverBase
from edelweissfe.solvers.nonlinearimplicitstatic import NIST
from edelweissfe.stepactions.base.loadbase import LoadBase
from edelweissfe.stepactions.base.stepactionbase import StepActionBase
from edelweissfe.stepactions.dirichlet import StepAction as Dirichlet
from edelweissfe.stepactions.nodeforces import StepAction as NodeForces
from edelweissfe.steps.adaptivestep import AdaptiveStep
from edelweissfe.steps.stepmanager import StepActionCollection
from edelweissfe.utils.fieldoutput import FieldOutputController
from edelweissfe.utils.inputfileparser import parseInputFile

#: A plate of two by two plane-strain quads on the unit square.
_PLATE = """
*material, name=LinearElastic, id=linearelastic, provider=edelweiss
100.0, 0.3

*modelGenerator, generator=planeRectQuad, name=gen
x0=0, y0=0, l=1.0, h=1.0, nX=2, nY=2, elType=CPE4, elProvider=edelweiss

*section, name=section1, thickness=1.0, material=linearelastic, type=plane
gen_all
"""


class _Load(LoadBase):
    """A load that assembles nothing: the collection only asks what an action is."""

    def __init__(self, name: str):
        self.name = name

    def assembleLoad(self, P, K, U_np, dU, timeStep, dofManager):
        pass

    def applyAtStepEnd(self, model, loadParameter: float | None = None):
        pass


class _NotALoad(StepActionBase):
    """A step action that is not a load, such as a Dirichlet boundary condition."""

    def __init__(self, name: str):
        self.name = name


class ConcentratedForces(NodeForces):
    """Node forces kept under a step action module of their own: as far as the solvers can tell, a new kind of
    load."""


def test_the_loads_of_every_module_are_collected_module_by_module_in_a_fixed_order():
    nodeForce, bodyForce, firstOwnLoad, secondOwnLoad, lastOwnLoad = (
        _Load(name) for name in ("nodeForce", "bodyForce", "first", "second", "last")
    )
    stepActions = {
        "zeroloads": {"last": lastOwnLoad},
        "dirichlet": {"clamp": _NotALoad("clamp")},
        "bodyforce": {"bodyForce": bodyForce},
        "distributedload": {},
        "concentratedforces": {"first": firstOwnLoad, "second": secondOwnLoad},
        "nodeforces": {"nodeForce": nodeForce},
    }

    loads = NonlinearSolverBase.collectLoads(stepActions)

    assert loads == [nodeForce, bodyForce, firstOwnLoad, secondOwnLoad, lastOwnLoad]


def _solve(tmp_path, loadModule: str) -> np.ndarray:
    """The displacements of the plate, clamped at its bottom and pulled at its top right corner by node forces kept
    under the step action module ``loadModule``, after one linear step of ``NIST``."""

    tmp_path.mkdir()
    path = tmp_path / "plate.inp"
    path.write_text(_PLATE)

    journal = Journal(verbose=False)
    model = fillFEModelFromInputFile(FEModel(2), parseInputFile(str(path)), journal)
    model.prepareYourself(journal)
    model.advanceToTime(0.0)
    for nodeField in model.nodeFields.values():
        nodeField.createFieldValueEntry("U")
        nodeField.createFieldValueEntry("P")
    model._linkFieldVariableObjects(model.nodeSets["all"])

    jobInfo = loadConfiguration(dict())
    fieldOutputController = FieldOutputController(model, journal)
    model.fieldOutputController = fieldOutputController
    fieldOutputController.initializeJob()

    stepActions = StepActionCollection()
    stepActions["dirichlet"]["clamp"] = Dirichlet(
        "clamp", model.nodeSets["gen_bottom"], "displacement", {0: 0.0, 1: 0.0}, model, journal
    )
    stepActions[loadModule]["pull"] = ConcentratedForces(
        "pull", model.nodeSets["gen_rightTop"], "displacement", np.array([1.0, 2.0]), model, journal
    )

    step = AdaptiveStep(
        0,
        model,
        fieldOutputController,
        journal,
        jobInfo,
        NIST(jobInfo, journal),
        [],
        stepActions,
        stepLength=1.0,
        startInc=1.0,
        maxInc=1.0,
    )
    step.solve()
    fieldOutputController.finalizeJob()

    return np.array(model.nodeFields["displacement"]["U"])


def test_a_load_kept_under_a_step_action_module_of_its_own_is_assembled(tmp_path):
    registry.register("stepaction", "concentratedforces", ConcentratedForces)

    U = _solve(tmp_path / "ownModule", "concentratedforces")
    reference = _solve(tmp_path / "nodeforces", "nodeforces")

    assert np.abs(reference).max() > 0.0, "the check is vacuous if the plate does not move"
    np.testing.assert_array_equal(U, reference)
