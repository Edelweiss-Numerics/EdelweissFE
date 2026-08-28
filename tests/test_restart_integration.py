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
"""An end-to-end restart test driving the real ``.inp``/driver/solver stack
(unlike ``tests/test_femodel_restart.py``, which only exercises ``FEModel.writeRestart``/
``readRestart`` directly).

Mirrors EdelweissMeshfree's ``examples/114_marmot_micropolar_snni_quad_restart_test/`` two-invocation
shape (run once uninterrupted for a reference, run again truncated + resumed, diff the final
result), adapted for EdelweissFE's ``*restart``/``*output, type=restart`` keywords instead of
Meshfree's ``solveStep(...)`` kwargs. Uses ``VonMises`` (a Marmot-backed material with real
per-quadrature-point plastic history), not ``LinearElastic``: a history-free material would not
exercise ``getStateVars``/``setStateVars`` round-tripping at all, so it could not catch a state
transfer bug.

This two-invocation shape does not fit ``run_tests_edelweissfe``'s single ``test.inp``/``U.ref``
model, so it lives here as a real pytest test instead (this repo's ``tests/`` already hosts one,
see ``PLAN_INPUT_SYSTEM.md``), driving ``finiteElementSimulation`` directly rather than the ``.inp``
files under ``testfiles/`` -- both invocations of the resumed run must share byte-identical step
definitions with the uninterrupted reference (see the module-level template), which a
``testfiles/marmot/RestartTest/test.inp`` alone could not enforce without duplicating it into two
near-identical input files anyway.
"""

from pathlib import Path

import h5py
import numpy as np

from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.utils.inputfileparser import parseInputFile

_MATERIAL_AND_MESH = """
*material, name=VonMises, id=myMaterial
210000, 0.3, 550, 1000, 200, 1400

*section, name=section1, material=myMaterial, type=plane, thickness=1
all

*job, name=restartTestJob, domain=2d
*solver, solver=NIST, name=theSolver

*modelGenerator, generator=planeRectQuad, name=gen
l=10
h=10
nX=2
nY=2
elType=CPE4
"""

_STEP_1 = """
*step, solver=theSolver
maxInc=1e0, minInc=1e-2, maxNumInc=100, maxIter=25, stepLength=1
>>dirichlet, name=left, nSet=gen_left, field=displacement, 1=0, 2=0
>>dirichlet, name=bottom, nSet=gen_leftBottom, field=displacement, 2=0
"""


def _step2(maxNumInc) -> str:
    return f"""
*step, solver=theSolver
maxInc=5e-2, minInc=1e-4, maxNumInc={maxNumInc}, maxIter=25, stepLength=1
>>dirichlet, name=right, nSet=gen_right, field=displacement, 2=0.2
"""


def _runInputFile(path: Path):
    inputfile = parseInputFile(str(path))
    model, fieldOutputController = finiteElementSimulation(inputfile, verbose=False, suppressPlots=True)
    return model


def _mostRecentCheckpoint(tmp_path: Path) -> Path:
    checkpoints = list(tmp_path.glob("ckpt_*.h5"))
    assert checkpoints, "the truncated run did not write any restart checkpoint"

    def checkpointTime(path):
        with h5py.File(path, "r") as f:
            return f.attrs["time"]

    return max(checkpoints, key=checkpointTime)


def test_restart_resume_matches_uninterrupted_reference(tmp_path):
    fullPath = tmp_path / "full.inp"
    fullPath.write_text(_MATERIAL_AND_MESH + _STEP_1 + _step2(maxNumInc=10000))
    referenceModel = _runInputFile(fullPath)
    referenceU = referenceModel.nodeFields["displacement"]["U"].copy()

    # Absolute checkpoint base name/path: the output manager and the driver's *restart, readFrom=
    # both resolve their file name relative to the process's current working directory, not this
    # test's tmp_path, so a bare "ckpt" would scatter checkpoint files outside tmp_path.
    checkpointBaseName = tmp_path / "ckpt"

    truncatedPath = tmp_path / "truncated.inp"
    truncatedPath.write_text(
        _MATERIAL_AND_MESH + f"\n*output, type=restart, name=restartwriter\n"
        f"writeInterval=1, baseName={checkpointBaseName}, numberOfFilesToKeep=3\n"
        + _STEP_1
        + _step2(maxNumInc=3)  # deliberately too low: truncates the job before step 2 finishes
    )
    _runInputFile(truncatedPath)

    checkpoint = _mostRecentCheckpoint(tmp_path)

    resumePath = tmp_path / "resume.inp"
    resumePath.write_text(
        _MATERIAL_AND_MESH + f"\n*restart, readFrom={checkpoint}\n" + _STEP_1 + _step2(maxNumInc=10000)
    )
    resumedModel = _runInputFile(resumePath)
    resumedU = resumedModel.nodeFields["displacement"]["U"]

    np.testing.assert_allclose(resumedU, referenceU, atol=1e-10)
