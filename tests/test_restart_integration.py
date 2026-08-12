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

Parametrized over ``linsolver`` (``pardiso``, the shipped default, and ``blockamg``, the
block-AMG-preconditioned outer solver from ``perf/linsolve-investigation``): restart's
reconstruct-then-overwrite model construction never touches the linear solver at all -- every
``NIST`` step builds a fresh one in ``solveStep`` regardless of whether the run is resumed -- so
this is less "does restart know about blockamg" (it doesn't need to) and more a pinned regression
proving that claim, on the one solver stack this branch actually adds.

This two-invocation shape does not fit ``run_tests_edelweissfe``'s single ``test.inp``/``U.ref``
model, so it lives here as a real pytest test instead, driving ``finiteElementSimulation`` directly
rather than the ``.inp``
files under ``testfiles/`` -- both invocations of the resumed run must share byte-identical step
definitions with the uninterrupted reference (see the module-level template), which a
``testfiles/marmot/RestartTest/test.inp`` alone could not enforce without duplicating it into two
near-identical input files anyway.
"""

from pathlib import Path

import h5py
import numpy as np
import pytest

from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.utils.inputfileparser import parseInputFile


def _materialAndMesh(linsolver: str) -> str:
    # "pardiso" is the shipped NIST default (edelweissfe/solvers/nonlinearimplicitstatic.py) and
    # needs no explicit option; only deviate from it for "blockamg".
    linsolverOption = "" if linsolver == "pardiso" else f"linsolver={linsolver}\n"
    return f"""
*material, name=VonMises, id=myMaterial
210000, 0.3, 550, 1000, 200, 1400

*section, name=section1, material=myMaterial, type=plane, thickness=1
all

*job, name=restartTestJob, domain=2d
*solver, solver=NIST, name=theSolver
{linsolverOption}
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


@pytest.mark.parametrize("linsolver", ["pardiso", "blockamg"])
def test_restart_resume_matches_uninterrupted_reference(tmp_path, linsolver):
    materialAndMesh = _materialAndMesh(linsolver)

    fullPath = tmp_path / "full.inp"
    fullPath.write_text(materialAndMesh + _STEP_1 + _step2(maxNumInc=10000))
    referenceModel = _runInputFile(fullPath)
    referenceU = referenceModel.nodeFields["displacement"]["U"].copy()

    # Absolute checkpoint base name/path: the output manager and the driver's *restart, readFrom=
    # both resolve their file name relative to the process's current working directory, not this
    # test's tmp_path, so a bare "ckpt" would scatter checkpoint files outside tmp_path.
    checkpointBaseName = tmp_path / "ckpt"

    truncatedPath = tmp_path / "truncated.inp"
    truncatedPath.write_text(
        materialAndMesh + f"\n*output, type=restart, name=restartwriter\n"
        f"writeInterval=1, baseName={checkpointBaseName}, numberOfFilesToKeep=3\n"
        + _STEP_1
        + _step2(maxNumInc=3)  # deliberately too low: truncates the job before step 2 finishes
    )
    _runInputFile(truncatedPath)

    checkpoint = _mostRecentCheckpoint(tmp_path)

    resumePath = tmp_path / "resume.inp"
    resumePath.write_text(materialAndMesh + f"\n*restart, readFrom={checkpoint}\n" + _STEP_1 + _step2(maxNumInc=10000))
    resumedModel = _runInputFile(resumePath)
    resumedU = resumedModel.nodeFields["displacement"]["U"]

    np.testing.assert_allclose(resumedU, referenceU, atol=1e-10)


def test_restart_ensight_output_continues_across_resume(tmp_path):
    """Found via a real end-to-end validation run (AnchorPryOut): a resumed run's Ensight output
    used to restart its transient sequence numbering from zero -- since a fresh OutputManager
    instance has no way to know what a previous, now-dead process already wrote -- silently
    orphaning the pre-resume portion (the .case file only ever describes what *this* process
    wrote) and misordering the on-disk frame files (the resumed run's frame 0 landing chronologically
    after the truncated run's later frames). This pins the fix: the .case file's declared step
    list, after resuming, must cover the *entire* run (pre- and post-checkpoint), not just the
    resumed portion, and must be in correct chronological order.
    """
    materialAndMesh = _materialAndMesh("pardiso")
    ensightName = tmp_path / "esTest"
    ensightBlock = (
        "\n*fieldOutput\n"
        ">>perNode, elSet=all, field=displacement, result=U, name=displacement\n"
        f"\n*output, type=ensight, name={ensightName}\n"
        ">>perNode, fieldOutput=displacement\n"
        ">>configuration, overwrite=yes\n"
    )

    fullPath = tmp_path / "full.inp"
    fullPath.write_text(materialAndMesh + ensightBlock + _STEP_1 + _step2(maxNumInc=10000))
    _runInputFile(fullPath)
    fullStepCount = _ensightStepCount(ensightName)

    checkpointBaseName = tmp_path / "ckpt"
    truncatedPath = tmp_path / "truncated.inp"
    truncatedPath.write_text(
        materialAndMesh + ensightBlock + f"\n*output, type=restart, name=restartwriter\n"
        f"writeInterval=1, baseName={checkpointBaseName}, numberOfFilesToKeep=3\n" + _STEP_1 + _step2(maxNumInc=3)
    )
    _runInputFile(truncatedPath)
    truncatedStepCount = _ensightStepCount(ensightName)
    assert 0 < truncatedStepCount < fullStepCount, "the truncated run should stop partway through"

    checkpoint = _mostRecentCheckpoint(tmp_path)
    resumePath = tmp_path / "resume.inp"
    resumePath.write_text(
        materialAndMesh + ensightBlock + f"\n*restart, readFrom={checkpoint}\n" + _STEP_1 + _step2(maxNumInc=10000)
    )
    _runInputFile(resumePath)
    resumedStepCount, resumedTimeValues = _ensightStepCountAndTimes(ensightName)

    assert resumedStepCount == fullStepCount, "resuming must cover the whole run, not just the resumed portion"
    assert list(resumedTimeValues) == sorted(resumedTimeValues), "time values must stay in chronological order"

    # every declared frame file must actually exist, in the range the .case file promises
    for i in range(resumedStepCount):
        assert (ensightName.parent / f"{ensightName.name}" / f"displacement.var_{i:04d}").exists()

    # the GEOMETRY section must still reference the model: since this mesh never changes, the
    # resumed run's own writeOutput never calls writeGeometryTrendChunk (mesh_changed stays False
    # the whole time), so geometryTrends -- which the "model:" line is written from -- would stay
    # empty unless it was restored too, even though timeAndFileSets/the sequence itself is correct.
    caseText = (ensightName.parent / f"{ensightName.name}.case").read_text()
    assert "model:" in caseText, "GEOMETRY section lost its model: reference after resume"


def _ensightStepCount(ensightName: Path) -> int:
    return _ensightStepCountAndTimes(ensightName)[0]


def _ensightStepCountAndTimes(ensightName: Path) -> tuple[int, list]:
    """The .case file's TIME block declares one sub-block per time/file set (geometry, which only
    grows when the mesh changes, and variables, which grow every write) -- parse per-set and
    return the variable set's (always >= geometry's, and the one file numbering/comparisons here
    actually care about), not just whichever "number of steps:" line comes first."""

    caseFile = ensightName.parent / f"{ensightName.name}.case"
    lines = caseFile.read_text().splitlines()

    setsByNumber = {}
    currentSetNumber = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("time set:"):
            currentSetNumber = int(stripped.split(":")[1].strip().split()[0])
            setsByNumber[currentSetNumber] = {"numberOfSteps": None, "timeValues": []}
        elif stripped.startswith("number of steps:"):
            setsByNumber[currentSetNumber]["numberOfSteps"] = int(stripped.split(":")[1].strip())
        elif stripped.startswith("time values:"):
            setsByNumber[currentSetNumber]["timeValues"].append(float(stripped.split(":")[1].strip()))
        elif stripped.startswith(("filename start number:", "filename increment:")):
            pass
        elif stripped.startswith(("GEOMETRY", "VARIABLE", "FORMAT")):
            currentSetNumber = None
        elif currentSetNumber is not None and stripped:
            setsByNumber[currentSetNumber]["timeValues"].append(float(stripped))

    variableSet = max(setsByNumber.values(), key=lambda s: s["numberOfSteps"])
    return variableSet["numberOfSteps"], variableSet["timeValues"]
