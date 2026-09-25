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
"""Bitwise restart equivalence of explicit contact with a throttled contact search."""

from pathlib import Path

import h5py
import numpy as np
import pytest

from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.journal.journal import Journal
from edelweissfe.utils.inputfileparser import parseInputFile


def _box(name, origin, length, elProvider, elType):
    x0, y0, z0 = origin
    lX, lY, lZ = length
    return f"""
*modelGenerator, generator=boxGen, name={name}
nX=3
nY=3
nZ=3
x0={x0}
y0={y0}
z0={z0}
lX={lX}
lY={lY}
lZ={lZ}
{elProvider}
elType={elType}
"""


def _writeRidgeStl(path: Path):
    """Closed prism along y whose top has a ridge at x = 0.6: sliding it under the box changes the
    rigid triangle, normal and offset a contact point projects onto."""

    section = [(-0.5, 0.0), (2.5, 0.0), (2.5, 0.9), (0.6, 1.0), (-0.5, 0.9)]
    front = [np.array([x, -0.5, z]) for x, z in section]
    back = [np.array([x, 2.5, z]) for x, z in section]
    triangles = [(side[0], side[i], side[i + 1]) for side in (front, back) for i in range(1, 4)]
    for i in range(5):
        j = (i + 1) % 5
        triangles += [(front[i], front[j], back[j]), (front[i], back[j], back[i])]

    centroid = np.mean(front + back, axis=0)
    lines = ["solid ridge"]
    for a, b, c in triangles:
        normal = np.cross(b - a, c - a)
        if normal @ ((a + b + c) / 3 - centroid) < 0.0:
            b, c, normal = c, b, -normal
        lines += [" facet normal {:} {:} {:}".format(*normal / np.linalg.norm(normal)), "  outer loop"]
        lines += ["   vertex {:} {:} {:}".format(*v) for v in (a, b, c)] + ["  endloop", " endfacet"]
    path.write_text("\n".join(lines + ["endsolid ridge"]) + "\n")


def _deck(directory: Path, constraintType: str, maxNumInc: int, extra: str = "", amr: bool = False) -> str:
    # Live refinement needs Marmot's C3D20R, whose refined surface also needs a smaller time increment.
    elProvider = "" if amr else "elProvider=edelweiss"
    upper = _box("upper", (0.3, 1.1, 1.05), (0.6, 0.6, 1), elProvider, "C3D20R" if amr else "C3D8")
    upper += "*modelGenerator, generator=surfaceElementGenerator, name=gen1\nsurface=upper_back\nname=slaveSurf\n"
    fixLower = ">>dirichlet, name=fixLower, nSet=lower_back, field=displacement, 1=0, 2=0, 3=0"

    if constraintType == "surfaceToDiscreteRigidBodyPenalty":
        _writeRidgeStl(directory / "ridge.stl")
        model = (
            upper
            + f"""
*modelGenerator, generator=discreteRigidBodyGenerator, name=support, executeAfterManualGeneration=True
filename={directory / 'ridge.stl'}
rpCoordinate='1.0, 1.0, 0.5'
mass=1.0
inertia="1.0, 1.0, 1.0"
*section, name=section1, material=linearelastic, type=solid
upper_all
"""
        )
        master = "rigidBody=support"
        # The ridge passes a contact point shortly before the checkpoint.
        supports = (
            ">>dirichlet, name=slideSupport, nSet=support_rp, field=displacement, 1=0.1, 2=0, 3=0\n"
            ">>dirichlet, name=fixSupportRotation, nSet=support_rp, field=rotation, 1=0, 2=0, 3=0"
        )
    else:
        model = upper + _box("lower", (0, 0, 0), (2, 2, 1), elProvider, "C3D8")
        model += "*modelGenerator, generator=surfaceElementGenerator, name=gen2\nsurface=lower_front\nname=masterSurf\n"
        model += "*section, name=section1, material=linearelastic, type=solid\nlower_all\nupper_all\n"
        master = "masterSurface=masterSurf_facets"
        master += ", sliding=small" if constraintType == "nodeToDeformableSurfacePenalty" else ""
        supports = fixLower

    return f"""
*material, name=LinearElastic, id=linearelastic{"" if amr else ", provider=edelweiss"}
1.8e4, 0.3, 1.0
*job, name=contactRestartJob, domain=3d
*solver, solver=NEDParallel, name=theSolver
courant-number={0.3 if amr else 0.8}
output-frequency=5
contact-update-frequency=7
{"topology-check-frequency=100" if amr else ""}
{model}
*constraint, name=contact, type={constraintType}
slaveSurface=slaveSurf_facets, {master}, penalty=5e4, type=linear, searchDistance=2.0
{extra}
*step, type=adaptiveForExplicitSimulations, solver=theSolver
maxInc=1, minInc=1e-12, maxNumInc={maxNumInc}, maxIter=25, stepLength=0.5
{supports}
>>dirichlet, name=pushDown, nSet=upper_front, field=displacement, 3=-0.2
>>dirichlet, name=pinUpperXY, nSet=upper_bottomLeftBack, field=displacement, 1=0.0, 2=0.0
>>dirichlet, name=pinUpperRotZ, nSet=upper_bottomRightBack, field=displacement, 2=0.0
"""


def _run(directory: Path, name: str, *args, **kwargs):
    path = directory / f"{name}.inp"
    path.write_text(_deck(directory, *args, **kwargs))
    model, _ = finiteElementSimulation(parseInputFile(str(path)), verbose=False, suppressPlots=True)
    return model


def _fields(model) -> dict:
    return {(name, entry): np.array(field[entry]) for name, field in model.nodeFields.items() for entry in "UP"}


def _truncateAndResume(directory: Path, constraintType: str, nFull: int, nTruncated: int, extra="", amr=False):
    """Run nTruncated increments writing checkpoints, then resume from the last one to nFull.

    The contact-update frequency (7) is no divisor of the output frequency (5), so a checkpoint
    usually falls between two searches."""

    writer = (
        f"*output, type=restart, name=restart\nwriteInterval=1, baseName={directory / 'ckpt'}, numberOfFilesToKeep=1\n"
    )
    _run(directory, "truncated", constraintType, nTruncated, extra + writer, amr)
    (checkpoint,) = directory.glob("ckpt_*.h5")
    resumed = _run(directory, "resumed", constraintType, nFull, extra + f"*restart, readFrom={checkpoint}", amr)
    return resumed, checkpoint


@pytest.fixture(
    scope="module",
    params=["nodeToDeformableSurfacePenalty", "surfaceToDeformableSurfacePenalty", "surfaceToDiscreteRigidBodyPenalty"],
)
def resumedContact(request, tmp_path_factory):
    """(uninterrupted fields, resumed fields, resumed model, checkpoint) of a 300-increment run
    resumed from increment 250."""

    directory = tmp_path_factory.mktemp(request.param)
    reference = _fields(_run(directory, "full", request.param, 300))
    resumed, checkpoint = _truncateAndResume(directory, request.param, 300, 250)
    return reference, _fields(resumed), resumed, checkpoint


def test_resumed_run_is_bitwise_identical(resumedContact):
    reference, resumed, _, _ = resumedContact
    for key, values in reference.items():
        assert np.array_equal(resumed[key], values), key


def _checkpointedConstraintData(checkpoint: Path) -> dict:
    with h5py.File(checkpoint, "r") as f:
        return {key: values[:] for key, values in f["constraints/contact"].items()}


def test_checkpointed_projection_is_adopted_not_searched(resumedContact):
    _, _, model, checkpoint = resumedContact
    constraint = model.constraints["contact"]
    # Shifted, so that a search could not reproduce it by coincidence.
    restored = {k: v + 0.125 if v.dtype.kind == "f" else v for k, v in _checkpointedConstraintData(checkpoint).items()}

    constraint.setRestartData(restored)
    constraint.resumeConnectivity(model)
    readBack = constraint.getRestartData()

    assert readBack.keys() == restored.keys()
    for key, values in restored.items():
        assert np.array_equal(readBack[key], values), key


def test_foreign_or_missing_projection_falls_back_to_a_search(resumedContact, capsys):
    _, _, model, checkpoint = resumedContact
    constraint = model.constraints["contact"]
    constraint.journal = Journal()
    written = _checkpointedConstraintData(checkpoint)

    constraint.updateConnectivity(model)
    searched = constraint.getRestartData()

    layoutKey = next(k for k in written if k.startswith("searchLayout_") and written[k].size > 1)
    foreign = written | {layoutKey: written[layoutKey][::-1].copy()}
    legacy = {k: v for k, v in written.items() if not k.startswith("searchLayout_")}

    for data in (foreign, legacy):
        constraint.setRestartData(data)
        constraint.resumeConnectivity(model)
        assert "afresh" in capsys.readouterr().out
        for key, values in constraint.getRestartData().items():
            assert np.array_equal(values, searched[key]), key
