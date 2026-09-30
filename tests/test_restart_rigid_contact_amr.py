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
"""Restart across a live refinement of a body in contact with a moving discrete rigid body.

A discrete rigid body writes its current position into the ``coordinates`` of its surface nodes
(:meth:`~edelweissfe.rigidbodies.discreterigidbody.DiscreteRigidBody.updateKinematics`), while every
other node keeps its reference coordinates there. The topology fingerprint of a refinement recorded
after the body had moved therefore contained the moved rigid surface. A resumed run replays the
refinement before it restores the displacement, i.e. with the body at its reference position, and the
replay was refused with a ``TopologyError`` (the fingerprint of the ``surfaceFacets`` record). The
fingerprint now hashes the rigid surface nodes' reference coordinates.

The standard is bitwise: the resumed run must end exactly where the uninterrupted one does.
"""

from pathlib import Path

import h5py
import numpy as np

from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.utils.inputfileparser import parseInputFile

#: A flat rigid plate under the box: a closed cuboid, as 12 triangles.
_PLATE_STL = """solid plate
{:}
endsolid plate
"""


def _writePlateStl(path: Path):
    corners = np.array([[x, y, z] for x in (-0.5, 2.5) for y in (-0.5, 2.5) for z in (0.0, 1.0)])
    center = corners.mean(axis=0)
    # Each face of the cuboid as two triangles, oriented outward.
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    facets = []
    for a, b, c, d in faces:
        for triangle in ((a, b, c), (a, c, d)):
            p0, p1, p2 = corners[list(triangle)]
            normal = np.cross(p1 - p0, p2 - p0)
            if normal @ ((p0 + p1 + p2) / 3 - center) < 0.0:
                p1, p2, normal = p2, p1, -normal
            normal /= np.linalg.norm(normal)
            facets.append(
                " facet normal {:} {:} {:}\n  outer loop\n".format(*normal)
                + "".join("   vertex {:} {:} {:}\n".format(*p) for p in (p0, p1, p2))
                + "  endloop\n endfacet"
            )
    path.write_text(_PLATE_STL.format("\n".join(facets)))


def _deck(stl: Path, maxNumInc: int, extra: str = "") -> str:
    return f"""
*material, name=LinearElastic, id=linearelastic
1.8e4, 0.3, 1.0

*job, name=rigidContactAmrRestart, domain=3d

*solver, solver=NEDParallel, name=theSolver
courant-number=0.3
output-frequency=5
contact-update-frequency=1
topology-check-frequency=100

*modelGenerator, generator=boxGen, name=upper
nX      =3
nY      =3
nZ      =3
x0      =0.3
y0      =1.1
z0      =1.05
lX      =0.6
lY      =0.6
lZ      =1
elType  =C3D20R

*modelGenerator, generator=surfaceElementGenerator, name=gen1
surface = upper_back
name    = slaveSurf

*modelGenerator, generator=discreteRigidBodyGenerator, name=support, executeAfterManualGeneration=True
filename={stl}
rpCoordinate='1.0, 1.0, 0.5'
mass=1.0
inertia="1.0, 1.0, 1.0"

*section, name=section1, material=linearelastic, type=solid
upper_all

*modelModifier, type=hAdaptivity, name=amr
>>marker, type=elementSet, elSet=upper_all, initialOnly=False
refineElSet=upper_all
maxLevel=1

*constraint, name=contact, type=surfaceToDiscreteRigidBodyPenalty
slaveSurface=slaveSurf_facets, rigidBody=support, penalty=5e4, type=linear, searchDistance=2.0
{extra}
*step, type=adaptiveForExplicitSimulations, solver=theSolver
maxInc=1, minInc=1e-12, maxNumInc={maxNumInc}, maxIter=25, stepLength=0.5
>>dirichlet, name=slideSupport, nSet=support_rp, field=displacement, 1=0.1, 2=0, 3=0
>>dirichlet, name=fixSupportRotation, nSet=support_rp, field=rotation, 1=0, 2=0, 3=0
>>dirichlet, name=pushDown, nSet=upper_front, field=displacement, 3=-0.2
>>dirichlet, name=pinUpperXY, nSet=upper_bottomLeftBack, field=displacement, 1=0.0, 2=0.0
>>dirichlet, name=pinUpperRotZ, nSet=upper_bottomRightBack, field=displacement, 2=0.0
"""


def _run(path: Path, text: str):
    path.write_text(text)
    model, _ = finiteElementSimulation(parseInputFile(str(path)), verbose=False, suppressPlots=True)
    return model


#: The refinement happens at the first topology check (increment 100), after the support has started
#: to slide; the checkpoint falls between two later checks.
N_INCREMENTS_FULL = 400
N_INCREMENTS_TRUNCATED = 340


def test_resume_across_refinement_next_to_a_moving_rigid_body(tmp_path):
    stl = tmp_path / "plate.stl"
    _writePlateStl(stl)

    reference = _run(tmp_path / "full.inp", _deck(stl, N_INCREMENTS_FULL))

    writer = (
        f"*output, type=restart, name=restart\nwriteInterval=1, baseName={tmp_path / 'ckpt'}, numberOfFilesToKeep=1\n"
    )
    _run(tmp_path / "truncated.inp", _deck(stl, N_INCREMENTS_TRUNCATED, writer))
    (checkpoint,) = tmp_path.glob("ckpt_*.h5")
    with h5py.File(checkpoint, "r") as f:
        assert f.attrs["time"] > 0.0

    resumed = _run(tmp_path / "resumed.inp", _deck(stl, N_INCREMENTS_FULL, f"*restart, readFrom={checkpoint}"))

    # The slave body was refined before the checkpoint, and the support had moved by then.
    assert len(resumed.elements) > 2 * 27
    assert resumed.topologyHistory and resumed.topologyHistory[0].time > 0.0
    for name, field in reference.nodeFields.items():
        for entry in ("U", "P"):
            assert np.array_equal(resumed.nodeFields[name][entry], field[entry]), f"{name}/{entry} differs"
