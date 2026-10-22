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
#  Paul Hofer paul.hofer@uibk.ac.at
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
"""Geometry and set invariants of the cylinder generator, for every extrusion axis."""

import math

import numpy as np
import pytest

from edelweissfe.generators.cylindergenerator import CylinderGeneratorSchema, Generator
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel

ORIGIN = np.array([1.0, 2.0, 3.0])
RADIUS = 2.0
SECTION_LENGTHS = [1.0, 2.5, 1.5]
SECTION_NY = [1, 2, 1]
LENGTH = sum(SECTION_LENGTHS)
TOL = 1e-9

# global in-plane axes (iU, iV) per extrusion axis
IN_PLANE = {"x": (1, 2), "y": (0, 2), "z": (0, 1)}

# parts in terms of signs of the local (u, v) axes
PART_SIGNS = [(0, 0), (1, 0), (0, -1), (1, -1), (-1, 1)]

# trilinear corner signs of the Abaqus 8-node hexahedron
HEX_CORNERS = np.array(
    [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=float
)
GAUSS = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]) / math.sqrt(3)


def partString(axis, signs):
    """Spell local (u, v) signs as the generator's ``part`` option, e.g. '+x-z'."""
    names = "".join(("+" if s > 0 else "-") + "xyz"[i] for s, i in zip(signs, IN_PLANE[axis]) if s)
    return names or "full"


def generate(axis, signs=(0, 0), elType="C3D8", **kwargs):
    """Generate a sectioned cylinder into a fresh model."""
    options = dict(
        axis=axis,
        x0=ORIGIN[0],
        y0=ORIGIN[1],
        z0=ORIGIN[2],
        radius=RADIUS,
        nR=3,
        sectionLengths=", ".join(map(str, SECTION_LENGTHS)),
        sectionNY=", ".join(map(str, SECTION_NY)),
        part=partString(axis, signs),
        elType=elType,
        elProvider="edelweiss",
    )
    options.update(kwargs)
    model = FEModel(3)
    with model.topology.changes():
        Generator("gen", model, Journal(verbose=False), configuration=CylinderGeneratorSchema(**options))
    return model


def coordinates(nodes):
    return np.array([n.coordinates for n in nodes])


def cornerCoordinates(element):
    return coordinates(element.nodes[:8])


def jacobianDeterminants(element):
    """Determinants of the trilinear corner map at the 2x2x2 Gauss points."""
    corners = cornerCoordinates(element)
    dets = []
    for xi in GAUSS:
        dN = np.empty((8, 3))
        for k in range(3):
            factors = 1 + HEX_CORNERS * xi
            factors[:, k] = HEX_CORNERS[:, k]
            dN[:, k] = factors.prod(axis=1) / 8
        dets.append(np.linalg.det(corners.T @ dN))
    return np.array(dets)


def elementVolume(element):
    return jacobianDeterminants(element).sum()


def modelVolume(model):
    return sum(elementVolume(el) for el in model.elements.values())


@pytest.fixture(params=["x", "y", "z"])
def axis(request):
    return request.param


@pytest.fixture(scope="module")
def fullVolumes():
    """Volume and element count of the full cylinder, per axis."""
    return {axis: (modelVolume(model), len(model.elements)) for axis, model in ((a, generate(a)) for a in IN_PLANE)}


@pytest.mark.parametrize("signs", PART_SIGNS, ids=lambda s: "u{:+d}v{:+d}".format(*s))
def test_part_geometry(axis, signs, fullVolumes):
    model = generate(axis, signs)
    iAxis = "xyz".index(axis)
    iU, iV = IN_PLANE[axis]

    nodes = coordinates(model.nodes.values())
    relative = nodes - ORIGIN

    # every node lies in the selected part, between the end faces and inside the cylinder
    for sign, i in zip(signs, (iU, iV)):
        if sign:
            assert np.all(sign * relative[:, i] >= -TOL)
    assert np.all(relative[:, iAxis] >= -TOL) and np.all(relative[:, iAxis] <= LENGTH + TOL)
    assert np.all(np.hypot(relative[:, iU], relative[:, iV]) <= RADIUS * (1 + 1e-9))

    # every element is positively oriented, i.e., the winding matches the extrusion direction
    for el in model.elements.values():
        assert np.all(jacobianDeterminants(el) > 0)

    # the part is exactly the corresponding fraction of the full cylinder
    fraction = 0.5 ** sum(1 for s in signs if s)
    fullVolume, fullCount = fullVolumes[axis]
    assert len(model.elements) == fraction * fullCount
    assert modelVolume(model) == pytest.approx(fraction * fullVolume, rel=1e-10)
    # the straight-edged mesh is inscribed in the circle, so it slightly underestimates the volume
    assert 0.95 < fullVolume / (math.pi * RADIUS**2 * LENGTH) < 1.0


@pytest.mark.parametrize("signs", PART_SIGNS, ids=lambda s: "u{:+d}v{:+d}".format(*s))
@pytest.mark.parametrize("elType", ["C3D8", "C3D20R"])
def test_symmetry_sets(axis, signs, elType):
    model = generate(axis, signs, elType=elType)
    allRelative = coordinates(model.nodes.values()) - ORIGIN

    for sign, i in zip(signs, IN_PLANE[axis]):
        setName = "gen_symmetry" + "XYZ"[i]
        if not sign:
            assert setName not in model.nodeSets
            continue

        # exactly the nodes on the cutting plane, snapped onto it
        onPlane = coordinates(model.nodeSets[setName]) - ORIGIN
        assert np.all(onPlane[:, i] == 0.0)
        assert len(model.nodeSets[setName]) == np.count_nonzero(np.abs(allRelative[:, i]) < TOL)

    for i in range(3):
        if i not in IN_PLANE[axis] or not signs[IN_PLANE[axis].index(i)]:
            assert "gen_symmetry" + "XYZ"[i] not in model.nodeSets


@pytest.mark.parametrize("signs", [(0, 0), (1, -1)], ids=["full", "quarter"])
@pytest.mark.parametrize("elType", ["C3D8", "C3D20R"])
def test_section_sets(axis, signs, elType):
    model = generate(axis, signs, elType=elType)
    iAxis = "xyz".index(axis)
    iU, iV = IN_PLANE[axis]

    starts = np.concatenate([[0.0], np.cumsum(SECTION_LENGTHS)])
    nPerLayer = len(model.elementSets["gen_bottom"])
    nOuterPerLayer = len(model.elementSets["gen_outer"]) // sum(SECTION_NY)

    allSectionElements = []
    for iSection, (lower, upper, nY) in enumerate(zip(starts[:-1], starts[1:], SECTION_NY)):
        name = "gen_section{:}".format(iSection)
        elements = list(model.elementSets[name])
        allSectionElements += elements
        assert len(elements) == nY * nPerLayer

        # elements and nodes lie within the section's axial bounds
        for el in elements:
            axial = cornerCoordinates(el)[:, iAxis] - ORIGIN[iAxis]
            assert np.all(axial >= lower - TOL) and np.all(axial <= upper + TOL)
        axial = coordinates(model.nodeSets[name])[:, iAxis] - ORIGIN[iAxis]
        assert np.all(axial >= lower - TOL) and np.all(axial <= upper + TOL)
        # including the shared end faces
        assert np.any(np.abs(axial - lower) < TOL) and np.any(np.abs(axial - upper) < TOL)

        # the outer part of the section lies on the mantle
        outerElements = model.elementSets[name + "_outer"]
        assert len(outerElements) == nY * nOuterPerLayer
        assert set(outerElements) <= set(elements) & set(model.elementSets["gen_outer"])
        outerNodes = coordinates(model.nodeSets[name + "_outer"]) - ORIGIN
        assert np.allclose(np.hypot(outerNodes[:, iU], outerNodes[:, iV]), RADIUS)
        assert list(model.surfaces[name + "_outer"].keys()) == [5]

    # the sections partition the elements
    assert len(allSectionElements) == len(set(allSectionElements)) == len(model.elements)


def test_lY_ignored_with_sections():
    model = generate("y", lY=-1.0)
    assert len(model.elementSets["gen_section1"]) > 0

    with pytest.raises(Exception, match="lY must be positive"):
        generate("y", lY=-1.0, sectionLengths=None, sectionNY=None)
