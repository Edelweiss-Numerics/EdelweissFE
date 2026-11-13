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
"""
The Ensight export of quadrature point results (``>>perQuadraturePoint``), read back from the
written files: each element is split into one linear sub-element per quadrature point, which
carries the result of the quadrature point it contains.

The decks prescribe the displacement ``u_i = A_i x_i x_j`` (``j`` the next direction) at all nodes
of a regular mesh, which bilinear and serendipity elements represent exactly. With a Poisson's
ratio of zero the normal stress ``sigma_ii = E A_i x_j`` hence tells where it was evaluated,
whatever order an element numbers its quadrature points in.
"""

import glob

import numpy as np
import pytest

from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.elements._hexa3dnodeordering import hexa8DNdXi
from edelweissfe.utils.inputfileparser import parseInputFile

E = 1000.0
AMPLITUDES = (1e-3, 2e-3, 3e-3)
LENGTHS = (3.0, 1.0, 2.0)
DIVISIONS = (3, 2, 2)

NODES_PER_SHAPE = {"point": 1, "bar2": 2, "quad4": 4, "quad8": 8, "hexa8": 8, "hexa20": 20}

DECK = """
*material, name={material}, id=mat{materialProvider}
{E}, 0.0

*section, name=section1, material=mat, {section}
all

*job, name=job, domain={dim}d
*solver, solver=NIST, name=theSolver

*modelGenerator, generator={generator}, name=gen
{mesh}, elType={elType}, elProvider={provider}

*fieldOutput
>>perNode, name=displacement, elSet=all, field=displacement, result=U
>>perElement, name=stress, elSet=all, result=stress, quadraturePoint={quadraturePoints}

*output, type=ensight, name=ensight
>>perNode, fieldOutput=displacement
>>perQuadraturePoint, fieldOutput=stress
>>configuration, overwrite=yes

{analyticalFields}

*step, solver=theSolver
maxInc=1.0, minInc=1.0, maxNumInc=10, maxIter=10, stepLength=1
{dirichlets}
"""

MATERIALS = {
    "edelweiss": dict(material="linearelastic", materialProvider=", provider=edelweiss"),
    "marmot": dict(material="LINEARELASTIC", materialProvider=""),
}

# (provider, element type, spatial dimensions): 2D quadrilaterals and 3D hexahedra, linear and quadratic
CASES = [
    (provider, elType, dim)
    for provider in ("edelweiss", "marmot")
    for elType, dim in (("CPE4", 2), ("CPS4", 2), ("CPE8R", 2), ("C3D8", 3), ("C3D20R", 3))
]


def _run(tmp_path, monkeypatch, provider: str, elType: str, dim: int, quadraturePoints: str = None):
    """Run the deck in a temporary directory, which then holds the Ensight export."""

    if provider == "marmot":
        pytest.importorskip("edelweissfe.elements.marmotelement.element")
        pytest.importorskip("edelweissfe.materials.marmot.marmothypoelastic")

    axes = "xyz"[:dim]
    if dim == 2:
        mesh = "x0=0, l={:}, y0=0, h={:}, nX={:}, nY={:}".format(*LENGTHS[:2], *DIVISIONS[:2])
    else:
        mesh = "x0=0, lX={:}, y0=0, lY={:}, z0=0, lZ={:}, nX={:}, nY={:}, nZ={:}".format(*LENGTHS, *DIVISIONS)

    deck = DECK.format(
        **MATERIALS[provider],
        E=E,
        section="thickness=1.0, type=plane" if dim == 2 else "type=solid",
        dim=dim,
        generator="planeRectQuad" if dim == 2 else "boxGen",
        mesh=mesh,
        elType=elType,
        provider=provider,
        quadraturePoints=quadraturePoints or "0:{:}".format(2**dim),
        analyticalFields="\n".join(
            '*analyticalField, name=f{:}, type=scalarExpression\n"f(x,y,z)"="{:}*{:}"'.format(
                i, axes[i], axes[(i + 1) % dim]
            )
            for i in range(dim)
        ),
        dirichlets="\n".join(
            ">>dirichlet, name=u{0:}, nSet=all, field=displacement, {1:}={2:}, analyticalField=f{0:}".format(
                i, i + 1, AMPLITUDES[i]
            )
            for i in range(dim)
        ),
    )

    monkeypatch.chdir(tmp_path)
    (tmp_path / "test.inp").write_text(deck)
    return finiteElementSimulation(parseInputFile("test.inp"), verbose=False, suppressPlots=True)


class _EnsightFile:
    """Sequential reader of the records of an Ensight Gold C binary file."""

    def __init__(self, fileName: str):
        with open(fileName, "rb") as f:
            self._data = f.read()
        self._offset = 0

    def atEnd(self) -> bool:
        return self._offset >= len(self._data)

    def string(self, peek: bool = False) -> str:
        raw = self._data[self._offset : self._offset + 80]
        if not peek:
            self._offset += 80
        return raw.rstrip(b"\x00 ").decode()

    def array(self, dtype, size: int) -> np.ndarray:
        values = np.frombuffer(self._data, dtype, size, self._offset)
        self._offset += 4 * size
        return values


def _readGeometry(fileName: str) -> dict:
    """The parts of a geometry file by name: number, node coordinates, connectivity per shape."""

    f = _EnsightFile(fileName)
    assert f.string() == "C Binary"
    f.string(), f.string()
    assert f.string() == "node id given"
    assert f.string() == "element id given"

    parts = {}
    while not f.atEnd():
        assert f.string() == "part"
        number = int(f.array(np.int32, 1)[0])
        name = f.string()
        assert f.string() == "coordinates"
        nNodes = int(f.array(np.int32, 1)[0])
        f.array(np.int32, nNodes)
        coordinates = f.array(np.float32, 3 * nNodes).reshape(3, nNodes).T.astype(float)
        elements, labels = {}, {}
        while not f.atEnd() and f.string(peek=True) != "part":
            shape = f.string()
            nElements = int(f.array(np.int32, 1)[0])
            labels[shape] = f.array(np.int32, nElements)
            elements[shape] = f.array(np.int32, nElements * NODES_PER_SHAPE[shape]).reshape(nElements, -1) - 1
        parts[name] = dict(number=number, coordinates=coordinates, elements=elements, labels=labels)

    return parts


def _readVariable(fileName: str, parts: dict, size: int) -> dict:
    """The values of a variable file with `size` components by part name: an array per node, or an
    array per element shape."""

    partsByNumber = {part["number"]: (name, part) for name, part in parts.items()}

    f = _EnsightFile(fileName)
    f.string()
    values = {}
    while not f.atEnd():
        assert f.string() == "part"
        name, part = partsByNumber[int(f.array(np.int32, 1)[0])]
        if f.string(peek=True) == "coordinates":
            f.string()
            n = len(part["coordinates"])
            values[name] = f.array(np.float32, size * n).reshape(size, n).T.astype(float)
            continue
        values[name] = {}
        while not f.atEnd() and f.string(peek=True) != "part":
            shape = f.string()
            n = len(part["elements"][shape])
            values[name][shape] = f.array(np.float32, size * n).reshape(size, n).T.astype(float)

    return values


def _lastFile(pattern: str) -> str:
    return sorted(glob.glob(pattern))[-1]


@pytest.mark.parametrize("provider, elType, dim", CASES, ids=lambda v: str(v))
def test_quadrature_point_results_on_sub_elements(tmp_path, monkeypatch, provider, elType, dim):
    model, _ = _run(tmp_path, monkeypatch, provider, elType, dim)

    nQp = 2**dim
    nElements = len(model.elementSets["all"])
    spacing = np.array(LENGTHS[:dim]) / np.array(DIVISIONS[:dim])
    amplitudes = np.array(AMPLITUDES[:dim])
    following = [(i + 1) % dim for i in range(dim)]

    parts = _readGeometry("ensight/geometry.geo_0000")
    base, part = parts["all"], parts["all_QP"]
    subShape = "quad4" if dim == 2 else "hexa8"

    # one linear sub-element per quadrature point, on the nodes of the element set and the added ones
    assert list(part["elements"]) == [subShape]
    connectivity = part["elements"][subShape]
    assert len(connectivity) == nQp * nElements
    assert len(np.unique(part["labels"][subShape])) == nQp * nElements

    coordinates = part["coordinates"][:, :dim]
    nBase = len(base["coordinates"])
    assert np.array_equal(part["coordinates"][:nBase], base["coordinates"])

    # nodes shared by neighboring elements are not duplicated: the nodes are those of the lattice of half the spacing
    assert len(coordinates) == np.prod(2 * np.array(DIVISIONS[:dim]) + 1)
    assert len(np.unique(np.round(coordinates / (spacing / 2)), axis=0)) == len(coordinates)

    # the sub-elements are the cells of that lattice, each one exactly once and none inverted
    X = coordinates[connectivity]
    assert np.allclose(np.ptp(X, axis=1), spacing / 2)
    centroids = X.mean(axis=1)
    assert len(np.unique(np.round(centroids / (spacing / 4)), axis=0)) == nQp * nElements
    if dim == 2:
        x, y = X[:, :, 0], X[:, :, 1]
        volumes = 0.5 * np.sum(x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y, axis=1)
    else:
        volumes = 8 * np.linalg.det(np.einsum("in,enj->eij", hexa8DNdXi(0.0, 0.0, 0.0), X))
    assert np.allclose(volumes, np.prod(spacing / 2))

    # each sub-element carries the stress at the Gauss point it contains: at 1/sqrt(3) of the element's
    # half length from the element center, the centroid of the sub-element being at 1/2
    elementCenters = (np.floor(centroids / spacing) + 0.5) * spacing
    quadraturePoints = elementCenters + (centroids - elementCenters) * 2 / np.sqrt(3)

    stress = _readVariable(_lastFile("ensight/stress.var_*"), parts, 6)
    assert list(stress) == ["all_QP"]
    assert np.allclose(stress["all_QP"][subShape][:, :dim], E * amplitudes * quadraturePoints[:, following], rtol=1e-5)

    # node results are written on the sub-element part as well, interpolated to the added nodes
    displacement = _readVariable(_lastFile("ensight/displacement.var_*"), parts, 3)
    assert set(displacement) == {"all", "all_QP"}
    assert np.array_equal(displacement["all"], displacement["all_QP"][:nBase])
    expected = amplitudes * coordinates * coordinates[:, following]
    assert np.allclose(displacement["all_QP"][:, :dim], expected, rtol=1e-5, atol=1e-9)
    assert not np.any(displacement["all_QP"][:, dim:])


def test_sub_element_of_each_quadrature_point_of_the_element(tmp_path, monkeypatch):
    """Sub-element g of an element contains the element's quadrature point g."""

    model, _ = _run(tmp_path, monkeypatch, "edelweiss", "CPE8R", 2)

    part = _readGeometry("ensight/geometry.geo_0000")["all_QP"]
    X = part["coordinates"][:, :2][part["elements"]["quad4"]].reshape(-1, 4, 4, 2)
    for element, subElements in zip(model.elementSets["all"], X):
        quadraturePoints = element.getCoordinatesAtQuadraturePoints()
        assert np.all(subElements.min(axis=1) < quadraturePoints)
        assert np.all(quadraturePoints < subElements.max(axis=1))


@pytest.mark.parametrize(
    "elType, dim, quadraturePoints, message",
    [
        ("CPE8", 2, "0:9", "2 quadrature points per direction"),
        ("C3D20", 3, "0:27", "2 quadrature points per direction"),
        ("CPE4", 2, "0:2", "quadraturePoint=0:4"),
        ("C3D8", 3, "0", "quadraturePoint=0:8"),
    ],
)
def test_unsupported_quadrature_point_output_is_refused(tmp_path, monkeypatch, elType, dim, quadraturePoints, message):
    with pytest.raises(Exception, match=message):
        _run(tmp_path, monkeypatch, "edelweiss", elType, dim, quadraturePoints)


AMR_DECK = """
*material, name=LINEARELASTIC, id=mat
1000.0, 0.0

*section, name=section1, material=mat, type=solid
gen_all

*modelGenerator, generator=boxGen, name=gen
nX=2, nY=1, nZ=1, lX=2, lY=1, lZ=1, elType=C3D20R

*modelModifier, type=hAdaptivity, name=amr
>>marker, type=nodeSet, nSet=gen_right, initialOnly=True
maxLevel=1

*job, name=job, domain=3d
*solver, solver=NIST, name=theSolver

*fieldOutput
>>perNode, name=displacement, elSet=gen_all, field=displacement, result=U
>>perElement, name=stress, elSet=gen_all, result=stress, quadraturePoint=0:8

*output, type=ensight, name=ensight
>>perNode, fieldOutput=displacement
>>perQuadraturePoint, fieldOutput=stress
>>configuration, overwrite=yes

*step, solver=theSolver
maxInc=1.0, minInc=1.0, maxNumInc=1, maxIter=25, stepLength=1
>>dirichlet, name=fix, nSet=gen_left, field=displacement, 1=0.0, 2=0.0, 3=0.0
>>dirichlet, name=move, nSet=gen_right, field=displacement, 1=0.1, 2=0.05
"""


def test_sub_elements_follow_a_mesh_refinement(tmp_path, monkeypatch):
    """The sub-element part is rebuilt for a mesh refined after the output manager was set up."""

    pytest.importorskip("edelweissfe.elements.marmotelement.element")
    pytest.importorskip("edelweissfe.materials.marmot.marmothypoelastic")

    monkeypatch.chdir(tmp_path)
    (tmp_path / "test.inp").write_text(AMR_DECK)
    model, fieldOutputController = finiteElementSimulation(
        parseInputFile("test.inp"), verbose=False, suppressPlots=True
    )

    # one of the two elements is refined into 8
    elements = list(model.elementSets["gen_all"])
    assert len(elements) == 9

    parts = _readGeometry("ensight/geometry.geo_0000")
    part = parts["gen_all_QP"]
    X = part["coordinates"][part["elements"]["hexa8"]].reshape(9, 8, 8, 3)
    for element, subElements in zip(elements, X):
        quadraturePoints = np.asarray(element.getCoordinatesAtQuadraturePoints())
        assert np.all(subElements.min(axis=1) < quadraturePoints)
        assert np.all(quadraturePoints < subElements.max(axis=1))

    stress = _readVariable(_lastFile("ensight/stress.var_*"), parts, 6)["gen_all_QP"]["hexa8"]
    expected = np.reshape(fieldOutputController.fieldOutputs["stress"].getLastResult(), (9 * 8, 6))
    assert np.any(expected)
    assert np.allclose(stress, expected, rtol=1e-5, atol=1e-5 * np.abs(expected).max())

    displacement = _readVariable(_lastFile("ensight/displacement.var_*"), parts, 3)
    nBase = len(parts["gen_all"]["coordinates"])
    assert np.array_equal(displacement["gen_all"], displacement["gen_all_QP"][:nBase])
    assert len(displacement["gen_all_QP"]) == len(part["coordinates"]) > nBase
