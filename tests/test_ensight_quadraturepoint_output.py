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
written files: each element is split into one linear sub-element per quadrature point of its
Gauss rule, in proportion to the weights, which carries the result of the quadrature point it
contains.

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
from edelweissfe.outputmanagers.ensight import (
    createQuadraturePointPartFromElementSet,
    createUnstructuredPartFromElementSet,
)
from edelweissfe.points.node import Node
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
>>perElement, name=stress, elSet=all, result=stress, quadraturePoint={quadraturePoints}{function}

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

# per number of Gauss points in a direction: the points, and the natural coordinates which split
# the element in proportion to the weights
GAUSS = {
    2: ((-1 / np.sqrt(3), 1 / np.sqrt(3)), (-1.0, 0.0, 1.0)),
    3: ((-np.sqrt(0.6), 0.0, np.sqrt(0.6)), (-1.0, -4 / 9, 4 / 9, 1.0)),
}

# (provider, element type, spatial dimensions, Gauss points per direction): 2D quadrilaterals and
# 3D hexahedra, linear and quadratic, the latter with reduced and with full integration
CASES = [
    (provider, elType, dim, n)
    for provider in ("edelweiss", "marmot")
    for elType, dim, n in (
        ("CPE4", 2, 2),
        ("CPS4", 2, 2),
        ("CPE8R", 2, 2),
        ("CPE8", 2, 3),
        ("C3D8", 3, 2),
        ("C3D20R", 3, 2),
        ("C3D20", 3, 3),
    )
]


def _run(
    tmp_path, monkeypatch, provider: str, elType: str, dim: int, quadraturePoints: str = None, function: str = None
):
    """Run the deck in a temporary directory, which then holds the Ensight export; `function` is
    the f(x) of the stress field output."""

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
        function=", f(x)='{:}'".format(function) if function else "",
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


def _cellsOfSubElements(part: dict, subShape: str, dim: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """For each sub-element of the regular mesh the Gauss point it contains and its extents, from
    where its centroid lies in its element."""

    spacing = np.array(LENGTHS[:dim]) / np.array(DIVISIONS[:dim])
    points, breakpoints = (np.array(values) for values in GAUSS[n])
    cellCenters = (breakpoints[:-1] + breakpoints[1:]) / 2

    centroids = part["coordinates"][:, :dim][part["elements"][subShape]].mean(axis=1)
    elementCenters = (np.floor(centroids / spacing) + 0.5) * spacing
    natural = (centroids - elementCenters) / (spacing / 2)
    cells = np.abs(natural[:, :, None] - cellCenters).argmin(axis=2)
    assert np.allclose(natural, cellCenters[cells], atol=1e-6)

    return elementCenters + points[cells] * spacing / 2, np.diff(breakpoints)[cells] * spacing / 2


@pytest.mark.parametrize("provider, elType, dim, n", CASES, ids=lambda v: str(v))
def test_quadrature_point_results_on_sub_elements(tmp_path, monkeypatch, provider, elType, dim, n):
    nQp = n**dim
    model, _ = _run(tmp_path, monkeypatch, provider, elType, dim, "0:{:}".format(nQp))

    nElements = len(model.elementSets["all"])
    divisions = np.array(DIVISIONS[:dim])
    spacing = np.array(LENGTHS[:dim]) / divisions
    amplitudes = np.array(AMPLITUDES[:dim])
    following = [(i + 1) % dim for i in range(dim)]

    parts = _readGeometry("ensight/geometry.geo_0000")
    base, part = parts["all"], parts["all_QP"]
    subShape = "quad4" if dim == 2 else "hexa8"

    # one linear sub-element per quadrature point
    assert list(part["elements"]) == [subShape]
    connectivity = part["elements"][subShape]
    assert len(connectivity) == nQp * nElements
    assert len(np.unique(part["labels"][subShape])) == nQp * nElements

    # the nodes are those of the lattice of n cells per element and direction: nodes shared by neighboring
    # elements are not duplicated, and nodes of the element set off that lattice are left out
    coordinates = part["coordinates"][:, :dim]
    assert len(coordinates) == np.prod(n * divisions + 1)
    assert len(np.unique(np.round(coordinates / spacing, 5), axis=0)) == len(coordinates)

    # the nodes of the element set come first; all of them, unless edge midpoints are off the lattice
    nBase = len(base["coordinates"])
    allBaseNodesKept = n % 2 == 0 or nBase == np.prod(divisions + 1)
    if allBaseNodesKept:
        assert np.array_equal(part["coordinates"][:nBase], base["coordinates"])
    else:
        nCorners = np.prod(divisions + 1)
        assert all((row == base["coordinates"]).all(axis=1).any() for row in part["coordinates"][:nCorners])

    # the sub-elements are the cells of that lattice, each one exactly once and none inverted
    quadraturePoints, extents = _cellsOfSubElements(part, subShape, dim, n)
    X = coordinates[connectivity]
    assert np.allclose(np.ptp(X, axis=1), extents)
    assert len(np.unique(np.round(X.mean(axis=1) / spacing, 5), axis=0)) == nQp * nElements
    if dim == 2:
        x, y = X[:, :, 0], X[:, :, 1]
        volumes = 0.5 * np.sum(x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y, axis=1)
    else:
        volumes = 8 * np.linalg.det(np.einsum("in,enj->eij", hexa8DNdXi(0.0, 0.0, 0.0), X))
    assert np.allclose(volumes, np.prod(extents, axis=1))

    # each sub-element carries the stress at the Gauss point it contains
    stress = _readVariable(_lastFile("ensight/stress.var_*"), parts, 6)
    assert list(stress) == ["all_QP"]
    assert np.allclose(stress["all_QP"][subShape][:, :dim], E * amplitudes * quadraturePoints[:, following], rtol=1e-5)

    # node results are written on the sub-element part as well, interpolated to the added nodes
    displacement = _readVariable(_lastFile("ensight/displacement.var_*"), parts, 3)
    assert set(displacement) == {"all", "all_QP"}
    if allBaseNodesKept:
        assert np.array_equal(displacement["all"], displacement["all_QP"][:nBase])
    expected = amplitudes * coordinates * coordinates[:, following]
    assert np.allclose(displacement["all_QP"][:, :dim], expected, rtol=1e-5, atol=1e-9)
    assert not np.any(displacement["all_QP"][:, dim:])


@pytest.mark.parametrize("elType, nQp", [("CPE8R", 4), ("CPE8", 9)])
def test_sub_element_of_each_quadrature_point_of_the_element(tmp_path, monkeypatch, elType, nQp):
    """Sub-element g of an element contains the element's quadrature point g."""

    model, _ = _run(tmp_path, monkeypatch, "edelweiss", elType, 2, "0:{:}".format(nQp))

    part = _readGeometry("ensight/geometry.geo_0000")["all_QP"]
    X = part["coordinates"][:, :2][part["elements"]["quad4"]].reshape(-1, nQp, 4, 2)
    for element, subElements in zip(model.elementSets["all"], X):
        quadraturePoints = element.getCoordinatesAtQuadraturePoints()
        assert np.all(subElements.min(axis=1) < quadraturePoints)
        assert np.all(quadraturePoints < subElements.max(axis=1))


@pytest.mark.parametrize(
    "elType, dim, quadraturePoints, message",
    [
        ("CPE4", 2, "0:2", "quadraturePoint=0:4"),
        ("CPE8", 2, "0:4", "quadraturePoint=0:9"),
        ("C3D8", 3, "0", "quadraturePoint=0:8"),
        ("C3D20", 3, "0:8", "quadraturePoint=0:27"),
    ],
)
def test_incomplete_quadrature_point_range_is_refused(tmp_path, monkeypatch, elType, dim, quadraturePoints, message):
    with pytest.raises(Exception, match=message):
        _run(tmp_path, monkeypatch, "edelweiss", elType, dim, quadraturePoints)


class _StubElement:
    """Just what the sub-element part needs of an element: a unit square with given quadrature points."""

    def __init__(self, elNumber: int, ensightType: str, quadraturePoints: list):
        corners = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        self.elNumber = elNumber
        self.ensightType = ensightType
        self.visualizationNodes = [Node(4 * elNumber + i, np.array(x)) for i, x in enumerate(corners)]
        self._quadraturePoints = np.array(quadraturePoints)

    def getNumberOfQuadraturePoints(self) -> int:
        return len(self._quadraturePoints)

    def getCoordinatesAtQuadraturePoints(self) -> np.ndarray:
        return self._quadraturePoints


_GAUSS_2X2 = [(0.5 + 0.5 * a / np.sqrt(3), 0.5 + 0.5 * b / np.sqrt(3)) for a in (-1, 1) for b in (-1, 1)]


@pytest.mark.parametrize(
    "elements, message",
    [
        # not a quadrilateral or hexahedron
        ([_StubElement(1, "tria3", _GAUSS_2X2)], "requires Quad4, Quad8, Hexa8 or Hexa20 elements"),
        # not as many quadrature points as a rule with n points per direction has
        ([_StubElement(1, "quad4", _GAUSS_2X2[:3])], "requires a Gauss rule"),
        # as many, but elsewhere
        ([_StubElement(1, "quad4", [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])], "requires a Gauss rule"),
        # as many, but two of them at the same Gauss point
        ([_StubElement(1, "quad4", _GAUSS_2X2[:3] + _GAUSS_2X2[:1])], "requires a Gauss rule"),
        # two rules in one element set
        (
            [_StubElement(1, "quad4", _GAUSS_2X2), _StubElement(2, "quad4", [(0.5, 0.5)])],
            "requires the same number of quadrature points in all elements",
        ),
    ],
)
def test_elements_without_a_sub_element_per_quadrature_point_are_refused(elements, message):
    basePart = createUnstructuredPartFromElementSet("set", elements, 1)
    with pytest.raises(Exception, match=message):
        createQuadraturePointPartFromElementSet("set", elements, basePart, 2)


def test_single_quadrature_point():
    """An element with one quadrature point is its own sub-element."""

    element = _StubElement(1, "quad4", [(0.5, 0.5)])
    basePart = createUnstructuredPartFromElementSet("set", [element], 1)
    part, mapping = createQuadraturePointPartFromElementSet("set", [element], basePart, 2)

    assert part.nodes == element.visualizationNodes
    assert [list(nodes) for nodes in part.elementTree["quad4"].values()] == [[0, 1, 2, 3]]
    assert np.array_equal(mapping.toarray(), np.eye(4))


def test_function_of_the_result_per_quadrature_point(tmp_path, monkeypatch):
    """An f(x) of the field output which keeps one entry per quadrature point: a stress component."""

    _run(tmp_path, monkeypatch, "edelweiss", "C3D8", 3, function="x[:, :, 1]")

    parts = _readGeometry("ensight/geometry.geo_0000")
    quadraturePoints, _ = _cellsOfSubElements(parts["all_QP"], "hexa8", 3, 2)

    stress = _readVariable(_lastFile("ensight/stress.var_*"), parts, 1)["all_QP"]["hexa8"]
    assert np.allclose(stress[:, 0], E * AMPLITUDES[1] * quadraturePoints[:, 2], rtol=1e-5)


@pytest.mark.parametrize(
    "elType, dim, function",
    [
        # reduced over the quadrature points
        ("C3D8", 3, "np.mean(x, axis=1)"),
        ("CPE4", 2, "np.max(x, axis=1)"),
        # the same number of values in another layout
        ("C3D8", 3, "np.swapaxes(x, 1, 2)"),
        ("CPE4", 2, "x.reshape(len(x), -1)"),
    ],
)
def test_result_not_laid_out_per_quadrature_point_is_refused(tmp_path, monkeypatch, elType, dim, function):
    with pytest.raises(Exception, match="does not start with .* elements and .* quadrature points"):
        _run(tmp_path, monkeypatch, "edelweiss", elType, dim, function=function)


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
