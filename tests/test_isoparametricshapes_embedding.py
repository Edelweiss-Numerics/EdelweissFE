"""Unit tests of the isoparametric shape functions, the inverse isoparametric mapping, the host element locator
and the splitting of bars at host boundaries (collected by CI from tests/)."""

from types import SimpleNamespace

import numpy as np
import pytest

from edelweissfe.utils.embedding import (
    HostElementLocator,
    barLength,
    barPoint,
    findBarPartsInHosts,
)
from edelweissfe.utils.isoparametricshapes import (
    NODE_PARAMETRIC_COORDINATES,
    SIMPLEX_SHAPES,
    inverseMapping,
    shapeFunctionsAndDerivatives,
)

SHAPES = list(NODE_PARAMETRIC_COORDINATES)


@pytest.mark.parametrize("shape", SHAPES)
def test_kronecker_delta_and_partition_of_unity(shape):
    nodes = NODE_PARAMETRIC_COORDINATES[shape]
    for a, xi in enumerate(nodes):
        N, _ = shapeFunctionsAndDerivatives(shape, xi)
        np.testing.assert_allclose(N, np.eye(len(nodes))[a], atol=1e-14)
    rng = np.random.default_rng(0)
    for _ in range(5):
        xi = rng.uniform(-0.5, 0.5, nodes.shape[1]) * (0.5 if shape in SIMPLEX_SHAPES else 1.0) + (
            0.25 if shape in SIMPLEX_SHAPES else 0.0
        )
        N, dN = shapeFunctionsAndDerivatives(shape, xi)
        assert abs(N.sum() - 1.0) < 1e-14
        np.testing.assert_allclose(dN.sum(axis=0), 0.0, atol=1e-13)


@pytest.mark.parametrize("shape", SHAPES)
def test_derivatives_match_finite_differences(shape):
    dim = NODE_PARAMETRIC_COORDINATES[shape].shape[1]
    xi = np.full(dim, 0.2) + 0.05 * np.arange(dim)
    _, dN = shapeFunctionsAndDerivatives(shape, xi)
    h = 1e-7
    for d in range(dim):
        e = np.zeros(dim)
        e[d] = h
        num = (shapeFunctionsAndDerivatives(shape, xi + e)[0] - shapeFunctionsAndDerivatives(shape, xi - e)[0]) / (
            2 * h
        )
        np.testing.assert_allclose(dN[:, d], num, atol=1e-8)


def _distorted(shape):
    """A distorted element: the parent nodes mapped by a smooth nonlinear map (curved edges for quadratic shapes)."""
    nodes = NODE_PARAMETRIC_COORDINATES[shape]
    dim = nodes.shape[1]
    A = np.eye(dim) * 2.0 + 0.3 * np.ones((dim, dim))
    return np.array([A @ p + 0.1 * np.sin(p.sum()) + 0.05 * p**2 for p in nodes])


@pytest.mark.parametrize("shape", ["quad4", "quad8", "hexa8", "hexa20", "tria3", "tria6", "tetra4", "tetra10"])
def test_inverse_mapping(shape):
    coords = _distorted(shape)
    rng = np.random.default_rng(1)
    dim = coords.shape[1]
    for _ in range(10):
        xi = rng.uniform(0.0, 0.3, dim) if shape in SIMPLEX_SHAPES else rng.uniform(-0.9, 0.9, dim)
        x = shapeFunctionsAndDerivatives(shape, xi)[0] @ coords
        np.testing.assert_allclose(inverseMapping(shape, coords, x), xi, atol=1e-10)


@pytest.mark.parametrize("shape", ["quad4", "quad8", "hexa8", "hexa20", "tria3", "tria6", "tetra4", "tetra10"])
def test_inverse_mapping_far_from_the_origin(shape):
    """EMB-01: an element of size h ~ 25 (mm) offset by 1e4 h from the origin, 5 % distorted: interior points are
    found with an error at round-off of the element size. An absolute step tolerance fails here (the residual carries
    eps |x|), which made interior points "outside of all hosts"."""
    rng = np.random.default_rng(2)
    h = 25.0
    for offset in (1e4 * h, 1e6 * h):
        coords = h * _distorted(shape) / 2.0 + offset * np.linspace(
            1.0, 0.7, NODE_PARAMETRIC_COORDINATES[shape].shape[1]
        )
        coords = coords + 0.05 * h * rng.uniform(-1, 1, coords.shape)
        for _ in range(20):
            xi = (
                rng.uniform(0.0, 0.3, coords.shape[1])
                if shape in SIMPLEX_SHAPES
                else rng.uniform(-0.9, 0.9, coords.shape[1])
            )
            x = shapeFunctionsAndDerivatives(shape, xi)[0] @ coords
            found = inverseMapping(shape, coords, x)
            assert found is not None, f"{shape}: not converged at offset {offset}"
            np.testing.assert_allclose(found, xi, atol=1e-8)


def test_inverse_mapping_of_the_adaptivity_uses_the_shared_newton():
    """the hex20 topology of the adaptivity uses the same Newton (its own node ordering): far from the origin too"""
    from edelweissfe.adaptivity.hex20topology import Hex20Topology

    topology = Hex20Topology()
    from edelweissfe.adaptivity.hex20shapefunctions import LOCAL_COORDS

    corners = 25.0 * np.asarray(LOCAL_COORDS, float) / 2.0 + 2.5e5
    xi = np.array([0.3, -0.4, 0.6])
    x = np.asarray(topology.shape_functions(*xi)) @ corners
    np.testing.assert_allclose(topology.inverse_map(x, corners), xi, atol=1e-9)


def _node(label, x):
    return SimpleNamespace(label=label, coordinates=np.asarray(x, dtype=float))


def _element(number, shape, nodes):
    return SimpleNamespace(elNumber=number, ensightType=shape, nodes=nodes)


def _quadGrid(nx, ny, lx, ly, origin=(0.0, 0.0)):
    """A structured grid of quad4 elements on [0, lx] x [0, ly], shifted by ``origin``."""
    x0, y0 = origin
    nodes = {
        (i, j): _node(i * (ny + 1) + j, (x0 + lx * i / nx, y0 + ly * j / ny))
        for i in range(nx + 1)
        for j in range(ny + 1)
    }
    elements = []
    for i in range(nx):
        for j in range(ny):
            elements.append(
                _element(
                    len(elements),
                    "quad4",
                    [nodes[(i, j)], nodes[(i + 1, j)], nodes[(i + 1, j + 1)], nodes[(i, j + 1)]],
                )
            )
    return elements


def test_locator_finds_host_and_shape_functions_interpolate():
    hosts = _quadGrid(4, 3, 4.0, 3.0)
    locator = HostElementLocator(hosts)
    x = np.array([2.3, 1.7])
    p = locator.locate(x)
    assert p.element is hosts[2 * 3 + 1]
    coords = np.array([n.coordinates for n in p.element.nodes])
    np.testing.assert_allclose(p.shapeFunctions() @ coords, x, atol=1e-12)
    assert locator.locate(np.array([5.0, 1.0])) is None


@pytest.mark.parametrize("barShape", ["bar2", "bar3"])
def test_split_inclined_bar(barShape):
    """an inclined bar crossing a grid is split into parts, one per crossed host element, that tile the bar"""
    hosts = _quadGrid(4, 3, 4.0, 3.0)
    locator = HostElementLocator(hosts)
    a, b = np.array([0.2, 0.35]), np.array([3.7, 2.6])
    barNodes = [_node(100, a), _node(101, b)] + ([_node(102, 0.5 * (a + b))] if barShape == "bar3" else [])
    bar = _element(1000, barShape, barNodes)

    parts, unbonded = findBarPartsInHosts(bar, locator)

    assert unbonded == 0.0
    assert parts[0].etaStart == -1.0 and parts[-1].etaEnd == 1.0
    for p, q in zip(parts[:-1], parts[1:]):
        assert abs(p.etaEnd - q.etaStart) < 1e-9
        assert p.hostElement is not q.hostElement
    # the crossed cells, by walking the line in fine steps
    crossed = []
    for t in np.linspace(0, 1, 20001):
        x = a + t * (b - a)
        cell = int(x[0]) * 3 + int(x[1])
        if not crossed or crossed[-1] != cell:
            crossed.append(cell)
    assert [p.hostElement.elNumber for p in parts] == crossed
    coords = np.array([n.coordinates for n in barNodes])
    total = sum(barLength(barShape, coords, p.etaStart, p.etaEnd) for p in parts)
    assert abs(total - np.linalg.norm(b - a)) < 1e-9


def test_split_bar_sticking_out():
    hosts = _quadGrid(2, 2, 2.0, 2.0)
    locator = HostElementLocator(hosts)
    bar = _element(1000, "bar2", [_node(100, (-1.0, 0.5)), _node(101, (1.5, 0.5))])
    parts, unbonded = findBarPartsInHosts(bar, locator)
    # the bar is outside on x in [-1, 0]: length 1, up to the boundary tolerance of the hosts (1e-8 in parametric
    # coordinates, i.e., 5e-9 here)
    assert abs(unbonded - 1.0) < 1e-8
    assert [p.hostElement.elNumber for p in parts] == [0, 2]
    assert abs(parts[0].etaStart - (-0.2)) < 1e-8


@pytest.mark.parametrize("start, end", [((1.0, 1.0), (3.0, 1.0)), ((1.0, 0.5), (3.0, 1.7)), ((1.0, 1.0), (1.0, 3.0))])
def test_split_bar_starting_on_host_boundary_has_no_degenerate_parts(start, end):
    """a bar starting on a host edge or node must not get a zero-length part in a host it only touches"""
    hosts = _quadGrid(4, 4, 4.0, 4.0)
    locator = HostElementLocator(hosts)
    bar = _element(1000, "bar2", [_node(100, start), _node(101, end)])
    parts, unbonded = findBarPartsInHosts(bar, locator)
    assert unbonded == 0.0
    assert all(p.etaEnd - p.etaStart > 1e-6 for p in parts)
    assert parts[0].etaStart == -1.0 and parts[-1].etaEnd == 1.0


@pytest.mark.parametrize("barShape", ["bar2", "bar3"])
def test_split_bar_through_mesh_corners_has_no_slivers(barShape):
    """a bar at 45 degrees through the nodes of a structured grid touches the side neighbors only at the corners: no
    sliver parts, and the parts tile the bar"""
    hosts = _quadGrid(6, 6, 6.0, 6.0)
    locator = HostElementLocator(hosts)
    a, b = np.array([0.5, 0.5]), np.array([5.5, 5.5])  # through the grid nodes (1,1), (2,2), ...
    barNodes = [_node(100, a), _node(101, b)] + ([_node(102, 0.5 * (a + b))] if barShape == "bar3" else [])
    parts, unbonded = findBarPartsInHosts(_element(1000, barShape, barNodes), locator)
    assert unbonded == 0.0
    assert all(p.etaEnd - p.etaStart > 1e-3 for p in parts)
    assert [p.hostElement.elNumber for p in parts] == [0, 7, 14, 21, 28, 35]  # the diagonal cells
    assert parts[0].etaStart == -1.0 and parts[-1].etaEnd == 1.0
    for p, q in zip(parts[:-1], parts[1:]):
        assert p.etaEnd == q.etaStart


def _assertPartsInTheirHosts(parts, bar, locator):
    """every part starts and ends inside its host (within the locator tolerance), and parts do not overlap"""
    shape = bar.ensightType
    coords = np.array([n.coordinates for n in bar.nodes])
    for p in parts:
        for eta in (p.etaStart, 0.5 * (p.etaStart + p.etaEnd), p.etaEnd):
            assert locator.contains(p.hostElement, barPoint(shape, coords, eta))
    for p, q in zip(parts[:-1], parts[1:]):
        assert p.etaEnd <= q.etaStart


@pytest.mark.parametrize("y", [1.0, 2.0])
def test_split_bar_along_a_host_edge(y):
    """a bar lying exactly on the edges between two rows of hosts is split into one part per column, each in one of the
    two hosts sharing the edge, without gaps"""
    hosts = _quadGrid(4, 3, 4.0, 3.0)
    locator = HostElementLocator(hosts)
    bar = _element(1000, "bar2", [_node(100, (0.3, y)), _node(101, (3.6, y))])
    parts, unbonded = findBarPartsInHosts(bar, locator)
    assert unbonded == 0.0
    assert parts[0].etaStart == -1.0 and parts[-1].etaEnd == 1.0
    assert len(parts) == 4
    for p, q in zip(parts[:-1], parts[1:]):
        assert p.etaEnd == q.etaStart
    _assertPartsInTheirHosts(parts, bar, locator)


def test_split_bar_far_from_the_origin():
    """the split far from the origin (offset 1e5 times the host size) equals the one at the origin"""
    a, b = np.array([0.2, 0.35]), np.array([3.7, 2.6])
    results = []
    for origin in (np.zeros(2), np.array([1e5, -4e5])):
        locator = HostElementLocator(_quadGrid(4, 3, 4.0, 3.0, origin))
        parts, unbonded = findBarPartsInHosts(_element(1000, "bar2", [_node(100, origin + a), _node(101, origin + b)]), locator)
        assert unbonded == 0.0
        results.append([(p.hostElement.elNumber, p.etaStart, p.etaEnd) for p in parts])
    assert [r[0] for r in results[0]] == [r[0] for r in results[1]]
    np.testing.assert_allclose([r[1:] for r in results[0]], [r[1:] for r in results[1]], atol=1e-9)


def test_split_long_bar_passing_beside_a_host_node():
    """a bar of 10 host sizes passing 1e-7 host sizes beside a host node (BOND-07): the sliver in the corner host is
    dropped as unbonded, and no part reaches outside of its host"""
    hosts = _quadGrid(10, 2, 10.0, 2.0)
    locator = HostElementLocator(hosts)
    # slightly inclined bar through (5, 1 + 1e-7): just above the node (5, 1)
    a, b = np.array([0.0, 1.0 + 1e-7 - 5 * 1e-3]), np.array([10.0, 1.0 + 1e-7 + 5 * 1e-3])
    bar = _element(1000, "bar2", [_node(100, a), _node(101, b)])
    parts, unbonded = findBarPartsInHosts(bar, locator)
    _assertPartsInTheirHosts(parts, bar, locator)
    assert unbonded < 1e-5
    total = sum(barLength("bar2", np.array([a, b]), p.etaStart, p.etaEnd) for p in parts)
    assert abs(total + unbonded - np.linalg.norm(b - a)) < 1e-9
