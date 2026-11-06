"""Unit tests of the isoparametric shape functions and of the embedding of bars in host elements."""

from types import SimpleNamespace

import numpy as np
import pytest

from edelweissfe.utils.embedding import (
    HostElementLocator,
    barLength,
    splitBarElementAtHostBoundaries,
)
from edelweissfe.utils.isoparametricshapes import (
    NODE_PARAMETRIC_COORDINATES,
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
        xi = rng.uniform(-0.5, 0.5, nodes.shape[1]) * (0.5 if shape == "tetra4" else 1.0) + (
            0.25 if shape == "tetra4" else 0.0
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


@pytest.mark.parametrize("shape", ["quad4", "quad8", "hexa8", "hexa20", "tetra4"])
def test_inverse_mapping(shape):
    coords = _distorted(shape)
    rng = np.random.default_rng(1)
    dim = coords.shape[1]
    for _ in range(10):
        xi = rng.uniform(0.0, 0.3, dim) if shape == "tetra4" else rng.uniform(-0.9, 0.9, dim)
        x = shapeFunctionsAndDerivatives(shape, xi)[0] @ coords
        np.testing.assert_allclose(inverseMapping(shape, coords, x), xi, atol=1e-10)


def _node(label, x):
    return SimpleNamespace(label=label, coordinates=np.asarray(x, dtype=float))


def _element(number, shape, nodes):
    return SimpleNamespace(elNumber=number, ensightType=shape, nodes=nodes)


def _quadGrid(nx, ny, lx, ly):
    """A structured grid of quad4 elements on [0, lx] x [0, ly]."""
    nodes = {(i, j): _node(i * (ny + 1) + j, (lx * i / nx, ly * j / ny)) for i in range(nx + 1) for j in range(ny + 1)}
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

    parts, unembedded = splitBarElementAtHostBoundaries(bar, locator)

    assert unembedded == 0.0
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
    parts, unembedded = splitBarElementAtHostBoundaries(bar, locator)
    # the bar is outside on x in [-1, 0]: parametric length 0.8 of 2
    assert abs(unembedded - 0.8) < 1e-9
    assert [p.hostElement.elNumber for p in parts] == [0, 2]
    assert abs(parts[0].etaStart - (-0.2)) < 1e-9


@pytest.mark.parametrize("start, end", [((1.0, 1.0), (3.0, 1.0)), ((1.0, 0.5), (3.0, 1.7)), ((1.0, 1.0), (1.0, 3.0))])
def test_split_bar_starting_on_host_boundary_has_no_degenerate_parts(start, end):
    """a bar starting on a host edge or node must not get a zero-length part in a host it only touches"""
    hosts = _quadGrid(4, 4, 4.0, 4.0)
    locator = HostElementLocator(hosts)
    bar = _element(1000, "bar2", [_node(100, start), _node(101, end)])
    parts, unembedded = splitBarElementAtHostBoundaries(bar, locator)
    assert unembedded == 0.0
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
    parts, unembedded = splitBarElementAtHostBoundaries(_element(1000, barShape, barNodes), locator)
    assert unembedded == 0.0
    assert all(p.etaEnd - p.etaStart > 1e-3 for p in parts)
    assert [p.hostElement.elNumber for p in parts] == [0, 7, 14, 21, 28, 35]  # the diagonal cells
    assert parts[0].etaStart == -1.0 and parts[-1].etaEnd == 1.0
    for p, q in zip(parts[:-1], parts[1:]):
        assert p.etaEnd == q.etaStart
