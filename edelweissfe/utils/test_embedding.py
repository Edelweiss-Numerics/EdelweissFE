"""Unit tests of the isoparametric shape functions and of the host element locator."""

from types import SimpleNamespace

import numpy as np
import pytest

from edelweissfe.utils.embedding import HostElementLocator
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
