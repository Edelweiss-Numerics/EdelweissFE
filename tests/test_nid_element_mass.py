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
"""The consistent mass of every element type used with the implicit dynamic solver (``NID``).

``NID`` does not check the assembled mass for symmetry at runtime -- that costs a transposed copy
of the whole matrix, gigabytes on a large model -- so the properties it relies on are pinned here,
once per element type, on a single element: the mass is symmetric, it carries the total mass
``density * volume`` on each displacement component, and it is positive semi-definite.
"""

import numpy as np
import pytest

from edelweissfe.config.phenomena import getFieldSize
from edelweissfe.points.node import Node

elementModule = pytest.importorskip("edelweissfe.elements.marmotelement.element")

#: Density and edge lengths of the test brick; not a unit cube, so that a mass off by a
#: coordinate permutation or a volume factor is caught.
DENSITY = 2.5
EDGES = np.array([2.0, 1.0, 0.5])

#: The corner and midside nodes of a hexahedron on the unit cube, in the node ordering Marmot
#: and EdelweissFE share (corners bottom then top, bottom edges, top edges, vertical edges).
_HEXA8 = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
_HEXA20 = _HEXA8 + [
    (0.5, 0, 0), (1, 0.5, 0), (0.5, 1, 0), (0, 0.5, 0),
    (0.5, 0, 1), (1, 0.5, 1), (0.5, 1, 1), (0, 0.5, 1),
    (0, 0, 0.5), (1, 0, 0.5), (1, 1, 0.5), (0, 1, 0.5),
]  # fmt: skip

#: (element type, reference nodes, material, material properties with the density). This list is
#: the whole guard: an element type not in it has its mass checked nowhere, so add a type here
#: before using it with ``NID``.
ELEMENTS = [
    ("C3D8", _HEXA8, "LINEARELASTIC", [100.0, 0.2, DENSITY]),
    ("C3D20", _HEXA20, "LINEARELASTIC", [100.0, 0.2, DENSITY]),
    ("C3D20R", _HEXA20, "LINEARELASTIC", [100.0, 0.2, DENSITY]),
    ("GC3D8", _HEXA8, "AT2PHASEFIELD", [100.0, 0.2, 1.0, 0.1, DENSITY, 1e-3]),
    ("GC3D20R", _HEXA20, "AT2PHASEFIELD", [100.0, 0.2, 1.0, 0.1, DENSITY, 1e-3]),
]


def _consistentMass(elType, referenceNodes, material, properties):
    """The consistent mass of one element on the test brick, and the field of each of its dofs."""

    element = elementModule.MarmotElementWrapper(elType, 1)
    element.setNodes([Node(i, np.array(xi, dtype=float) * EDGES) for i, xi in enumerate(referenceNodes)])
    element.setMaterial(material, np.array(properties, dtype=float))
    element.initializeElement()

    M = np.zeros(element.nDof * element.nDof)
    element.computeConsistentInertia(M)
    M = M.reshape(element.nDof, element.nDof)

    # Node-major, then the element's own permutation -- the order the DofManager scatters with.
    dofField = np.array(
        [
            (fieldName, component)
            for nodeFields in element.fields
            for fieldName in nodeFields
            for component in range(getFieldSize(fieldName, 3))
        ],
        dtype=object,
    )
    if element.dofIndicesPermutation is not None:
        dofField = dofField[np.asarray(element.dofIndicesPermutation)]

    return M, dofField


@pytest.mark.parametrize("elType, referenceNodes, material, properties", ELEMENTS, ids=[e[0] for e in ELEMENTS])
def test_consistent_mass_is_symmetric_and_carries_the_total_mass(elType, referenceNodes, material, properties):
    M, dofField = _consistentMass(elType, referenceNodes, material, properties)

    scale = np.max(np.abs(M))
    assert scale > 0.0
    np.testing.assert_allclose(M, M.T, rtol=0.0, atol=1e-12 * scale)
    assert np.min(np.linalg.eigvalsh(0.5 * (M + M.T))) > -1e-10 * scale

    totalMass = DENSITY * np.prod(EDGES)
    for component in range(3):
        isComponent = np.array([f == "displacement" and c == component for f, c in dofField])
        assert isComponent.sum() == len(referenceNodes)
        np.testing.assert_allclose(M[np.ix_(isComponent, isComponent)].sum(), totalMass, rtol=1e-12)
        # a displacement component couples to no other component and to no other field
        np.testing.assert_array_equal(M[np.ix_(isComponent, ~isComponent)], 0.0)


def test_hexa8_consistent_mass_is_the_textbook_matrix():
    """The one reference matrix: the trilinear brick's consistent mass is ``rho V / 216`` times
    ``2^(number of shared coordinates)``, the tensor product of the 1D ``[[2, 1], [1, 2]] / 6``."""

    M, dofField = _consistentMass(*ELEMENTS[0])
    isX = np.array([f == "displacement" and c == 0 for f, c in dofField])
    Mx = M[np.ix_(isX, isX)]

    corners = np.array(_HEXA8)
    sharedCoordinates = (corners[:, None, :] == corners[None, :, :]).sum(axis=2)
    reference = DENSITY * np.prod(EDGES) / 216.0 * 2.0**sharedCoordinates
    np.testing.assert_allclose(Mx, reference, rtol=1e-12)
