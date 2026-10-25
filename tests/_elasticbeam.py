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
"""Small linear-elastic beams, shared by the tests of the rigid-body coarse level: a structured mesh of trilinear
hexahedra (3D) or of bilinear quadrilaterals in plane stress (2D), clamped at x = 0 and propped (one component
constrained) at the far end -- bending-dominated structures, whose smooth error rotates the cross-sections, with fully
and partially constrained nodes."""

import functools
import itertools

import numpy as np
import scipy.sparse as sp


def _elasticityMatrix(dimension: int, youngsModulus: float, poissonsRatio: float) -> np.ndarray:
    """The isotropic elasticity matrix C in Voigt notation with engineering shear strains: 3D, or plane stress in
    2D."""
    E, nu = youngsModulus, poissonsRatio
    if dimension == 2:
        return E / (1.0 - nu**2) * np.array([[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, 0.5 * (1.0 - nu)]])
    lame = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    shear = E / (2.0 * (1.0 + nu))
    C = np.zeros((6, 6))
    C[:3, :3] = lame
    C[np.arange(3), np.arange(3)] += 2.0 * shear
    C[np.arange(3, 6), np.arange(3, 6)] = shear
    return C


def _elementStiffness(elementSize: np.ndarray, C: np.ndarray) -> np.ndarray:
    """The stiffness K = sum over the Gauss points of B^T C B det J of a rectangular (2D) or brick (3D) element of the
    given edge lengths, with 2 x 2 (x 2) Gauss points; its nodes are the corners in lexicographic order of their
    reference coordinates (x fastest)."""
    dimension = len(elementSize)
    corners = np.array(list(itertools.product((-1.0, 1.0), repeat=dimension)))[:, ::-1]
    # the element is axis-aligned: the Jacobian dx/dxi of its map from the reference element is diagonal
    dx_dxi = 0.5 * np.asarray(elementSize, dtype=float)
    detJ = np.prod(dx_dxi)
    K = np.zeros((dimension * len(corners), dimension * len(corners)))
    for point in itertools.product((-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)), repeat=dimension):
        point = np.asarray(point)
        # the shape functions N_i = prod_d (1 + xi_d xi_id) / 2 and their derivatives, by the product rule
        factors = 0.5 * (1.0 + corners * point)
        dN_dxi = np.empty((len(corners), dimension))
        for d in range(dimension):
            dN_dxi[:, d] = 0.5 * corners[:, d] * np.prod(np.delete(factors, d, axis=1), axis=1)
        dN_dx = dN_dxi / dx_dxi
        B = np.zeros((C.shape[0], dimension * len(corners)))
        for i, g in enumerate(dN_dx):
            columns = slice(dimension * i, dimension * (i + 1))
            if dimension == 2:
                B[:, columns] = [[g[0], 0.0], [0.0, g[1]], [g[1], g[0]]]
            else:
                B[:, columns] = [
                    [g[0], 0.0, 0.0],
                    [0.0, g[1], 0.0],
                    [0.0, 0.0, g[2]],
                    [0.0, g[2], g[1]],
                    [g[2], 0.0, g[0]],
                    [g[1], g[0], 0.0],
                ]
        K += B.T @ C @ B * detJ
    return K


def cantilever(elements: tuple, lengths: tuple, youngsModulus: float = 30000.0, poissonsRatio: float = 0.2):
    """A beam of ``elements`` (per direction, x the axis) over ``lengths``, clamped at x = 0 and propped at x = L: the
    component 1 (the y displacement) of the nodes on the edge x = L, y = 0 is constrained. The constraints are
    eliminated symmetrically: zero rows and columns, unit diagonal.

    Returns
    -------
    tuple
        ``(K, coordinates, constrainedDofs)``: the stiffness (CSR, node-major, sorted columns), the node coordinates
        and the constrained dofs.
    """
    dimension = len(elements)
    nodesPerDirection = np.asarray(elements) + 1
    elementSize = np.asarray(lengths, dtype=float) / np.asarray(elements)
    grids = np.meshgrid(*[np.arange(n) for n in nodesPerDirection], indexing="ij")
    # nodes numbered with x fastest
    indices = np.stack([g.ravel(order="F") for g in grids], axis=1)
    coordinates = indices * elementSize

    def nodeNumber(index):
        number = np.zeros(len(index), dtype=np.int64)
        for d in reversed(range(dimension)):
            number = number * nodesPerDirection[d] + index[:, d]
        return number

    origins = np.stack(
        [g.ravel(order="F") for g in np.meshgrid(*[np.arange(n) for n in elements], indexing="ij")], axis=1
    )
    corners = np.array(list(itertools.product((0, 1), repeat=dimension)))[:, ::-1]
    connectivity = np.stack([nodeNumber(origins + corner) for corner in corners], axis=1)
    elementDofs = (dimension * connectivity[:, :, None] + np.arange(dimension)).reshape(len(connectivity), -1)

    Ke = _elementStiffness(elementSize, _elasticityMatrix(dimension, youngsModulus, poissonsRatio))
    rows = np.repeat(elementDofs, elementDofs.shape[1], axis=1).ravel()
    columns = np.tile(elementDofs, elementDofs.shape[1]).ravel()
    n = dimension * len(coordinates)
    K = sp.coo_matrix((np.tile(Ke.ravel(), len(elementDofs)), (rows, columns)), shape=(n, n)).tocsr()

    clamped = np.nonzero(coordinates[:, 0] == 0.0)[0]
    propped = np.nonzero((coordinates[:, 0] == lengths[0]) & (coordinates[:, 1] == 0.0))[0]
    constrainedDofs = np.unique(
        np.r_[(dimension * clamped[:, None] + np.arange(dimension)).ravel(), dimension * propped + 1]
    )
    free = np.ones(n)
    free[constrainedDofs] = 0.0
    K = (sp.diags(free) @ K @ sp.diags(free) + sp.diags(1.0 - free)).tocsr()
    K.eliminate_zeros()
    K.sort_indices()
    return K, coordinates, constrainedDofs


@functools.lru_cache
def slenderBeam(dimension: int):
    """The propped cantilever of the tests, 16:1 slender: 32 x 2 x 2 hexahedra in 3D, 64 x 4 quadrilaterals in 2D
    (see :func:`cantilever`, whose result it returns; cached, so not to be changed in place)."""
    if dimension == 3:
        return cantilever((32, 2, 2), (16.0, 1.0, 1.0))
    return cantilever((64, 4), (16.0, 1.0))
