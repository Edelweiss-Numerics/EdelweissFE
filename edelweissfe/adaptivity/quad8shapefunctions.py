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
"""Reference shape functions and topology data for the 8-node serendipity quadrilateral (Marmot
Quad8, e.g. CPS8R, GCPS8R, GMSCPE8R). Pure math, the 2D counterpart of
:mod:`~edelweissfe.adaptivity.hex20shapefunctions`.

Local node ordering (matches Marmot/Abaqus CPS8):
  0-3   corners, counter-clockwise from (-1, -1)
  4-7   midsides of the edges 0-1, 1-2, 2-3, 3-0

All parametric coordinates live on the reference square [-1, 1]^2.
"""

from fractions import Fraction

import numpy as np

# Local node parametric coordinates, CPS8 order.
LOCAL_COORDS = np.array([(-1, -1), (1, -1), (1, 1), (-1, 1), (0, -1), (1, 0), (0, 1), (-1, 0)], dtype=float)

N_NODES = 8

# Marmot Quad8 faceID (1-based) -> local slots [corner a, corner b, midside], from
# MarmotFiniteElement2D.cpp Quad8::getBoundaryElementIndices; FACES is in faceID order.
BOUNDARY_FACEID_INDICES = {1: [0, 1, 4], 2: [1, 2, 5], 3: [2, 3, 6], 4: [3, 0, 7]}
FACES = [BOUNDARY_FACEID_INDICES[faceID] for faceID in sorted(BOUNDARY_FACEID_INDICES)]
# the same four edges as [corner a, midside, corner b]
EDGES = [[a, m, b] for a, b, m in FACES]
FACEID_TO_FACE = {faceID: faceID - 1 for faceID in BOUNDARY_FACEID_INDICES}

_COORDS = [tuple(int(v) for v in row) for row in LOCAL_COORDS.tolist()]


def quad8_box_coords(x0, x1, y0, y1):
    """Return the 8 node coordinates of an axis-aligned QUAD8 box, in CPS8 order."""
    xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
    return np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1), (xm, y0), (x1, ym), (xm, y1), (x0, ym)], dtype=float)


def _quad8_shape(xi, eta, quarter, half) -> list:
    """The 8 serendipity shape functions, with the constants 1/4 and 1/2 passed in so that one
    formula serves floating-point and exact rational evaluation alike."""
    N = [0] * N_NODES
    for i, (xa, ya) in enumerate(_COORDS):
        if xa != 0 and ya != 0:
            N[i] = quarter * (1 + xi * xa) * (1 + eta * ya) * (xi * xa + eta * ya - 1)
        elif xa == 0:
            N[i] = half * (1 - xi**2) * (1 + eta * ya)
        else:
            N[i] = half * (1 + xi * xa) * (1 - eta**2)
    return N


def quad8_shape(xi, eta):
    """Evaluate the 8 serendipity shape functions at (xi, eta)."""
    return np.array(_quad8_shape(xi, eta, 0.25, 0.5), dtype=float)


def quad8_shape_exact(xi, eta) -> list:
    """The 8 serendipity shape functions at an exact rational point, as :class:`~fractions.Fraction`."""
    return _quad8_shape(Fraction(xi), Fraction(eta), Fraction(1, 4), Fraction(1, 2))


def quad8_shape_grad(xi, eta):
    """The 8 shape functions and their gradient w.r.t. (xi, eta): ``(N, dN)``, ``dN`` of shape ``(8, 2)``."""
    N = np.zeros(N_NODES)
    dN = np.zeros((N_NODES, 2))
    for i, (xa, ya) in enumerate(_COORDS):
        if xa != 0 and ya != 0:
            A, B, S = 1 + xi * xa, 1 + eta * ya, xi * xa + eta * ya - 1
            N[i] = 0.25 * A * B * S
            dN[i] = 0.25 * xa * B * (S + A), 0.25 * ya * A * (S + B)
        elif xa == 0:
            B = 1 + eta * ya
            N[i] = 0.5 * (1 - xi**2) * B
            dN[i] = -xi * B, 0.5 * (1 - xi**2) * ya
        else:
            A = 1 + xi * xa
            N[i] = 0.5 * A * (1 - eta**2)
            dN[i] = 0.5 * xa * (1 - eta**2), -eta * A
    return N, dN


def subdivision_children_param(n: int = 2):
    """Parametric coordinates (in the parent square [-1, 1]^2) of the 8 nodes of each of the ``n**2``
    children of an ``n``-per-axis split, ordered ``index = ix * n + iy`` (x outer)."""
    slabs = [(-1.0 + 2.0 * i / n, -1.0 + 2.0 * (i + 1) / n) for i in range(n)]
    return [quad8_box_coords(sx[0], sx[1], sy[0], sy[1]) for sx in slabs for sy in slabs]


def face_child_indices(face_index, n: int = 2):
    """The ``n`` child indices (in :func:`subdivision_children_param` order) that tile a given face
    (edge); each child covers it with its same local face."""
    rows = LOCAL_COORDS[FACES[face_index]]
    axis = next(k for k in range(2) if np.all(rows[:, k] == rows[0, k]))
    side = n - 1 if rows[0, axis] > 0 else 0
    return [ix * n + iy for ix in range(n) for iy in range(n) if (ix, iy)[axis] == side]
