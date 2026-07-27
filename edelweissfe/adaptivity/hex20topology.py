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

"""Reference topology and shape functions for the 20-node serendipity hexahedron (Marmot C3D20).

Local node ordering (matches Marmot/Abaqus C3D20):
  0-3   corners of the xi=-1 face          (loop)
  4-7   corners of the xi=+1 face           (loop, node i+4 above node i)
  8-11  midsides of the xi=-1 face edges    (0-1, 1-2, 2-3, 3-0)
  12-15 midsides of the xi=+1 face edges    (4-5, 5-6, 6-7, 7-4)
  16-19 midsides of the connecting edges    (0-4, 1-5, 2-6, 3-7)

All parametric coordinates live on the reference cube [-1, 1]^3.
"""

import math

import numpy as np

# Local node parametric coordinates, C3D20 order.
LOCAL_COORDS = np.array(
    [
        (-1, 1, 1),
        (-1, -1, 1),
        (-1, -1, -1),
        (-1, 1, -1),  # 0-3   corners xi=-1
        (1, 1, 1),
        (1, -1, 1),
        (1, -1, -1),
        (1, 1, -1),  # 4-7   corners xi=+1
        (-1, 0, 1),
        (-1, -1, 0),
        (-1, 0, -1),
        (-1, 1, 0),  # 8-11  mids xi=-1 face
        (1, 0, 1),
        (1, -1, 0),
        (1, 0, -1),
        (1, 1, 0),  # 12-15 mids xi=+1 face
        (0, 1, 1),
        (0, -1, 1),
        (0, -1, -1),
        (0, 1, -1),  # 16-19 connecting mids
    ],
    dtype=float,
)

N_NODES = 20


def hex20_box_coords(x0, x1, y0, y1, z0, z1):
    """Return the 20 node coordinates of an axis-aligned HEX20 box, in C3D20 order."""
    xm, ym, zm = (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2
    return np.array(
        [
            (x0, y1, z1),
            (x0, y0, z1),
            (x0, y0, z0),
            (x0, y1, z0),
            (x1, y1, z1),
            (x1, y0, z1),
            (x1, y0, z0),
            (x1, y1, z0),
            (x0, ym, z1),
            (x0, y0, zm),
            (x0, ym, z0),
            (x0, y1, zm),
            (x1, ym, z1),
            (x1, y0, zm),
            (x1, ym, z0),
            (x1, y1, zm),
            (xm, y1, z1),
            (xm, y0, z1),
            (xm, y0, z0),
            (xm, y1, z0),
        ],
        dtype=float,
    )


def hex20_from_corners(corners8):
    """Build the 20 node coordinates of a straight-edged HEX20 from its 8 corners (C3D20 corner
    order), placing each midside node at the midpoint of its edge."""
    corners8 = np.asarray(corners8, dtype=float)
    out = np.zeros((N_NODES, 3))
    out[:8] = corners8
    for ca, mid, cb in _build_edges():
        out[mid] = 0.5 * (corners8[ca] + corners8[cb])
    return out


# QUAD8 serendipity reference nodes: 4 corners then 4 edge midsides (matches a FACES entry).
_QUAD8_NODES = np.array([(-1, -1), (1, -1), (1, 1), (-1, 1), (0, -1), (1, 0), (0, 1), (-1, 0)], dtype=float)


def quad8_shape(xi, eta):
    """Evaluate the 8 QUAD8 serendipity shape functions at (xi, eta), in FACES node order."""
    N = np.zeros(8)
    for a, (xi_a, eta_a) in enumerate(_QUAD8_NODES):
        if xi_a != 0 and eta_a != 0:  # corner
            N[a] = 0.25 * (1 + xi * xi_a) * (1 + eta * eta_a) * (xi * xi_a + eta * eta_a - 1)
        elif xi_a == 0:  # midside on eta = +-1
            N[a] = 0.5 * (1 - xi**2) * (1 + eta * eta_a)
        else:  # midside on xi = +-1
            N[a] = 0.5 * (1 + xi * xi_a) * (1 - eta**2)
    return N


def _partition_local_coords():
    """Group node indices by shape-function family (corner / midside-x / midside-y / midside-z),
    each entry pre-flattened to plain-Python scalars. Evaluated once at import time so
    :func:`hex20_shape` -- called ~10^5-10^6 times per refinement event via the Newton-Raphson
    inverse map (:func:`~edelweissfe.adaptivity.statetransfer.base.hex20InverseMap`) -- never pays
    per-call numpy-array-indexing/branching overhead for a fixed, tiny (20-entry) topology."""
    coords = [tuple(row) for row in LOCAL_COORDS.tolist()]
    corner = [(i, xa, ya, za) for i, (xa, ya, za) in enumerate(coords) if xa != 0 and ya != 0 and za != 0]
    midX = [(i, ya, za) for i, (xa, ya, za) in enumerate(coords) if xa == 0]
    midY = [(i, xa, za) for i, (xa, ya, za) in enumerate(coords) if xa != 0 and ya == 0]
    midZ = [(i, xa, ya) for i, (xa, ya, za) in enumerate(coords) if xa != 0 and ya != 0 and za == 0]
    return corner, midX, midY, midZ


_CORNER_NODES, _MIDX_NODES, _MIDY_NODES, _MIDZ_NODES = _partition_local_coords()


def hex20_shape(xi, eta, zeta):
    """Evaluate the 20 serendipity shape functions at (xi, eta, zeta)."""
    N = [0.0] * N_NODES
    for i, xa, ya, za in _CORNER_NODES:
        N[i] = 0.125 * (1 + xi * xa) * (1 + eta * ya) * (1 + zeta * za) * (xi * xa + eta * ya + zeta * za - 2)
    xi2 = 1 - xi**2
    for i, ya, za in _MIDX_NODES:
        N[i] = 0.25 * xi2 * (1 + eta * ya) * (1 + zeta * za)
    eta2 = 1 - eta**2
    for i, xa, za in _MIDY_NODES:
        N[i] = 0.25 * eta2 * (1 + xi * xa) * (1 + zeta * za)
    zeta2 = 1 - zeta**2
    for i, xa, ya in _MIDZ_NODES:
        N[i] = 0.25 * zeta2 * (1 + xi * xa) * (1 + eta * ya)
    return np.array(N)


def _corner_indices():
    return [i for i in range(N_NODES) if np.count_nonzero(LOCAL_COORDS[i] == 0) == 0]


def _mid_indices():
    return [i for i in range(N_NODES) if np.count_nonzero(LOCAL_COORDS[i] == 0) == 1]


def _build_faces():
    """Return the 6 faces, each as 8 local indices in QUAD8 order [c0,c1,c2,c3, m01,m12,m23,m30]."""
    faces = []
    for axis in range(3):
        for sign in (-1, 1):
            onface = [i for i in range(N_NODES) if LOCAL_COORDS[i, axis] == sign]
            corners = [i for i in onface if np.count_nonzero(LOCAL_COORDS[i] == 0) == 0]
            mids = [i for i in onface if np.count_nonzero(LOCAL_COORDS[i] == 0) == 1]
            inplane = [a for a in range(3) if a != axis]
            center = LOCAL_COORDS[corners][:, inplane].mean(0)

            def ang(i):
                v = LOCAL_COORDS[i][inplane] - center
                return math.atan2(v[1], v[0])

            corners = sorted(corners, key=ang)
            ordered_mids = []
            for k in range(4):
                mp = (LOCAL_COORDS[corners[k]] + LOCAL_COORDS[corners[(k + 1) % 4]]) / 2
                ordered_mids.append(next(m for m in mids if np.allclose(LOCAL_COORDS[m], mp)))
            faces.append(corners + ordered_mids)
    return faces


def _build_edges():
    """Return the 12 edges, each as [corner_a, midside, corner_b]."""
    edges = []
    corners = _corner_indices()
    mids = _mid_indices()
    for ii in range(len(corners)):
        for jj in range(ii + 1, len(corners)):
            ca, cb = corners[ii], corners[jj]
            if np.count_nonzero(LOCAL_COORDS[ca] - LOCAL_COORDS[cb]) == 1:  # differ in one axis
                mp = (LOCAL_COORDS[ca] + LOCAL_COORDS[cb]) / 2
                m = next((m for m in mids if np.allclose(LOCAL_COORDS[m], mp)), None)
                if m is not None:
                    edges.append([ca, m, cb])
    return edges


def subdivision_children_param(n: int = 2):
    """Parametric coordinates (in the parent cube [-1, 1]^3) of the 20 nodes of each of the ``n**3``
    children produced by subdividing a HEX20 into ``n`` equal parts per axis.

    ``n = 2`` is octree bisection (8 children); ``n = 3`` gives a 3x3x3 split (27 children), etc.
    Children are ordered ``index = ix*n*n + iy*n + iz`` with ``ix, iy, iz`` in ``0..n-1`` (x outer,
    z inner) -- the ordering assumed by :func:`face_child_octants` and the warm-start interpolation.
    """
    slabs = [(-1.0 + 2.0 * i / n, -1.0 + 2.0 * (i + 1) / n) for i in range(n)]
    children = []
    for sx in slabs:
        for sy in slabs:
            for sz in slabs:
                children.append(hex20_box_coords(sx[0], sx[1], sy[0], sy[1], sz[0], sz[1]))
    return children


def octant_children_param():
    """Octree (2x2x2) children -- backward-compatible alias for ``subdivision_children_param(2)``."""
    return subdivision_children_param(2)


FACES = _build_faces()  # 6 x 8
EDGES = _build_edges()  # 12 x 3

# Marmot Hexa20 faceID (1-based) -> local slot indices of that face, from
# MarmotFiniteElement3D.cpp Hexa20::getBoundaryElementIndices. Used to translate an FE surface
# faceID to our internal face index and back.
BOUNDARY_FACEID_INDICES = {
    1: [3, 2, 1, 0, 10, 9, 8, 11],
    2: [4, 5, 6, 7, 12, 13, 14, 15],
    3: [0, 1, 5, 4, 8, 17, 12, 16],
    4: [6, 5, 1, 2, 13, 17, 9, 18],
    5: [7, 6, 2, 3, 14, 18, 10, 19],
    6: [4, 7, 3, 0, 15, 19, 11, 16],
}


def _faceid_to_face():
    """Map each Marmot faceID to our FACES index by matching the 4 corner slots."""
    out = {}
    for fid, idx in BOUNDARY_FACEID_INDICES.items():
        cornerset = set(idx[:4])
        out[fid] = next(fi for fi, f in enumerate(FACES) if set(f[:4]) == cornerset)
    return out


FACEID_TO_FACE = _faceid_to_face()  # faceID (1-6) -> FACES index (0-5)
FACE_TO_FACEID = {v: k for k, v in FACEID_TO_FACE.items()}


def face_child_octants(face_index, n: int = 2):
    """The ``n**2`` child indices (in :func:`subdivision_children_param` order) that tile a given face
    of an ``n``-per-axis subdivision.

    A child tiles a parent face iff it is on that face's side of the split; the child covers it with
    its SAME local face (and hence the SAME faceID)."""
    axis = face_index // 2
    side = n - 1 if (face_index % 2 == 1) else 0  # even index = minus side, odd = plus side
    out = []
    for ix in range(n):
        for iy in range(n):
            for iz in range(n):
                if (ix, iy, iz)[axis] == side:
                    out.append(ix * n * n + iy * n + iz)
    return out


# Which FACES entry is the +xi face (all corners at xi=+1): used a lot.
FACE_XPLUS = next(f for f in FACES if all(LOCAL_COORDS[i, 0] == 1 for i in f[:4]))
FACE_XMINUS = next(f for f in FACES if all(LOCAL_COORDS[i, 0] == -1 for i in f[:4]))


if __name__ == "__main__":
    # partition of unity + nodal delta property for hex20_shape
    rng_pts = LOCAL_COORDS.tolist() + [[0.3, -0.7, 0.1], [0, 0, 0], [0.5, 0.5, -0.5]]
    for p in rng_pts:
        assert abs(hex20_shape(*p).sum() - 1.0) < 1e-13, ("PoU fail", p)
    M = np.array([hex20_shape(*p) for p in LOCAL_COORDS])
    assert np.allclose(M, np.eye(N_NODES), atol=1e-13), "nodal delta fail"

    # linear + quadratic completeness: reproduce f exactly from nodal samples
    for f in [
        lambda x, y, z: 1.0,
        lambda x, y, z: 2 - x + 3 * y - 0.5 * z,
        lambda x, y, z: x * y - 2 * z**2 + 0.7 * x * z + y**2,
        lambda x, y, z: x**2 * y - 0.3 * y * z**2 + x * y * z,  # serendipity term
    ]:
        vals = np.array([f(*p) for p in LOCAL_COORDS])
        err = max(abs(hex20_shape(*p) @ vals - f(*p)) for p in rng_pts)
        assert err < 1e-12, ("completeness fail", err)

    assert len(FACES) == 6 and all(len(f) == 8 for f in FACES)
    assert len(EDGES) == 12 and all(len(e) == 3 for e in EDGES)
    # every face mid is the midpoint of its bracketing corners
    for f in FACES:
        for k in range(4):
            mp = (LOCAL_COORDS[f[k]] + LOCAL_COORDS[f[(k + 1) % 4]]) / 2
            assert np.allclose(LOCAL_COORDS[f[4 + k]], mp)
    print("hex20topology self-tests PASSED")
    print("FACE_XPLUS  =", FACE_XPLUS)
    print("FACE_XMINUS =", FACE_XMINUS)
