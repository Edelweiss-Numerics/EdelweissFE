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

"""Coordinate-system-agnostic geometry helpers for planar quadrilateral faces of straight-edged
hexahedra. Used for topological (shared-face) neighbour identification and for computing exact
hanging-node weights on arbitrarily oriented / non-parallelogram (trapezoidal) faces.

Assumes planar quadrilateral faces of straight or curved hexahedra.
Used for topological (shared-face) neighbour identification and for computing exact
hanging-node weights on arbitrarily oriented / non-parallelogram (trapezoidal) faces.
"""

import math

import numpy as np

# --- plain-float 3-vector primitives -------------------------------------------------------------
# These hot helpers operate on plain (x, y, z) float tuples. On the tiny 3-vectors of hanging-node
# classification, numpy's np.cross / np.linalg.norm are dominated by dispatch overhead (moveaxis,
# normalize_axis_tuple, ...); explicit scalar math is ~10-30x faster and is what the O(interface)
# classification loop runs millions of times.


def _sub3(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot3(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross3(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _norm3(a):
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def quadratic_edge_parameter(p, ca, cm, cb, tol=1e-8, itmax=25):
    """Find parameter t in [-1, 1] for point p on 3-node quadratic edge [ca, cm, cb].
    Returns (t, distance_to_edge). Robust to curved 3-node edges.

    A straight edge (midside node at the corner midpoint, the octree-refinement case) is handled by
    a closed-form linear projection; only genuinely curved edges fall through to Newton iteration.
    """
    ca = (float(ca[0]), float(ca[1]), float(ca[2]))
    cm = (float(cm[0]), float(cm[1]), float(cm[2]))
    cb = (float(cb[0]), float(cb[1]), float(cb[2]))
    p = (float(p[0]), float(p[1]), float(p[2]))

    e = _sub3(cb, ca)
    e2 = _dot3(e, e)
    if e2 == 0.0:
        return 0.0, _norm3(_sub3(p, cm))

    # straight edge? midside at the chord midpoint within tol -> exact closed-form linear projection
    mid = ((ca[0] + cb[0]) * 0.5, (ca[1] + cb[1]) * 0.5, (ca[2] + cb[2]) * 0.5)
    if _norm3(_sub3(cm, mid)) < tol * (1.0 + math.sqrt(e2)):
        s = _dot3(_sub3(p, ca), e) / e2  # in [0, 1] along the chord
        sc = 0.0 if s < 0.0 else (1.0 if s > 1.0 else s)
        proj = (ca[0] + sc * e[0], ca[1] + sc * e[1], ca[2] + sc * e[2])
        return 2.0 * sc - 1.0, _norm3(_sub3(p, proj))

    # curved edge: Newton on the quadratic map
    t = 2.0 * _dot3(_sub3(p, ca), e) / e2 - 1.0
    t = -1.0 if t < -1.0 else (1.0 if t > 1.0 else t)
    d2xdt2 = (ca[0] - 2.0 * cm[0] + cb[0], ca[1] - 2.0 * cm[1] + cb[1], ca[2] - 2.0 * cm[2] + cb[2])
    for _ in range(itmax):
        Na, Nm, Nb = 0.5 * t * (t - 1.0), 1.0 - t * t, 0.5 * t * (t + 1.0)
        x = (
            Na * ca[0] + Nm * cm[0] + Nb * cb[0],
            Na * ca[1] + Nm * cm[1] + Nb * cb[1],
            Na * ca[2] + Nm * cm[2] + Nb * cb[2],
        )
        r = _sub3(x, p)
        dNa, dNm, dNb = t - 0.5, -2.0 * t, t + 0.5
        dxdt = (
            dNa * ca[0] + dNm * cm[0] + dNb * cb[0],
            dNa * ca[1] + dNm * cm[1] + dNb * cb[1],
            dNa * ca[2] + dNm * cm[2] + dNb * cb[2],
        )
        f = _dot3(r, dxdt)
        dxdt2 = _dot3(dxdt, dxdt)
        if dxdt2 < 1e-14 or abs(f) < 1e-12:
            break
        df = dxdt2 + _dot3(r, d2xdt2)
        if abs(df) < 1e-14:
            break
        dt = -f / df
        t += dt
        if abs(dt) < 1e-10:
            break

    t = -1.0 if t < -1.0 else (1.0 if t > 1.0 else t)
    Na, Nm, Nb = 0.5 * t * (t - 1.0), 1.0 - t * t, 0.5 * t * (t + 1.0)
    x = (
        Na * ca[0] + Nm * cm[0] + Nb * cb[0],
        Na * ca[1] + Nm * cm[1] + Nb * cb[1],
        Na * ca[2] + Nm * cm[2] + Nb * cb[2],
    )
    return t, _norm3(_sub3(x, p))


def face_frame(corners4):
    """In-plane orthonormal frame (origin, unit normal, e1, e2) of a planar quad from its 4 corners."""
    c = np.asarray(corners4, dtype=float)
    o = c[0]
    n = np.cross(c[1] - c[0], c[3] - c[0])
    n = n / np.linalg.norm(n)
    e1 = c[1] - c[0]
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    return o, n, e1, e2


def _to2d(p, o, e1, e2):
    d = np.asarray(p, dtype=float) - o
    return np.array([d @ e1, d @ e2])


def point_in_convex_quad(p, corners4, tol=1e-8):
    """True if p is (near-)coplanar with and inside the convex quad given by 4 corners (loop order).

    Hot path of hanging-node classification: implemented in plain-float scalar math (no numpy) to
    avoid per-call dispatch overhead on 3-vectors.
    """
    c0 = (float(corners4[0][0]), float(corners4[0][1]), float(corners4[0][2]))
    c1 = (float(corners4[1][0]), float(corners4[1][1]), float(corners4[1][2]))
    c2 = (float(corners4[2][0]), float(corners4[2][1]), float(corners4[2][2]))
    c3 = (float(corners4[3][0]), float(corners4[3][1]), float(corners4[3][2]))
    p = (float(p[0]), float(p[1]), float(p[2]))

    u = _sub3(c1, c0)
    w = _sub3(c3, c0)
    n = _cross3(u, w)
    nn = _norm3(n)
    if nn == 0.0:
        return False
    n = (n[0] / nn, n[1] / nn, n[2] / nn)
    if abs(_dot3(_sub3(p, c0), n)) > tol:
        return False  # not coplanar

    e1n = _norm3(u)
    e1 = (u[0] / e1n, u[1] / e1n, u[2] / e1n)
    e2 = _cross3(n, e1)

    def to2d(x):
        d = _sub3(x, c0)
        return (_dot3(d, e1), _dot3(d, e2))

    poly = [to2d(c0), to2d(c1), to2d(c2), to2d(c3)]
    pt = to2d(p)
    sign = 0
    for k in range(4):
        a, b = poly[k], poly[(k + 1) % 4]
        cross = (b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0])
        if abs(cross) <= tol:
            continue  # on an edge
        s = 1 if cross > 0 else -1
        if sign == 0:
            sign = s
        elif s != sign:
            return False
    return True


def bilinear_inverse(p, corners4, tol=1e-13, itmax=50):
    """Return (xi, eta) in [-1,1]^2 with the Q1 bilinear map of corners4 sending it to p.

    Corner order matches the (xi,eta) reference nodes (-1,-1),(1,-1),(1,1),(-1,1). Exact for planar
    straight-edged quads (parallelograms and trapezoids alike). Newton iteration in the face plane.
    """
    o, n, e1, e2 = face_frame(corners4)
    P = [_to2d(c, o, e1, e2) for c in corners4]
    target = _to2d(p, o, e1, e2)
    xi = eta = 0.0
    converged = False
    for _ in range(itmax):
        N = np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta), (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)]) / 4
        x = sum(N[i] * P[i] for i in range(4))
        r = x - target
        if np.linalg.norm(r) < tol:
            converged = True
            break
        dNdxi = np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)]) / 4
        dNdeta = np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)]) / 4
        J = np.zeros((2, 2))
        for i in range(4):
            J[:, 0] += dNdxi[i] * P[i]
            J[:, 1] += dNdeta[i] * P[i]
        d = np.linalg.solve(J, -r)
        xi += d[0]
        eta += d[1]
    if not converged:
        raise RuntimeError(
            "bilinear_inverse: Newton iteration did not converge (residual {:.3e} after {:} iterations); "
            "the face may be degenerate or badly warped.".format(float(np.linalg.norm(r)), itmax)
        )
    return xi, eta


if __name__ == "__main__":
    # trapezoid: bilinear_inverse must invert the bilinear map exactly
    trap = np.array([[0, 0, 0], [2, 0, 0], [1.5, 1, 0], [0.5, 1, 0]], dtype=float)
    for xi_t, eta_t in [(-1, -1), (1, -1), (1, 1), (-1, 1), (0, 0), (0.3, -0.6), (-0.4, 0.7)]:
        Nq = (
            np.array(
                [(1 - xi_t) * (1 - eta_t), (1 + xi_t) * (1 - eta_t), (1 + xi_t) * (1 + eta_t), (1 - xi_t) * (1 + eta_t)]
            )
            / 4
        )
        p = Nq @ trap
        xi, eta = bilinear_inverse(p, trap)
        assert abs(xi - xi_t) < 1e-10 and abs(eta - eta_t) < 1e-10, (xi, eta, xi_t, eta_t)
    assert point_in_convex_quad([1.0, 0.5, 0.0], trap)
    assert not point_in_convex_quad([1.0, 0.5, 0.3], trap)  # off-plane
    assert not point_in_convex_quad([5.0, 0.5, 0.0], trap)  # outside
    # rotated plane still works
    R = np.linalg.qr(np.array([[1, 0.3, -0.2], [0.1, 1, 0.4], [-0.3, 0.2, 1.0]]))[0]
    trapR = trap @ R.T
    xi, eta = bilinear_inverse(trapR[2], trapR)
    assert abs(xi - 1) < 1e-10 and abs(eta - 1) < 1e-10
    print("geometry self-tests PASSED")
