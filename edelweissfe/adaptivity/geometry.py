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

import numpy as np


def quadratic_edge_parameter(p, ca, cm, cb, tol=1e-8, itmax=25):
    """Find parameter t in [-1, 1] for point p on 3-node quadratic edge [ca, cm, cb].
    Returns (t, distance_to_edge). Robust to curved 3-node edges.
    """
    ca, cm, cb = np.asarray(ca, dtype=float), np.asarray(cm, dtype=float), np.asarray(cb, dtype=float)
    p = np.asarray(p, dtype=float)

    # initial guess from linear projection onto chord [ca, cb]
    e = cb - ca
    e2 = float(e @ e)
    if e2 == 0.0:
        return 0.0, float(np.linalg.norm(p - cm))
    t = 2.0 * float((p - ca) @ e) / e2 - 1.0
    t = float(np.clip(t, -1.0, 1.0))

    for _ in range(itmax):
        Na = 0.5 * t * (t - 1.0)
        Nm = 1.0 - t**2
        Nb = 0.5 * t * (t + 1.0)
        x = Na * ca + Nm * cm + Nb * cb
        r = x - p

        dNa = t - 0.5
        dNm = -2.0 * t
        dNb = t + 0.5
        dxdt = dNa * ca + dNm * cm + dNb * cb

        f = float(r @ dxdt)
        dxdt2 = float(dxdt @ dxdt)
        if dxdt2 < 1e-14 or abs(f) < 1e-12:
            break
        d2xdt2 = ca - 2.0 * cm + cb
        df = dxdt2 + float(r @ d2xdt2)
        if abs(df) < 1e-14:
            break
        dt = -f / df
        t += dt
        if abs(dt) < 1e-10:
            break

    t = float(np.clip(t, -1.0, 1.0))
    Na = 0.5 * t * (t - 1.0)
    Nm = 1.0 - t**2
    Nb = 0.5 * t * (t + 1.0)
    x = Na * ca + Nm * cm + Nb * cb
    dist = float(np.linalg.norm(x - p))
    return t, dist


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
    """True if p is (near-)coplanar with and inside the convex quad given by 4 corners (loop order)."""
    o, n, e1, e2 = face_frame(corners4)
    if abs((np.asarray(p, dtype=float) - o) @ n) > tol:
        return False
    poly = [_to2d(c, o, e1, e2) for c in corners4]
    pt = _to2d(p, o, e1, e2)
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
