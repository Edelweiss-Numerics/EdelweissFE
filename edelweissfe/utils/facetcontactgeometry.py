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

import numpy as np

"""
Exact gap function, gradient, and full Hessian (including the second-derivative term arising
from the cross-product-then-normalize/rotate-then-normalize construction of the facet normal) for
a slave point against a flat contact facet (Tria3 in 3D, Line2 in 2D), expressed directly in terms
of the current nodal coordinates.

Both facets are exactly flat (a plane through 3 points, or a line through 2 points), so the
curvature/second-fundamental-form contribution present in curved-surface contact vanishes
identically -- the only surviving second-derivative term is a pose-dependent nonlinearity of the
normal's own construction from its defining nodes' positions, not a curvature effect.

The closed forms below were derived by hand and cross-verified against exact symbolic
differentiation (SymPy) at many random, non-degenerate configurations before being transcribed
here -- see the derivation/verification scripts referenced in the class docstring of
:mod:`~edelweissfe.constraints.nodetodeformablesurfacepenalty` for the underlying methodology.
Do not hand-edit these formulas without re-verifying them the same way; this kind of
normalize/rotate second-derivative algebra is very easy to get subtly wrong.
"""


def _skew(v: np.ndarray) -> np.ndarray:
    """The skew-symmetric cross-product matrix of a 3-vector, such that ``_skew(v) @ x == v x x``."""
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ]
    )


def tria3GapGradientHessian(
    xs: np.ndarray, x1: np.ndarray, x2: np.ndarray, x3: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Exact gap, gradient, and Hessian of a slave point against a flat Tria3 facet (3D).

    The facet plane is spanned by ``x1, x2, x3`` (in this local order); the outward normal is
    ``cross(x2-x1, x3-x1)``, normalized. The gap is positive outside the facet's half-space,
    negative when penetrating.

    Parameters
    ----------
    xs, x1, x2, x3
        Current coordinates (each shape ``(3,)``) of the slave point and the facet's three nodes,
        in this fixed local order.

    Returns
    -------
    tuple[float, numpy.ndarray, numpy.ndarray]
        The gap ``g``, its gradient ``w`` (shape ``(12,)``, blocks ``[xs, x1, x2, x3]``), and its
        Hessian ``H`` (shape ``(12, 12)``, same block order).
    """

    r = xs - x1
    e1 = x2 - x1
    e2 = x3 - x1
    c = np.cross(e1, e2)
    m = np.linalg.norm(c)
    n = c / m
    g = n.dot(r)

    blocks = ("xs", "x1", "x2", "x3")

    dr_dBlock = {"xs": np.eye(3), "x1": -np.eye(3), "x2": np.zeros((3, 3)), "x3": np.zeros((3, 3))}
    dc_dBlock = {"xs": np.zeros((3, 3)), "x1": -_skew(x2 - x3), "x2": -_skew(e2), "x3": _skew(e1)}

    projectorOntoTangentPlane = np.eye(3) - np.outer(n, n)
    dn_dBlock = {k: (projectorOntoTangentPlane @ dc_dBlock[k]) / m for k in blocks}
    dm_dBlock = {k: n @ dc_dBlock[k] for k in blocks}  # row vector

    w_dBlock = {k: dn_dBlock[k].T @ r + dr_dBlock[k].T @ n for k in blocks}
    w = np.concatenate([w_dBlock[k] for k in blocks])

    # The tangential (in-plane) component of r: normal-projected-out, used below since the
    # curvature-like second-derivative pieces only couple through r's in-plane part.
    rTangential = r - g * n

    # d(dc_dBlock[a])/d(block b) -- dc_dBlock[a] is +/- skew(u_a) for a fixed linear combination
    # u_a of the facet's nodes; this is the constant Jacobian du_a/d(block b) for each a.
    du_dBlock = {
        "x1": {"xs": np.zeros((3, 3)), "x1": np.zeros((3, 3)), "x2": -np.eye(3), "x3": np.eye(3)},  # u = x3-x2
        "x2": {"xs": np.zeros((3, 3)), "x1": -np.eye(3), "x2": np.zeros((3, 3)), "x3": np.eye(3)},  # u = x3-x1
        "x3": {"xs": np.zeros((3, 3)), "x1": -np.eye(3), "x2": np.eye(3), "x3": np.zeros((3, 3))},  # u = x2-x1
    }
    dcSign = {"x1": +1.0, "x2": -1.0, "x3": +1.0}  # dc_dBlock[a] = dcSign[a] * skew(u_a)

    H = np.zeros((12, 12))
    blockSlice = {"xs": slice(0, 3), "x1": slice(3, 6), "x2": slice(6, 9), "x3": slice(9, 12)}

    H[blockSlice["xs"], blockSlice["xs"]] = 0.0
    for b in ("x1", "x2", "x3"):
        H[blockSlice["xs"], blockSlice[b]] = dn_dBlock[b]

    for a in ("x1", "x2", "x3"):
        for b in blocks:
            # d(dn_dBlock[a])/d(block b), contracted with r on the normal's own index -- the
            # exact second derivative of the cross-product-then-normalize construction of n.
            crossNormalizeTerm = -(1.0 / m) * (
                np.outer(dm_dBlock[a], dn_dBlock[b].T @ r) + g * (dc_dBlock[a].T @ dn_dBlock[b])
            )
            skewArgumentTerm = dcSign[a] * (1.0 / m) * (_skew(rTangential) @ du_dBlock[a][b])
            normalizeDenominatorTerm = -(1.0 / m**2) * np.outer(rTangential @ dc_dBlock[a], dm_dBlock[b])

            d2n_a_contractedWithR = crossNormalizeTerm + skewArgumentTerm + normalizeDenominatorTerm
            H[blockSlice[a], blockSlice[b]] = (
                d2n_a_contractedWithR + dn_dBlock[a].T @ dr_dBlock[b] + dr_dBlock[a].T @ dn_dBlock[b]
            )

    return g, w, H


def line2GapGradientHessian(xs: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Exact gap, gradient, and Hessian of a slave point against a flat Line2 facet (2D).

    The facet edge runs from ``x1`` to ``x2``; the outward normal is the edge direction rotated
    by -90 degrees, normalized. The gap is positive outside the facet's half-plane, negative when
    penetrating.

    Parameters
    ----------
    xs, x1, x2
        Current coordinates (each shape ``(2,)``) of the slave point and the facet's two nodes,
        in this fixed local order.

    Returns
    -------
    tuple[float, numpy.ndarray, numpy.ndarray]
        The gap ``g``, its gradient ``w`` (shape ``(6,)``, blocks ``[xs, x1, x2]``), and its
        Hessian ``H`` (shape ``(6, 6)``, same block order).
    """

    r = xs - x1
    e = x2 - x1
    length = np.linalg.norm(e)
    eHat = e / length
    rotateMinus90 = np.array([[0.0, 1.0], [-1.0, 0.0]])
    n = rotateMinus90 @ eHat
    g = n.dot(r)

    blocks = ("xs", "x1", "x2")

    dr_dBlock = {"xs": np.eye(2), "x1": -np.eye(2), "x2": np.zeros((2, 2))}
    de_dBlock = {"xs": np.zeros((2, 2)), "x1": -np.eye(2), "x2": np.eye(2)}

    projectorOntoNormal = np.eye(2) - np.outer(eHat, eHat)
    dEHat_dBlock = {k: (projectorOntoNormal @ de_dBlock[k]) / length for k in blocks}
    dn_dBlock = {k: rotateMinus90 @ dEHat_dBlock[k] for k in blocks}
    dLength_dBlock = {k: eHat @ de_dBlock[k] for k in blocks}  # row vector

    w_dBlock = {k: dn_dBlock[k].T @ r + dr_dBlock[k].T @ n for k in blocks}
    w = np.concatenate([w_dBlock[k] for k in blocks])

    # r rotated into the eHat/normal frame -- required because dn_dBlock = rotateMinus90 @
    # dEHat_dBlock, so contracting r with n's own index is equivalent to contracting
    # rotateMinus90^T @ r with eHat's index instead (rotateMinus90 is constant, applied on the
    # left, so it commutes out of the contraction this way).
    rRotated = rotateMinus90.T @ r
    rRotatedDotEHat = rRotated.dot(eHat)

    H = np.zeros((6, 6))
    blockSlice = {"xs": slice(0, 2), "x1": slice(2, 4), "x2": slice(4, 6)}

    for a in blocks:
        for b in blocks:
            normalizeTerm = -(1.0 / length) * (
                np.outer(eHat @ de_dBlock[a], dEHat_dBlock[b].T @ rRotated)
                + rRotatedDotEHat * (de_dBlock[a].T @ dEHat_dBlock[b])
                + np.outer(dEHat_dBlock[a].T @ rRotated, dLength_dBlock[b])
            )
            H[blockSlice[a], blockSlice[b]] = (
                normalizeTerm + dn_dBlock[a].T @ dr_dBlock[b] + dr_dBlock[a].T @ dn_dBlock[b]
            )

    return g, w, H
