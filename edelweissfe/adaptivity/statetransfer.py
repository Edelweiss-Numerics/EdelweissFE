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

"""State-variable transfer on refinement (WS-F, tier F1): nearest-quadrature-point block copy.

When a parent element is subdivided, each child quadrature point inherits the material state of the
geometrically closest parent quadrature point (a verbatim block copy of the per-QP state buffer).
Because it copies an already-admissible state -- rather than interpolating -- it cannot produce an
inadmissible internal state (off-yield stress, inconsistent loading/unloading), unlike an L2
projection. Weights are unnecessary: it is piecewise-constant per child.

Requires the elements to expose their per-QP state buffer via ``getStateVars`` / ``setStateVars``
(see :class:`edelweissfe.elements.base.baseelement.BaseElement`) and the buffer to be laid out as
``nQuadraturePoints`` equal contiguous blocks.
"""

import numpy as np

from edelweissfe.adaptivity.hex20topology import hex20_shape


def _hex20_inverse(point, elementNodeCoords, tol=1e-11, itmax=50):
    """Map a physical point into the reference cube [-1,1]^3 of a HEX20 element (Newton with a
    numerical Jacobian). Robust to element distortion, unlike raw physical distance."""
    coords = np.asarray(elementNodeCoords, dtype=float)
    xi = np.zeros(3)
    h = 1e-7
    for _ in range(itmax):
        x = hex20_shape(*xi) @ coords
        residual = x - point
        if np.linalg.norm(residual) < tol:
            return xi
        jac = np.zeros((3, 3))
        for k in range(3):
            xip = xi.copy()
            xip[k] += h
            jac[:, k] = ((hex20_shape(*xip) @ coords) - x) / h
        xi = xi - np.linalg.solve(jac, residual)
    return xi  # best effort; nearest-QP matching tolerates a slightly imperfect inverse


def _perQpBlockSize(element):
    n = element.getStateVars().shape[0]
    nqp = element.getNumberOfQuadraturePoints()
    if nqp == 0 or n % nqp != 0:
        raise ValueError(
            "state transfer: state buffer of size {:} is not an integer multiple of {:} "
            "quadrature points (element-level state variables are not supported).".format(n, nqp)
        )
    return n // nqp


def transferStateNearestQp(parent, children):
    """Copy the parent's per-QP state into each child by nearest quadrature point.

    Parameters
    ----------
    parent
        The (converged) parent element being refined.
    children
        The freshly created & initialized child elements (same type/material as the parent).
    """
    parentState = parent.getStateVars()
    parentBlock = _perQpBlockSize(parent)
    parentNodeCoords = np.array([n.coordinates for n in parent.nodes], dtype=float)
    # parent QP positions in the parent reference cube [-1,1]^3 (distortion-independent)
    parentQpPhys = np.asarray(parent.getCoordinatesAtQuadraturePoints()).reshape(
        parent.getNumberOfQuadraturePoints(), -1
    )
    parentQpRef = np.array([_hex20_inverse(p, parentNodeCoords) for p in parentQpPhys])

    for child in children:
        childBlock = _perQpBlockSize(child)
        if childBlock != parentBlock:
            raise ValueError("state transfer: parent and child per-QP state sizes differ.")
        childQpPhys = np.asarray(child.getCoordinatesAtQuadraturePoints()).reshape(
            child.getNumberOfQuadraturePoints(), -1
        )
        childState = child.getStateVars()
        for cq in range(child.getNumberOfQuadraturePoints()):
            # nearest parent QP in the parent's REFERENCE cube -> selects the geometrically
            # corresponding octant even for distorted / high-aspect-ratio hexes
            childRef = _hex20_inverse(childQpPhys[cq], parentNodeCoords)
            pq = int(np.argmin(np.linalg.norm(parentQpRef - childRef, axis=1)))
            childState[cq * childBlock : (cq + 1) * childBlock] = parentState[pq * parentBlock : (pq + 1) * parentBlock]
        child.setStateVars(childState)
