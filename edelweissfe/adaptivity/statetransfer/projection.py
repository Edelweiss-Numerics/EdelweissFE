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

"""Least-squares polynomial projection of quadrature-point state (tier F2)."""

import numpy as np

from edelweissfe.adaptivity.statetransfer.base import StateTransferStrategy


def _monomialBasis(refCoords, degree):
    """Tensor-product monomial basis :math:`\\xi^i \\eta^j \\zeta^k`, :math:`0 \\le i,j,k \\le`
    ``degree``, evaluated at the given reference coordinates. Returns ``(nPoints, (degree+1)**3)``.
    """
    refCoords = np.atleast_2d(refCoords)
    powers = [refCoords[:, d][:, None] ** np.arange(degree + 1)[None, :] for d in range(3)]
    cols = [
        powers[0][:, i] * powers[1][:, j] * powers[2][:, k]
        for i in range(degree + 1)
        for j in range(degree + 1)
        for k in range(degree + 1)
    ]
    return np.column_stack(cols)


class PolynomialProjection(StateTransferStrategy):
    """Reconstruct a tensor-product polynomial from the parent's quadrature-point values (in the
    parent reference cube) by least squares, then resample it at the child quadrature points.

    The polynomial degree is chosen so the basis is at most as large as the number of parent
    quadrature points -- ``degree = round(nQpParent ** (1/3)) - 1`` (trilinear for 8 points,
    triquadratic for 27), capped so the fit is never underdetermined -- and the least-squares solve
    tolerates rank deficiency gracefully. Unlike :class:`~edelweissfe.adaptivity.statetransfer.nearestquadraturepoint.NearestQuadraturePointCopy`
    this is smooth across octants, but it may produce an *inadmissible* internal state (e.g. a
    stress off the yield surface), so it is best applied per state variable to genuinely
    smooth fields via :class:`~edelweissfe.adaptivity.statetransfer.perstatevar.PerStateVarStateTransfer`.
    """

    def __init__(self, degree: int = None):
        self._degree = degree

    def _transferColumns(self, parentValues, parentRefCoords, childRefCoords, childInitValues, columns):
        nQpParent = parentRefCoords.shape[0]
        if self._degree is not None:
            degree = self._degree
        else:
            degree = int(round(nQpParent ** (1.0 / 3.0))) - 1
        # never let the basis outgrow the samples (would make the fit underdetermined)
        while degree > 0 and (degree + 1) ** 3 > nQpParent:
            degree -= 1

        parentBasis = _monomialBasis(parentRefCoords, degree)
        childBasis = _monomialBasis(childRefCoords, degree)
        coeffs, *_ = np.linalg.lstsq(parentBasis, parentValues[:, columns], rcond=None)
        return childBasis @ coeffs
