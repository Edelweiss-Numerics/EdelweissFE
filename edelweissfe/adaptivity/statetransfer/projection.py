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

from itertools import product

import numpy as np

from edelweissfe.adaptivity.statetransfer.base import StateTransferStrategy


def _monomialBasis(refCoords, degree):
    """Tensor-product monomial basis :math:`\\xi^i \\eta^j \\zeta^k`, :math:`0 \\le i,j,k \\le`
    ``degree`` (in 2D :math:`\\xi^i \\eta^j`), evaluated at the given reference coordinates. Returns
    ``(nPoints, (degree+1)**dim)``.
    """
    refCoords = np.atleast_2d(refCoords)
    dim = refCoords.shape[1]
    powers = [refCoords[:, d][:, None] ** np.arange(degree + 1)[None, :] for d in range(dim)]
    cols = []
    for exponents in product(range(degree + 1), repeat=dim):
        column = np.ones(refCoords.shape[0])
        for d, e in enumerate(exponents):
            column = column * powers[d][:, e]
        cols.append(column)
    return np.column_stack(cols)


class PolynomialProjection(StateTransferStrategy):
    """Reconstruct a tensor-product polynomial from the parent's quadrature-point values (in the
    parent reference cube or square) by least squares, then resample it at the child quadrature points.

    The polynomial degree is chosen so the basis is at most as large as the number of parent
    quadrature points -- ``degree = round(nQpParent ** (1/dim)) - 1`` (trilinear for 8 points,
    triquadratic for 27; in 2D bilinear for 4, biquadratic for 9), capped so the fit is never
    underdetermined -- and the least-squares solve tolerates rank deficiency gracefully. Unlike :class:`~edelweissfe.adaptivity.statetransfer.nearestquadraturepoint.NearestQuadraturePointCopy`
    this is smooth across children, but it may produce an *inadmissible* internal state (e.g. a
    stress off the yield surface), so it is best applied per state variable to genuinely
    smooth fields via :class:`~edelweissfe.adaptivity.statetransfer.perstatevar.PerStateVarStateTransfer`.
    """

    def __init__(self, degree: int = None):
        self._degree = degree

    def _transferColumns(self, parentValues, parentRefCoords, childRefCoords, childInitValues, columns):
        nQpParent, dim = parentRefCoords.shape
        if self._degree is not None:
            degree = self._degree
        else:
            degree = int(round(nQpParent ** (1.0 / dim))) - 1
        # never let the basis outgrow the samples (would make the fit underdetermined)
        while degree > 0 and (degree + 1) ** dim > nQpParent:
            degree -= 1

        parentBasis = _monomialBasis(parentRefCoords, degree)
        childBasis = _monomialBasis(childRefCoords, degree)
        coeffs, *_ = np.linalg.lstsq(parentBasis, parentValues[:, columns], rcond=None)
        res = childBasis @ coeffs
        if np.all(parentValues[:, columns] >= -1e-12):
            res = np.maximum(res, 0.0)
        return res
