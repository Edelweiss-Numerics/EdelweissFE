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
"""
Symmetric quadrature rules on a triangle with positive weights (Dunavant, Int. J. Numer. Meth. Engng. 21 (1985)
1129-1148), in barycentric coordinates. The weights sum to one, so an integral over a triangle of area :math:`A` is
:math:`A \\sum_q w_q f(\\mathbf{b}_q)`.

=======  ======  =================================================
degree   points  points (orbits)
=======  ======  =================================================
1        1       centroid
2        3       :math:`(2/3, 1/6, 1/6)`
4        6       two orbits :math:`(1 - 2a, a, a)`
5        7       centroid and two orbits :math:`(1 - 2a, a, a)`
=======  ======  =================================================

The one place of these rules in EdelweissFE: the beam section points of polygons
(:mod:`edelweissfe.utils.beamsections`, by degree) and the contact facets (:mod:`edelweissfe.utils.parentfacegeometry`,
by number of points) use them.
"""

import numpy as np


def _orbit(a: float) -> list:
    """The three permutations of the barycentric point (1 - 2a, a, a)."""
    b = 1 - 2 * a
    return [(b, a, a), (a, b, a), (a, a, b)]


#: The rules by polynomial degree of exactness: (barycentric points (n, 3), weights (n,)).
TRIANGLE_RULES_BY_DEGREE = {
    degree: (np.array(points, dtype=float), np.array(weights, dtype=float))
    for degree, (points, weights) in {
        1: ([(1 / 3, 1 / 3, 1 / 3)], [1.0]),
        2: (_orbit(1 / 6), [1 / 3] * 3),
        4: (
            _orbit(0.091576213509771) + _orbit(0.445948490915965),
            [0.109951743655322] * 3 + [0.223381589678011] * 3,
        ),
        5: (
            [(1 / 3, 1 / 3, 1 / 3)] + _orbit(0.470142064105115) + _orbit(0.101286507323456),
            [0.225] + [0.132394152788506] * 3 + [0.125939180544827] * 3,
        ),
    }.items()
}

#: The same rules by number of points.
TRIANGLE_RULES_BY_POINTS = {len(w): (b, w) for b, w in TRIANGLE_RULES_BY_DEGREE.values()}


def triangleRuleByDegree(degree: int) -> tuple[np.ndarray, np.ndarray]:
    """The rule exact to the given polynomial degree: (barycentric points (n, 3), weights (n,), summing to one)."""
    if degree not in TRIANGLE_RULES_BY_DEGREE:
        raise ValueError(f"No triangle rule of degree {degree}; available: {sorted(TRIANGLE_RULES_BY_DEGREE)}.")
    return TRIANGLE_RULES_BY_DEGREE[degree]
