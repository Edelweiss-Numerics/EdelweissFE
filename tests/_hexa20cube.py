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
"""A single hexa20 cube fixture, shared by the integrated contact tests."""

import numpy as np

_HEXA20_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)

_SIDE = 2.0

#: Face numbers of the hexa20 node ordering, per the generator's face tables.
_YMIN, _YMAX = 1, 2


def _hexa20Coordinates(yOffset: float) -> list:
    """The 20 node coordinates of a side-``_SIDE`` cube whose y span starts at ``yOffset``.

    The corner ring order is boxGen's own -- (0,0), (0,S), (S,S), (S,0) in the (x, z) plane -- which
    is what the generator's face tables were verified against. Transposing two corners still yields
    a geometrically valid cube whose face areas and shape-function integrals are unchanged, but it
    reverses the face normals, so a contact fixture built that way reports penetration where there
    is separation. Hence the orientation assertion in the tests below.
    """

    corners = [
        np.array([0.0, yOffset, 0.0]),
        np.array([0.0, yOffset, _SIDE]),
        np.array([_SIDE, yOffset, _SIDE]),
        np.array([_SIDE, yOffset, 0.0]),
        np.array([0.0, yOffset + _SIDE, 0.0]),
        np.array([0.0, yOffset + _SIDE, _SIDE]),
        np.array([_SIDE, yOffset + _SIDE, _SIDE]),
        np.array([_SIDE, yOffset + _SIDE, 0.0]),
    ]
    return corners + [0.5 * (corners[a] + corners[b]) for a, b in _HEXA20_EDGES]
