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

"""Nearest-quadrature-point block copy (tier F1)."""

import numpy as np

from edelweissfe.adaptivity.statetransfer.base import StateTransferStrategy


class NearestQuadraturePointCopy(StateTransferStrategy):
    """Each child quadrature point inherits, verbatim, the state of the parent quadrature point
    closest to it in the parent reference cube.

    Copying an already-admissible state -- rather than interpolating it -- cannot produce an
    inadmissible internal state (off-yield stress, inconsistent loading / unloading), unlike an
    :math:`L_2` projection. Matching in the parent's *reference* cube (not physical space) selects
    the geometrically corresponding octant even for distorted / high-aspect-ratio hexes. This is
    the default strategy.
    """

    def _transferColumns(self, parentValues, parentRefCoords, childRefCoords, childInitValues, columns):
        out = np.empty((childRefCoords.shape[0], len(columns)))
        for cq in range(childRefCoords.shape[0]):
            pq = int(np.argmin(np.linalg.norm(parentRefCoords - childRefCoords[cq], axis=1)))
            out[cq, :] = parentValues[pq][columns]
        return out
