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
"""The implicit dynamic solver assembles a CONSISTENT mass matrix from every element, so a
``PointMass`` (the reference point of every ``DiscreteRigidBody``) must provide one. A point mass
has no shape functions: its consistent inertia is its lumped one on the diagonal.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from edelweissfe.elements.pointmass import PointMass


@pytest.mark.parametrize("square", [False, True])
@pytest.mark.parametrize("inertia", [None, [2.0, 3.0, 4.0]])
def test_consistentInertiaIsTheLumpedOneOnTheDiagonal(inertia, square):
    pointMass = PointMass(1, [MagicMock()], MagicMock(domainSize=3), mass=5.0, inertia=inertia)

    lumped = np.zeros(pointMass.nDof)
    pointMass.computeLumpedInertia(lumped)
    shape = (pointMass.nDof, pointMass.nDof) if square else (pointMass.nDof * pointMass.nDof,)
    consistent = np.full(shape, np.nan)
    pointMass.computeConsistentInertia(consistent)

    np.testing.assert_array_equal(consistent.reshape(pointMass.nDof, pointMass.nDof), np.diag(lumped))
    np.testing.assert_array_equal(lumped[:3], 5.0)
    if inertia is not None:
        np.testing.assert_array_equal(lumped[3:], inertia)
