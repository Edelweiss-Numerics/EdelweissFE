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
"""The current displacements and coordinates of nodes, read from the model's displacement field."""

import numpy as np

from edelweissfe.models.femodel import FEModel
from edelweissfe.points.node import Node
from edelweissfe.utils.meshtools import currentNodeCoordinates, currentNodeDisplacements


def test_without_a_displacement_field_the_nodes_stay_where_they_are():
    """A model without a displacement field displaces no node: each stays at its reference coordinates."""

    referenceCoordinates = np.array([[0.0, 0.0], [1.0, 0.5], [2.0, 1.0]])
    nodes = [Node(label, coordinates) for label, coordinates in enumerate(referenceCoordinates, start=1)]
    model = FEModel(2)

    np.testing.assert_array_equal(currentNodeDisplacements(nodes, model, 2), np.zeros((3, 2)))
    np.testing.assert_array_equal(currentNodeCoordinates(nodes, model, referenceCoordinates), referenceCoordinates)
