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


class ElementProperty:
    """Represents a single named element property, assigned to all elements
    of an element set via the ``*elementproperty`` input file keyword.

    Parameters
    ----------
    elSetName
        Name of the element set the property is assigned to.
    propertyName
        Name of the property, as understood by the target elements'
        ``assignProperty`` method.
    values
        A numpy array containing the property values.
    """

    def __init__(self, elSetName: str, propertyName: str, values: np.ndarray):
        self.elSetName = elSetName
        self.propertyName = propertyName
        self.values = values

    def assignElementPropertiesToModel(self, model):
        """Assign this property to all elements of the referenced element set.

        Parameters
        ----------
        model
            The model object.
        """
        for el in model.elementSets[self.elSetName]:
            el.assignProperty(self.propertyName, self.values)
