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
    def __init__(self, elSetName: str, propertyName: str, values: np.ndarray):
        self.elSetName = elSetName
        self.propertyName = propertyName
        self.values = values

    def assignElementPropertiesToModel(self, model):
        if self.elSetName not in model.elementSets:
            raise Exception(f"Element set '{self.elSetName}' not found in model.")

        elSet = model.elementSets[self.elSetName]
        for el in elSet:
            if hasattr(el, "assignProperty"):
                el.assignProperty(self.propertyName, self.values)
            elif hasattr(el, "setProperties"):
                el.setProperties(self.values)
