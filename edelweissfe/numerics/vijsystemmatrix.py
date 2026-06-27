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


class VIJSystemMatrix(np.ndarray):
    """
    This class represents the V Vector of VIJ triple (sparse matrix in COO format),
    which

      * also contains the I and J vectors as class members,
      * allows to directly access (contiguous read and write) access of each entity via the [] operator

    Parameters
    ----------
    nDof
        The size of the system.
    I
        The I vector for the VIJ triple.
    J
        The J vector for the VIJ triple.
    entitiesInVIJ
        A dictionary containing the indices of an entity in the value vector.
    """

    def __new__(cls, nDof: int, I: np.ndarray, J: np.ndarray, entitiesInVIJ: dict):  # noqa: E741
        obj = np.zeros_like(I, dtype=float).view(cls)

        obj.nDof = nDof
        obj.I = I  # noqa: E741
        obj.J = J
        obj.entitiesInVIJ = entitiesInVIJ

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.nDof = getattr(obj, "nDof", None)
        self.I = getattr(obj, "I", None)  # noqa: E741
        self.J = getattr(obj, "J", None)  # noqa: E741
        self.entitiesInVIJ = getattr(obj, "entitiesInVIJ", None)

    def __getitem__(self, key):
        if isinstance(key, (int, slice, np.ndarray, list)):
            return super().__getitem__(key)

        try:
            # Entity Lookup
            idxInVIJ = self.entitiesInVIJ[key]
            size = key.getVIJContributionSize()
            flat_view = super().__getitem__(slice(idxInVIJ, idxInVIJ + size))
            return key.shapeVIJContribution(flat_view)
        except (KeyError, TypeError, AttributeError):
            # Fallback for any other weird key types
            return super().__getitem__(key)
