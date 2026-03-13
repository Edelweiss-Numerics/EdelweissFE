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

import edelweissfe.numerics.scatterdofvector


class DofVector(np.ndarray):
    """
    Represents a Dof Vector with entity-aware indexing.

    Parameters
    ----------
    nDof
        The total number of degrees of freedom.
    entitiesInDofVector
        A dictionary mapping entities to their indices in the DofVector.
    """

    def __new__(cls, nDof: int, entitiesInDofVector: dict):
        obj = np.zeros(nDof, dtype=float).view(cls)
        obj.entitiesInDofVector = entitiesInDofVector
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.entitiesInDofVector = getattr(obj, "entitiesInDofVector", None)

    def __getitem__(self, key):
        if isinstance(key, (int, slice, np.ndarray, list)):
            return super().__getitem__(key)

        try:
            return super().__getitem__(self.entitiesInDofVector[key])
        except (KeyError, TypeError):
            return super().__getitem__(key)

    def __setitem__(self, key, value):
        if isinstance(key, (int, slice, np.ndarray, list)):
            super().__setitem__(key, value)
            return

        try:
            super().__setitem__(self.entitiesInDofVector[key], value)
        except (KeyError, TypeError):
            super().__setitem__(key, value)

    def copy(self, order="C"):
        """
        Create a copy of this DofVector.

        Parameters
        ----------
        order
            The memory layout order.
        Returns
        -------
        DofVector
            The copied DofVector.
        """
        newDofVector = super().copy(order).view(DofVector)
        if self.entitiesInDofVector is not None:
            newDofVector.entitiesInDofVector = self.entitiesInDofVector.copy()
        return newDofVector

    def createScatterVector(self) -> "ScatterDofVector":  # noqa: F821
        """
        Create a scatter vector for ALL entities in this DofVector.

        Returns
        -------
        ScatterDofVector
            The ScatterDofVector.
        """
        return edelweissfe.numerics.scatterdofvector.ScatterDofVector(self.entitiesInDofVector, self.size)
