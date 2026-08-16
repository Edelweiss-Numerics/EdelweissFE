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
    Represents a Degree-of-Freedom (DOF) vector with entity-aware indexing.

    A `DofVector` is a 1D NumPy array subclass augmented with metadata mapping simulation
    entities (such as elements, constraints, or node sets) to their respective DOF indices.
    This enables direct indexing and assignment using entity objects as keys.

    Parameters
    ----------
    nDof : int
        The total number of degrees of freedom (length of the vector).
    entitiesInDofVector : dict, optional
        A dictionary mapping entities (or entity keys) to their index arrays or slices
        in the DofVector.
    """

    def __new__(cls, nDof: int, entitiesInDofVector: dict | None = None):
        obj = np.zeros(nDof, dtype=float).view(cls)
        obj.entitiesInDofVector = entitiesInDofVector if entitiesInDofVector is not None else {}
        obj._scatterTemplate = None
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.entitiesInDofVector = getattr(obj, "entitiesInDofVector", None)
        self._scatterTemplate = getattr(obj, "_scatterTemplate", None)

    def __getitem__(self, key):
        # try the entity lookup first; it is by far the most common access pattern.
        # Non-entity keys (ints, slices, arrays, lists) miss the dictionary cheaply
        # via KeyError/TypeError and fall through to plain ndarray indexing.
        if isinstance(key, (int, slice, np.ndarray, list, tuple)):
            return super().__getitem__(key)

        try:
            return super().__getitem__(self.entitiesInDofVector[key])
        except (KeyError, TypeError):
            return super().__getitem__(key)

    def __setitem__(self, key, value):
        if isinstance(key, (int, slice, np.ndarray, list, tuple)):
            super().__setitem__(key, value)
            return

        try:
            super().__setitem__(self.entitiesInDofVector[key], value)
        except (KeyError, TypeError):
            super().__setitem__(key, value)

    def copy(self, order: str = "C") -> "DofVector":
        """
        Create a copy of this DofVector.

        Parameters
        ----------
        order : str, optional
            The memory layout order ('C' for C-contiguous, 'F' for Fortran-contiguous). Default is 'C'.

        Returns
        -------
        DofVector
            A new `DofVector` instance with copied array data and an independent copy of `entitiesInDofVector`.
        """
        newDofVector = super().copy(order).view(DofVector)
        if self.entitiesInDofVector is not None:
            newDofVector.entitiesInDofVector = self.entitiesInDofVector.copy()
        return newDofVector

    def createScatterVector(self) -> "edelweissfe.numerics.scatterdofvector.ScatterDofVector":
        """
        Create a scatter vector for ALL entities registered in this DofVector.

        The underlying layout (entity lookup map and scatter indices) is computed once
        and cached, so repeated calls (e.g. once per Newton iteration) only allocate
        the zero-initialized data buffer.

        Returns
        -------
        ScatterDofVector
            A `ScatterDofVector` initialized with this vector's entity mapping and total DOF count.

        Raises
        ------
        ValueError
            If `entitiesInDofVector` is None or uninitialized.
        """
        if self.entitiesInDofVector is None:
            raise ValueError("Cannot create a ScatterDofVector: entitiesInDofVector is None.")

        # __array_finalize__ inherits _scatterTemplate across views/copies, but a view's
        # entitiesInDofVector may later be replaced by a different mapping (e.g. copy()
        # assigns a fresh dict) - guard against reusing a template built for a stale one.
        if self._scatterTemplate is None or self._scatterTemplate.entitiesInDofVector is not self.entitiesInDofVector:
            self._scatterTemplate = edelweissfe.numerics.scatterdofvector.ScatterDofVectorTemplate(
                self.entitiesInDofVector, self.size
            )
        return edelweissfe.numerics.scatterdofvector.ScatterDofVector(self._scatterTemplate)
