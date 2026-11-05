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
    The value vector `V` of a VIJ (COO) system matrix: one value per entry of the block of every entity, the entities
    one after the other, in the layout of a
    :class:`~edelweissfe.numerics.systemmatrixassembly.VIJSystemMatrixAssembly`, which holds the row and column
    indices `I` and `J` of the values.

    Indexed by an entity, it is the entity's view of its block, so that an entity writes its block into it directly.

    Parameters
    ----------
    nEntries : int
        The number of values.
    vijSliceOfEntity : dict
        A dictionary mapping entities to the slice of `V` that holds the values of their block.
    """

    def __new__(cls, nEntries: int, vijSliceOfEntity: dict):
        obj = np.zeros(nEntries, dtype=float).view(cls)
        obj.vijSliceOfEntity = vijSliceOfEntity
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.vijSliceOfEntity = getattr(obj, "vijSliceOfEntity", None)

    def __getitem__(self, key):
        """
        Get value(s) or shaped entity matrix slice.

        Parameters
        ----------
        key : int, slice, np.ndarray, list, tuple, or entity
            Standard array index/slice or entity instance.

        Returns
        -------
        ndarray or float
            Sub-array or shaped entity matrix view.
        """
        if isinstance(key, (int, slice, np.ndarray, list, tuple)):
            return super().__getitem__(key)

        if self.vijSliceOfEntity is not None:
            try:
                flat_view = super().__getitem__(self.vijSliceOfEntity[key])
                return key.shapeBlock(flat_view)
            except (KeyError, TypeError, AttributeError):
                pass
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        """
        Set value(s) in the system matrix.

        Parameters
        ----------
        key : int, slice, np.ndarray, list, tuple, or entity
            Standard array index/slice or entity instance.
        value : array_like or float
            Value(s) to assign.
        """
        if isinstance(key, (int, slice, np.ndarray, list, tuple)):
            super().__setitem__(key, value)
            return

        if self.vijSliceOfEntity is not None:
            try:
                super().__setitem__(self.vijSliceOfEntity[key], value)
                return
            except (KeyError, TypeError, AttributeError):
                pass
        super().__setitem__(key, value)

    def copy(self, order: str = "C") -> "VIJSystemMatrix":
        """
        Create a copy of this VIJSystemMatrix.

        Parameters
        ----------
        order : str, optional
            The memory layout order ('C' for C-contiguous, 'F' for Fortran-contiguous). Default is 'C'.

        Returns
        -------
        VIJSystemMatrix
            A new `VIJSystemMatrix` instance with copied values and metadata.
        """
        new_mat = super().copy(order).view(VIJSystemMatrix)
        if self.vijSliceOfEntity is not None:
            new_mat.vijSliceOfEntity = self.vijSliceOfEntity.copy()
        return new_mat
