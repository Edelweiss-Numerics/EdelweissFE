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
    Represents the value vector `V` of a COO (VIJ triple) sparse matrix.

    `VIJSystemMatrix` stores matrix entry values for sparse global systems (Coordinate format),
    maintaining associated row indices `I`, column indices `J`, and an `entitiesInVIJ` mapping.
    It supports entity-aware indexing to slice, reshape, and assign local entity matrices directly.

    Parameters
    ----------
    nDof : int
        The dimension (number of rows/columns) of the system matrix.
    I : ndarray
        The 1D integer array of global row indices.
    J : ndarray
        The 1D integer array of global column indices.
    entitiesInVIJ : dict
        A dictionary mapping entities to their starting offset index in `V`.
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

        if self.entitiesInVIJ is not None:
            try:
                idxInVIJ = self.entitiesInVIJ[key]
                size = key.getVIJContributionSize()
                flat_view = super().__getitem__(slice(idxInVIJ, idxInVIJ + size))
                return key.shapeVIJContribution(flat_view)
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

        if self.entitiesInVIJ is not None:
            try:
                idxInVIJ = self.entitiesInVIJ[key]
                size = key.getVIJContributionSize()
                super().__setitem__(slice(idxInVIJ, idxInVIJ + size), value)
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
        new_mat.nDof = self.nDof
        if self.I is not None:
            new_mat.I = self.I.copy()  # noqa: E741
        if self.J is not None:
            new_mat.J = self.J.copy()
        if self.entitiesInVIJ is not None:
            new_mat.entitiesInVIJ = self.entitiesInVIJ.copy()
        return new_mat
