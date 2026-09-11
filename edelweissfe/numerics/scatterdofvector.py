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

import edelweissfe.numerics.dofvector


class ScatterDofVectorTemplate:
    """
    The precomputed, immutable layout of a :class:`ScatterDofVector`:
    the entity lookup map and the gather/scatter index array.

    Since the layout only depends on the entities and their indices in the DofVector,
    it can be computed once and shared by all scatter vectors of the same equation
    system, e.g. across Newton iterations.

    Parameters
    ----------
    entitiesInDofVector : dict
        The dictionary mapping entities to their global index arrays in the `DofVector`.
    nDof : int
        The total number of global degrees of freedom.
    """

    def __init__(self, entitiesInDofVector: dict, nDof: int):
        sizes = np.array([len(v) for v in entitiesInDofVector.values()], dtype=np.intc)
        total_size = int(np.sum(sizes))

        offsets = np.zeros(len(entitiesInDofVector) + 1, dtype=np.intc)
        np.cumsum(sizes, out=offsets[1:])

        # map entity -> (offset, size), such that a single lookup suffices for access
        self.offsetMap = {
            entity: (offset, size) for entity, offset, size in zip(entitiesInDofVector.keys(), offsets, sizes)
        }

        self.globalIndices = np.empty(total_size, dtype=np.int32)

        current_offset = 0
        for entity, indices in entitiesInDofVector.items():
            n = len(indices)
            self.globalIndices[current_offset : current_offset + n] = indices
            current_offset += n

        self.entitiesInDofVector = entitiesInDofVector
        self.nDof = nDof
        self.totalSize = total_size


class ScatterDofVector(np.ndarray):
    """
    A Scatter Vector that stores data for entities contiguously.
    Includes a fast lookup map to support random access by Entity.

    Parameters
    ----------
    template
        The precomputed layout of the scatter vector.
    """

    def __new__(cls, template: ScatterDofVectorTemplate):
        obj = np.zeros(template.totalSize, dtype=float).view(cls)

        obj._offset_map = template.offsetMap
        obj._global_indices = template.globalIndices
        obj._entitiesInDofVector = template.entitiesInDofVector
        obj._nDof = template.nDof
        #: Plain-ndarray alias of the same buffer; see __getitem__.
        obj._plainView = None

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._offset_map = getattr(obj, "_offset_map", None)
        self._entitiesInDofVector = getattr(obj, "_entitiesInDofVector", None)
        self._nDof = getattr(obj, "_nDof", None)
        self._global_indices = getattr(obj, "_global_indices", None)
        self._plainView = None

    def __getitem__(self, key):
        """
        Returns a view into the contiguous scatter buffer.

        Parameters
        ----------
        key : int, slice, np.ndarray, list, tuple, or entity
            The key for indexing, either standard numpy index/slice or an entity key.

        Returns
        -------
        ndarray or float
            The slice view corresponding to the key.
        """
        # Hot path: see DofVector.__getitem__ for the measurement. The slice itself is
        # cheap here, so wrapping the result back into a ScatterDofVector subclass (which
        # runs __array_finalize__ and its four attribute lookups) was almost the entire
        # cost. A plain ndarray view of the same buffer aliases the same memory, so the
        # element kernels still write straight into the scatter buffer.
        try:
            offset, size = self._offset_map[key]
        except (KeyError, TypeError):
            return super().__getitem__(key)

        plainView = self._plainView
        if plainView is None:
            plainView = self._plainView = self.view(np.ndarray)
        return plainView[offset : offset + size]

    def __setitem__(self, key, value):
        """
        Assign values to a slice in the scatter buffer.

        Parameters
        ----------
        key : int, slice, np.ndarray, list, tuple, or entity
            The key for indexing, either standard numpy index/slice or an entity key.
        value : array_like or float
            The value(s) to assign.
        """
        if isinstance(key, (int, slice, np.ndarray, list, tuple)):
            super().__setitem__(key, value)
            return

        try:
            offset, size = self._offset_map[key]
            super().__setitem__(slice(offset, offset + size), value)
        except (KeyError, TypeError):
            super().__setitem__(key, value)

    def copy(self, order: str = "C") -> "ScatterDofVector":
        """
        Create a copy of this ScatterDofVector.

        Parameters
        ----------
        order : str, optional
            The memory layout order ('C' for C-contiguous, 'F' for Fortran-contiguous). Default is 'C'.

        Returns
        -------
        ScatterDofVector
            A new `ScatterDofVector` instance with copied array data and metadata.
        """
        new_vec = super().copy(order).view(ScatterDofVector)
        if self._offset_map is not None:
            new_vec._offset_map = self._offset_map.copy()
        if self._entitiesInDofVector is not None:
            new_vec._entitiesInDofVector = self._entitiesInDofVector.copy()
        if self._global_indices is not None:
            new_vec._global_indices = self._global_indices.copy()
        new_vec._nDof = self._nDof
        return new_vec

    def assembleInto(self, targetDofVector: np.ndarray, absolute: bool = False) -> None:
        """
        Scatter-add values into a global DOF vector.

        Parameters
        ----------
        targetDofVector : ndarray
            The target vector (e.g. `DofVector`) to assemble into.
        absolute : bool, optional
            If True, accumulate absolute values. Default is False.
        """
        data = np.abs(self) if absolute else self
        # bincount performs the duplicate-resolving scatter-add in a single pass,
        # considerably faster than the unbuffered np.add.at
        targetDofVector += np.bincount(self._global_indices, weights=data, minlength=self._nDof)

    def toDofVector(self, absolute: bool = False) -> "edelweissfe.numerics.dofvector.DofVector":
        """
        Create a new global DofVector by scattering values from this scatter vector.

        Parameters
        ----------
        absolute : bool, optional
            If True, use absolute values. Default is False.

        Returns
        -------
        DofVector
            A new `DofVector` containing the accumulated entries.
        """
        new_dof_vector = edelweissfe.numerics.dofvector.DofVector(self._nDof, self._entitiesInDofVector)
        self.assembleInto(new_dof_vector, absolute=absolute)
        return new_dof_vector
