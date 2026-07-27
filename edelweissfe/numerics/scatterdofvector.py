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
    entitiesInDofVector
        The dictionary mapping entities to their indices in the DofVector.
    nDof
        The total number of degrees of freedom.
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

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._offset_map = getattr(obj, "_offset_map", None)
        self._entitiesInDofVector = getattr(obj, "_entitiesInDofVector", None)
        self._nDof = getattr(obj, "_nDof", None)
        self._global_indices = getattr(obj, "_global_indices", None)

    def __getitem__(self, key):
        """
        Returns a VIEW into the expanded buffer.

        Parameters
        ----------
        key
            The key for indexing, either an entity or an integer index.
        """
        # try the entity lookup first; it is by far the most common access pattern
        try:
            offset, size = self._offset_map[key]
            return super().__getitem__(slice(offset, offset + size))
        except (KeyError, TypeError):
            return super().__getitem__(key)

    def assembleInto(self, targetDofVector, absolute=False):
        """Scatter-Add into the global vector.

        Parameters
        ----------
        targetDofVector
            The target DofVector to assemble into.
        absolute
            If True, assemble the absolute values.
        """
        data = np.abs(self) if absolute else self
        # bincount performs the duplicate-resolving scatter-add in a single pass,
        # considerably faster than the unbuffered np.add.at
        targetDofVector += np.bincount(self._global_indices, weights=data, minlength=self._nDof)

    def toDofVector(self, absolute=False) -> "DofVector":  # noqa: F821
        """Create a new DofVector from this scatter vector.

        Parameters
        ----------
        absolute
            If True, use absolute values.

        Returns
        -------
        DofVector
            The new DofVector.
        """
        new_dof_vector = edelweissfe.numerics.dofvector.DofVector(self._nDof, self._entitiesInDofVector)
        self.assembleInto(new_dof_vector, absolute=absolute)
        return new_dof_vector
