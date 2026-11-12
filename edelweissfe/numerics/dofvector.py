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

from typing import TYPE_CHECKING

import numpy as np

import edelweissfe.numerics.scatterdofvector

if TYPE_CHECKING:
    from edelweissfe.numerics.doflocations import DofLocations


class _NoLocations:
    """The locations of a vector that is wrapped with none: no entity is located."""

    #: No entity has a location.
    ofEntity = {}


_WITHOUT_LOCATIONS = _NoLocations()


class DofVector(np.ndarray):
    """
    Represents a Degree-of-Freedom (DOF) vector with entity-aware indexing.

    A `DofVector` is a 1D NumPy array subclass wrapped with the locations of the entities (elements, constraints, ...)
    in the numbering of its discrete system
    (:class:`~edelweissfe.numerics.doflocations.DofLocations`). This enables direct indexing and assignment using
    entity objects as keys: ``vector[entity]`` are the values at the entity's location.

    The vector refers to the locations it was wrapped with. When the locations change but the numbering does not -- a
    contact search that couples other nodes -- the values stay the values of the same degrees of freedom: the vector is
    wrapped on the new locations, see :meth:`onLocations`.

    Parameters
    ----------
    nDof : int
        The total number of degrees of freedom (length of the vector).
    locations : DofLocations, optional
        The locations of the entities, which map every entity to its indices in the vector: a
        :class:`~edelweissfe.numerics.doflocations.DofLocations`, or any object with a mapping ``ofEntity``.
    """

    def __new__(cls, nDof: int, locations: "DofLocations | None" = None):
        obj = np.zeros(nDof, dtype=float).view(cls)
        obj.locations = locations if locations is not None else _WITHOUT_LOCATIONS
        obj._scatterTemplate = None
        #: Plain-ndarray alias of the same buffer, built lazily on first entity access so
        #: that the hot path never triggers numpy subclass wrapping. See __getitem__.
        obj._plainView = None
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.locations = getattr(obj, "locations", _WITHOUT_LOCATIONS)
        self._scatterTemplate = getattr(obj, "_scatterTemplate", None)
        self._plainView = None

    def onLocations(self, locations: "DofLocations") -> "DofVector":
        """The same values, wrapped with other locations: a view of this vector's buffer, not a copy. For a discrete
        system whose locations changed on the same numbering
        (:meth:`~edelweissfe.models.discretesystem.DiscreteSystem.withNewLocations`): the degrees of freedom are the
        same, and so are their values, only where the entities find theirs has changed. Writes through either vector
        are writes to both.

        Parameters
        ----------
        locations
            The new locations, on the same numbering.

        Returns
        -------
        DofVector
            The vector wrapped with the new locations.
        """

        rewrapped = self.view(DofVector)
        rewrapped.locations = locations
        rewrapped._scatterTemplate = None
        rewrapped._plainView = None

        return rewrapped

    def __getitem__(self, key):
        # Hot path (called per element per increment): try the entity dictionary first, skipping an isinstance()
        # chain, and return a plain ndarray view, skipping numpy's subclass wrapping. Non-entity keys (ints,
        # slices, arrays) fall through to normal ndarray indexing at the cost of one exception.
        try:
            indices = self.locations.ofEntity[key]
        except (KeyError, TypeError):
            return super().__getitem__(key)

        return self.asPlainArray()[indices]

    def asPlainArray(self) -> np.ndarray:
        """The same buffer seen as a plain :class:`numpy.ndarray`, built once and cached.

        Indexing through this bypasses the subclass wrapping that dominates hot-path access -- see
        :meth:`__getitem__`. It aliases this vector rather than copying it, so writes through it are
        writes to this vector. Callers that index a DofVector repeatedly in a loop should take this
        once outside the loop instead of calling ``view(np.ndarray)`` themselves.

        Returns
        -------
        np.ndarray
            A plain-ndarray view of this vector's buffer.
        """

        plainView = self._plainView
        if plainView is None:
            plainView = self._plainView = self.view(np.ndarray)
        return plainView

    def __setitem__(self, key, value):
        if isinstance(key, (int, slice, np.ndarray, list, tuple)):
            super().__setitem__(key, value)
            return

        try:
            super().__setitem__(self.locations.ofEntity[key], value)
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
            A new `DofVector` instance with copied array data, wrapped with the same locations.
        """
        return super().copy(order).view(DofVector)

    def createScatterVector(self) -> "edelweissfe.numerics.scatterdofvector.ScatterDofVector":
        """
        Create a scatter vector for ALL entities located in the locations of this DofVector.

        The underlying layout (entity lookup map and scatter indices) is computed once
        and cached, so repeated calls (e.g. once per Newton iteration) only allocate
        the zero-initialized data buffer.

        Returns
        -------
        ScatterDofVector
            A `ScatterDofVector` initialized with this vector's locations and total DOF count.
        """

        # The template is inherited across views/copies, but the vector may since have been wrapped with other
        # locations: do not reuse a template built for stale locations.
        if self._scatterTemplate is None or self._scatterTemplate.locations is not self.locations:
            self._scatterTemplate = edelweissfe.numerics.scatterdofvector.ScatterDofVectorTemplate(
                self.locations, self.size
            )
        return edelweissfe.numerics.scatterdofvector.ScatterDofVector(self._scatterTemplate)
