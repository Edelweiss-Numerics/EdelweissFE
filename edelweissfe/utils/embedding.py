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
"""
Embedding of reinforcement (bar, beam) elements in host continuum elements.

:class:`HostElementLocator` finds the host element containing a point and its parametric coordinates there; it is
used by the embedded region constraint (perfect bond, :mod:`edelweissfe.constraints.embeddedregion`).
"""

from dataclasses import dataclass

import numpy as np

from edelweissfe.utils.isoparametricshapes import (
    PARAMETRIC_DIMENSION,
    inverseMapping,
    isInside,
    shapeFunctions,
)


@dataclass(frozen=True)
class HostPoint:
    """A point located in a host element."""

    #: The host element.
    element: object
    #: The parametric coordinates in the host element.
    xi: np.ndarray

    def shapeFunctions(self) -> np.ndarray:
        """The shape functions of the host element at the point."""
        return shapeFunctions(self.element.ensightType, self.xi)


class HostElementLocator:
    """Locates points in a set of host elements.

    Parameters
    ----------
    hostElements
        The host elements. Their ``ensightType`` must be supported by
        :mod:`~edelweissfe.utils.isoparametricshapes`, and its parametric dimension must equal the spatial
        dimension.
    tolerance
        The tolerance of the parametric coordinates of a point on the boundary of a host element.
    """

    def __init__(self, hostElements, tolerance: float = 1e-10):
        self.elements = list(hostElements)
        if not self.elements:
            raise ValueError("No host elements given.")
        self.tolerance = tolerance
        self._index = {id(el): i for i, el in enumerate(self.elements)}

        self.coordinates = []
        for el in self.elements:
            shape = el.ensightType
            coords = np.array([n.coordinates for n in el.nodes], dtype=float)
            if shape not in PARAMETRIC_DIMENSION:
                raise NotImplementedError(f"Host element {el.elNumber}: '{shape}' elements cannot host embedded bars.")
            if PARAMETRIC_DIMENSION[shape] != coords.shape[1]:
                raise ValueError(f"Host element {el.elNumber}: '{shape}' is not a continuum element.")
            self.coordinates.append(coords)

        mins = np.array([c.min(axis=0) for c in self.coordinates])
        maxs = np.array([c.max(axis=0) for c in self.coordinates])
        # quadratic elements may bulge beyond the bounding box of their nodes; be generous
        margin = 0.1 * (maxs - mins).max(axis=1, keepdims=True)
        self._boxMin = mins - margin
        self._boxMax = maxs + margin
        self.characteristicSize = float(np.min((maxs - mins).max(axis=1)))

    def locate(self, x: np.ndarray, exclude=None) -> HostPoint | None:
        """The host element containing the point ``x``, and the parametric coordinates of ``x`` therein.

        Parameters
        ----------
        x
            The point.
        exclude
            An optional host element not to be considered.

        Returns
        -------
        HostPoint
            The located point, or ``None`` if no host element contains ``x``.
        """
        candidates = np.flatnonzero(np.all((self._boxMin <= x) & (x <= self._boxMax), axis=1))
        best, bestExcess = None, np.inf
        for i in candidates:
            el = self.elements[i]
            if el is exclude:
                continue
            xi = inverseMapping(el.ensightType, self.coordinates[i], x)
            if xi is None:
                continue
            excess = _parentDomainExcess(el.ensightType, xi)
            if excess <= self.tolerance and excess < bestExcess:
                best, bestExcess = HostPoint(el, xi), excess
                if excess < 0:  # strictly inside: unique
                    break
        return best

    def contains(self, element, x: np.ndarray) -> bool:
        """Whether the host ``element`` contains the point ``x``."""
        xi = inverseMapping(element.ensightType, self.coordinates[self._index[id(element)]], x)
        return xi is not None and isInside(element.ensightType, xi, self.tolerance)


def _parentDomainExcess(shape: str, xi: np.ndarray) -> float:
    """By how much the parametric coordinates exceed the parent domain; negative if strictly inside."""
    if shape == "tetra4":
        return float(max(-np.min(xi), np.sum(xi) - 1.0))
    return float(np.max(np.abs(xi)) - 1.0)
