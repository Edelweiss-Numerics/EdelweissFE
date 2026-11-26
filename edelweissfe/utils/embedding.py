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
    INSIDE_TOLERANCE,
    PARAMETRIC_DIMENSION,
    inverseMapping,
    parentDomainExcess,
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

    def __init__(self, hostElements, tolerance: float = INSIDE_TOLERANCE):
        self.elements = list(hostElements)
        if not self.elements:
            raise ValueError("No host elements given.")
        self.tolerance = tolerance

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

    def _candidates(self, x: np.ndarray):
        """The hosts whose (enlarged) bounding box contains ``x``, with the parametric coordinates of ``x``."""
        for i in np.flatnonzero(np.all((self._boxMin <= x) & (x <= self._boxMax), axis=1)):
            el = self.elements[i]
            xi = inverseMapping(el.ensightType, self.coordinates[i], x)
            if xi is not None:
                yield el, xi, parentDomainExcess(el.ensightType, xi)

    def locate(self, x: np.ndarray) -> HostPoint | None:
        """The host element containing the point ``x``, and the parametric coordinates of ``x`` therein. On a
        boundary between hosts, the one it lies deepest in (round-off decides between equals: use :meth:`locateAll`
        for quantities that jump between hosts).

        Parameters
        ----------
        x
            The point.

        Returns
        -------
        HostPoint
            The located point, or ``None`` if no host element contains ``x``.
        """
        best, bestExcess = None, np.inf
        for el, xi, excess in self._candidates(x):
            if excess <= self.tolerance and excess < bestExcess:
                best, bestExcess = HostPoint(el, xi), excess
                if excess < -self.tolerance:  # strictly inside: unique
                    break
        return best

    def locateAll(self, x: np.ndarray) -> list[HostPoint]:
        """All host elements containing the point ``x`` within the tolerance: one for an interior point, several for
        a point on a face, an edge or a node shared by hosts.

        Parameters
        ----------
        x
            The point.

        Returns
        -------
        list[HostPoint]
            The located points, in the order of the host elements.
        """
        return [HostPoint(el, xi) for el, xi, excess in self._candidates(x) if excess <= self.tolerance]
