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
Embedding of reinforcement (bar) elements in host continuum elements.

:class:`HostElementLocator` finds the host element containing a point and its parametric coordinates there.
:func:`splitBarElementAtHostBoundaries` splits a bar element into the parts lying in individual host elements. Both
are shared by the embedded region constraint (perfect bond, :mod:`edelweissfe.constraints.embeddedregion`) and the
embedded bond generator (bond-slip, :mod:`edelweissfe.generators.embeddedbond`).
"""

from dataclasses import dataclass

import numpy as np

from edelweissfe.utils.isoparametricshapes import (
    PARAMETRIC_DIMENSION,
    inverseMapping,
    isInside,
    shapeFunctions,
    shapeFunctionsAndDerivatives,
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


def barGeometry(barElement) -> tuple[str, np.ndarray]:
    """The shape (``bar2`` or ``bar3``) and the nodal coordinates of a bar element."""
    shape = barElement.ensightType
    if shape not in ("bar2", "bar3"):
        raise ValueError(f"Element {barElement.elNumber} is not a bar element, but a '{shape}' element.")
    return shape, np.array([n.coordinates for n in barElement.nodes], dtype=float)


def barPoint(shape: str, coordinates: np.ndarray, eta: float) -> np.ndarray:
    """The point at the parametric coordinate ``eta`` of a bar."""
    return shapeFunctions(shape, eta) @ coordinates


def barLength(shape: str, coordinates: np.ndarray, etaStart: float = -1.0, etaEnd: float = 1.0) -> float:
    """The length of the part ``[etaStart, etaEnd]`` of a bar (5 point Gauss rule, exact for bar2/bar3 up to the
    square root)."""
    gp, gw = np.polynomial.legendre.leggauss(5)
    length = 0.0
    for p, w in zip(gp, gw):
        eta = etaStart + 0.5 * (etaEnd - etaStart) * (p + 1)
        dN = shapeFunctionsAndDerivatives(shape, eta)[1][:, 0]
        length += np.linalg.norm(dN @ coordinates) * 0.5 * (etaEnd - etaStart) * w
    return length


@dataclass(frozen=True)
class BarPart:
    """The part ``[etaStart, etaEnd]`` of a bar element lying in a host element."""

    barElement: object
    hostElement: object
    etaStart: float
    etaEnd: float


def splitBarElementAtHostBoundaries(barElement, locator: HostElementLocator, samplesPerHostSize: float = 8.0):
    """Split a bar element into the parts lying in individual host elements.

    The bar is sampled along its parametric coordinate, with a spacing well below the size of the smallest host
    element, and the exit points of the bar from each host element are found by bisection. Inside the host domain,
    every crossed host element is found; a bar running outside of all host elements may miss a re-entry shorter than
    the sample spacing (e.g., clipping a host corner), which then remains unbonded.

    Parameters
    ----------
    barElement
        The bar element (``bar2`` or ``bar3``).
    locator
        The locator of the host elements.
    samplesPerHostSize
        The number of samples per characteristic size of the smallest host element.

    Returns
    -------
    tuple[list[BarPart], float]
        The parts of the bar lying in host elements, ordered along the bar, and the parametric length of the bar not
        lying in any host element.
    """
    shape, coords = barGeometry(barElement)
    length = barLength(shape, coords)
    nSamples = max(4, int(np.ceil(samplesPerHostSize * length / locator.characteristicSize)))
    etas = np.linspace(-1.0, 1.0, nSamples + 1)
    tol = 1e-13

    def hostAt(eta, exclude=None):
        p = locator.locate(barPoint(shape, coords, eta), exclude=exclude)
        return p.element if p else None

    def exitPoint(host, etaInside, etaOutside):
        """Bisection for the exit of the bar from host between a point inside and a point outside."""
        while abs(etaOutside - etaInside) > tol:
            mid = 0.5 * (etaInside + etaOutside)
            if locator.contains(host, barPoint(shape, coords, mid)):
                etaInside = mid
            else:
                etaOutside = mid
        return etaInside

    parts = []
    unembedded = 0.0
    eta = -1.0
    # the host is chosen just ahead of the start, not at it: a bar starting on a host face, edge or node must not be
    # assigned to a neighbor it only touches
    host = hostAt(min(1.0, eta + 1e-9)) or hostAt(eta)
    iSample = 0
    while eta < 1.0:
        # march to the first sample beyond eta not in the current host (or not in any host)
        while iSample < nSamples and etas[iSample + 1] <= eta:
            iSample += 1
        nextOutside = None
        for j in range(iSample + 1, nSamples + 1):
            inside = locator.contains(host, barPoint(shape, coords, etas[j])) if host else hostAt(etas[j]) is None
            if not inside:
                nextOutside = etas[j]
                break

        if host is None:
            # unembedded stretch: find where the bar enters a host
            if nextOutside is None:
                unembedded += 1.0 - eta
                break
            lo, hi = eta, nextOutside
            while hi - lo > tol:
                mid = 0.5 * (lo + hi)
                if hostAt(mid) is None:
                    lo = mid
                else:
                    hi = mid
            unembedded += hi - eta
            eta = hi
            host = hostAt(eta)
            continue

        etaExit = 1.0 if nextOutside is None else exitPoint(host, max(eta, etas[j - 1]), nextOutside)
        progress = etaExit > eta
        if etaExit - eta > 1e-9:  # skip degenerate parts, e.g., where the bar only touches a host
            parts.append(BarPart(barElement, host, eta, etaExit))
        if etaExit >= 1.0:
            break
        # continue in the next host, behind the exit point; a host touched in a single point only is skipped
        etaNext = min(1.0, etaExit + 10 * tol)
        newHost = hostAt(etaNext, exclude=host)
        eta = etaExit if (newHost is not None and progress) else etaNext
        host = newHost

    return _removeSlivers(parts), unembedded


#: Parts of a bar element shorter than this (in its parametric coordinate, of length 2) are slivers, e.g., where a bar
#: passes exactly through a corner of the host mesh and touches a neighbor only within the tolerance of the search.
MIN_PART_LENGTH = 1e-6


def _removeSlivers(parts: list) -> list:
    """Drop sliver parts and close the gaps they leave between adjacent parts, so that the remaining parts still tile
    the embedded stretches of the bar."""
    kept = []
    for part in parts:
        if part.etaEnd - part.etaStart < MIN_PART_LENGTH:
            continue
        if kept and part.etaStart - kept[-1].etaEnd < MIN_PART_LENGTH:
            # adjacent (up to a dropped sliver or the search tolerance): meet in the middle
            middle = 0.5 * (kept[-1].etaEnd + part.etaStart)
            kept[-1] = BarPart(kept[-1].barElement, kept[-1].hostElement, kept[-1].etaStart, middle)
            part = BarPart(part.barElement, part.hostElement, middle, part.etaEnd)
        kept.append(part)
    return kept
