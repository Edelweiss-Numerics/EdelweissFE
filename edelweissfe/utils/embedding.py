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
:func:`findBarPartsInHosts` finds the parts of a bar element lying in individual host elements. Both
are shared by the embedded region constraint (perfect bond, :mod:`edelweissfe.constraints.embeddedregion`) and the
embedded bond generator (bond-slip, :mod:`edelweissfe.generators.embeddedbond`).
"""

from dataclasses import dataclass

import numpy as np

from edelweissfe.utils.isoparametricshapes import (
    INSIDE_TOLERANCE,
    PARAMETRIC_DIMENSION,
    inverseMapping,
    isInside,
    parentDomainExcess,
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

    def __init__(self, hostElements, tolerance: float = INSIDE_TOLERANCE):
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
        #: The size of the smallest host element (largest extent of its nodes).
        self.characteristicSize = float(np.min((maxs - mins).max(axis=1)))

    def _candidates(self, x: np.ndarray, exclude=None):
        """The hosts whose (enlarged) bounding box contains ``x``, with the parametric coordinates of ``x``."""
        for i in np.flatnonzero(np.all((self._boxMin <= x) & (x <= self._boxMax), axis=1)):
            el = self.elements[i]
            if el is exclude:
                continue
            xi = inverseMapping(el.ensightType, self.coordinates[i], x)
            if xi is not None:
                yield el, xi, parentDomainExcess(el.ensightType, xi)

    def locate(self, x: np.ndarray, exclude=None) -> HostPoint | None:
        """The host element containing the point ``x``, and the parametric coordinates of ``x`` therein. On a
        boundary between hosts, the one it lies deepest in (round-off decides between equals: use :meth:`locateAll`
        for quantities that jump between hosts).

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
        best, bestExcess = None, np.inf
        for el, xi, excess in self._candidates(x, exclude):
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

    def contains(self, element, x: np.ndarray) -> bool:
        """Whether the host ``element`` contains the point ``x`` (within the tolerance).

        Parameters
        ----------
        element
            One of the host elements.
        x
            The point.
        """
        xi = inverseMapping(element.ensightType, self.coordinates[self._index[id(element)]], x)
        return xi is not None and isInside(element.ensightType, xi, self.tolerance)


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


#: Precision (in the bar parametric coordinate, of range 2) of the points where a bar leaves or enters a host.
ETA_BISECTION_TOLERANCE = 1e-13
#: Look-ahead (in the bar parametric coordinate) that decides which host a bar continues in from a point on a host
#: boundary: the host containing the bar just ahead of the point, not one it only touches.
ETA_LOOK_AHEAD = 1e-9
#: Parts of a bar element shorter than this (in its parametric coordinate) are slivers, e.g., where a bar passes
#: exactly through a corner of the host mesh and touches a neighbor only within the tolerance of the search. They are
#: dropped and their length counts as unbonded.
MIN_PART_LENGTH = 1e-6


def _bisect(isInside, etaInside: float, etaOutside: float) -> tuple[float, float]:
    """Bisection for the point where ``isInside(eta)`` switches from true (at ``etaInside``) to false (at
    ``etaOutside``).

    Returns
    -------
    tuple[float, float]
        The last point found inside and the first point found outside, ``ETA_BISECTION_TOLERANCE`` apart.
    """
    while abs(etaOutside - etaInside) > ETA_BISECTION_TOLERANCE:
        middle = 0.5 * (etaInside + etaOutside)
        if isInside(middle):
            etaInside = middle
        else:
            etaOutside = middle
    return etaInside, etaOutside


def findBarPartsInHosts(barElement, locator: HostElementLocator, samplesPerHostSize: float = 8.0):
    """Find the parts of a bar element lying in individual host elements.

    The algorithm marches along the bar parametric coordinate :math:`\\eta` from -1 to 1:

    1. Sample the bar with a spacing of the smallest host size / ``samplesPerHostSize``.
    2. In a host: find the first sample ahead that lies outside of it, and bisect between it and the last sample inside
       for the exit point. The part from the current point to the exit point lies in this host.
    3. Continue in the host that contains the bar just ahead of the exit point (``ETA_LOOK_AHEAD``). If there is none,
       the bar leaves the host domain: find the next sample in any host, and bisect for the entry point; the stretch in
       between is unbonded.
    4. Parts shorter than ``MIN_PART_LENGTH`` (slivers at host corners) are dropped and count as unbonded, so every
       part lies in its host.

    Inside the host domain, every crossed host element is found; a bar running outside of all host elements may miss
    a re-entry shorter than the sample spacing (e.g., clipping a host corner), which then remains unbonded.

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
        The parts of the bar lying in host elements, ordered along the bar, and the (physical) length of the bar not
        lying in any of them.
    """
    shape, coords = barGeometry(barElement)
    nSamples = max(4, int(np.ceil(samplesPerHostSize * barLength(shape, coords) / locator.characteristicSize)))
    samples = np.linspace(-1.0, 1.0, nSamples + 1)

    def hostAt(eta: float, exclude=None):
        located = locator.locate(barPoint(shape, coords, eta), exclude=exclude)
        return located.element if located else None

    def firstSampleOutside(eta: float, isInside):
        """The first sample beyond ``eta`` for which ``isInside`` is false, and the sample before it (or ``eta``);
        ``None`` if all samples up to the bar end are inside."""
        for k in np.flatnonzero(samples > eta):
            if not isInside(samples[k]):
                return max(eta, samples[k - 1]), samples[k]
        return None

    parts, unbonded = [], []
    eta = -1.0
    host = hostAt(min(1.0, eta + ETA_LOOK_AHEAD)) or hostAt(eta)
    while eta < 1.0:
        if host is None:
            # outside of all hosts: march to the next sample in a host, and bisect for the entry point
            def isUnembedded(e):
                return hostAt(e) is None

            bracket = firstSampleOutside(eta, isUnembedded)
            if bracket is None:
                unbonded.append((eta, 1.0))
                break
            _, entry = _bisect(isUnembedded, *bracket)
            unbonded.append((eta, entry))
            eta, host = entry, hostAt(entry) or hostAt(min(1.0, entry + ETA_LOOK_AHEAD))
            continue

        def isInHost(e, host=host):
            return locator.contains(host, barPoint(shape, coords, e))

        bracket = firstSampleOutside(eta, isInHost)
        etaExit = 1.0 if bracket is None else _bisect(isInHost, *bracket)[0]

        if etaExit <= eta:
            # the bar only touches this host at eta: step past the touching point
            stepped = min(1.0, eta + ETA_LOOK_AHEAD)
            unbonded.append((eta, stepped))
            eta, host = stepped, hostAt(stepped)
            continue
        if etaExit - eta >= MIN_PART_LENGTH:
            parts.append(BarPart(barElement, host, eta, etaExit))
        else:
            unbonded.append((eta, etaExit))
        eta, host = etaExit, hostAt(min(1.0, etaExit + ETA_LOOK_AHEAD), exclude=host)

    return parts, sum(barLength(shape, coords, a, b) for a, b in unbonded)
