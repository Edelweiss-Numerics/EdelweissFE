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

from edelweissfe.elements.base.baseelement import BaseElement
from edelweissfe.points.node import Node

"""
Thin, geometry-only "contact facet" elements: flat (linear) surface patches attached to the
existing nodes of a deformable body's boundary, used as the master side of node-to-deformable-
surface penalty contact. They carry no material, no volume, and no independent DOFs of their own
-- they only expose the current position/normal of a flat facet as a function of their nodes'
ordinary displacement DOFs, which are shared with (and driven by) whatever real element(s) also
reference those nodes.
"""


def facetNormalAndMeasure(coords: np.ndarray) -> tuple[np.ndarray, float]:
    """The (non-unit-normalized only in intermediate steps) outward normal and measure (area for a
    Tria3 facet, length for a Line2 facet) of a flat facet, as a function of its current node
    coordinates.

    Parameters
    ----------
    coords
        Array of shape ``(3, 3)`` (Tria3, 3D) or ``(2, 2)`` (Line2, 2D) with the facet's current
        node coordinates in its fixed local order.

    Returns
    -------
    tuple[numpy.ndarray, float]
        The outward unit normal, and the facet's measure (area or length).
    """

    nNodes, domainSize = coords.shape

    if nNodes == 3 and domainSize == 3:
        e1 = coords[1] - coords[0]
        e2 = coords[2] - coords[0]
        c = np.cross(e1, e2)
        cNorm = np.linalg.norm(c)
        return c / cNorm, 0.5 * cNorm

    elif nNodes == 2 and domainSize == 2:
        e = coords[1] - coords[0]
        eNorm = np.linalg.norm(e)
        # Outward normal is e rotated by -90 degrees, consistent with a counter-clockwise
        # (node 1 -> node 2) traversal of the solid's boundary.
        n = np.array([e[1], -e[0]]) / eNorm
        return n, eNorm

    raise ValueError(f"facetNormalAndMeasure: unsupported facet shape {coords.shape}.")


class ContactFacetElementBase(BaseElement):
    """Base class for flat contact facet elements. Subclasses only need to set the class
    attributes ``_nNodes``, ``_domainSize`` and ``_ensightType``.
    """

    _nNodes = None
    _domainSize = None
    _ensightType = None

    def __init__(self, elementType: str, elNumber: int):
        self._elType = elementType
        self._elNumber = elNumber
        self._nDof = self._nNodes * self._domainSize
        self._fields = [["displacement"] for _ in range(self._nNodes)]
        self._dofIndicesPermutation = np.arange(self._nDof, dtype=int)
        self._hasMaterial = True  # dummy: this element carries no material, see setMaterial()

    @property
    def elNumber(self) -> int:
        return self._elNumber

    @property
    def elType(self) -> str:
        return self._elType

    @property
    def nNodes(self) -> int:
        return self._nNodes

    @property
    def nodes(self) -> list[Node]:
        return self._nodes

    @property
    def nDof(self) -> int:
        return self._nDof

    @property
    def fields(self) -> list[list[str]]:
        return self._fields

    @property
    def dofIndicesPermutation(self) -> np.ndarray:
        return self._dofIndicesPermutation

    @property
    def ensightType(self) -> str:
        return self._ensightType

    @property
    def visualizationNodes(self) -> list[Node]:
        return self._nodes

    @property
    def hasMaterial(self) -> bool:
        return self._hasMaterial

    def setNodes(self, nodes: list[Node]):
        self._nodes = nodes

    def setProperties(self, elementProperties: np.ndarray):
        if len(elementProperties):
            raise ValueError(f"{self._elType} does not accept any element properties.")

    def initializeElement(self):
        self._referenceCoordinates = np.array([n.coordinates for n in self._nodes])

    def setMaterial(self, materialName: str, materialProperties: np.ndarray):
        raise ValueError(
            f"{self._elType} is a geometry-only contact facet and cannot be assigned a material "
            "-- do not reference its element set in a *material/*section definition."
        )

    def setInitialCondition(self, stateType: str, values: np.ndarray):
        pass

    def computeDistributedLoad(
        self,
        loadType: str,
        P: np.ndarray,
        K: np.ndarray,
        faceID: int,
        load: np.ndarray,
        U: np.ndarray,
        time: float,
        dT: float,
    ):
        pass

    def computeKernels(self, P: np.ndarray, K: np.ndarray, U: np.ndarray, dU: np.ndarray, time: float, dT: float):
        pass

    def computeKernelsExplicit(self, P: np.ndarray, U: np.ndarray, dU: np.ndarray, time: float, dT: float):
        pass

    def computeLumpedInertia(self, M: np.ndarray):
        M[:] = 0.0

    def computeCriticalTimeStepForExplicitDynamics(self, Q: np.ndarray) -> float:
        return 1e99

    def computeInternalEnergy(self) -> float:
        return 0.0

    def computeBodyForce(
        self, P: np.ndarray, K: np.ndarray, load: np.ndarray, U: np.ndarray, time: float, dTime: float
    ):
        pass

    def acceptLastState(self):
        pass

    def resetToLastValidState(self):
        pass

    def getResultArray(self, result: str, quadraturePoint: int, getPersistentView: bool = True) -> np.ndarray:
        """Only reference-configuration diagnostics are available here (this element never
        updates any state from the current solution) -- the contact constraint itself computes
        current-configuration normals/positions directly from the live solution vector."""

        normal, measure = facetNormalAndMeasure(self._referenceCoordinates)
        if result == "normal":
            return normal
        if result in ("area", "length"):
            return np.array([measure])
        raise ValueError(f"{self._elType} does not provide a result named '{result}'.")

    def getCoordinatesAtCenter(self) -> np.ndarray:
        return np.mean(self._referenceCoordinates, axis=0)

    def getCoordinatesAtQuadraturePoints(self) -> np.ndarray:
        return self.getCoordinatesAtCenter()[np.newaxis, :]

    def getNumberOfQuadraturePoints(self) -> int:
        return 1


class Tria3ContactFacet(ContactFacetElementBase):
    """A flat, 3-node triangular contact facet (3D)."""

    _nNodes = 3
    _domainSize = 3
    _ensightType = "tria3"


class Line2ContactFacet(ContactFacetElementBase):
    """A flat, 2-node linear contact facet (2D)."""

    _nNodes = 2
    _domainSize = 2
    _ensightType = "bar2"
