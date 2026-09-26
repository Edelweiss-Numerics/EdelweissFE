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
The contact points of an integrated (Gauss-point-to-segment) contact formulation: quadrature points
on the curved parent faces of a slave surface.

Shared by every constraint that integrates contact over a slave surface, whatever the master side
is -- a deformable surface or a rigid body. See
:mod:`~edelweissfe.constraints.surfacetodeformablesurfacepenalty` for why the pressure must be
distributed with the *parent* face's shape functions.
"""

import numpy as np

from edelweissfe.models.femodel import FEModel
from edelweissfe.models.modelchange import ModelChange
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.utils.facetcontactgeometry import facetNormalAndMeasure
from edelweissfe.utils.meshtools import currentNodeCoordinates
from edelweissfe.utils.parentfacegeometry import (
    facetQuadratureRule,
    parentFaceShapeFunctions,
)


def checkFacetsCarryParentFaces(facets: list, side: str, setName: str, constraintName: str):
    """Reject contact facets that cannot be used with parent-face shape functions.

    Parameters
    ----------
    facets
        The contact facet elements.
    side
        'slave' or 'master', for the error message.
    setName
        The name of the facet element set, for the error message.
    constraintName
        The name of the constraint, for the error message.
    """

    unstamped = [facet.elNumber for facet in facets if facet.parentFaceType is None]
    if unstamped:
        raise ValueError(
            f"Constraint '{constraintName}': {side} facet {unstamped[0]} carries no parent face. "
            "The integrated contact formulation distributes the contact pressure with the "
            "parent element face's shape functions, so the facets must come from the "
            "surface element generator, which stamps them."
        )

    # A per-node weighting has no meaning in this formulation and, unlike the node-based
    # constraint, there is no code path that could apply one: the pressure is distributed with
    # the parent face's own shape functions, evaluated at the quadrature points. Accepting a
    # surface stamped with a weight transform would silently give a different answer from the
    # sibling constraint on the same input, which is the one outcome worth refusing outright --
    # the whole point of the parent-face basis is that it fixes the corner mismatch the
    # weighting only minimises.
    weighted = [facet.elNumber for facet in facets if facet.weightTransform is not None]
    if weighted:
        raise ValueError(
            f"Constraint '{constraintName}': {side} surface '{setName}' was generated with "
            f"nodalWeights='serendipityOptimal' (facet {weighted[0]} and "
            f"{len(weighted) - 1} more carry a weight transform), which this constraint "
            "cannot honour -- it distributes the contact pressure with the parent face's "
            "shape functions, not with per-node weights, and needs no corner reweighting "
            "to begin with. Generate these facets with the default "
            "nodalWeights='facetConsistent'."
        )


class ContactPointsOnSlaveSurface:
    """The quadrature points of a slave surface, used as contact points.

    Each slave facet carries an ``nQuadraturePoints``-point rule. A point at facet-barycentric
    coordinates :math:`b` sits at parent-face parametric coordinates :math:`\\xi = b \\cdot \\Xi`
    (with :math:`\\Xi` the facet vertices' own parametric coordinates in the parent face, stamped by
    the surface element generator), and carries the constant weight
    :math:`J_q = A_{\\text{facet}}^{\\text{ref}} w_q`. Its position follows the *curved* parent
    surface, :math:`x_q = N^s(\\xi_q) \\cdot x^s`.

    The contact points are numbered by slave facet and, within a facet, by quadrature point.

    Parameters
    ----------
    slaveSurface
        The element set of contact facet elements forming the slave surface.
    nQuadraturePoints
        The number of quadrature points per slave facet.
    constraintName
        The name of the owning constraint, for error messages.

    Attributes
    ----------
    facets : list
        The slave contact facet elements.
    parentNodes : list[list]
        Per slave facet, the nodes of its parent element face.
    parentReferenceCoordinates : list[numpy.ndarray]
        Per slave facet, the reference coordinates of its parent face nodes.
    shapeFunctions : list[numpy.ndarray]
        Per slave facet, the parent-face shape functions at each of its quadrature points, of shape
        (nQuadraturePoints, nParentNodes). Constant, since the parametric locations are fixed.
    integrationWeights : list[numpy.ndarray]
        Per slave facet, the integration weight of each of its quadrature points in the reference
        configuration (consistent with the small-deformation setting, as the node-based
        constraint's tributary areas are).
    pointFacet : numpy.ndarray
        The slave facet index of each contact point.
    pointQuadraturePoint : numpy.ndarray
        The quadrature point index of each contact point within its facet.
    nPoints : int
        The number of contact points.
    surfaceNodes : list
        The unique nodes of the slave facets, in the same first-encounter order the surface element
        generator uses for its '<prefix>_nodes' node set -- so a fromExpression field output over
        that set lines up with :meth:`nodalNormalForces`.
    """

    def __init__(self, slaveSurface: ElementSet, nQuadraturePoints: int, constraintName: str):
        self.setName = slaveSurface.name
        self.nQuadraturePoints = nQuadraturePoints
        self.facets = list(slaveSurface)

        checkFacetsCarryParentFaces(self.facets, "slave", self.setName, constraintName)

        self.parentNodes = []
        self.parentReferenceCoordinates = []
        self.shapeFunctions = []
        self.integrationWeights = []

        for facet in self.facets:
            barycentric, weights = facetQuadratureRule(facet.nNodes, nQuadraturePoints)
            _, measure = facetNormalAndMeasure(np.array([n.coordinates for n in facet.nodes]))
            parametric = barycentric @ facet.vertexParametricCoords
            self.parentNodes.append(list(facet.parentFaceNodes))
            self.parentReferenceCoordinates.append(np.array([n.coordinates for n in facet.parentFaceNodes]))
            self.shapeFunctions.append(
                np.array([parentFaceShapeFunctions(facet.parentFaceType, xi) for xi in parametric])
            )
            self.integrationWeights.append(measure * weights)

        self.pointFacet = np.repeat(np.arange(len(self.facets)), nQuadraturePoints)
        self.pointQuadraturePoint = np.tile(np.arange(nQuadraturePoints), len(self.facets))
        self.nPoints = len(self.pointFacet)

        self.surfaceNodes = list(dict.fromkeys(node for facet in self.facets for node in facet.nodes))

    def allParentNodes(self) -> set:
        """The set of all parent face nodes of the slave surface."""

        return {node for nodes in self.parentNodes for node in nodes}

    def currentPointCoordinates(self, model: FEModel) -> np.ndarray:
        """The current positions of all contact points, on the *curved* slave parent surface.

        Parameters
        ----------
        model
            The model tree, holding the displacement field.

        Returns
        -------
        numpy.ndarray
            The coordinates, of shape (nPoints, nDim).
        """

        nDim = model.domainSize
        coords = np.empty((self.nPoints, nDim))
        for f, nodes in enumerate(self.parentNodes):
            currentCoords = currentNodeCoordinates(nodes, model, self.parentReferenceCoordinates[f])
            first = f * self.nQuadraturePoints
            coords[first : first + self.nQuadraturePoints] = self.shapeFunctions[f] @ currentCoords
        return coords

    def normalPressures(self, normalForces: np.ndarray) -> np.ndarray:
        """Convert the normal forces of the contact points into pressures (positive in compression).

        Parameters
        ----------
        normalForces
            The normal force of each contact point (negative in compression).

        Returns
        -------
        numpy.ndarray
            The normal pressure of each contact point.
        """

        weights = np.concatenate(self.integrationWeights) if self.nPoints else np.zeros(0)
        return np.divide(-normalForces, weights, out=np.zeros_like(normalForces), where=weights > 0)

    def nodalNormalForces(self, normalForces: np.ndarray) -> np.ndarray:
        """Distribute the normal forces of the contact points to the slave surface nodes.

        Positive means repelled; **negative means pulled**, which is exactly the tensile corner load
        a serendipity face requires and a node-based scheme cannot produce. Ordered like
        :attr:`surfaceNodes`. A parent-face node that is not itself a facet node -- possible only
        with ``triangulation=corner`` -- receives contact force but has no slot there, and is
        omitted.

        Parameters
        ----------
        normalForces
            The normal force of each contact point (negative in compression, zero if inactive).

        Returns
        -------
        numpy.ndarray
            The normal force at each slave surface node.
        """

        forceOfNode = dict.fromkeys(self.surfaceNodes, 0.0)
        for p in range(self.nPoints):
            if normalForces[p] == 0.0:
                continue
            f = self.pointFacet[p]
            for node, shapeFunction in zip(self.parentNodes[f], self.shapeFunctions[f][self.pointQuadraturePoint[p]]):
                if node in forceOfNode:
                    forceOfNode[node] += -normalForces[p] * shapeFunction
        return np.array([forceOfNode[node] for node in self.surfaceNodes])

    def isTouchedBy(self, model: FEModel, change: ModelChange) -> bool:
        """Whether a mesh change touched the source elements of the slave surface, so the contact
        points must be rebuilt from the regenerated facets.

        Parameters
        ----------
        model
            The model tree, holding the recipes of the generated contact facet sets.
        change
            The mesh change.

        Returns
        -------
        bool
            True if the slave surface must be rebuilt.
        """

        recipe = model.contactFacetRecipes.get(self.setName)
        return recipe is not None and change.touchesSurface(recipe[0])
