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
from scipy.sparse import csr_matrix

from edelweissfe.constraints.base.contactpoints import ContactPoints, outOfPlaneExtentAt
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

    # Per-node weights cannot be applied here (the pressure is distributed with the parent face's
    # shape functions); accepting them would silently differ from the node-based constraint.
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


class ContactPointsOnSlaveSurface(ContactPoints):
    """The quadrature points of a slave surface, used as contact points.

    Each slave facet carries an ``nQuadraturePoints``-point rule. A point at facet-barycentric
    coordinates :math:`b` sits at parent-face parametric coordinates :math:`\\xi = b \\cdot \\Xi`
    (with :math:`\\Xi` the facet vertices' own parametric coordinates in the parent face, stamped by
    the surface element generator), and carries the constant weight :math:`L_p = w_q J_q`, the quadrature weight
    times the reference measure :math:`J_q` of the facet. Its position follows the *curved* parent
    surface, :math:`x_q = N^s(\\xi_q) \\cdot x^s`. The area it stands for is :math:`L_p` times the out-of-plane
    extent of the body at its origin (:func:`~edelweissfe.constraints.base.contactpoints.outOfPlaneExtentAt`),
    measured from the model when the points are made: times the thickness of the section of its facet's element in a
    plane model, :math:`2 \\pi r_q L_p`, the area of its ring, in an axisymmetric model.

    The contact points are numbered by slave facet and, within a facet, by quadrature point. The faces of the slave
    surface must be of one type (one number of parent face nodes), so that the located points are evaluated as one
    array. A located point takes the nodes of its parent face into the block of the constraint, its own copy of them:
    its interpolation vector is :math:`N^s(\\xi_q)` over them, and it moves with their total displacement, from its
    origin in the reference configuration, :math:`N^s(\\xi_q) \\cdot X^s`.

    Parameters
    ----------
    slaveSurface
        The element set of contact facet elements forming the slave surface.
    nQuadraturePoints
        The number of quadrature points per slave facet.
    model
        The model, whose elements the areas are measured as.
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
        constraint's tributary areas are), in the model's own space: a length in 2D.
    pointFacet : numpy.ndarray
        The slave facet index of each contact point.
    pointQuadraturePoint : numpy.ndarray
        The quadrature point index of each contact point within its facet.
    areas : numpy.ndarray
        The area each contact point stands for: its integration weight (:attr:`integrationWeights`) times the
        out-of-plane extent of the body at its origin.
    """

    def __init__(self, slaveSurface: ElementSet, nQuadraturePoints: int, model: FEModel, constraintName: str):
        self.setName = slaveSurface.name
        self._constraintName = constraintName
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

        self._surfaceNodes = list(dict.fromkeys(node for facet in self.facets for node in facet.nodes))

        nodeCounts = {len(nodes) for nodes in self.parentNodes}
        if len(nodeCounts) > 1:
            raise ValueError(
                f"Constraint '{constraintName}': the slave surface '{self.setName}' mixes parent faces with "
                f"{sorted(nodeCounts)} nodes; a single element type is required, since the contact points of a "
                "slave surface are evaluated as one array."
            )
        self.nDim = self.parentReferenceCoordinates[0].shape[1] if self.facets else 0
        self.nParentNodes = nodeCounts.pop() if nodeCounts else 0

        self.areas = (np.concatenate(self.integrationWeights) if self.nPoints else np.zeros(0)) * outOfPlaneExtentAt(
            self.referenceCoordinates(), [self.facets[f].parentElement for f in self.pointFacet], model
        )
        self.axisymmetric = model.axisymmetric

        self.locatePoints(np.zeros(0, dtype=int))

    @property
    def surfaceNodes(self) -> list:
        """The unique nodes of the slave facets, in the same first-encounter order the surface element generator uses
        for its '<prefix>_nodes' node set -- so a fromExpression field output over that set lines up with
        :meth:`nodalNormalForces`.

        Returns
        -------
        list
            The nodes.
        """

        return self._surfaceNodes

    def describe(self) -> str:
        """The number of points, slave facets and quadrature points per facet, for the journal.

        Returns
        -------
        str
            The description.
        """

        return f"{self.nPoints} points ({len(self.facets)} slave facets x {self.nQuadraturePoints} quadrature points)"

    def finiteElementNodes(self) -> set:
        """All parent face nodes of the slave surface.

        Returns
        -------
        set
            The nodes.
        """

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

        forceOfNode = dict.fromkeys(self._surfaceNodes, 0.0)
        for p in range(self.nPoints):
            if normalForces[p] == 0.0:
                continue
            f = self.pointFacet[p]
            for node, shapeFunction in zip(self.parentNodes[f], self.shapeFunctions[f][self.pointQuadraturePoint[p]]):
                if node in forceOfNode:
                    forceOfNode[node] += -normalForces[p] * shapeFunction
        return np.array([forceOfNode[node] for node in self._surfaceNodes])

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

    def regenerated(self, model: FEModel) -> "ContactPointsOnSlaveSurface":
        """The contact points of the regenerated slave facets, after a change of the mesh touched them.

        Parameters
        ----------
        model
            The model tree, holding the regenerated facet set.

        Returns
        -------
        ContactPointsOnSlaveSurface
            The new contact points.
        """

        return ContactPointsOnSlaveSurface(
            model.elementSets[self.setName], self.nQuadraturePoints, model, self._constraintName
        )

    def referenceCoordinates(self) -> np.ndarray:
        """:math:`N^s(\\xi_q) \\cdot X^s`, all points in the reference configuration, on the *curved* parent faces, in
        the order of the points (by facet, then by quadrature point).

        Returns
        -------
        numpy.ndarray
            The coordinates, of shape (nPoints, nDim).
        """

        return np.concatenate(
            [N @ X for N, X in zip(self.shapeFunctions, self.parentReferenceCoordinates)] + [np.zeros((0, self.nDim))]
        )

    def locatePoints(self, points: np.ndarray) -> list:
        """Locate the given points: each takes the nodes of its parent face into the block, a copy of its own, and its
        shape functions and its origin, :math:`N^s(\\xi_q) \\cdot X^s` in the reference configuration, are gathered
        into arrays, once per update of the connectivity, so that an evaluation is a fixed handful of array operations
        rather than a loop over the points.

        Parameters
        ----------
        points
            The indices of the contact points to locate, ascending.

        Returns
        -------
        list
            The nodes of the parent faces of the points, point after point.
        """

        facets = self.pointFacet[points]
        self._shapeFunctionsOfLocated = np.array(
            [self.shapeFunctions[f][q] for f, q in zip(facets, self.pointQuadraturePoint[points])]
        ).reshape((len(points), self.nParentNodes))
        referenceCoordinatesOfLocated = np.array([self.parentReferenceCoordinates[f] for f in facets]).reshape(
            (len(points), self.nParentNodes, self.nDim)
        )
        self._originsOfLocated = np.einsum("ai,aij->aj", self._shapeFunctionsOfLocated, referenceCoordinatesOfLocated)
        return [node for f in facets for node in self.parentNodes[f]]

    def originsOfLocatedPoints(self) -> np.ndarray:
        """:math:`N^s(\\xi_q) \\cdot X^s`, the points in the reference configuration, on the *curved* parent face.

        Returns
        -------
        numpy.ndarray
            The coordinates, of shape (nLocated, nDim).
        """

        return self._originsOfLocated

    def displacementsOfLocatedPoints(self, U: np.ndarray, dU: np.ndarray) -> np.ndarray:
        """:math:`N^s(\\xi_q) \\cdot U^s`, from the total displacement of the parent face nodes.

        Parameters
        ----------
        U
            The total displacement of the slave part of the block; the points move with it.
        dU
            The increment of the slave part of the block; unused.

        Returns
        -------
        numpy.ndarray
            The displacements, of shape (nLocated, nDim).
        """

        return np.einsum(
            "ai,aij->aj",
            self._shapeFunctionsOfLocated,
            U.reshape((len(self._shapeFunctionsOfLocated), self.nParentNodes, self.nDim)),
        )

    def addForceResidual(self, P: np.ndarray, points: np.ndarray, forces: np.ndarray, directions: np.ndarray):
        """Add :math:`f\\, N^s \\otimes n` to the parent face nodes of each given point; see
        :meth:`ContactPoints.addForceResidual`.

        Parameters
        ----------
        P
            The slave part of the residual contribution; augmented.
        points
            The located points that carry a force.
        forces
            The forces of their springs.
        directions
            The unit directions of their springs.
        """

        Ns = self._shapeFunctionsOfLocated[points]
        w = (Ns[:, :, None] * directions[:, None, :]).reshape((len(points), -1))
        P.reshape((-1, self.nParentNodes * self.nDim))[points] += forces[:, None] * w

    def interpolationOfLocatedPoint(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """The degrees of freedom of the parent face nodes of a located point, and its shape functions there.

        Parameters
        ----------
        k
            The index of the located point.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            The degrees of freedom and the interpolation vector.
        """

        nDof = self.nParentNodes * self.nDim
        return np.arange(k * nDof, (k + 1) * nDof), self._shapeFunctionsOfLocated[k]

    def interpolationMatrixOfLocatedPoints(self) -> csr_matrix:
        """The shape functions :math:`N^s(\\xi_q)` of the located points, each over its own copy of the nodes of its
        parent face; see :meth:`ContactPoints.interpolationMatrixOfLocatedPoints`.

        Returns
        -------
        scipy.sparse.csr_matrix
            The block diagonal matrix of the shape functions.
        """

        nLocated = len(self._shapeFunctionsOfLocated)
        return csr_matrix(
            (
                self._shapeFunctionsOfLocated.ravel(),
                np.arange(nLocated * self.nParentNodes),
                np.arange(0, nLocated * self.nParentNodes + 1, self.nParentNodes),
            ),
            shape=(nLocated, nLocated * self.nParentNodes),
        )

    def searchLayout(self) -> dict[str, np.ndarray]:
        """The slave facets and the quadrature rule identify the contact points in a checkpoint.

        Returns
        -------
        dict[str, numpy.ndarray]
            The element numbers of the slave facets, and the number of quadrature points per facet.
        """

        return {
            "slaveFacets": np.array([facet.elNumber for facet in self.facets], dtype=np.int64),
            "nQuadraturePoints": np.array([self.nQuadraturePoints], dtype=np.int64),
        }
