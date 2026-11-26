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
The contact points of a node-based contact formulation: the nodes of a slave surface, each standing for its tributary
area.
"""

import numpy as np
from scipy.sparse import csr_matrix, identity

from edelweissfe.constraints.base.contactpoints import ContactPoints, outOfPlaneExtentAt
from edelweissfe.models.femodel import FEModel
from edelweissfe.models.modelchange import ModelChange
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.utils.meshtools import currentNodeCoordinates


def tributaryAreasOfSlaveNodes(slaveFacets: list, model: FEModel) -> tuple[list, np.ndarray]:
    """The unique nodes of slave contact facets and the area each stands for, in the reference configuration, as the
    elements integrate: the sum over its incident facets of its share :math:`s_{s,f}` of the measure of each facet
    (``nodalAreaShares``, set by the surface element generator) times the out-of-plane extent of the body at the node
    on that facet's element, :math:`A_s = \\sum_f b_f(\\boldsymbol{X}_s)\\, s_{s,f}`
    (:func:`~edelweissfe.constraints.base.contactpoints.outOfPlaneExtentAt`): a node between elements of two
    thicknesses stands for its share of the area of each, a node of an axisymmetric model for its share of the rings.

    Parameters
    ----------
    slaveFacets
        The slave contact facet elements.
    model
        The model, with the sections of the elements the facets bound.

    Returns
    -------
    tuple[list, numpy.ndarray]
        The nodes, in first-encounter order (the order of the generator's '<prefix>_nodes' node set), and their
        tributary areas. A node with a zero share (``nodalWeights=serendipityOptimal``) is kept, inert, so that the
        order is unaffected.
    """

    nodesOnFacets = [
        (facet, node, share) for facet in slaveFacets for node, share in zip(facet.nodes, facet.nodalAreaShares)
    ]
    extents = outOfPlaneExtentAt(
        np.array([node.coordinates for _, node, _ in nodesOnFacets]).reshape(len(nodesOnFacets), model.domainSize),
        [facet.parentElement for facet, _, _ in nodesOnFacets],
        model,
    )

    tributaryAreaOfSlaveNode = {}
    for (_, node, share), extent in zip(nodesOnFacets, extents):
        tributaryAreaOfSlaveNode[node] = tributaryAreaOfSlaveNode.get(node, 0.0) + share * extent
    return list(tributaryAreaOfSlaveNode.keys()), np.array(list(tributaryAreaOfSlaveNode.values()))


class ContactPointsAtSlaveNodes(ContactPoints):
    """The nodes of a slave surface, used as contact points.

    A node is its own contact point: its interpolation vector is :math:`c = [1]`, it sits at
    :math:`\\boldsymbol{X} + \\boldsymbol{U}` (its origin :math:`\\boldsymbol{X}`, its total displacement
    :math:`\\boldsymbol{U}`), and it stands for its tributary area (:func:`tributaryAreasOfSlaveNodes`), measured
    from the model when the nodes are taken: times the thickness in a plane model, the area of its share of the rings
    in an axisymmetric model. A located node takes itself into the block of the constraint, once.

    Parameters
    ----------
    slaveSurface
        The element set of contact facet elements forming the slave surface.
    model
        The model, whose elements the areas are measured as.
    constraintName
        The name of the owning constraint, for error messages.
    """

    def __init__(self, slaveSurface: ElementSet, model: FEModel, constraintName: str):
        self.setName = slaveSurface.name
        self._constraintName = constraintName
        self.slaveNodes, self.areas = tributaryAreasOfSlaveNodes(list(slaveSurface), model)
        self.axisymmetric = model.axisymmetric
        self.nPoints = len(self.slaveNodes)
        self._referenceCoordinates = np.array([node.coordinates for node in self.slaveNodes])
        self.locatePoints(np.zeros(0, dtype=int))

    @property
    def surfaceNodes(self) -> list:
        """The slave nodes, in the order of the surface element generator's '<prefix>_nodes' node set.

        Returns
        -------
        list
            The nodes.
        """

        return self.slaveNodes

    def describe(self) -> str:
        """The number of slave nodes, for the journal.

        Returns
        -------
        str
            The description.
        """

        return f"{self.nPoints} slave nodes"

    def finiteElementNodes(self) -> set:
        """The slave nodes.

        Returns
        -------
        set
            The nodes.
        """

        return set(self.slaveNodes)

    def currentPointCoordinates(self, model: FEModel) -> np.ndarray:
        """The current coordinates of the slave nodes.

        Parameters
        ----------
        model
            The model tree, holding the displacement field.

        Returns
        -------
        numpy.ndarray
            The coordinates, of shape (nPoints, nDim).
        """

        return currentNodeCoordinates(self.slaveNodes, model, self._referenceCoordinates)

    def locatePoints(self, points: np.ndarray) -> list:
        """Locate the given nodes: each takes itself into the block.

        Parameters
        ----------
        points
            The indices of the slave nodes to locate, ascending.

        Returns
        -------
        list
            The nodes.
        """

        self._referenceCoordinatesOfLocated = self._referenceCoordinates[points]
        return [self.slaveNodes[p] for p in points]

    def originsOfLocatedPoints(self) -> np.ndarray:
        """:math:`\\boldsymbol{X}`, the reference coordinates of the located nodes.

        Returns
        -------
        numpy.ndarray
            The coordinates, of shape (nLocated, nDim).
        """

        return self._referenceCoordinatesOfLocated

    def displacementsOfLocatedPoints(self, U: np.ndarray, dU: np.ndarray) -> np.ndarray:
        """:math:`\\boldsymbol{U}`, the total displacement of the located nodes.

        Parameters
        ----------
        U
            The total displacement of the slave part of the block; the nodes move with it.
        dU
            The increment of the slave part of the block; unused.

        Returns
        -------
        numpy.ndarray
            The displacements, of shape (nLocated, nDim).
        """

        return U.reshape(self._referenceCoordinatesOfLocated.shape)

    def addForceResidual(self, P: np.ndarray, points: np.ndarray, forces: np.ndarray, directions: np.ndarray):
        """Add :math:`f\\, n` at each given node; see :meth:`ContactPoints.addForceResidual`.

        Parameters
        ----------
        P
            The slave part of the residual contribution; augmented.
        points
            The located nodes that carry a force.
        forces
            The forces of their springs.
        directions
            The unit directions of their springs.
        """

        P.reshape(self._referenceCoordinatesOfLocated.shape)[points] += forces[:, None] * directions

    def interpolationOfLocatedPoint(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """The degrees of freedom of a located node, and :math:`c = [1]`.

        Parameters
        ----------
        k
            The index of the located node.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            The degrees of freedom and the interpolation vector.
        """

        nDim = self._referenceCoordinates.shape[1]
        return np.arange(k * nDim, (k + 1) * nDim), np.ones(1)

    def interpolationMatrixOfLocatedPoints(self) -> csr_matrix:
        """The identity: every located node is its own point; see
        :meth:`ContactPoints.interpolationMatrixOfLocatedPoints`.

        Returns
        -------
        scipy.sparse.csr_matrix
            The identity over the located nodes.
        """

        return identity(len(self._referenceCoordinatesOfLocated), format="csr")

    def searchLayout(self) -> dict[str, np.ndarray]:
        """The labels of the slave nodes identify the contact points in a checkpoint.

        Returns
        -------
        dict[str, numpy.ndarray]
            The node labels.
        """

        return {"slaveNodes": np.array([node.label for node in self.slaveNodes], dtype=np.int64)}

    def nodalNormalForces(self, normalForces: np.ndarray) -> np.ndarray:
        """The normal force at each slave node, positive when repelled.

        Parameters
        ----------
        normalForces
            The normal force of each node (negative in compression).

        Returns
        -------
        numpy.ndarray
            The normal force at each slave node.
        """

        return -normalForces

    def isTouchedBy(self, model: FEModel, change: ModelChange) -> bool:
        """Whether a change of the mesh touched the source elements of the slave surface.

        Parameters
        ----------
        model
            The model tree, holding the recipes of the generated contact facet sets.
        change
            The change of the mesh.

        Returns
        -------
        bool
            True if the slave nodes must be regenerated.
        """

        recipe = model.contactFacetRecipes.get(self.setName)
        return recipe is not None and change.touchesSurface(recipe[0])

    def regenerated(self, model: FEModel) -> "ContactPointsAtSlaveNodes":
        """The slave nodes of the regenerated slave facets.

        Parameters
        ----------
        model
            The model tree, holding the regenerated facet set.

        Returns
        -------
        ContactPointsAtSlaveNodes
            The new contact points.
        """

        return ContactPointsAtSlaveNodes(model.elementSets[self.setName], model, self._constraintName)

    def referenceCoordinates(self) -> np.ndarray:
        """:math:`\\boldsymbol{X}`, the reference coordinates of all slave nodes.

        Returns
        -------
        numpy.ndarray
            The coordinates, of shape (nPoints, nDim).
        """

        return self._referenceCoordinates
