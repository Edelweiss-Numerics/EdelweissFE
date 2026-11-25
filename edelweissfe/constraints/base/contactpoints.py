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
The contact points of a contact constraint: the points of the slave side at which contact is enforced.

A contact constraint pairs points of a slave body with a master surface, measures the gap of each point along the
master's normal, and turns the gap into a force (see :mod:`~edelweissfe.constraints.base.penaltylaw`). How the points
move, which nodes their force goes to, and the area each of them stands for depends on the kind of point, and on
nothing else:

- a node of a finite element: :math:`c = [1]`, at :math:`\\boldsymbol{x} = \\boldsymbol{X} + \\boldsymbol{U}`;
- a quadrature point :math:`\\xi_q` of a finite element face: :math:`c = N^s(\\xi_q)`, the shape functions of the
  face, at :math:`\\boldsymbol{x} = N^s(\\xi_q) \\cdot (\\boldsymbol{X}^s + \\boldsymbol{U}^s)`.

A point moves with its interpolation vector :math:`c` and its force is distributed to the nodes with the same vector,
the transpose of its motion: then the penalty spring at the point has a symmetric stiffness and does no work over a
closed cycle.

A point's position is told in two parts: its *origin* :math:`\\boldsymbol{x}^0`, where it is when the displacement it
moves with is zero, and its *displacement* :math:`c \\cdot \\boldsymbol{u}` from there. For the points of finite
elements, :math:`\\boldsymbol{u}` is ``U``, the total displacement of their nodes, and the origin is the point in the
reference configuration. A kind of point whose nodes carry the increment rather than the total displacement (points
on the faces of updated-Lagrangian particles) fits the same interface: for it, :math:`\\boldsymbol{u}` would be ``dU``,
the increment, and the origin the point at the start of the increment. A constraint therefore passes both, and each
kind takes the one it moves with. The split also lets a constraint that measures a small distance between two points
-- the stretch of a stiff penalty spring -- take the difference of the origins once and the difference of the
displacements in every evaluation, rather than the difference of the coordinates, whose round-off scales with the
coordinates (see :class:`~edelweissfe.constraints.surfacetodeformablesurfacepenalty.Constraint`, *Measured from the
origins*).

:class:`ContactPoints` is what a constraint needs from its contact points; a constraint written against it -- the
pairing with the master, the gap, the law -- is written once for every kind of point. Two kinds are here: the nodes of
a slave surface (:mod:`~edelweissfe.constraints.base.contactpointsatslavenodes`) and the quadrature points of its faces
(:mod:`~edelweissfe.constraints.base.contactpointsonslavesurface`). The contact with a deformable master surface,
:mod:`~edelweissfe.constraints.surfacetodeformablesurfacepenalty`, and the contact with a discrete rigid body,
:mod:`~edelweissfe.constraints.surfacetodiscreterigidbodypenalty`, are written against it, each once for both kinds.
"""

from abc import ABC, abstractmethod

import numpy as np
from scipy.sparse import csr_matrix

from edelweissfe.models.femodel import FEModel
from edelweissfe.models.modelchange import ModelChange


class ContactPoints(ABC):
    """The contact points of the slave side of a contact constraint.

    A constraint uses them so. In every update of the connectivity, it asks for the current coordinates of all points
    (:meth:`currentPointCoordinates`), pairs those within reach with the master surface, and then *locates* the paired
    ones (:meth:`locatePoints`): they name the nodes they act on, which become the slave part of the constraint's
    block. Within the increment the constraint evaluates the located points only: where they are, from their origins
    (:meth:`originsOfLocatedPoints`) and their displacements (:meth:`displacementsOfLocatedPoints`), the residual
    contribution of the forces on them
    (:meth:`addForceResidual`) and, for the tangent, their interpolation vectors
    (:meth:`interpolationOfLocatedPoint`), all at once for the bound of the stiffness
    (:meth:`interpolationMatrixOfLocatedPoints`).

    Attributes
    ----------
    nPoints : int
        The number of contact points.
    areas : numpy.ndarray
        The area each contact point stands for, in the reference configuration; the penalty modulus per unit area times
        this area is the stiffness of the point's spring.
    """

    nPoints: int
    areas: np.ndarray

    #: Whether the points can be located only once the model has connected the entities they belong to, at an update
    #: of the connectivity: points whose nodes follow from where they are (the kernel functions of updated-Lagrangian
    #: particles). A checkpoint does not hold that connectivity, so a constraint restored from one keeps their
    #: projection and locates them at the next update of the connectivity, which the resumed run makes before its
    #: first increment. False for the points of finite elements, which act on nodes of their own.
    locatedOnlyWhenConnected: bool = False

    @abstractmethod
    def currentPointCoordinates(self, model: FEModel) -> np.ndarray:
        """The coordinates of all contact points in the configuration of the model, for the contact search.

        Parameters
        ----------
        model
            The model.

        Returns
        -------
        numpy.ndarray
            The coordinates, of shape (nPoints, nDim).
        """

    @abstractmethod
    def locatePoints(self, points: np.ndarray) -> list:
        """Locate the given contact points in the block of the constraint: the nodes their interpolation vectors act
        on, in the order they take in the block. The located points are the ones the evaluation methods below refer
        to, in the order given here.

        Parameters
        ----------
        points
            The indices of the contact points to locate, ascending.

        Returns
        -------
        list
            The nodes, the slave part of the constraint's block; a node may appear more than once.
        """

    @abstractmethod
    def originsOfLocatedPoints(self) -> np.ndarray:
        """The origins of the located points: where they are when the displacement they move with is zero -- in the
        reference configuration for the points of finite elements, which move with the total displacement ``U``; at
        the start of the increment for points whose nodes carry the increment ``dU``. Constant from one location to
        the next.

        Returns
        -------
        numpy.ndarray
            The coordinates, of shape (nLocated, nDim).
        """

    @abstractmethod
    def displacementsOfLocatedPoints(self, U: np.ndarray, dU: np.ndarray) -> np.ndarray:
        """The displacements of the located points from their origins (:meth:`originsOfLocatedPoints`):
        :math:`C \\boldsymbol{u}`, the interpolation vectors of the points times the displacement
        :math:`\\boldsymbol{u}` they move with -- ``U`` for the points of finite elements, ``dU`` for points whose
        nodes carry the increment. Both are passed; each kind takes the one it moves with.

        Parameters
        ----------
        U
            The total displacement of the slave part of the constraint's block.
        dU
            The increment of the slave part of the constraint's block.

        Returns
        -------
        numpy.ndarray
            The displacements, of shape (nLocated, nDim).
        """

    def coordinatesOfLocatedPoints(self, U: np.ndarray, dU: np.ndarray) -> np.ndarray:
        """The current coordinates of the located points: their origins plus their displacements.

        Parameters
        ----------
        U
            The total displacement of the slave part of the constraint's block.
        dU
            The increment of the slave part of the constraint's block.

        Returns
        -------
        numpy.ndarray
            The coordinates, of shape (nLocated, nDim).
        """

        return self.originsOfLocatedPoints() + self.displacementsOfLocatedPoints(U, dU)

    @abstractmethod
    def addForceResidual(self, P: np.ndarray, points: np.ndarray, forces: np.ndarray, directions: np.ndarray):
        """Add the residual contribution of the forces of penalty springs at located points, :math:`f\\, c \\otimes n`
        per point: the negative of the force :math:`-f\\, n` on the point, distributed with its interpolation vector.
        A spring acts along a unit direction :math:`n`: the normal of a contact, an axis of a tie.

        Parameters
        ----------
        P
            The slave part of the constraint's residual contribution; augmented.
        points
            The located points that carry a force (indices into the located points).
        forces
            The forces :math:`f` of their springs, the normal forces of a contact (negative in contact).
        directions
            The unit directions of their springs, of shape (len(points), nDim).
        """

    @abstractmethod
    def interpolationOfLocatedPoint(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Where a located point sits in the slave part of the block, and how it moves with it.

        Parameters
        ----------
        k
            The index of the located point.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            Its degrees of freedom in the slave part of the block, node by node, and its interpolation vector :math:`c`
            over these nodes.
        """

    @abstractmethod
    def interpolationMatrixOfLocatedPoints(self) -> csr_matrix:
        """The interpolation vectors :math:`c` of all located points, one row per point, over the nodes of the slave
        part of the block: the matrix :math:`C` with :math:`\\boldsymbol{x}_p = (C \\boldsymbol{x})_p`.

        Returns
        -------
        scipy.sparse.csr_matrix
            The matrix, of shape (nLocated, number of nodes of the slave part of the block).
        """

    @abstractmethod
    def searchLayout(self) -> dict[str, np.ndarray]:
        """What identifies the contact points in a checkpoint, see
        :class:`~edelweissfe.constraints.base.frozencontactsearch.FrozenContactSearch`.

        Returns
        -------
        dict[str, numpy.ndarray]
            The identifying arrays.
        """

    def isTouchedBy(self, model: FEModel, change: ModelChange) -> bool:
        """Whether a change of the mesh touched the contact points, so that they must be regenerated
        (:meth:`regenerated`). By default no change does.

        Parameters
        ----------
        model
            The model.
        change
            The change of the mesh.

        Returns
        -------
        bool
            True if the contact points must be regenerated.
        """

        return False

    def regenerated(self, model: FEModel) -> "ContactPoints":
        """The contact points after a change of the mesh that touched them (:meth:`isTouchedBy`).

        Parameters
        ----------
        model
            The model.

        Returns
        -------
        ContactPoints
            The new contact points.
        """

        raise NotImplementedError(f"{type(self).__name__} is never touched by a change of the mesh.")

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

        return np.divide(-normalForces, self.areas, out=np.zeros_like(normalForces), where=self.areas > 0)

    def describe(self) -> str:
        """What the contact points are, for the journal.

        Returns
        -------
        str
            A short description.
        """

        return f"{self.nPoints} contact points"

    def finiteElementNodes(self) -> set:
        """The nodes of finite elements the contact points can act on, so that a constraint can refuse a master
        surface on the same nodes (self-contact). The empty set for points that act on no finite element nodes.

        Returns
        -------
        set
            The nodes.
        """

        return set()

    @property
    def surfaceNodes(self) -> list:
        """The nodes of the slave surface, in the order of :meth:`nodalNormalForces`. Points that act on no finite
        element nodes have none.

        Returns
        -------
        list
            The nodes.
        """

        raise TypeError(f"{type(self).__name__} has no surface nodes.")

    def nodalNormalForces(self, normalForces: np.ndarray) -> np.ndarray:
        """The normal forces of the contact points, distributed to the nodes of the slave surface. Points that act on
        no finite element nodes have no surface nodes.

        Parameters
        ----------
        normalForces
            The normal force of each contact point.

        Returns
        -------
        numpy.ndarray
            The normal force at each slave surface node.
        """

        raise TypeError(f"{type(self).__name__} has no surface nodes to distribute the normal forces to.")
