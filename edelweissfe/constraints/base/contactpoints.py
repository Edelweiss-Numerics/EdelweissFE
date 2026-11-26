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
reference configuration. A kind of point whose nodes carry the increment rather than the total displacement fits the
same interface: for it, :math:`\\boldsymbol{u}` is ``dU``, the increment, and the origin the point at the start of the
increment. EdelweissMeshfree adds such a kind, the points on the faces of its updated-Lagrangian particles
(``ContactPointsOnParticleFaces``). A constraint therefore passes both, and each
kind takes the one it moves with. The split also lets a constraint that measures a small distance between two points
-- the stretch of a stiff penalty spring -- take the difference of the origins once and the difference of the
displacements in every evaluation, rather than the difference of the coordinates, whose round-off scales with the
coordinates (see :ref:`contact-origins`).

Every contact point stands for an area :math:`A_p`, its share of the surface of the body: the penalty per unit area
times it is the stiffness of its spring. The area is the one the elements integrate over. They integrate over the body
with :math:`\\mathrm{d}V = b\\, \\mathrm{d}A`, with the out-of-plane extent :math:`b` of the body at a point
(:func:`outOfPlaneExtentAt`): :math:`b = 1` in a 3D model, the thickness :math:`t` of the plane section in a **plane
model**, the circumference :math:`2 \\pi r` at the radius :math:`r` in an **axisymmetric model**, the section r-z of
a body of revolution. So the surface of the body is :math:`\\mathrm{d}S = b\\, \\mathrm{d}L`, and a contact point
stands for its measure :math:`L_p` in the model's own space -- an area in 3D, a length in 2D: its quadrature weight
times the Jacobian of its face, or the tributary measure of a node -- times :math:`b` at the point,

.. math::
    A_p = b(\\boldsymbol{X}_p) \\, L_p .

A node sums this over its faces, :math:`A_s = \\sum_f b_f(\\boldsymbol{X}_s)\\, s_{s,f}` with its shares
:math:`s_{s,f}` of their measures: a node between faces of two thicknesses stands for its share of the area of each.
The springs then act on the same body as the elements -- in an axisymmetric model on the full rings -- and the two
balance at the nodes. A point on the axis of an axisymmetric model stands for a ring of radius 0, no area: the axis is
a line, not a surface of the body. The pressure and the traction of a point are its force per :math:`A_p`, the area in
the reference configuration: the nominal pressure, not per radian, not per unit thickness. The axial forces of the
points of an axisymmetric model add up to the total axial force; the radial forces of a ring cancel about the axis, so
a sum of radial components is no resultant.

Every kind of point measures its areas when its points are made. The two kinds of finite element points take the model
and measure with :func:`outOfPlaneExtentAt`. A kind of point of another library measures them as its own bodies
integrate -- in an axisymmetric model with :func:`circumferenceAt`, the same rule of the axis --, and says with
:attr:`ContactPoints.axisymmetric` whether it was made for an axisymmetric model; a constraint refuses points made for a
model of another kind (:func:`refusePointsMadeForAnotherKindOfModel`). Whether the bodies of such a kind are slices of
the same thickness as the finite elements they meet in a plane model is not checked: that is for the model to ensure.

:class:`ContactPoints` is what a constraint needs from its contact points; a constraint written against it -- the
pairing with the master, the gap, the law -- is written once for every kind of point. Two kinds are here: the nodes of
a slave surface (:mod:`~edelweissfe.constraints.base.contactpointsatslavenodes`) and the quadrature points of its faces
(:mod:`~edelweissfe.constraints.base.contactpointsonslavesurface`). The contact with a deformable master surface,
:mod:`~edelweissfe.constraints.surfacetodeformablesurfacepenalty`, and the contact with a discrete rigid body,
:mod:`~edelweissfe.constraints.surfacetodiscreterigidbodypenalty`, are written against it, each once for both kinds.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np
from scipy.sparse import csr_matrix

from edelweissfe.models.femodel import FEModel
from edelweissfe.models.modelchange import ModelChange

#: How far below the axis of an axisymmetric model a point may lie, relative to the size :math:`h` of its element, and
#: still be on the axis: :math:`r \\geq -\\epsilon\\, h` is the round-off of a mesher and counts as :math:`r = 0`,
#: :math:`r < -\\epsilon\\, h` is a mesh drawn across the axis and is refused. It is the rule of Marmot's axisymmetric
#: elements for their nodes (``Spatial2D::Axisymmetric::axisTolerance``, with :math:`h` the diagonal of the bounding box
#: of the element), so that a contact point accepts the nodes its element accepts.
axisTolerance = 1e-12


def outOfPlaneExtentAt(coordinates: np.ndarray, elements: list, model: FEModel) -> np.ndarray:
    """The out-of-plane extent :math:`b` of the body at points on the boundaries of elements: the factor of
    :math:`\\mathrm{d}V = b\\, \\mathrm{d}A` the elements integrate with, and so the factor of the area of a contact
    point, :math:`A_p = b(\\boldsymbol{X}_p)\\, L_p` (see the module documentation):

    ========================  =================================================  ======================
    model                     the elements integrate                             :math:`b`
    ========================  =================================================  ======================
    3D                        over the volume                                    1
    plane                     over the slice of the thickness of their section   :math:`t`
    axisymmetric              over the rings about the axis                      :math:`2 \\pi r`
    ========================  =================================================  ======================

    In a plane model, :math:`t` is the thickness of the section of the element
    (:meth:`~edelweissfe.models.femodel.FEModel.thicknessOf`). In an axisymmetric model, :math:`b` is the
    circumference at the radius of the point (:func:`circumferenceAt`), with the size of the element for the round-off
    of the axis.

    Parameters
    ----------
    coordinates
        The points, (nPoints, nDim), in the reference configuration.
    elements
        For each point, the element on whose boundary it lies (the parent element of its contact facet): its section
        gives the thickness in a plane model, its size the round-off of the axis in an axisymmetric model.
    model
        The model.

    Returns
    -------
    numpy.ndarray
        :math:`b` at each point.

    Raises
    ------
    ValueError
        If a point of a 2D model lies on no element (a contact facet not made by the surface element generator), or a
        point of an axisymmetric model lies below the axis by more than the round-off.
    """

    if model.domainSize != 2 or not len(elements):
        return np.ones(len(elements))
    if any(element is None for element in elements):
        raise ValueError(
            "A contact point of a 2D model lies on a contact facet that knows no element (one not made by the surface "
            "element generator); the element tells the thickness, or the round-off of the axis."
        )

    if not model.axisymmetric:
        thicknessOf = {element: model.thicknessOf(element) for element in dict.fromkeys(elements)}
        return np.array([thicknessOf[element] for element in elements], dtype=float)

    sizeOf = {element: _sizeOf(element) for element in dict.fromkeys(elements)}
    return circumferenceAt(
        coordinates[:, 0],
        np.array([sizeOf[element] for element in elements]),
        lambda p: f"element {elements[p].elNumber}",
    )


def circumferenceAt(radii: np.ndarray, sizes: np.ndarray, bodyOf: Callable[[int], str]) -> np.ndarray:
    """The circumference :math:`2 \\pi r` of the ring about the axis of an axisymmetric model at each point: the
    out-of-plane extent :math:`b` of an axisymmetric body (:func:`outOfPlaneExtentAt`). The round-off of the axis
    scales with the size :math:`h` of the body a point lies on: :math:`r \\geq -\\epsilon\\, h` is on the axis,
    :math:`b = 0`, and a point further below is refused (:data:`axisTolerance`). Every kind of contact point of an
    axisymmetric model takes :math:`b` from here: the kinds of finite elements through :func:`outOfPlaneExtentAt`,
    with the sizes of their elements, a kind of another library with the sizes of its own bodies.

    Parameters
    ----------
    radii
        The radius :math:`r` of each point, its first coordinate in the reference configuration, (nPoints,).
    sizes
        The size :math:`h` of the body each point lies on, the diagonal of the bounding box of its nodes or vertices,
        (nPoints,).
    bodyOf
        What point ``p`` lies on, for the message of a refusal: ``bodyOf(p)`` reads, for example, ``"element 7"``.

    Returns
    -------
    numpy.ndarray
        :math:`2 \\pi r` at each point, 0 on the axis.

    Raises
    ------
    ValueError
        If a point lies below the axis by more than the round-off.
    """

    belowTheAxis = radii < -axisTolerance * sizes
    if belowTheAxis.any():
        p = np.flatnonzero(belowTheAxis)[0]
        raise ValueError(
            f"A contact point of an axisymmetric model lies at the radius r = {radii[p]:.3e} < 0, on {bodyOf(p)}; the "
            "section r-z of a body of revolution lies in r >= 0."
        )
    return 2.0 * np.pi * np.maximum(radii, 0.0)


def _sizeOf(element) -> float:
    """The size of an element: the diagonal of the bounding box of its nodes, as Marmot's axisymmetric elements measure
    it for :data:`axisTolerance`."""

    coordinates = np.array([node.coordinates for node in element.nodes])
    return float(np.linalg.norm(coordinates.max(axis=0) - coordinates.min(axis=0)))


def refusePointsMadeForAnotherKindOfModel(contactPoints: "ContactPoints", model: FEModel, constraintName: str):
    """Refuse contact points made for a model of another kind than the constraint's: their areas would be those of
    rings in a plane model, or of slices in an axisymmetric one (:attr:`ContactPoints.axisymmetric`).

    Parameters
    ----------
    contactPoints
        The contact points the constraint takes.
    model
        The model of the constraint.
    constraintName
        The name of the constraint, for the error message.

    Raises
    ------
    ValueError
        If the points were made for a model of another kind.
    """

    if contactPoints.axisymmetric != model.axisymmetric:
        kind = "axisymmetric" if model.axisymmetric else "not axisymmetric"
        raise ValueError(
            f"Constraint '{constraintName}': its model is {kind}, but the contact points were made with "
            f"axisymmetric={contactPoints.axisymmetric}; their areas are those of another body."
        )


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
        The area each contact point stands for, in the reference configuration, measured when the points are made: its
        measure times the out-of-plane extent of the body at it (:func:`outOfPlaneExtentAt`); the penalty modulus per
        unit area times this area is the stiffness of the point's spring.
    """

    nPoints: int
    areas: np.ndarray

    #: True if the points were made for an axisymmetric model, the section r-z of a body of revolution, where each
    #: stands for the ring it sweeps about the axis; a constraint refuses points made for a model of another kind
    #: (:func:`refusePointsMadeForAnotherKindOfModel`).
    axisymmetric: bool = False

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

    def referenceCoordinates(self) -> np.ndarray:
        """Where all the contact points are in the reference configuration, for a kind of contact points that a change
        of the mesh regenerates (:meth:`regenerated`): what a constraint carries the state of the old points over to
        the new ones by (a tie, see :mod:`~edelweissfe.constraints.surfacetodeformablesurfacepenalty`).

        Returns
        -------
        numpy.ndarray
            The coordinates, of shape (nPoints, nDim).
        """

        raise NotImplementedError(f"{type(self).__name__} is never touched by a change of the mesh.")

    def normalPressures(self, normalForces: np.ndarray) -> np.ndarray:
        """Convert the normal forces of the contact points into pressures (positive in compression), per the areas of
        the points (:meth:`perArea`).

        Parameters
        ----------
        normalForces
            The normal force of each contact point (negative in compression).

        Returns
        -------
        numpy.ndarray
            The normal pressure of each contact point.
        """

        return self.perArea(-normalForces)

    def perArea(self, forces: np.ndarray) -> np.ndarray:
        """Forces on the contact points per the areas they stand for: nominal pressures or tractions. A point that
        stands for no area, on the axis of an axisymmetric model, carries no force and reports zero.

        Parameters
        ----------
        forces
            The force on each contact point, (nPoints,) or (nPoints, nDim).

        Returns
        -------
        numpy.ndarray
            The forces per area, of the shape of ``forces``.
        """

        areas = self.areas.reshape((-1,) + (1,) * (forces.ndim - 1))
        return np.divide(forces, areas, out=np.zeros_like(forces), where=areas > 0)

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
