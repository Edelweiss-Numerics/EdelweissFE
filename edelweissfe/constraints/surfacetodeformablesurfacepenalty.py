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

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from scipy.spatial import cKDTree

from edelweissfe.constraints.base.constraintbase import ConstraintBase
from edelweissfe.constraints.base.contactpoints import ContactPoints
from edelweissfe.constraints.base.contactpointsonslavesurface import (
    checkFacetsCarryParentFaces,
)
from edelweissfe.constraints.base.forcesonlyexplicitevaluation import (
    ForcesOnlyExplicitEvaluation,
)
from edelweissfe.constraints.base.frozencontactsearch import FrozenContactSearch
from edelweissfe.constraints.base.penaltylaw import penaltyLaw, penaltyLawsAndTheTie
from edelweissfe.constraints.base.penaltysprings import (
    PenaltySprings,
    SpringGradientOnOwnNodes,
    SpringGradientOnSharedNodes,
    addBoundOfAbsoluteStiffnessOfSprings,
    smallestGapsWithin,
)
from edelweissfe.constraints.base.surfacecontactpenaltyschema import (
    SurfaceContactPenaltySchema,
    contactPointsOfTheSlaveSurface,
)
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.models.meshdependent import MeshDependent
from edelweissfe.numerics.systemmatrixcontributor import localEntriesOfSubBlocks
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.solvers.base.criticaltimestep import NextIncrement
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.facetcontactgeometry import (
    FacetedSurface,
    closestFacetCandidates,
    closestFacets,
    facetNormalAndMeasure,
)
from edelweissfe.utils.meshtools import currentNodeCoordinates, currentNodeDisplacements
from edelweissfe.utils.parentfacegeometry import parentFaceShapeFunctions
from edelweissfe.utils.schema import buildSchemaFromOptions, schemaField

"""
A penalty based unilateral contact constraint between the contact points of a slave body and a deformable master
surface, whose force is distributed with the *parent element face* shape functions of the master -- for every kind of
contact point: the quadrature points of the faces of finite elements (a Gauss-point-to-segment, GPTS, formulation) and
the nodes of finite elements. What a kind of point contributes -- how it moves, which nodes its force goes to, the area
it stands for -- is its :class:`~edelweissfe.constraints.base.contactpoints.ContactPoints`; everything else is written
once, here.

Why the quadrature points exist alongside :mod:`~edelweissfe.constraints.nodetodeformablesurfacepenalty`: a
node-based penalty scheme cannot transmit a correct pressure across a *serendipity* (quad8/hexa20)
face, and no choice of nodal weights can repair it. The consistent nodal loads of a uniform
pressure :math:`p` on a quad8 face of area :math:`A` are :math:`-pA/12` at the corners and
:math:`+pA/3` at the midsides -- the corner load is *tensile*, while a unilateral spring can only
push. The discrete solution resolves that contradiction by opening the corner gaps: on a matched
hexa20 patch test all corner nodes lift off, carrying no load, and the transmitted force falls ~1.7%
short with a visibly undulating interface. See issue #112.

Integrating instead removes the contradiction at its root. The pressure is evaluated *pointwise*
at quadrature points, where it is non-negative as physics requires, and is then distributed with
the parent face's own shape functions, so the nodal force at node :math:`a`,

.. math::
    f_a = \\sum_q w_q \\, J_q \\, p_q \\, N_a(\\xi_q) ,

is negative wherever :math:`N_a` is -- a tensile *nodal* force arising from a non-negative
*pressure field*, with no unilateral condition ever imposed at a corner node. Note that quadrature
alone would achieve nothing: the flat facets' own linear shape functions are non-negative, so
distributing with those would leave the corners lifting off exactly as before. The parent-face
basis (:mod:`~edelweissfe.utils.parentfacegeometry`) is the load-bearing half of the formulation.

Scope of this implementation: frictionless normal penalty contact under ``sliding=small``, for both the implicit
and the explicit solver path. The springs come from an exchangeable law
(:mod:`~edelweissfe.constraints.base.penaltylaw`), which owns what differs between laws -- its springs, and whether it
pairs the points once or anew in every search; the constraint owns everything the laws share. Coulomb friction and
augmented Lagrange are *not* implemented (the node-based constraint has both); friction enters as a law with a
per-point history (the module documentation of :mod:`~edelweissfe.constraints.base.penaltylaw` says what it
implements). With the law ``type=tied`` the same constraint *ties* its points to the master surface instead, in every
direction, with the gap or overlap at the pairing as the state at rest -- e.g. two independently meshed bodies, see
:class:`Constraint`, *The tie*. ``sliding=finite`` is rejected rather than approximated, see :class:`Constraint`.

What this formulation does *not* fix: pointwise enforcement at quadrature points still
over-constrains a non-matching interface, and the integrand is discontinuous inside a slave facet
wherever the assigned master facet changes, so that quadrature error does not vanish under
refinement of the rule. Exactness on non-matching meshes requires segment clipping and weak
enforcement through a multiplier field -- mortar. This is the strictly weaker, strictly simpler
member of that family, and it is what the corner-liftoff defect actually calls for.
"""

#: The fraction of the search distance beyond which the depth at which a point enters contact is reported: a point
#: that enters that deep between two updates of the connectivity has almost outrun the search.
_entryDepthWarningFraction = 0.5

#: The most steps a point in contact walks over the master in one update of the connectivity, see
#: :meth:`Constraint._closestFacetsOnThisSideOfTheConvexCreases`. Every step goes to a closer facet, so the walk ends
#: anyway; it takes about as many steps as the point passed rings of facets since the last update, one or two when the
#: connectivity is updated often enough. A point that needs more has moved over much of the master between two
#: updates: it keeps the facet of the last step, and the update is reported, as it then pairs differently when it is
#: repeated (a resumed step repeats it).
maximumStepsOfThePairingWalk = 16


class ContactState(IntEnum):
    """Where a contact point stands with respect to the master surface, as of the last update of the connectivity."""

    #: No facet within the search distance.
    FREE = 0
    #: A facet within the search distance, the point outside of it: the point is seen before it touches.
    APPROACHING = 1
    #: The point lies inside the master body by the geometric gap, seen outside before or overlapping it within the
    #: search distance at the first update, and is pushed back out through its paired facet. Whether it carries a force
    #: is the law's: its gap less an initial overlap it does not count (see :attr:`Constraint.numberOfPointsWithASpring`).
    IN_CONTACT = 2
    #: The point lies inside the master body without having been seen outside of it before: it has tunnelled through
    #: the search distance between two updates of the connectivity, or it overlaps the master deeper than the search
    #: distance at the first update. It gets no force (the closest facet is no indication of the way out), and the
    #: constraint reports it. It stays so until it is seen outside within the search distance: deeper inside than the
    #: search distance, it is out of sight, not free.
    FOUND_INSIDE = 3
    #: The point is paired once, with a material point of the master surface, for good: it is tied to it (a law that
    #: pairs once, :attr:`~edelweissfe.constraints.base.penaltylaw.PenaltyLaw.pairsOnce`: the tie).
    TIED = 4


@dataclass(frozen=True)
class TieOfThePoints:
    """What a tie has decided for its contact points, independent of the mesh of the master, and what a new mesh hands
    on (see :class:`Constraint`, *A new mesh*): which points are tied, the material point of the master each one is
    tied to, and how far each one is displaced from rest at the last accepted state.

    Attributes
    ----------
    tied
        Which points are tied, a mask over all points.
    masterPoints
        The master point of each tied point, in the reference configuration, (nTied, nDim).
    displacementsFromRest
        The displacement of each tied point from rest at the last accepted state, (nTied, nDim).
    """

    tied: np.ndarray
    masterPoints: np.ndarray
    displacementsFromRest: np.ndarray


def affineInterpolationAlongASurface(points: np.ndarray, at: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The weights that interpolate a field given at points of a surface onto other points of it: at each point of
    ``at``, the value of the affine function of the position along the surface that fits the field at the
    :math:`3^{n_\\mathrm{dim} - 1}` nearest ``points`` best, in the least-squares sense. The position along the surface
    is measured in the plane that fits those points best, along their principal directions. The interpolation is
    exact for a uniform field, and for a linear one on a flat surface; otherwise its error is of second order in the
    spacing of the points, from the change of the gradient of the field and the curvature of the surface, within a
    smooth part of the surface. At an edge or a corner of the surface, the nearest points may lie on faces on either
    side of it, the plane that fits them best does not describe the surface there, and the error is of first order.

    Parameters
    ----------
    points
        Where the field is given, (nPoints, nDim).
    at
        Where it is wanted, (nAt, nDim).

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        The neighbours of each point of ``at`` among ``points`` and their weights, both (nAt, nNeighbours): the field
        at ``at[i]`` is ``sum_j weights[i, j] * field[neighbours[i, j]]``.
    """

    nDim = points.shape[1]
    nNeighbours = min(len(points), 3 ** (nDim - 1))
    if not len(at):
        return np.zeros((0, nNeighbours), dtype=np.intp), np.zeros((0, nNeighbours))

    _, neighbours = cKDTree(points).query(at, k=nNeighbours)
    neighbours = np.asarray(neighbours).reshape((len(at), nNeighbours))
    patch = points[neighbours]
    centre = patch.mean(axis=1)
    _, _, principalDirections = np.linalg.svd(patch - centre[:, None, :])
    alongTheSurface = principalDirections[:, : nDim - 1, :]

    # the affine basis [1, s] at the neighbours and at the point, s the position along the surface
    basisAtTheNeighbours = np.concatenate(
        (np.ones((len(at), nNeighbours, 1)), np.einsum("akd,asd->aks", patch - centre[:, None, :], alongTheSurface)),
        axis=2,
    )
    basisAtThePoint = np.concatenate(
        (np.ones((len(at), 1)), np.einsum("ad,asd->as", at - centre, alongTheSurface)), axis=1
    )
    weights = np.einsum("ab,abk->ak", basisAtThePoint, np.linalg.pinv(basisAtTheNeighbours))
    return neighbours, weights


class ContactStiffnessView:
    """Provides structured 2-D sub-views for the sparse stiffness matrix slice of :class:`Constraint`.

    Each located contact point couples the nodes its interpolation vector acts on to the nodes of its assigned master
    facet's parent face, and to nothing else -- so the contribution is one dense square block per located point, over
    that point's own ``[slave nodes, master parent face nodes]``, zero while the point has no spring.

    Attributes
    ----------
    blocks : list[numpy.ndarray]
        One view per located contact point, of shape ``(m, m)`` with
        ``m = (nSlaveNodesOfThePoint + nMasterParentNodes) * nDim``.
    """

    def __init__(self, flat_array: np.ndarray, blockSizes: list[int]):
        """Carve the flat value slice into one dense square block per located contact point.

        Parameters
        ----------
        flat_array
            This constraint's slice of the global VIJ value array.
        blockSizes
            The DOF count ``m`` of each located contact point's block, in assembly order.
        """

        self.blocks = []
        offset = 0
        for m in blockSizes:
            self.blocks.append(flat_array[offset : offset + m * m].reshape((m, m)))
            offset += m * m


@dataclass(frozen=True)
class SurfaceToDeformableSurfacePenaltySchema(SurfaceContactPenaltySchema):
    """The options this constraint accepts: the shared slave-side and penalty options of
    :class:`~edelweissfe.constraints.base.surfacecontactpenaltyschema.SurfaceContactPenaltySchema`,
    plus the master surface and the kind of contact points. ``penalty`` here is an interface stiffness modulus per
    unit area exactly as in the node-based constraint, multiplied by the area each contact point stands for.
    """

    masterSurface: str | None = schemaField(
        description="The element set of contact facet elements (Tria3ContactFacet/Line2ContactFacet) "
        "forming the master surface.",
        dtype=str,
        default=None,
        required=True,
    )
    initialOverlap: str = schemaField(
        description="What an overlap of a contact point found at the first contact search stands for. 'pushApart': a "
        "real interference fit (a groove cut smaller than the thread), pushed apart -- at once, or over "
        "initialOverlapRampTime. 'adjust': an error of the geometry (flat faces standing for a curved surface): the "
        "overlap is kept as a stress-free offset of the point's gap for good, so that the point touches the master at "
        "the start without a force, and a point that separates and comes back meets the surface where it touched it. "
        "Not with type=tied, which keeps any gap or overlap at the pairing as the state at rest.",
        dtype=str,
        default="pushApart",
    )
    initialOverlapRampTime: float = schemaField(
        description="With initialOverlap=pushApart: the time over which the overlap found at the first contact "
        "search is pushed apart: it is taken out of the gap at first and fed back in smoothly (starting and ending "
        "at rest) over this time. 0: pushed apart at once.",
        dtype=float,
        default=0.0,
    )
    contactType: str = schemaField(
        description="The penalty law: 'linear' (linear force, constant stiffness with jump) or 'quadratic' "
        "(quadratic force, linear stiffness), or 'tied': the contact points are tied to the master surface, in "
        "tension and compression and in every direction, by linear springs of the penalty stiffness; each point is "
        "paired once, at the first contact search, and the gap or overlap it has then is the state at rest. "
        "searchDistance is then the distance within which points are tied, once: a point beyond it is never tied.",
        dtype=str,
        default="linear",
        optionName="type",
    )


class Constraint(FrozenContactSearch, ForcesOnlyExplicitEvaluation, ConstraintBase, MeshDependent):
    """
    Penalty based unilateral contact between the contact points of a slave body and a deformable master surface, or a
    tie of the two (``type=tied``), distributed with the master's parent element face shape functions.

    Theoretical background
    -----------------------
    A contact point :math:`p` moves with an interpolation vector :math:`c` over the nodes it acts on: the parent face
    shape functions :math:`N^s(\\xi_q)` of a quadrature point of a slave facet, :math:`[1]` for a node (see
    :mod:`~edelweissfe.constraints.base.contactpoints`); it stands for an area :math:`A_p`. A quadrature point
    follows the *curved* parent surface, :math:`x_q = N^s(\\xi_q) \\cdot x^s`, so the slave-side faceting error of
    the node-based formulation disappears as well.

    Under ``sliding=small`` the projection is frozen at every update of the connectivity: the assigned master facet,
    the parametric location of the closest point *in that facet's parent face*, and the direction of the gap
    :math:`\\bar{n}`, the facet's unit normal except at a crease of the master (see *The gap at the corners of the
    master*). The gap is then

    .. math::
        g = \\bar{n} \\cdot \\left( \\boldsymbol{x}_p - N^m(\\xi_m) \\cdot x^m \\right) ,

    which is *linear* in the displacement DOFs, so the gradient
    :math:`w = [c, -N^m] \\otimes \\bar{n}` is constant, the geometric Hessian vanishes
    identically, and the tangent :math:`\\mathrm{d}f_n/\\mathrm{d}g \\, w \\otimes w` is
    symmetric. The force is :math:`-f_n\\, w` on the nodes, with the normal force :math:`f_n` of the penalty law at
    the stiffness ``penalty`` times :math:`A_p`.
    Structurally this is the node-based constraint's frozen form, with that formulation's slave coefficient ``1``
    replaced by the interpolation vector of the contact point -- which is precisely the generalization that lets a
    corner node of a serendipity face carry a tensile force.

    Note that the gap is measured to the *parent surface* point :math:`N^m(\\xi_m) \\cdot x^m`, not
    to the flat facet point :math:`b \\cdot x^{\\text{facet}}`. This matters: distributing the force
    with :math:`N^m` while measuring the gap to the flat point would make the force distribution
    differ from the transpose of the gap gradient, i.e. a non-symmetric, variationally inconsistent
    operator. The flat facets serve only as the search and parametrization scaffold; the contact
    geometry itself is the curved parent surface. For straight-edged faces the two points coincide
    identically, so nothing is lost in the common case.

    Measured from the origins
    -------------------------
    The vector :math:`\\boldsymbol{x}_p - N^m \\cdot x^m` is a difference of two nearly equal positions: across a closed
    contact, the two points are as far apart as a spring is stretched, :math:`t / k` for a traction :math:`t`, and a
    tied point is as far from where it rests, while their coordinates are as large as the model. Subtracting the
    coordinates would lose the stretch to their round-off, :math:`\\varepsilon |\\boldsymbol{x}|` for the machine
    epsilon :math:`\\varepsilon`, and the penalty turns that into a force of :math:`k A_p \\varepsilon
    |\\boldsymbol{x}|` per point: the force would jump from one bit of the coordinates to the next, a staircase in the
    displacements, and a Newton iteration would stall on that staircase instead of converging -- however small its
    corrections, its residual would scatter over the height of a step, with any change of the rounding (another order of
    summation) drawing it anew.

    The constraint therefore measures every point from its *origin*
    (:meth:`~edelweissfe.constraints.base.contactpoints.ContactPoints.originsOfLocatedPoints`), where it is when the
    displacement it moves with is zero -- the reference configuration, for the points of finite elements --, and the
    master from its reference configuration. A contact point rests on its master point, a tied point at its resting
    offset :math:`\\boldsymbol{r}_0` from it (see *The tie*), so its displacement from rest is

    .. math::
        \\boldsymbol{d} = \\boldsymbol{x}_p - N^m \\cdot x^m - \\boldsymbol{r}_0
        = \\underbrace{\\left( \\boldsymbol{x}^0_p - N^m \\cdot X^m \\right) - \\boldsymbol{r}_0}_{\\text{the same in
        every evaluation}}
        + \\left( c \\cdot \\boldsymbol{u} - N^m \\cdot U^m \\right) ,

    with :math:`c \\cdot \\boldsymbol{u}` the displacement of the point from its origin
    (:meth:`~edelweissfe.constraints.base.contactpoints.ContactPoints.displacementsOfLocatedPoints`, with
    :math:`\\boldsymbol{u}` the displacement the point moves with, the total displacement for the points of finite
    elements) and :math:`U^m` the total displacement of the master nodes. The gap is its normal part, :math:`g =
    \\bar{n} \\cdot \\boldsymbol{d}`. The relative position of the origins is taken once, when the points are located,
    and the first part is the same, to the last bit, in every evaluation until they are located anew: its round-off is a
    constant, which shifts the state at rest by a fraction of a bit of the coordinates and stalls nothing. The second
    part is a difference of displacements, rounded relative to the displacements, not to the coordinates. The force is
    then a smooth function of the solution, and a Newton iteration converges to the round-off of the forces themselves.
    What a pairing records from where a point is -- the resting offset of a tie, an initial overlap (see *The first
    update*) -- is measured the same way, at the solution the model holds, so that a tied point, or a point whose
    overlap is adjusted, rests exactly where it was found.

    The law
    -------
    The law (:mod:`~edelweissfe.constraints.base.penaltylaw`) owns what differs between laws: the springs it puts at
    the located points (:meth:`~edelweissfe.constraints.base.penaltylaw.PenaltyLaw.springs`), from their gaps, their
    frozen normals, their areas and their displacements from rest, and how they are paired
    (:attr:`~edelweissfe.constraints.base.penaltylaw.PenaltyLaw.pairsOnce`). Everything else is this constraint's,
    the same for every law: the contact points, the search and the pairing, the relative position and the gap, the
    force and the tangent of the springs on the nodes, the bound of their stiffness for the critical time step, the
    state of the points in a checkpoint, and the outputs, which are taken from the springs of the last evaluation
    alone: the forces on the points (:meth:`getTractions`, their normal part :meth:`getNormalPressures`), the
    resultants on either side, the stored energy, the number of points with a spring.

    The gap at the corners of the master
    ------------------------------------
    The gap is frozen as the first-order expansion of the gap of the point to the master surface at the update
    (:meth:`~edelweissfe.utils.facetcontactgeometry.FacetedSurface.gapsAndGradients`): :math:`\\bar{n}` is its
    gradient with respect to the point. Where the closest point lies inside a facet, or on a seam of a smooth surface,
    that is the facet's normal, and the gap the distance along it. Where it lies on a *crease* -- an edge or a vertex
    where the facets meet at more than the angle of
    :data:`~edelweissfe.utils.facetcontactgeometry.cosOfSmoothNormalChangeLimit`, such as the crest and the root of a
    thread or the tip of a cone -- the gap is the signed distance, its gradient the line from the closest point to the
    contact point, and whether the point is inside or outside is told by the angle-weighted pseudonormal of the crease,
    which is right at convex and re-entrant corners alike. So a point beside a crest is outside, with the distance to
    the crest as its gap, and a point inside a root is pushed out the shortest way. On the boundary of a master surface
    that ends, the plane of its last facet goes on, and a crease that ends there goes on along its last edge. Where the
    closest point passes from one rule to the other -- from a crease vertex onto an edge that is none, or across a seam
    -- the gap frozen at the next update jumps by up to :math:`1 - \\cos 30^\\circ` (about 13 %) of the distance, and
    the force of a closed point with it; its sign never changes.

    The pairing
    -----------
    In every update of the connectivity, the contact points within ``searchDistance`` of the master surface (all of
    them, if no search distance is given) are paired with their closest facet, but only while they are outside the
    master body (:attr:`ContactState.APPROACHING`). Once a point is in contact (:attr:`ContactState.IN_CONTACT`), it is
    paired anew in every update with its closest facet, however deep it is, as it slides along the surface, from one
    face of the master onto the next and around a re-entrant corner -- but never across a convex crease (see
    :meth:`~edelweissfe.utils.facetcontactgeometry.FacetedSurface.acrossAConvexCrease`): the closest facet of a point
    near a convex corner of the master may be on its side, of a point deep inside a thin body on its far side, and would
    push the point out of the wrong side. Such a point walks over the master instead: to the closest facet on this side
    of the crease, and on from there, until its facet no longer changes -- so that an update repeated at the same
    configuration, as a resumed step repeats one, pairs it the same, however far it slid since the last update. A point
    that leaves the master body is released at once, also beside a crest, and is free if it is then beyond the search
    distance. A point that was out of reach at the previous update and is found inside the master body has tunnelled
    through the search distance between two updates of the connectivity (:attr:`ContactState.FOUND_INSIDE`): it gets no
    force, is reported, and counts as found inside until it is seen outside again, also while it is deeper inside than
    the search distance. The search distance must therefore exceed the distance a point travels towards the surface
    between two updates -- every ``contact-update-frequency`` increments of ``NED`` (by default 100), every increment in
    a model with particles or in an implicit analysis -- and the constraint warns when a point enters deeper than half
    of it.

    The first update
    ----------------
    At the first update, the bodies are as the model gives them. Every point is searched at any depth, once, and a
    point that overlaps within the search distance is in contact. A point that overlaps deeper than the search
    distance has not been seen outside: it is found inside (:attr:`ContactState.FOUND_INSIDE`), gets no force, and is
    reported. An overlap that deep is either real, and the search distance too small for it, or a point beyond the end
    of a master surface that ends, behind the continued plane of its last facet; pushed out, it would be pushed by its
    whole depth at once. What the overlap :math:`g_0` of a point in contact stands for, ``initialOverlap`` says:

    - ``pushApart`` (the default): a real **interference fit**, such as a groove cut smaller than the thread. The
      overlap is pushed apart: at once, or with ``initialOverlapRampTime`` gradually -- :math:`g_0` is subtracted
      from the gap at first, and fed back in over that time, smoothly, starting and ending at rest.
    - ``adjust``: an **error of the geometry**, such as flat faces standing for the curved hole a screw fits, whose
      chords lie inside the screw's curved faces. :math:`g_0` is subtracted from the gap for good: the
      master surface is taken as adjusted to touch the point, the point rests there without a force, and its gap is
      measured from there -- a point that separates and comes back meets the surface where it touched it.

    Either way the offset is a constant of the gap, so the tangent does not change, and the law (with it which
    points are closed, and the bound of the stiffness for the critical time step) sees the offset gap. The offsets
    belong to the contact points and are checkpointed (see *A new mesh* for a refinement).

    The tie
    -------
    With the law ``type=tied`` (:class:`~edelweissfe.constraints.base.penaltylaw.TiedPenaltyLaw`) the constraint
    does not let its points touch the master surface, it *ties* them to it -- e.g. a finely meshed part to the coarse
    mesh of the body around it. It is the same constraint, the same contact points, the same frozen projection, the
    same parent face shape functions; what differs is the law, which pairs once and has its springs:

    - **paired once.** At the first update, every point within ``searchDistance`` of the master surface (every point,
      without a search distance), in the configuration of the model, is paired with the closest point of its closest
      facet, mapped onto the facet's parent face, and stays paired with that *material* point of the master,
      :math:`N^m(\\xi_m) \\cdot x^m`, for good: tension does not release it, sliding does not move it on, and none of
      the rules of *The pairing* apply. A point beyond the search distance then is never tied. A new mesh does not
      decide anew, it hands the tie on (see *A new mesh*).
    - **at rest where it is.** The vector :math:`\\boldsymbol{r}_0` from the master point to the point at the pairing
      is where the point rests: a gap or an overlap of the two surfaces -- of a coarse mesh, whose faceted surface
      lies inside the curved surface of a fine one -- is part of the stress-free state, not closed or pushed apart.
      ``initialOverlap`` is therefore an option of contact only.
    - **held in every direction.** A spring of the penalty stiffness along every axis holds the point there, in
      tension as in compression; together, with the displacement from rest
      :math:`\\boldsymbol{d} = \\boldsymbol{x}_p - N^m \\cdot x^m - \\boldsymbol{r}_0`, the force on the point is

      .. math::
          \\boldsymbol{f} = -k A_p \\, \\boldsymbol{d} , \\qquad
          \\boldsymbol{K} = k A_p \\, [c, -N^m] \\otimes [c, -N^m] \\otimes \\boldsymbol{I} ,

      each spring along its axis :math:`\\boldsymbol{e}_i` exactly the spring of a contact with :math:`\\bar{n}`
      replaced by :math:`\\boldsymbol{e}_i`. The force is linear in the displacements, its tangent constant and
      symmetric, its energy :math:`\\tfrac{1}{2} k A_p |\\boldsymbol{d}|^2`.

    The interface is as compliant as the springs: a traction :math:`t` opens it by :math:`t / k`. Choose ``penalty``
    well above the stiffness of the adjoining material over the size of its elements, :math:`E / h`, and within what
    the time step of an explicit analysis can carry. The rest offset is a fixed vector, so a rotation of the interface
    by :math:`\\theta` is resisted by :math:`k |\\boldsymbol{r}_0| \\theta`: small for a small gap at rest, and zero
    without one.

    This tie is not the ``tie`` constraint (:mod:`~edelweissfe.constraints.tie`), which eliminates the degrees of
    freedom of the slave nodes: exact, without a penalty and without a spring that lowers the critical time step, for
    nodes only, on the flat facets. This one holds both kinds of contact point on the parent faces by springs:
    compliant by :math:`t / k`, and the stick of a friction law that never slips. Both keep a gap at rest, the ``tie``
    with ``adjust=False``. The documentation page of the theory of contact compares the two (*Two ties*).

    A new mesh
    ----------
    Like the node-based constraint, this is a :class:`~edelweissfe.models.meshdependent.MeshDependent`: if either
    surface's source solid elements are refined mid-run, it rebuilds that side from the regenerated facets
    (:meth:`refresh`), and the update of the connectivity that follows every rebuild pairs the points anew. What the
    pairing has learnt belongs to the points: their states (:class:`ContactState`), the facet each point is paired
    with, the offsets of an initial overlap, and the displacements from rest of a tie.

    - A new mesh of the master keeps the states and the offsets, and carries the facet of each paired point over to
      the facet of the new master where the old one lay (in the reference configuration). A point in contact walks on
      from there as from the facet of the previous update: it is searched at any depth, is not paired across a convex
      crease, and a point found inside stays found inside.
    - Regenerated contact points are new points, with nothing to carry -- quadrature points, unlike nodes, have no
      identity that survives a retiling. They are paired as at the first update: a point inside within the search
      distance is in contact, with its closest facet, a point deeper inside is found inside; an overlap they have is
      pushed apart at once. The constraint reports it.
    - A tie is decided once, at the first update (*The tie*): which points are tied, and to which material point of
      the master. A new mesh hands that on (:class:`TieOfThePoints`), with the displacement of each tied point from
      rest at the last accepted state, promoted in :meth:`acceptLastState` and checkpointed, so that a run resumed and
      refined before its first evaluation, or a run fallen back to a checkpoint, hands on the state it goes on from.
      The next update ties the points by it (:meth:`_tieAgainAfterANewMesh`): each one to the point of the new master
      closest to its master point in the reference configuration, and with the resting offset that leaves it
      displaced from rest as it was. A new mesh of the master keeps the points, and so every point keeps its tie and
      the forces of the tie stay as they were. A refined slave side has new points: in the reference configuration, a
      new point is tied if the old point nearest to it was, and its master point and its displacement from rest are
      interpolated from the old tied points, by the affine function of the position along the surface that fits the
      nearest of them best (:func:`affineInterpolationAlongASurface`). That is exact for a uniform displacement from
      rest, and for a linear one on a flat interface; otherwise its error is of second order in the spacing of the old
      points, from the change of the gradient of the displacement from rest and the curvature of the interface, within
      a smooth part of the slave surface; at an edge of the slave surface, of first order.

    A law with a history, such as friction, will carry its history the same way.

    ``sliding=finite`` is rejected rather than approximated. It would require the exact gap gradient
    and Hessian of a point against a *curved* quadratic surface, whose closest-point projection is
    itself an iterative sub-problem -- a substantially different formulation from the flat-facet
    algebra of :mod:`~edelweissfe.utils.facetcontactgeometry`, not a variation on it.

    The methods follow the order of a first reading: after the construction, (1) the gap and the springs at the paired
    points, (2) the pairing, (3) an initial overlap, and (4) the bound of the stiffness for the critical time step.

    Parameters
    ----------
    name
        The name of this constraint.
    model
        The model tree.
    contactPoints
        The contact points of the slave side.
    masterSurface
        The element set of contact facet elements forming the master surface.
    journal
        The journal instance.
    configuration
        The options this constraint accepts, see :class:`SurfaceToDeformableSurfacePenaltySchema`. Its
        ``slaveSurface``, ``nQuadraturePoints`` and ``contactPoints`` say which contact points an input file builds
        (:meth:`fromConstraintDefinition`); here they are given.
    """

    #: Option schema for this constraint, per OptionSchemaProvider.
    schema = SurfaceToDeformableSurfacePenaltySchema

    #: The contact searches for the surface it touches.
    connectivityChangesWithMotion = True

    #: Journal sender tag: short and class-level, so that the fixed-width sender column never truncates it.
    identification = "SurfaceContact"

    def __init__(
        self,
        name: str,
        model: FEModel,
        contactPoints: ContactPoints,
        masterSurface: ElementSet,
        journal: Journal,
        *,
        configuration: SurfaceToDeformableSurfacePenaltySchema = SurfaceToDeformableSurfacePenaltySchema(),
    ):
        super().__init__(name, model)

        self.name = name
        self.journal = journal
        self._lastSeenTopologyVersion = model.topology.version
        model.topology.registerMeshDependent(self)

        self.nDim = model.domainSize

        self._masterSurfaceSetName = masterSurface.name

        self.penalty = configuration.penalty
        if self.penalty <= 0.0:
            raise ValueError("The penalty must be positive: a non-positive penalty silently disables contact.")

        self.law = penaltyLaw(configuration.contactType, penaltyLawsAndTheTie)

        self.sliding = configuration.sliding.lower()
        if self.sliding != "small":
            raise ValueError(
                f"Constraint sliding '{self.sliding}' is not supported by "
                "surfaceToDeformableSurfacePenalty: only 'small' is implemented. Finite sliding "
                "would require the exact gap gradient and Hessian against a curved quadratic "
                "surface, whose closest-point projection is itself an iterative sub-problem."
            )

        self.searchDistance = configuration.searchDistance

        initialOverlap = configuration.initialOverlap.lower()
        if initialOverlap not in ("pushapart", "adjust"):
            raise ValueError(
                f"Constraint '{name}': initialOverlap '{configuration.initialOverlap}' is not supported. Use "
                "'pushApart' or 'adjust'."
            )
        #: True if an overlap at the first contact search is kept as a stress-free offset of the gap for good
        #: (``initialOverlap=adjust``), False if it is pushed apart (``pushApart``).
        self.adjustsTheInitialOverlap = initialOverlap == "adjust"
        self.initialOverlapRampTime = configuration.initialOverlapRampTime
        if self.initialOverlapRampTime < 0.0:
            raise ValueError("The initial overlap ramp time must not be negative.")
        if self.adjustsTheInitialOverlap and self.initialOverlapRampTime > 0.0:
            raise ValueError(
                f"Constraint '{name}': an adjusted initial overlap is never pushed apart; it has no ramp time "
                "(initialOverlapRampTime is for initialOverlap=pushApart)."
            )
        if self.law.pairsOnce and (self.adjustsTheInitialOverlap or self.initialOverlapRampTime > 0.0):
            raise ValueError(
                f"Constraint '{name}': a law that pairs once (type={self.law.name}) keeps the gap or overlap of its "
                "points at the pairing as the state at rest; initialOverlap and initialOverlapRampTime are for laws "
                "that pair anew in every search."
            )
        # The time of the first contact search, from which the initial overlap is offset; None before it, and
        # without an offset (an overlap pushed apart at once).
        self._initialOverlapTime = None

        self.slave = contactPoints
        self._buildMaster(masterSurface)
        self._resetContactPoints()
        self._resetPairing()

        self.journal.message(
            f"contact '{self.name}': {self.slave.describe()}, {len(self.facetElements)} master facets, "
            f"sliding={self.sliding}, type={self.law.name}",
            self.identification,
        )

    @classmethod
    def fromConstraintDefinition(cls, name: str, definition: dict, model: FEModel, journal: Journal) -> "Constraint":
        """Build this constraint from a parsed ``*constraint`` definition, with the contact points the option
        ``contactPoints`` chooses on the slave surface. See
        :class:`~edelweissfe.constraints.base.constraintbase.ConstraintBase` for why this is
        separate from ``__init__``."""
        configuration = buildSchemaFromOptions(cls.schema, definition)
        return cls(
            name,
            model,
            contactPointsOfTheSlaveSurface(configuration, model, name),
            model.elementSets[configuration.masterSurface],
            journal,
            configuration=configuration,
        )

    def _buildMaster(self, masterSurface: ElementSet):
        """Build (or rebuild) everything cached about the master surface: its facets and their parent faces, the
        master nodes, each once, and the facets as rows of indices into them.

        Parameters
        ----------
        masterSurface
            The element set of master contact facets.
        """

        self.facetElements = list(masterSurface)

        checkFacetsCarryParentFaces(self.facetElements, "master", self._masterSurfaceSetName, self.name)

        parentFaceTypes = {facet.parentFaceType for facet in self.facetElements}
        if len(parentFaceTypes) > 1:
            raise ValueError(
                f"Constraint '{self.name}': the master surface '{self._masterSurfaceSetName}' mixes parent faces "
                f"{sorted(parentFaceTypes)}; a single element type is required."
            )
        self._masterParentFaceType = parentFaceTypes.pop() if parentFaceTypes else None

        masterParentNodes = {node for facet in self.facetElements for node in facet.parentFaceNodes}
        if not masterParentNodes.isdisjoint(self.slave.finiteElementNodes()):
            raise ValueError(
                f"Constraint '{self.name}': the contact points and the master surface "
                f"'{self._masterSurfaceSetName}' share nodes -- self-contact is not supported."
            )

        indexOfMasterNode = {}
        for facet in self.facetElements:
            for node in facet.nodes:
                indexOfMasterNode.setdefault(node, len(indexOfMasterNode))
        self._masterNodes = list(indexOfMasterNode)
        self._referenceCoordinatesOfMasterNodes = np.array([node.coordinates for node in self._masterNodes])
        self._masterNodesOfFacet = np.array(
            [[indexOfMasterNode[node] for node in facet.nodes] for facet in self.facetElements], dtype=np.intp
        ).reshape((len(self.facetElements), self.nDim))
        self._masterFacets = FacetedSurface(self._masterNodesOfFacet)

        self._vertexParametricCoordsOfFacet = np.array([facet.vertexParametricCoords for facet in self.facetElements])
        self._masterParentRefCoords = np.array(
            [[node.coordinates for node in facet.parentFaceNodes] for facet in self.facetElements]
        )
        self.nMasterParentNodes = self._masterParentRefCoords.shape[1] if self.facetElements else 0

    def _resetContactPoints(self):
        """Size what the pairing learns about each contact point for the current contact points, which have not been
        searched yet: their states and the offsets of an initial overlap. It belongs to the points, so a new mesh of
        the master keeps it (see the class documentation, *A new mesh*)."""

        nPoints = self.slave.nPoints

        self._stateOfPoint = np.full(nPoints, ContactState.FREE, dtype=np.int8)
        self._searchedBefore = False
        self._deepestEntryReported = 0.0

        # The overlap of each point at the first contact search (zero or negative), recorded if it is not pushed apart
        # at once: the gap does not count it, for good (adjusted) or until the ramp has fed it in (pushed apart).
        self._initialOverlap = np.zeros(nPoints)

        # For a law that pairs once: the displacement of each point from rest at the last accepted state (zero for a
        # point not tied), promoted in acceptLastState; and the tie a new mesh handed on to the points, until the next
        # update of the connectivity ties them by it (see the class documentation, *A new mesh*).
        self._displacementsFromRestAtTheLastAcceptedState = np.zeros((nPoints, self.nDim))
        self._tieHandedOver = None

    def _resetPairing(self):
        """Unpair every contact point from the master: no facet, no frozen projection, no resting offset, nothing
        located."""

        nPoints = self.slave.nPoints

        # The frozen projection, refreshed in every update of the connectivity: the assigned master facet (-1 if
        # none), the parent face shape functions of the closest point within that facet's parent face, and the
        # facet's unit normal.
        self._assignedFacetIdx = np.full(nPoints, -1, dtype=np.intp)
        self._frozenMasterShapeFunctions = np.zeros((nPoints, self.nMasterParentNodes))
        self._frozenNormals = np.zeros((nPoints, self.nDim))

        # Where each point rests relative to the master point it is paired with: zero for a law that pairs anew in
        # every search (the point rests on its master point), the vector from its master point to it at the pairing
        # for a law that pairs once.
        self._restingOffset = np.zeros((nPoints, self.nDim))

        # The gap of every point at the last evaluation, as the law sees it, for output.
        self._gapCurrent = np.zeros(nPoints)

        self._locatePairedPoints()
        self._lastReportedActivity = None

    @property
    def nodes(self) -> list:
        return self._nodes

    @property
    def fields(self) -> list:
        return [["displacement"]] * len(self._nodes)

    @property
    def nDof(self) -> int:
        return self.nDim * len(self._nodes)

    @property
    def nPoints(self) -> int:
        """The number of contact points."""
        return self.slave.nPoints

    @property
    def slaveSurfaceNodes(self) -> list:
        """The nodes of an FE slave surface, see
        :attr:`~edelweissfe.constraints.base.contactpoints.ContactPoints.surfaceNodes`."""
        return self.slave.surfaceNodes

    @property
    def stateOfPoints(self) -> np.ndarray:
        """The :class:`ContactState` of every contact point, as of the last update of the connectivity.

        Returns
        -------
        numpy.ndarray
            The states.
        """

        return self._stateOfPoint

    @property
    def numberOfPointsFoundInside(self) -> int:
        """The number of contact points found inside the master body without having been seen outside of it, which
        get no force (:attr:`ContactState.FOUND_INSIDE`); a point counts until it is seen outside again.

        Returns
        -------
        int
            The number.
        """

        return int(np.count_nonzero(self._stateOfPoint == ContactState.FOUND_INSIDE))

    # --- 1. The gap and the springs: what an evaluation computes at the paired points ---------------------------------

    def _locatePairedPoints(self):
        """Locate the paired points in the block: the nodes their contact points act on (the slave part), then, point
        after point, the parent face nodes of their master facets (the master part). Their frozen projections, and
        where they are relative to their master points at the zero solution of the block (see the class documentation,
        *Measured from the origins*), are gathered into arrays, once per update of the connectivity, so that an
        evaluation is a fixed handful of array operations rather than a loop over the points.
        """

        located = np.flatnonzero(self._assignedFacetIdx >= 0)
        self._locatedPoints = located
        self.withinReach = bool(len(located))

        slaveNodes = self.slave.locatePoints(located)
        facets = self._assignedFacetIdx[located]
        masterNodes = [node for f in facets for node in self.facetElements[f].parentFaceNodes]
        self._nodes = slaveNodes + masterNodes
        self._nSlaveDof = self.nDim * len(slaveNodes)

        self._masterShapeFunctionsOfLocated = self._frozenMasterShapeFunctions[located]
        self._normalsOfLocated = self._frozenNormals[located]
        self._initialOverlapOfLocated = self._initialOverlap[located]
        self._restingOffsetOfLocated = self._restingOffset[located]
        # Where the points are relative to their master points when the solution of the block is zero: from the
        # origins of the points and the reference configuration of the master, once per location.
        masterReferenceCoordinatesOfLocated = self._masterParentRefCoords[facets].reshape(
            (len(located), self.nMasterParentNodes, self.nDim)
        )
        self._relativePositionsAtZeroSolutionOfLocated = self.slave.originsOfLocatedPoints() - np.einsum(
            "ai,aij->aj", self._masterShapeFunctionsOfLocated, masterReferenceCoordinatesOfLocated
        )
        self._areasOfLocated = self.slave.areas[located]
        # For a law that pairs once: the displacements from rest of the located points at the last evaluation, which
        # acceptLastState promotes; none yet at these points.
        self._displacementsFromRestOfLocatedAtTheLastEvaluation = None
        self._blockSizes = None
        # The springs of the last evaluation, at these located points: what the outputs report.
        self._springs = []

    def _activeBlockSizes(self) -> list[int]:
        """The DOF count of each located contact point's dense block, in assembly order; cached per update of the
        connectivity, since the implicit path asks for it twice per evaluation."""

        if self._blockSizes is None:
            self._blockSizes = [len(dofs) for dofs in self._dofsOfLocatedPoints()]
        return self._blockSizes

    def _dofsOfLocatedPoints(self) -> list[np.ndarray]:
        """Per located point, its degrees of freedom in the block: those of its slave nodes, then those of the parent
        face nodes of its master facet."""

        nMasterDof = self.nMasterParentNodes * self.nDim
        return [
            np.concatenate(
                (
                    self.slave.interpolationOfLocatedPoint(k)[0],
                    self._nSlaveDof + k * nMasterDof + np.arange(nMasterDof),
                )
            )
            for k in range(len(self._locatedPoints))
        ]

    def localEntriesOfBlock(self) -> tuple[np.ndarray, np.ndarray]:
        """One dense block per located contact point, row by row, over its slave and master DOFs."""
        return localEntriesOfSubBlocks([(dofs, dofs) for dofs in self._dofsOfLocatedPoints()])

    def shapeBlock(self, values: np.ndarray) -> ContactStiffnessView:
        return ContactStiffnessView(values, self._activeBlockSizes())

    def computeKernels(
        self,
        U_np: np.ndarray,
        dU: np.ndarray,
        P: np.ndarray,
        K: ContactStiffnessView | None,
        timeStep: TimeStep,
    ):
        """Add the residual contribution of the penalty springs at the located points, and the tangent unless ``K``
        is None.

        All located points are evaluated at once: their displacements from rest, from the origins
        (:meth:`_displacementsFromRestOfLocatedPoints`), the springs the law puts there
        (:meth:`_springsOfLocatedPoints`: the contact's one along :math:`\\bar{n}` at every closed point, the tie's one
        along every axis at every point), and the forces of the springs on the nodes (:meth:`_addForcesOfSprings`).
        The springs are kept for the outputs, and for a law that pairs once the displacements from rest, which
        :meth:`acceptLastState` promotes if this is the evaluation at the accepted configuration.
        """

        self._springs = []
        located = self._locatedPoints
        if not len(located):
            return

        displacementsFromRest = self._displacementsFromRestOfLocatedPoints(U_np, dU)
        gaps, springs = self._springsOfLocatedPoints(displacementsFromRest, timeStep)
        for spring in springs:
            self._addForcesOfSprings(P, K, spring)

        # What the outputs report.
        self._springs = springs
        self._gapCurrent[located] = gaps
        if self.law.pairsOnce:
            # What the acceptance of the increment promotes, if this is the evaluation at the accepted configuration.
            self._displacementsFromRestOfLocatedAtTheLastEvaluation = displacementsFromRest

    def acceptLastState(self):
        """For a law that pairs once, promote the displacements from rest of the last evaluation, the one at the
        accepted configuration (see
        :meth:`~edelweissfe.constraints.base.constraintbase.ConstraintBase.acceptLastState`), to the state of the
        points, which a checkpoint holds and a new mesh hands on (see the class documentation, *A new mesh*). An
        iterate, or an increment that is cut back, is never handed on."""

        if self._displacementsFromRestOfLocatedAtTheLastEvaluation is not None:
            self._displacementsFromRestAtTheLastAcceptedState[self._locatedPoints] = (
                self._displacementsFromRestOfLocatedAtTheLastEvaluation
            )

    def _displacementsFromRestOfLocatedPoints(self, U_np: np.ndarray, dU: np.ndarray) -> np.ndarray:
        """The displacements of the located points from where they rest, :math:`\\boldsymbol{d} = \\boldsymbol{x}_p - N^m
        \\cdot x^m - \\boldsymbol{r}_0`, told from the origins (see the class documentation, *Measured from the
        origins*): the displacement from rest at the zero solution of the block, the same in every evaluation until the
        points are located anew, plus the difference of the displacements of the points and of their master points
        (:meth:`_relativeDisplacementsOfLocatedPoints`). The resting offset :math:`\\boldsymbol{r}_0` is zero unless
        the law pairs once: a contact point rests on its master point.

        Parameters
        ----------
        U_np
            The solution of the constraint's block.
        dU
            The increment of the constraint's block.

        Returns
        -------
        numpy.ndarray
            The displacements from rest, (nLocated, nDim).
        """

        displacementsFromRestAtZeroSolution = (
            self._relativePositionsAtZeroSolutionOfLocated - self._restingOffsetOfLocated
        )
        return displacementsFromRestAtZeroSolution + self._relativeDisplacementsOfLocatedPoints(U_np, dU)

    def _relativeDisplacementsOfLocatedPoints(self, U_np: np.ndarray, dU: np.ndarray) -> np.ndarray:
        """How far the located points have moved relative to their master points since the zero solution of the block:
        :math:`c \\cdot \\boldsymbol{u} - N^m \\cdot U^m`, the displacements of the points from their origins less
        those of their master points, which ride the *curved* parent surfaces (see the class docstring on why the master
        point must not be taken on the flat facet instead).

        Parameters
        ----------
        U_np
            The solution of the constraint's block.
        dU
            The increment of the constraint's block.

        Returns
        -------
        numpy.ndarray
            The relative displacements, (nLocated, nDim).
        """

        nSlaveDof = self._nSlaveDof
        displacementsOfPoints = self.slave.displacementsOfLocatedPoints(U_np[:nSlaveDof], dU[:nSlaveDof])
        displacementsOfMasterPoints = np.einsum(
            "ai,aij->aj",
            self._masterShapeFunctionsOfLocated,
            U_np[nSlaveDof:].reshape((len(self._locatedPoints), self.nMasterParentNodes, self.nDim)),
        )
        return displacementsOfPoints - displacementsOfMasterPoints

    def _springsOfLocatedPoints(
        self, displacementsFromRest: np.ndarray, timeStep: TimeStep | None, smallestGaps: np.ndarray | None = None
    ) -> tuple[np.ndarray, list[PenaltySprings]]:
        """The penalty springs the law puts at the located points, from their displacements from rest.

        What the constraint gives the law is the same for every law: the penalty times the area of each point, its
        frozen normal :math:`\\bar{n}`, its displacement from rest :math:`\\boldsymbol{d} = \\boldsymbol{x} -
        \\boldsymbol{x}^m - \\boldsymbol{r}_0` (the resting offset :math:`\\boldsymbol{r}_0` is zero unless the law
        pairs once), and its gap :math:`g = \\bar{n} \\cdot \\boldsymbol{d}`, less the part of the initial overlap
        it does not count (all of an adjusted one, the part of one pushed apart that the ramp has not fed in yet;
        :meth:`_gapsOfLocatedPoints`). Every one of them is affine in the displacements with a constant gradient, so
        the tangent is that of the springs alone. The contact law puts one spring along :math:`\\bar{n}` at every
        closed point, the tie one along every axis :math:`\\boldsymbol{e}_i` at every point, of the stretch
        :math:`d_i`.

        The force (:meth:`_addForcesOfSprings`), the bound of the stiffness
        (:meth:`addBoundOfAbsoluteStiffnessTimes`) and the outputs go through the same springs. The bound over the next
        increment gives the law the smallest gap of each point within it
        (:func:`~edelweissfe.constraints.base.penaltysprings.smallestGapsWithin`).

        Parameters
        ----------
        displacementsFromRest
            The displacements of the located points from rest, see :meth:`_displacementsFromRestOfLocatedPoints`.
        timeStep
            The time step of the evaluation, for the ramp of an initial overlap.
        smallestGaps
            The smallest gaps of the located points within the next increment, for the bound over it, which the law
            sees instead of those of ``displacementsFromRest``; None for the state now.

        Returns
        -------
        tuple
            The gaps the law sees (nLocated) of the located points, and the springs.
        """

        gaps = self._gapsOfLocatedPoints(displacementsFromRest, timeStep) if smallestGaps is None else smallestGaps
        springs = self.law.springs(
            self.penalty * self._areasOfLocated, gaps, self._normalsOfLocated, displacementsFromRest
        )
        return gaps, springs

    def _gapsOfLocatedPointsAt(self, U_np: np.ndarray, dU: np.ndarray, timeStep: TimeStep | None) -> np.ndarray:
        """The gaps of the located points as the law sees them at a state (:meth:`_gapsOfLocatedPoints`).

        Parameters
        ----------
        U_np
            The solution of the constraint's block.
        dU
            The increment of the constraint's block.
        timeStep
            The time step of the state.

        Returns
        -------
        numpy.ndarray
            The gaps.
        """

        return self._gapsOfLocatedPoints(self._displacementsFromRestOfLocatedPoints(U_np, dU), timeStep)

    def _gapsOfLocatedPoints(self, displacementsFromRest: np.ndarray, timeStep: TimeStep | None) -> np.ndarray:
        """The gaps of the located points as the law sees them: :math:`g = \\bar{n} \\cdot \\boldsymbol{d}`, less the
        part of the initial overlap not counted at the time of ``timeStep`` -- all of an adjusted one, the part of one
        pushed apart over a ramp time that the ramp has not fed in yet (see the class documentation, *The first
        update*).

        Parameters
        ----------
        displacementsFromRest
            The displacements of the located points from rest, see :meth:`_springsOfLocatedPoints`.
        timeStep
            The time step of the evaluation.

        Returns
        -------
        numpy.ndarray
            The gaps.
        """

        gaps = np.einsum("aj,aj->a", self._normalsOfLocated, displacementsFromRest)
        uncountedFraction = self._uncountedFractionOfInitialOverlapAt(timeStep)
        if uncountedFraction:
            gaps = gaps - uncountedFraction * self._initialOverlapOfLocated
        return gaps

    def _addForcesOfSprings(self, P: np.ndarray, K: ContactStiffnessView | None, springs: PenaltySprings):
        """Add the residual contribution of penalty springs at located points, each along a unit direction
        :math:`\\bar{n}` (the normal of a contact point, an axis of a tie), and their tangent unless ``K`` is None.

        The residual contribution is the negative of the force on the nodes, :math:`f\\, w` with
        :math:`w = [c, -N^m] \\otimes \\bar{n}`: the contact points distribute their part with their interpolation
        vectors, the master facets theirs with their parent face shape functions. The tangent is
        :math:`\\mathrm{d}f/\\mathrm{d}g \\; w \\otimes w`.

        Parameters
        ----------
        P
            The residual contribution of the constraint; augmented.
        K
            The tangent of the constraint, one dense block per located point; augmented. None on the explicit path.
        springs
            The springs, see :meth:`_springsOfLocatedPoints`.
        """

        points, directions, forces, stiffness = springs.points, springs.directions, springs.forces, springs.stiffness
        nSlaveDof = self._nSlaveDof
        self.slave.addForceResidual(P[:nSlaveDof], points, forces, directions)
        wMaster = ((-self._masterShapeFunctionsOfLocated[points])[:, :, None] * directions[:, None, :]).reshape(
            (len(points), -1)
        )
        P[nSlaveDof:].reshape((len(self._locatedPoints), -1))[points] += forces[:, None] * wMaster

        if K is not None:
            # The tangent is only requested by the implicit path: one dense block per located point.
            for i, k in enumerate(points):
                c = self.slave.interpolationOfLocatedPoint(k)[1]
                w = np.concatenate(((c[:, None] * directions[i][None, :]).ravel(), wMaster[i]))
                K.blocks[k] += stiffness[i] * np.outer(w, w)

    def _sumOverTheSprings(self, termOfSprings) -> np.ndarray:
        """The sum of a vector over the springs of the last evaluation; zero without a spring.

        Parameters
        ----------
        termOfSprings
            The vector of one set of springs, a function of the :class:`PenaltySprings`.

        Returns
        -------
        numpy.ndarray
            The sum, (nDim).
        """

        terms = [termOfSprings(springs) for springs in self._springs]
        return sum(terms[1:], start=terms[0]) if terms else np.zeros(self.nDim)

    def _springForcesOfLocatedPoints(self) -> np.ndarray:
        """The force of the springs at each located point at the last evaluation, :math:`\\sum_s f_s
        \\boldsymbol{n}_s` over its springs: :math:`f_n \\bar{n}` of a closed contact point, :math:`k A_p
        \\boldsymbol{d}` of a tied one. The force on the point is its negative.

        Returns
        -------
        numpy.ndarray
            The forces, (nLocated, nDim).
        """

        forces = np.zeros((len(self._locatedPoints), self.nDim))
        for springs in self._springs:
            forces[springs.points] += springs.forces[:, None] * springs.directions
        return forces

    def _normalForcesOfPoints(self) -> np.ndarray:
        """The normal force of every contact point at the last evaluation: the component of the force of its springs
        along its frozen normal, negative in compression; zero where a point has no spring.

        Returns
        -------
        numpy.ndarray
            The normal forces, (nPoints).
        """

        normalForces = np.zeros(self.nPoints)
        normalForces[self._locatedPoints] = np.einsum(
            "aj,aj->a", self._normalsOfLocated, self._springForcesOfLocatedPoints()
        )
        return normalForces

    def _locatedPointsWithSprings(self) -> np.ndarray:
        """The located points with a spring at the last evaluation (indices into the located points), each once.

        Returns
        -------
        numpy.ndarray
            The points, ascending.
        """

        return np.unique(np.concatenate([springs.points for springs in self._springs] + [np.zeros(0, dtype=np.intp)]))

    @property
    def totalNormalForce(self) -> float:
        """The sum of the normal forces of the contact points at the last evaluation (see :meth:`getNormalPressures`),
        negative in compression.

        Returns
        -------
        float
            The force.
        """

        return float(self._normalForcesOfPoints().sum())

    @property
    def contactForceOnSlave(self) -> np.ndarray:
        """The resultant force of the springs on the contact points at the last evaluation, :math:`-\\sum f\\,
        \\boldsymbol{n}` over the springs.

        Returns
        -------
        numpy.ndarray
            The force.
        """

        return -self._sumOverTheSprings(lambda springs: (springs.forces[:, None] * springs.directions).sum(axis=0))

    @property
    def contactForceOnMaster(self) -> np.ndarray:
        """The resultant force of the springs on the master nodes at the last evaluation, the sum of the nodal forces
        :math:`f N^m_a \\boldsymbol{n}` over the springs; equal and opposite to :attr:`contactForceOnSlave` to
        round-off, since the parent face shape functions sum to one.

        Returns
        -------
        numpy.ndarray
            The force.
        """

        def onTheMaster(springs: PenaltySprings) -> np.ndarray:
            sumOfNm = self._masterShapeFunctionsOfLocated[springs.points].sum(axis=1)
            return ((springs.forces * sumOfNm)[:, None] * springs.directions).sum(axis=0)

        return self._sumOverTheSprings(onTheMaster)

    @property
    def penaltyEnergy(self) -> float:
        """The energy stored in the springs at the last evaluation, for an energy balance.

        Returns
        -------
        float
            The energy.
        """

        return float(sum(springs.energies.sum() for springs in self._springs))

    @property
    def deepestPenetration(self) -> float:
        """The largest penetration of a contact point with a spring at the last evaluation, a positive length,
        measured as the law sees it (beyond an adjusted initial overlap); zero without one.

        Returns
        -------
        float
            The penetration.
        """

        points = self._locatedPoints[self._locatedPointsWithSprings()]
        return float(-self._gapCurrent[points].min()) if len(points) else 0.0

    @property
    def numberOfPointsWithASpring(self) -> int:
        """The number of contact points with a spring at the last evaluation: those that carried a force, the closed
        points of the contact law, every tied point of the tie. Not the points :attr:`ContactState.IN_CONTACT`: a point in contact at rest on an
        adjusted initial overlap has no spring.

        Returns
        -------
        int
            The number.
        """

        return len(self._locatedPointsWithSprings())

    def getNormalPressures(self) -> np.ndarray:
        """The current normal contact pressures (positive in compression), one per contact point: the normal force of
        its springs per unit of its area.

        Ordered like the contact points: for quadrature points, by slave facet and, within a facet, by quadrature
        point -- *not* by node, unlike the node-based constraint's accessor of the same name. A ``fromExpression``
        field output reading this therefore cannot be tied to a node set; use a reduction (``np.max``, ``np.mean``,
        ...) or :meth:`getSlaveNodalNormalForces` instead.
        """

        return self.slave.normalPressures(self._normalForcesOfPoints())

    def getTractions(self) -> np.ndarray:
        """The traction on each contact point at the last evaluation: the force of its springs on it per unit of the
        area it stands for, :math:`-\\sum f\\, \\boldsymbol{n} / A_p` -- :math:`-f_n \\bar{n} / A_p` in contact,
        :math:`-k \\boldsymbol{d}` in a tie --, with its normal and its tangential part; zero where a point has no
        spring. Ordered like :meth:`getNormalPressures`.

        Returns
        -------
        numpy.ndarray
            The tractions, (nPoints, nDim).
        """

        located = self._locatedPoints
        tractions = np.zeros((self.nPoints, self.nDim))
        tractions[located] = -self._springForcesOfLocatedPoints() / self.slave.areas[located][:, None]
        return tractions

    def getGaps(self) -> np.ndarray:
        """The current gap at each contact point (negative when penetrating) as the law sees it: along the normal
        frozen at the pairing, from where the point rests, less an initial overlap it does not count; ordered like
        :meth:`getNormalPressures`. For a tie, the normal part of the displacement from rest.

        A point with no assigned master facet reports 0, not a stale value from when it last had
        one -- see :meth:`updateConnectivity`. Returned as a copy, so a caller cannot corrupt the
        state a later augmented-Lagrange update would read.
        """

        return self._gapCurrent.copy()

    def getSlaveNodalNormalForces(self) -> np.ndarray:
        """The normal component of the contact force delivered to each slave surface node, for contact points on a
        slave surface of finite elements.

        Positive means pushed along the master's outward normal, i.e. repelled; **negative means
        pulled**, which is exactly the tensile corner load a serendipity face requires and a
        node-based scheme cannot produce. This is the accessor that demonstrates the formulation
        works, and the one a patch test should assert on.

        Ordered like the surface element generator's ``<prefix>_nodes`` node set of the slave
        surface (see :attr:`slaveSurfaceNodes`), so a ``fromExpression`` output over that set lines
        up. A parent-face node that is not itself a facet node -- possible only with
        ``triangulation=corner``, where the facets keep no midside nodes -- receives contact force
        but has no slot in that set, and is omitted here; use ``triangulation=midside`` for a
        complete picture.
        """

        return self.slave.nodalNormalForces(self._normalForcesOfPoints())

    # --- 2. The pairing: the master facet of every contact point, at every update of the connectivity -----------------

    def updateConnectivity(self, model: FEModel) -> bool:
        """Pair the contact points with the master surface at the configuration of the model, freeze their
        projections, and locate the paired points in the block (see the class documentation, *The pairing* and *The
        first update*). A law that pairs once pairs at the first update (:meth:`_tieAtTheFirstUpdate`), and after a new
        mesh by the tie it hands on (:meth:`_tieAgainAfterANewMesh`); see *The tie* and *A new mesh*.

        Called once per increment by the solver, before the discrete system is (re)built. The
        candidate search runs on the *flat* master facets, which keeps it robust; what is frozen from it is the
        parametric location of the closest point within the facet's *parent* face, from which the parent-face
        shape functions -- the ones that may go negative at a corner -- are evaluated.

        Parameters
        ----------
        model
            The model, whose node field ``displacement`` places the master surface.

        Returns
        -------
        bool
            True if the nodes of the constraint changed.
        """

        if self.law.pairsOnce:
            if self._tieHandedOver is not None:
                return self._tieAgainAfterANewMesh(model)
            if not self._searchedBefore:
                return self._tieAtTheFirstUpdate(model)
            # A tied point stays with its master point; it is only located anew, for a kind of contact point whose
            # nodes change as it moves.
            return self._assignMasterFacets(self._assignedFacetIdx)

        coordinatesOfPoint = self.slave.currentPointCoordinates(model)
        coordinatesOfFacet, normalOfFacet, features = self._masterFacetsInTheConfigurationOf(model)
        facetCoords = list(coordinatesOfFacet)

        previousState = self._stateOfPoint
        firstSearch = not self._searchedBefore
        wasInContact = previousState == ContactState.IN_CONTACT

        # A point in contact looks for its closest facet however deep it is, and so does every point at the first
        # update, so that an overlap deeper than the search distance is found; every other point looks within the
        # search distance.
        searchedAtAnyDepth = wasInContact | firstSearch
        candidatesOfPoint = [None] * self.nPoints
        for points, searchDistance in (
            (np.flatnonzero(searchedAtAnyDepth), None),
            (np.flatnonzero(~searchedAtAnyDepth), self.searchDistance),
        ):
            if len(points):
                for point, candidates in zip(
                    points, closestFacetCandidates(coordinatesOfPoint[points], facetCoords, searchDistance)
                ):
                    candidatesOfPoint[point] = candidates
        closestFacet, weights, distance = closestFacets(coordinatesOfPoint, facetCoords, candidatesOfPoint)

        withinTheSearchDistance = (
            np.ones(self.nPoints, dtype=bool) if self.searchDistance is None else distance <= self.searchDistance
        )
        reached = (closestFacet >= 0) & (searchedAtAnyDepth | withinTheSearchDistance)
        # Seen outside of the master body before: approaching or in contact at the previous update, or, at the first
        # update, where the bodies are as the model gives them, within the search distance.
        wasSeen = wasInContact | (previousState == ContactState.APPROACHING) | (firstSearch & withinTheSearchDistance)

        # The gap to the closest facet tells inside from outside, also at the edges and the vertices of a master that
        # is not convex.
        gap = np.zeros(self.nPoints)
        dGap_dPoint = np.zeros((self.nPoints, self.nDim))
        gap[reached], dGap_dPoint[reached] = self._masterFacets.gapsAndGradients(
            coordinatesOfPoint[reached],
            closestFacet[reached],
            weights[reached],
            coordinatesOfFacet,
            normalOfFacet,
            features,
        )
        penetrating = reached & (gap < 0.0)

        # A point in contact is pushed back out the way it came: it slides on over the master surface where that is
        # smooth and around a re-entrant crease, but does not move across a convex crease, onto the side of a corner
        # it is near or the far side of a thin master body.
        previousFacet = self._assignedFacetIdx
        mustStay = np.flatnonzero(penetrating & wasInContact)
        mustStay = mustStay[
            self._masterFacets.acrossAConvexCrease(
                previousFacet[mustStay], closestFacet[mustStay], coordinatesOfFacet, normalOfFacet
            )
        ]
        if len(mustStay):
            closestFacet[mustStay], weights[mustStay] = self._closestFacetsOnThisSideOfTheConvexCreases(
                coordinatesOfPoint[mustStay],
                previousFacet[mustStay],
                [candidatesOfPoint[point] for point in mustStay],
                coordinatesOfFacet,
                normalOfFacet,
            )
            gap[mustStay], dGap_dPoint[mustStay] = self._masterFacets.gapsAndGradients(
                coordinatesOfPoint[mustStay],
                closestFacet[mustStay],
                weights[mustStay],
                coordinatesOfFacet,
                normalOfFacet,
                features,
            )

        state = np.full(self.nPoints, ContactState.FREE, dtype=np.int8)
        # (a point searched at any depth that is outside the master body is free beyond the search distance)
        state[reached & ~penetrating & withinTheSearchDistance] = ContactState.APPROACHING
        state[penetrating & wasSeen] = ContactState.IN_CONTACT
        state[penetrating & ~wasSeen] = ContactState.FOUND_INSIDE
        # a point found inside that no facet reaches any more has gone deeper, not out
        state[~reached & (previousState == ContactState.FOUND_INSIDE)] = ContactState.FOUND_INSIDE

        self._reportEntriesAndPointsFoundInside(previousState, state, gap, firstSearch)
        self._stateOfPoint = state
        self._searchedBefore = True

        paired = (state == ContactState.APPROACHING) | (state == ContactState.IN_CONTACT)
        newAssignment = np.where(paired, closestFacet, -1)
        pairedFacet = newAssignment[paired]
        self._freezeProjections(paired, pairedFacet, weights[paired], dGap_dPoint[paired])
        hasChanged = self._assignMasterFacets(newAssignment)

        offsetsTheInitialOverlap = self.adjustsTheInitialOverlap or self.initialOverlapRampTime > 0.0
        if firstSearch and self._initialOverlapTime is None and offsetsTheInitialOverlap:
            self._recordTheInitialOverlap(model)

        return hasChanged

    def _closestFacetsOnThisSideOfTheConvexCreases(
        self,
        coordinatesOfPoints: np.ndarray,
        previousFacets: np.ndarray,
        candidatesOfPoints: list,
        coordinatesOfFacet: np.ndarray,
        normalOfFacet: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """The facets of points in contact whose closest facet lies across a convex crease from their facet: each
        point walks over the master, step by step, until its facet no longer changes (see the class documentation,
        *The pairing*). A step goes from the facet to the closest of the facets around it (its patch, see
        :meth:`~edelweissfe.utils.facetcontactgeometry.FacetedSurface.patchOfFacet`) and of the point's candidates
        that do not lie across a convex crease from it.

        One step is not enough: a point that slid past several turns of the master since the last update reaches, in
        one step, only the facets on this side of the first. It would move on by one more step in every update, also
        in an update repeated at the same configuration -- a resumed step repeats one -- and pair differently from the
        uninterrupted run. At the facet where the walk ends, it pairs the same however often the update is repeated.

        Every step goes to a closer facet (or an equally close one listed earlier), so the walk ends; it is bounded by
        :data:`maximumStepsOfThePairingWalk`, and reaching the bound is reported.

        Parameters
        ----------
        coordinatesOfPoints
            The current coordinates of the points.
        previousFacets
            The facet each point was paired with.
        candidatesOfPoints
            The candidate facets of each point, see
            :func:`~edelweissfe.utils.facetcontactgeometry.closestFacetCandidates`.
        coordinatesOfFacet
            The coordinates of all master facets.
        normalOfFacet
            Their unit normals.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            The facet of each point and the weights of its closest point in that facet.
        """

        facetCoords = list(coordinatesOfFacet)
        facets = previousFacets.copy()
        weights = np.zeros((len(facets), coordinatesOfFacet.shape[1]))
        walking = np.arange(len(facets))
        for _ in range(maximumStepsOfThePairingWalk):
            onThisSide = []
            for k in walking:
                candidates = np.union1d(candidatesOfPoints[k], self._masterFacets.patchOfFacet(facets[k])).astype(
                    np.intp
                )
                across = self._masterFacets.acrossAConvexCrease(
                    np.full(len(candidates), facets[k]), candidates, coordinatesOfFacet, normalOfFacet
                )
                onThisSide.append(candidates[~across].tolist())
            nextFacets, weights[walking], _ = closestFacets(coordinatesOfPoints[walking], facetCoords, onThisSide)
            moved = nextFacets != facets[walking]
            facets[walking] = nextFacets
            walking = walking[moved]
            if not len(walking):
                return facets, weights

        self.journal.message(
            f"contact '{self.name}': {len(walking)} contact point(s) still moved on over the master after "
            f"{maximumStepsOfThePairingWalk} steps of the pairing; they keep the facet of the last step, and an update "
            "repeated at this configuration may pair them differently. Update the connectivity more often.",
            self.identification,
            level=1,
        )
        return facets, weights

    def _masterFacetsInTheConfigurationOf(self, model: FEModel) -> tuple[np.ndarray, np.ndarray, dict]:
        """The master facets in the configuration of the model: their coordinates, unit normals, and features (see
        :meth:`~edelweissfe.utils.facetcontactgeometry.FacetedSurface.features`).

        Parameters
        ----------
        model
            The model, whose node field ``displacement`` places the master surface.

        Returns
        -------
        tuple
            The coordinates of the facets (nFacets, nVerticesOfAFacet, nDim), their normals, and their features.
        """

        coordinatesOfFacet = currentNodeCoordinates(self._masterNodes, model, self._referenceCoordinatesOfMasterNodes)[
            self._masterNodesOfFacet
        ]
        normalOfFacet, _ = facetNormalAndMeasure(coordinatesOfFacet)
        return coordinatesOfFacet, normalOfFacet, self._masterFacets.features(coordinatesOfFacet, normalOfFacet)

    def _freezeProjections(self, paired: np.ndarray, facets: np.ndarray, weights: np.ndarray, dGap_dPoint: np.ndarray):
        """Freeze the projections of the paired points, and clear those of the others.

        The clamped barycentric weights locate the closest point in the flat facet; that location is mapped into the
        parent face's parametric space, and the parent face's shape functions are evaluated there. Frozen with them is
        the gradient of the gap with respect to the point, :math:`\\bar{n}`.

        Parameters
        ----------
        paired
            Which points are paired, a mask over all points.
        facets
            The facet of each paired point.
        weights
            The barycentric weights of the closest point of each paired point in its facet.
        dGap_dPoint
            The gradient of the gap of each paired point with respect to the point, a unit vector.
        """

        parametric = np.einsum("pa,pad->pd", weights, self._vertexParametricCoordsOfFacet[facets])
        self._frozenMasterShapeFunctions[:] = 0.0
        self._frozenNormals[:] = 0.0
        if len(facets):
            self._frozenMasterShapeFunctions[paired] = parentFaceShapeFunctions(self._masterParentFaceType, parametric)
            self._frozenNormals[paired] = dGap_dPoint

    def _tieAtTheFirstUpdate(self, model: FEModel) -> bool:
        """The first pairing of a law that pairs once (see the class documentation, *The tie*): every contact point
        within ``searchDistance`` of the master surface (every point, without a search distance), in the configuration
        of the model, is tied to the closest point of its closest facet, mapped onto the facet's parent face, and rests
        where it is. This decides, once, which points are tied and what to: a new mesh hands it on
        (:meth:`_tieAgainAfterANewMesh`).

        Parameters
        ----------
        model
            The model, whose node field ``displacement`` places both sides.

        Returns
        -------
        bool
            True if the nodes of the constraint changed.
        """

        coordinatesOfPoint = self.slave.currentPointCoordinates(model)
        masterSurface = self._masterFacetsInTheConfigurationOf(model)
        facetCoords = list(masterSurface[0])

        closestFacet, weights, distance = closestFacets(
            coordinatesOfPoint,
            facetCoords,
            closestFacetCandidates(coordinatesOfPoint, facetCoords, self.searchDistance),
        )
        tied = closestFacet >= 0
        if self.searchDistance is not None:
            tied &= distance <= self.searchDistance

        hasChanged = self._tie(
            model,
            coordinatesOfPoint,
            masterSurface,
            tied,
            closestFacet[tied],
            weights[tied],
            np.zeros((np.count_nonzero(tied), self.nDim)),
        )

        nTied = int(np.count_nonzero(tied))
        largestGap = float(np.linalg.norm(self._restingOffset[tied], axis=1).max()) if nTied else 0.0
        self.journal.message(
            f"contact '{self.name}' (type={self.law.name}): {nTied}/{self.nPoints} points paired for good, largest gap "
            f"or overlap at rest {largestGap:.3e}",
            self.identification,
        )
        if nTied < self.nPoints and self.searchDistance is not None:
            self.journal.message(
                f"contact '{self.name}' (type={self.law.name}): {self.nPoints - nTied} point(s) farther than the "
                f"search distance ({self.searchDistance:.3e}) from the master surface are not paired, for good.",
                self.identification,
                level=1,
            )

        return hasChanged

    def _tieAgainAfterANewMesh(self, model: FEModel) -> bool:
        """Tie the points by the tie a new mesh handed on to them (:class:`TieOfThePoints`, see the class
        documentation, *A new mesh*): the points it ties, no others, whatever their distance to the master now; each
        one to the closest point of the master surface to its master point in the reference configuration, of the
        closest facet, mapped onto the facet's parent face; and displaced from rest as it was handed on.

        Parameters
        ----------
        model
            The model, whose node field ``displacement`` places both sides.

        Returns
        -------
        bool
            True if the nodes of the constraint changed.
        """

        handedOver, self._tieHandedOver = self._tieHandedOver, None
        facets, weights = self._closestPointsOfTheMasterAt(handedOver.masterPoints)
        return self._tie(
            model,
            self.slave.currentPointCoordinates(model),
            self._masterFacetsInTheConfigurationOf(model),
            handedOver.tied,
            facets,
            weights,
            handedOver.displacementsFromRest,
        )

    def _tie(
        self,
        model: FEModel,
        coordinatesOfPoint: np.ndarray,
        masterSurface: tuple,
        tied: np.ndarray,
        facets: np.ndarray,
        weights: np.ndarray,
        displacementsFromRest: np.ndarray,
    ) -> bool:
        """Tie the given points to the given points of the master facets, displaced from rest by
        ``displacementsFromRest`` in the configuration of the model: freeze their projections, locate them, and record
        their resting offsets :math:`\\boldsymbol{r}_0 = (\\boldsymbol{x}_p - N^m \\cdot x^m) - \\boldsymbol{d}`, with
        the relative position measured from the origins as an evaluation measures it (see the class documentation,
        *Measured from the origins*), so that the next evaluation at the model's state finds exactly ``d``.

        Parameters
        ----------
        model
            The model, whose node field ``displacement`` places both sides.
        coordinatesOfPoint
            The coordinates of all points in the configuration of the model.
        masterSurface
            The master facets in the configuration of the model (:meth:`_masterFacetsInTheConfigurationOf`).
        tied
            Which points are tied, a mask over all points.
        facets
            The facet of each tied point.
        weights
            The barycentric weights of its master point in that facet.
        displacementsFromRest
            The displacement from rest of each tied point, (nTied, nDim).

        Returns
        -------
        bool
            True if the nodes of the constraint changed.
        """

        _, dGap_dPoint = self._masterFacets.gapsAndGradients(coordinatesOfPoint[tied], facets, weights, *masterSurface)
        self._freezeProjections(tied, facets, weights, dGap_dPoint)
        self._stateOfPoint = np.where(tied, ContactState.TIED, ContactState.FREE).astype(np.int8)
        self._searchedBefore = True
        assignment = np.full(self.nPoints, -1, dtype=np.intp)
        assignment[tied] = facets
        hasChanged = self._assignMasterFacets(assignment)

        relativePositions = self._relativePositionsAtZeroSolutionOfLocated + self._relativeDisplacementsOfLocatedPoints(
            *self._solutionOfTheBlockInTheModel(model)
        )
        self._restingOffset[tied] = relativePositions - displacementsFromRest
        self._restingOffsetOfLocated = self._restingOffset[self._locatedPoints]
        self._displacementsFromRestAtTheLastAcceptedState[tied] = displacementsFromRest
        return hasChanged

    def _tieOfThePoints(self) -> TieOfThePoints | None:
        """What the tie has decided for the current points: the tie a new mesh handed on to them, if they have not been
        tied by it yet, otherwise the points tied, their master points in the reference configuration, and their
        displacements from rest at the last accepted state. None before the first update, which decides it.

        Returns
        -------
        TieOfThePoints | None
            The tie of the points.
        """

        if self._tieHandedOver is not None:
            return self._tieHandedOver
        if not self._searchedBefore:
            return None
        tied = self._stateOfPoint == ContactState.TIED
        facets = self._assignedFacetIdx[tied]
        masterPoints = np.einsum(
            "ai,aij->aj",
            self._frozenMasterShapeFunctions[tied],
            self._masterParentRefCoords[facets].reshape((len(facets), self.nMasterParentNodes, self.nDim)),
        )
        return TieOfThePoints(tied, masterPoints, self._displacementsFromRestAtTheLastAcceptedState[tied])

    def _tieOfTheRegeneratedPoints(
        self, tieOfTheOldPoints: TieOfThePoints, originsOfTheOldPoints: np.ndarray
    ) -> TieOfThePoints:
        """The tie of regenerated contact points, from the tie of the points they replace, in the reference
        configuration: a new point is tied if the old point nearest to it was, and its master point and its
        displacement from rest are interpolated from the old tied points (:func:`affineInterpolationAlongASurface`).

        Parameters
        ----------
        tieOfTheOldPoints
            The tie of the old points.
        originsOfTheOldPoints
            Where the old points are in the reference configuration, (nOldPoints, nDim).

        Returns
        -------
        TieOfThePoints
            The tie of the current, regenerated points.
        """

        if not tieOfTheOldPoints.tied.any():
            nothing = np.zeros((0, self.nDim))
            return TieOfThePoints(np.zeros(self.nPoints, dtype=bool), nothing, nothing)

        origins = self.slave.referenceCoordinates()
        _, nearestOldPoint = cKDTree(originsOfTheOldPoints).query(origins)
        tied = tieOfTheOldPoints.tied[nearestOldPoint]
        neighbours, weights = affineInterpolationAlongASurface(
            originsOfTheOldPoints[tieOfTheOldPoints.tied], origins[tied]
        )

        def interpolated(values: np.ndarray) -> np.ndarray:
            return np.einsum("ak,akj->aj", weights, values[neighbours])

        return TieOfThePoints(
            tied, interpolated(tieOfTheOldPoints.masterPoints), interpolated(tieOfTheOldPoints.displacementsFromRest)
        )

    def _reportEntriesAndPointsFoundInside(
        self, previousState: np.ndarray, state: np.ndarray, gap: np.ndarray, firstSearch: bool
    ):
        """Report in the journal the points newly found inside the master body, and a point that enters contact
        deeper than any reported before, beyond half of the search distance.

        Parameters
        ----------
        previousState
            The states before this update.
        state
            The states after it.
        gap
            The gaps after it.
        firstSearch
            True at the first contact search of these contact points (the first update, or the update after they were
            regenerated), where a point found inside overlaps the master deeper than the search distance.
        """

        newlyFoundInside = (state == ContactState.FOUND_INSIDE) & (previousState != ContactState.FOUND_INSIDE)
        if newlyFoundInside.any():
            if firstSearch:
                how = (
                    "overlap the master body deeper than the search distance at the first contact search of these "
                    "contact points (the first update, or after they were regenerated)"
                )
                remedy = "If the overlap is real, increase the search distance."
            else:
                how = "found inside the master body without having been seen outside of it within the search distance"
                remedy = "Increase the search distance, or reduce the time increment."
            self.journal.message(
                f"contact '{self.name}': {np.count_nonzero(newlyFoundInside)} contact point(s) {how} "
                f"({self.searchDistance:.3e}); they get no force. {remedy}",
                self.identification,
                level=1,
            )

        entering = (state == ContactState.IN_CONTACT) & (previousState == ContactState.APPROACHING)
        if self.searchDistance is not None and entering.any():
            entryDepth = float(-gap[entering].min())
            if (
                entryDepth > _entryDepthWarningFraction * self.searchDistance
                and entryDepth > self._deepestEntryReported
            ):
                self._deepestEntryReported = entryDepth
                self.journal.message(
                    f"contact '{self.name}': a contact point entered {entryDepth:.3e} deep since the last update of the "
                    f"connectivity, more than {_entryDepthWarningFraction:.0%} of the search distance "
                    f"({self.searchDistance:.3e}): "
                    "points may soon tunnel through it unseen.",
                    self.identification,
                    level=1,
                )

    def _assignMasterFacets(self, newAssignment: np.ndarray) -> bool:
        """Assign each contact point its master facet (searched or restored), and locate the paired points."""

        hasChanged = not np.array_equal(newAssignment, self._assignedFacetIdx)
        self._assignedFacetIdx = newAssignment
        withASpringAtTheLastEvaluation = self.numberOfPointsWithASpring

        previousNodes = self._nodes
        self._locatePairedPoints()
        if self._nodes != previousNodes:
            hasChanged = True

        # A point that lost its facet has no gap: reset it to zero, so that getGaps() does not keep its last value.
        self._gapCurrent[newAssignment < 0] = 0.0

        # Assembly is paid per paired point, load is carried by the points with a spring (counted at the last force
        # evaluation). Reported only when it changes.
        activity = (len(self._locatedPoints), withASpringAtTheLastEvaluation)
        if activity != self._lastReportedActivity:
            self._lastReportedActivity = activity
            self.journal.message(
                f"contact '{self.name}': {activity[0]}/{self.nPoints} paired, {activity[1]} with a spring",
                self.identification,
                level=2,
            )

        return hasChanged

    def _searchLayout(self) -> dict[str, np.ndarray]:
        return self.slave.searchLayout() | {
            "masterFacets": np.array([facet.elNumber for facet in self.facetElements], dtype=np.int64),
        }

    def _frozenProjection(self) -> dict[str, np.ndarray]:
        return {
            "assignedFacetIdx": self._assignedFacetIdx,
            "stateOfPoint": self._stateOfPoint,
            "masterShapeFunctions": self._frozenMasterShapeFunctions,
            "normals": self._frozenNormals,
            "initialOverlap": self._initialOverlap,
            "restingOffset": self._restingOffset,
            "displacementsFromRestAtTheLastAcceptedState": self._displacementsFromRestAtTheLastAcceptedState,
            # Empty before the first search.
            "initialOverlapTime": np.array([] if self._initialOverlapTime is None else [self._initialOverlapTime]),
        }

    def _adoptFrozenProjection(self, projection: dict[str, np.ndarray]) -> bool:
        self._stateOfPoint = np.array(projection["stateOfPoint"], dtype=np.int8)
        self._searchedBefore = True
        self._frozenMasterShapeFunctions = np.array(projection["masterShapeFunctions"]).reshape(
            self._frozenMasterShapeFunctions.shape
        )
        self._frozenNormals = np.array(projection["normals"]).reshape(self._frozenNormals.shape)
        self._initialOverlap = np.array(projection["initialOverlap"]).reshape(self._initialOverlap.shape)
        self._restingOffset = np.array(projection["restingOffset"]).reshape(self._restingOffset.shape)
        # The state of the points the run goes on from: a new mesh hands it on, not what was evaluated since (a
        # fall-back), nor a tie handed on before (a resume, whose replay of the topology rebuilt the points).
        self._displacementsFromRestAtTheLastAcceptedState = np.array(
            projection["displacementsFromRestAtTheLastAcceptedState"]
        ).reshape(self._displacementsFromRestAtTheLastAcceptedState.shape)
        self._displacementsFromRestOfLocatedAtTheLastEvaluation = None
        self._tieHandedOver = None
        initialOverlapTime = np.asarray(projection["initialOverlapTime"]).ravel()
        self._initialOverlapTime = float(initialOverlapTime[0]) if len(initialOverlapTime) else None
        assignment = np.array(projection["assignedFacetIdx"], dtype=np.intp)
        if self.slave.locatedOnlyWhenConnected:
            # Located at the next update of the connectivity, once the entities of the points are connected.
            self._assignedFacetIdx = assignment
            return False
        return self._assignMasterFacets(assignment)

    def refresh(self, model: FEModel, change) -> bool:
        """Rebuild the sides ``change`` touched (see the class documentation, *A new mesh*): the contact points are
        regenerated from their regenerated facets if touched, and the master surface is rebuilt from its facet set.

        The facets themselves were already regenerated in the topology-update phase by the implicit ``surfaceFacets``
        modifier; this constraint is a pure reader of the resulting element sets. What is cached about the surfaces is
        derived from the reference configuration and rebuilt wholesale. What the pairing has learnt about the points
        stays with them if only the master was retiled -- the facet of a paired point carried over to the facet of the
        new master where the old one lay -- and is lost with regenerated contact points (reported). A law that pairs
        once hands on what it decided for the points (:class:`TieOfThePoints`), to the same points after a new mesh of
        the master, interpolated onto the new points of a regenerated slave side. The update of the connectivity that
        follows every rebuild pairs the points anew: until then, no point is located.

        Parameters
        ----------
        model
            The model, whose element sets hold the regenerated facets.
        change
            The change of the mesh.

        Returns
        -------
        bool
            True if ``change`` touched either side.
        """

        masterRecipe = model.contactFacetRecipes.get(self._masterSurfaceSetName)
        touchedSlave = self.slave.isTouchedBy(model, change)
        touchedMaster = masterRecipe is not None and change.touchesSurface(masterRecipe[0])
        if not (touchedSlave or touchedMaster):
            return False

        # What a tie has decided for the points, taken before either side changes: it is handed on.
        tieOfThePoints = self._tieOfThePoints() if self.law.pairsOnce else None

        if touchedSlave:
            initialOverlap = self._initialOverlap
            originsOfTheOldPoints = self.slave.referenceCoordinates() if tieOfThePoints is not None else None
            self.slave = self.slave.regenerated(model)
            lost = (
                f"; the offsets of their initial overlap (deepest {-initialOverlap.min():.3e}) are lost, and an "
                "overlap they still have is pushed apart at once"
                if initialOverlap.any()
                else ""
            )
            how = (
                "tied where the points they replace were tied, to the master points and displaced from rest as "
                "interpolated from them"
                if self.law.pairsOnce
                else "paired as at the first update, so a point that overlaps the master deeper than the search "
                "distance is found inside and loses its force"
            )
            self.journal.message(
                f"contact '{self.name}': the contact points were regenerated; they are {how}{lost}.",
                self.identification,
                level=1,
            )

        # Regenerated contact points, and the points of a law that pairs once, are paired from scratch: the contact as
        # at the first update, the tie by the tie handed on.
        pairedFromScratch = touchedSlave or self.law.pairsOnce

        # The paired points that are kept, and where their facets lay on the master (the centroids).
        keptPairedPoints = (
            np.flatnonzero(self._assignedFacetIdx >= 0) if not pairedFromScratch else np.zeros(0, np.intp)
        )
        whereTheirFacetsLay = self._referenceCoordinatesOfMasterNodes[
            self._masterNodesOfFacet[self._assignedFacetIdx[keptPairedPoints]]
        ].mean(axis=1)

        self._buildMaster(model.elementSets[self._masterSurfaceSetName])
        if pairedFromScratch:
            self._resetContactPoints()
        self._resetPairing()
        if tieOfThePoints is not None:
            self._tieHandedOver = (
                self._tieOfTheRegeneratedPoints(tieOfThePoints, originsOfTheOldPoints)
                if touchedSlave
                else tieOfThePoints
            )
        if len(keptPairedPoints):
            # Where the next update starts the walk of a point in contact, as from the facet of the previous update.
            self._assignedFacetIdx[keptPairedPoints], _ = self._closestPointsOfTheMasterAt(whereTheirFacetsLay)
        return True

    def _closestPointsOfTheMasterAt(self, referencePoints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The closest point of the master to each point, in the reference configuration: its facet, and its
        barycentric weights in it.

        Parameters
        ----------
        referencePoints
            The points, (nPoints, nDim).

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            The facets, (nPoints,), and the weights, (nPoints, nVerticesOfAFacet).
        """

        facetCoords = list(self._referenceCoordinatesOfMasterNodes[self._masterNodesOfFacet])
        if not len(referencePoints):
            return np.zeros(0, dtype=np.intp), np.zeros((0, self._masterNodesOfFacet.shape[1]))
        facets, weights, _ = closestFacets(
            referencePoints, facetCoords, closestFacetCandidates(referencePoints, facetCoords, None)
        )
        return facets, weights

    # --- 3. An initial overlap: the part of it the gap does not count -------------------------------------------------

    def _recordTheInitialOverlap(self, model: FEModel):
        """Record the overlap of the paired points at the first contact search as the offsets of their gaps (see the
        class documentation, *The first update*), and report the deepest. The paired points are located; their gaps
        are measured as an evaluation measures them (:meth:`_displacementsFromRestOfLocatedPoints`), at the solution
        the model holds (:meth:`_solutionOfTheBlockInTheModel`), so that an adjusted overlap leaves a point at rest
        exactly.

        Parameters
        ----------
        model
            The model, whose node field ``displacement`` places both sides.
        """

        self._initialOverlapTime = model.time
        located = self._locatedPoints
        gaps = np.einsum(
            "aj,aj->a",
            self._normalsOfLocated,
            self._displacementsFromRestOfLocatedPoints(*self._solutionOfTheBlockInTheModel(model)),
        )
        self._initialOverlap[located] = np.minimum(gaps, 0.0)
        self._initialOverlapOfLocated = self._initialOverlap[located]

        overlapping = np.count_nonzero(self._initialOverlap)
        if overlapping:
            treatment = (
                "adjusted: kept as the state at rest"
                if self.adjustsTheInitialOverlap
                else f"pushed apart over {self.initialOverlapRampTime:.3e}"
            )
            self.journal.message(
                f"contact '{self.name}': {overlapping} point(s) overlap the master at the first search, the deepest by "
                f"{-self._initialOverlap.min():.3e}; {treatment}",
                self.identification,
            )

    def _solutionOfTheBlockInTheModel(self, model: FEModel) -> tuple[np.ndarray, np.ndarray]:
        """The solution of the block at the state of the model, as at an update of the connectivity, where the points
        are paired: taken from the model's displacement field, without an increment.

        Parameters
        ----------
        model
            The model, whose node field ``displacement`` places both sides.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            The solution and the (zero) increment of the block.
        """

        U = currentNodeDisplacements(self._nodes, model, self.nDim).ravel()
        return U, np.zeros_like(U)

    def _uncountedFractionOfInitialOverlapAt(self, timeStep: TimeStep | None) -> float:
        """The fraction of the initial overlap the gap does not count at the time of ``timeStep``. Adjusted: one, for
        good. Pushed apart: zero at once; over a ramp time one at the first contact search, zero after the ramp time,
        in between :math:`1 - s + \\sin(2 \\pi s) / (2 \\pi)` of the elapsed fraction :math:`s` of the ramp
        time, which starts and ends at rest.

        Parameters
        ----------
        timeStep
            The time step of the evaluation; without an initial overlap to ramp, it may be None.

        Returns
        -------
        float
            The fraction.
        """

        if self._initialOverlapTime is None:
            return 0.0
        if self.adjustsTheInitialOverlap:
            return 1.0
        if timeStep is None:
            return 0.0
        s = min(max((timeStep.totalTime - self._initialOverlapTime) / self.initialOverlapRampTime, 0.0), 1.0)
        return 1.0 - s + np.sin(2.0 * np.pi * s) / (2.0 * np.pi)

    # --- 4. The bound of the stiffness for the critical time step of NED ----------------------------------------------

    def addBoundOfAbsoluteStiffnessTimes(
        self,
        U_np: np.ndarray,
        dU: np.ndarray,
        weights: np.ndarray,
        bound: np.ndarray,
        timeStep: TimeStep,
        nextIncrement: NextIncrement | None = None,
    ):
        """Add the row sums of the absolute value of the stiffness of the springs, weighted by column; see
        :meth:`~edelweissfe.constraints.base.constraintbase.ConstraintBase.addBoundOfAbsoluteStiffnessTimes`.

        The springs are those of the force (:meth:`_springsOfLocatedPoints`), over the next increment with the smallest
        gap of each point within it: a point that could close within it has its spring. A spring :math:`p` acts with the
        vector :math:`\\boldsymbol{a}_p = [c, -N^m] \\otimes \\bar{n}_p`, the gradient of its gap, along its direction
        :math:`\\bar{n}_p`, with the stiffness :math:`k_p = \\mathrm{d}f/\\mathrm{d}g` its law gives it (the one of the
        tangent); the bound is the one of every penalty spring
        (:func:`~edelweissfe.constraints.base.penaltysprings.addBoundOfAbsoluteStiffnessOfSprings`), with the contact
        points' interpolation over their slave nodes and the parent face nodes of each point's master facet; each point
        has a copy of its own of both in the block.
        The tie's springs along different axes act on different components of the nodes.
        """

        located = self._locatedPoints
        if not len(located):
            return

        smallestGaps = None
        if nextIncrement is not None:
            smallestGaps = smallestGapsWithin(nextIncrement, self._gapsOfLocatedPointsAt, U_np, dU, timeStep)
        _, springs = self._springsOfLocatedPoints(
            self._displacementsFromRestOfLocatedPoints(U_np, dU), timeStep, smallestGaps
        )
        if not springs:
            return

        nSlaveDof = self._nSlaveDof
        slaveInterpolationOfLocated = self.slave.interpolationMatrixOfLocatedPoints()
        dofsOfSlaveNodes = np.arange(nSlaveDof).reshape((-1, self.nDim))
        dofsOfMasterNodesOfLocated = nSlaveDof + np.arange(len(located) * self.nMasterParentNodes * self.nDim).reshape(
            (len(located), self.nMasterParentNodes, self.nDim)
        )
        for spring in springs:
            points = spring.points
            gradient = [
                SpringGradientOnSharedNodes(slaveInterpolationOfLocated[points], spring.directions, dofsOfSlaveNodes),
                SpringGradientOnOwnNodes(
                    -self._masterShapeFunctionsOfLocated[points], spring.directions, dofsOfMasterNodesOfLocated[points]
                ),
            ]
            addBoundOfAbsoluteStiffnessOfSprings(bound, weights, spring.stiffness, gradient)
