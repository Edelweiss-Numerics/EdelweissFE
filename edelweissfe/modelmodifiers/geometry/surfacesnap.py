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

"""Analytical-surface node snapping for AMR-refined curved boundaries (HEX20 only).

A coarse mesh represents a curved boundary (e.g. a borehole) as a polygon of flat facets. When
AMR h-refines elements attached to that boundary, the new nodes it creates do NOT improve
geometric fidelity toward the true curve -- ``Hex20Topology.subdivide`` evaluates the PARENT
element's own (possibly already-flat) isoparametric map, so a straight parent edge only ever
produces more straight sub-edges. This purely reactive model modifier reacts to an AMR
modifier's :class:`~edelweissfe.models.modelchange.ModelChange` and snaps exactly the newly
created nodes on a tracked boundary onto a specified analytical surface (currently: a cylinder),
leaving every pre-existing node untouched.

See ``PLAN_BOREHOLE_GEOMETRY_SNAP.md`` (EdelweissFE repo root) for the full design rationale, the
Phase 1 prototype that validated this approach against a real mesh, and open items.

**Restart safety.** Which element faces are "on the tracked surface" is decision-side state that
must survive a checkpoint/resume exactly like hAdaptivity's own tracked node sets do: by living in
a genuine :class:`~edelweissfe.sets.nodeset.NodeSet` (:attr:`ModelModifier._wallSetName`, private
to this modifier instance) that a restart rebuilds by REPLAYING this modifier's own recorded
:class:`SnapPlan` history -- the same mechanism, not a parallel one. This is why
:meth:`ModelModifier.apply` grows that node set (via ``model.nodeSets[...].add(...)``) for every
newly-confirmed wall node, independent of whether that node's coordinates actually moved this
round: a round can legitimately snap nothing (every candidate is a hanging-node collision, or the
quality safeguard vetoed all of them) while still discovering real new wall-face membership that
a LATER round needs to know about. Growing the tracked node set makes
:class:`~edelweissfe.models.modelchange.ModelChange` non-empty (via ``changedNodeSets``) even when
``movedNodes`` is empty, so that round is always recorded and always replays -- no state is ever
silently lost across a restart.
"""

from dataclasses import dataclass

import numpy as np

from edelweissfe.adaptivity.hex20topology import Hex20Topology
from edelweissfe.constraints.hangingnode import Constraint as HangingNodeConstraint
from edelweissfe.journal.journal import Journal
from edelweissfe.modelmodifiers.base.modelmodifierbase import ModelModifierBase
from edelweissfe.models.femodel import FEModel
from edelweissfe.models.modelchange import ModelChange
from edelweissfe.models.modelchangeobserver import ModelChangeType
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.utils.schema import buildSchemaFromOptions, schemaField


@dataclass(frozen=True)
class SurfaceSnapSchema:
    """The options this model modifier accepts, owned by this module."""

    moduleOptions: dict = schemaField(description="Internal", dtype=dict, default_factory=dict)
    nodeSet: str | None = schemaField(
        description=(
            "Node set naming the boundary to snap onto the analytical surface, e.g. a borehole "
            "wall exported by the mesh generator (its nodes must already lie exactly on the true "
            "surface -- only nodes AMR creates later on faces fully within this set are ever "
            "moved)."
        ),
        dtype=str,
        default=None,
        required=True,
    )
    originX: float = schemaField(description="X coordinate of a point on the cylinder axis.", dtype=float, default=0.0)
    originY: float = schemaField(description="Y coordinate of a point on the cylinder axis.", dtype=float, default=0.0)
    originZ: float = schemaField(description="Z coordinate of a point on the cylinder axis.", dtype=float, default=0.0)
    axisX: float = schemaField(description="X component of the cylinder axis direction.", dtype=float, default=0.0)
    axisY: float = schemaField(description="Y component of the cylinder axis direction.", dtype=float, default=1.0)
    axisZ: float = schemaField(description="Z component of the cylinder axis direction.", dtype=float, default=0.0)
    radius: float | None = schemaField(description="Cylinder radius.", dtype=float, default=None, required=True)
    midsideNodes: str = schemaField(
        description=(
            "'straight' (default): recompute a new midside node as the mean of its two "
            "(already-snapped) corner endpoints, keeping the edge a straight chord. 'curved': "
            "independently project the midside node onto the cylinder too, giving a true curved "
            "quadratic edge."
        ),
        dtype=str,
        default="straight",
    )
    qualityDropThreshold: float = schemaField(
        description=(
            "An affected element's minimum corner Jacobian determinant must stay above this "
            "fraction of its pre-snap value (and stay positive), or the ENTIRE round's snap is "
            "skipped (nothing is moved, but the boundary-face bookkeeping still advances so a "
            "later refinement pass gets another chance)."
        ),
        dtype=float,
        default=0.5,
    )


@dataclass(frozen=True)
class SnapPlan:
    """One geometry-snap decision.

    ``labels``/``coords`` are the exact final coordinates to write -- computed once, in
    :meth:`ModelModifier.plan`, so :meth:`ModelModifier.apply` is a pure "write these values"
    operation with no decision logic of its own. ``newWallNodes`` are EVERY node label newly
    confirmed to lie on the tracked analytical surface this round -- snapped, hanging-node-
    collision-skipped, and quality-vetoed alike, since topological membership on a wall face does
    not depend on whether the node's coordinates actually moved. :meth:`ModelModifier.apply` adds
    all of them to the modifier's own tracked :class:`~edelweissfe.sets.nodeset.NodeSet`, which is
    what makes wall-face detection (and this whole class of decision) restart-replay-safe -- see
    the module docstring.
    """

    labels: tuple
    coords: tuple
    newWallNodes: tuple

    def __init__(self, labels, coords, newWallNodes):
        object.__setattr__(self, "labels", tuple(int(label) for label in labels))
        object.__setattr__(self, "coords", tuple(tuple(float(x) for x in coord) for coord in coords))
        object.__setattr__(self, "newWallNodes", tuple(int(label) for label in newWallNodes))


def _projectOntoCylinder(point, origin, axis, radius):
    """Radially project ``point`` onto a cylinder given by a unit ``axis`` through ``origin``."""
    d = np.asarray(point, dtype=float) - origin
    axial = float(d @ axis) * axis
    radial = d - axial
    rho = float(np.linalg.norm(radial))
    if rho < 1e-9:
        raise ValueError(f"node at {point} sits on the cylinder axis; cannot project it radially onto the surface")
    return origin + axial + radial * (radius / rho)


class ModelModifier(ModelModifierBase):
    """Snaps newly AMR-created HEX20 boundary nodes onto an analytical cylinder."""

    #: Purely reactive: plan() returns None unless another modifier already refined the mesh.
    initiatesTopologyChanges = False

    #: Option schema for this model modifier, per OptionSchemaProvider.
    schema = SurfaceSnapSchema

    def __init__(self, name: str, model: FEModel, journal: Journal, *args, **kwargs):
        super().__init__(name, model, journal, *args, **kwargs)
        options = buildSchemaFromOptions(SurfaceSnapSchema, kwargs)

        if options.midsideNodes not in ("straight", "curved"):
            raise ValueError(
                f"surfaceSnap modifier {name!r}: 'midsideNodes' must be 'straight' or 'curved', "
                f"got {options.midsideNodes!r}."
            )
        if options.nodeSet not in model.nodeSets:
            raise ValueError(f"surfaceSnap modifier {name!r}: node set {options.nodeSet!r} does not exist.")

        self._nodeSetName = options.nodeSet
        self._origin = np.array([options.originX, options.originY, options.originZ], dtype=float)
        axis = np.array([options.axisX, options.axisY, options.axisZ], dtype=float)
        axisNorm = np.linalg.norm(axis)
        if axisNorm < 1e-12:
            raise ValueError(f"surfaceSnap modifier {name!r}: the cylinder axis direction must be nonzero.")
        self._axis = axis / axisNorm
        self._radius = options.radius
        self._midsideMode = options.midsideNodes
        self._qualityDropThreshold = options.qualityDropThreshold

        self._topology = Hex20Topology()
        refParams = self._topology.reference_node_param()
        #: local slot indices (0-19) that are corners, i.e. the nodes a face/edge actually spans.
        self._cornerSlots = {i for i, p in enumerate(refParams) if np.all(np.isclose(np.abs(p), 1.0))}

        # The tracked "known to lie exactly on the analytical surface" node set -- a genuine model
        # NodeSet, not private Python state, precisely so it is restart-replay-safe (see module
        # docstring). Private to this modifier instance: never referenced by a marker, BC, or any
        # other consumer, so growing it has no side effect beyond this modifier's own bookkeeping.
        self._wallSetName = f"__surfaceSnap_{name}_wallNodes"
        seedNodes = list(model.nodeSets[self._nodeSetName])
        model.nodeSets[self._wallSetName] = NodeSet(self._wallSetName, seedNodes)

        if not seedNodes:
            self._journal.message(
                f"surfaceSnap modifier {name!r}: node set {self._nodeSetName!r} is empty; nothing "
                "will ever be snapped unless it gains members.",
                "surfaceSnap",
                1,
            )

    def _hangingSlaveLabels(self, model: FEModel) -> set:
        labels = set()
        for constraint in model.multiPointConstraints.values():
            if isinstance(constraint, HangingNodeConstraint):
                labels |= {n.label for n in constraint.claimedSlaveNodes()}
        return labels

    def _minCornerJacobian(self, coords) -> float:
        dets = []
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    _, dN = self._topology.shape_functions_and_grad(sx, sy, sz)
                    dets.append(np.linalg.det(np.asarray(coords).T @ dN))
        return min(dets)

    def plan(self, model: FEModel, change: "ModelChange | None", step, timeStep: float) -> "SnapPlan | None":
        """Find newly-created nodes on a tracked wall face and decide their snapped positions.

        Read-only: computes the exact final coordinates to write (so :meth:`apply` needs no
        decision logic of its own), but does not mutate the model. See
        :meth:`~edelweissfe.modelmodifiers.base.modelmodifierbase.ModelModifierBase.plan`.
        """

        if change is None or not change.faceMap or not change.addedNodes:
            return None

        # Pass 1: every child face tiling a tracked wall face, and which of its 8 nodes are new.
        #
        # Cannot identify the wall face by looking up the PARENT element (change.faceMap's own
        # key): hAdaptivity has already REMOVED the parent from model.elements by the time this
        # runs -- only its children (change.faceMap's values) still exist. Instead: a face was on
        # the tracked wall iff every one of its REUSED (non-added) node labels, gathered across
        # every child tiling it, is already a member of the tracked wall set -- reused corners and
        # some reused edge-midpoints always exist for a genuine HEX20 face subdivision, so this
        # never needs the parent itself.
        wallLabels = {n.label for n in model.nodeSets[self._wallSetName]}
        wallFaceEntries = []  # (childElement, faceSlots, newLabelsOnFace)
        newWallNodes = set()  # every new node topologically on a wall face, snapped or not
        for (parentLabel, faceID), childPairs in change.faceMap.items():
            faceIdx = self._topology.faceid_to_face[faceID]
            faceSlots = self._topology.faces[faceIdx]
            childData = []  # (childElement, newOnFace)
            reusedFaceLabels = set()
            allChildrenExist = True
            for childLabel, childFaceID in childPairs:
                childEl = model.elements.get(childLabel)
                if childEl is None:
                    allChildrenExist = False
                    break
                faceLabels = {childEl.nodes[i].label for i in faceSlots}
                reusedFaceLabels |= faceLabels - change.addedNodes
                childData.append((childEl, faceLabels & change.addedNodes))
            if not allChildrenExist or not reusedFaceLabels or not reusedFaceLabels <= wallLabels:
                continue
            for childEl, newOnFace in childData:
                if newOnFace:
                    newWallNodes |= newOnFace
                    wallFaceEntries.append((childEl, faceSlots, newOnFace))

        if not newWallNodes:
            return None

        hangingSlaves = self._hangingSlaveLabels(model)
        collisions = set()
        labelsToCoord = {}

        # Pass 2: corners first (always radially projected, regardless of midside mode) -- later
        # midside computation in "straight" mode needs their resolved (post-snap) positions.
        for childEl, faceSlots, newOnFace in wallFaceEntries:
            for i in faceSlots:
                if i not in self._cornerSlots:
                    continue
                label = childEl.nodes[i].label
                if label not in newOnFace or label in labelsToCoord:
                    continue
                if label in hangingSlaves:
                    collisions.add(label)
                    continue
                labelsToCoord[label] = _projectOntoCylinder(
                    model.nodes[label].coordinates, self._origin, self._axis, self._radius
                )

        # Pass 3: midsides.
        for childEl, faceSlots, newOnFace in wallFaceEntries:
            edgeOf = None
            for i in faceSlots:
                if i in self._cornerSlots:
                    continue
                label = childEl.nodes[i].label
                if label not in newOnFace or label in labelsToCoord:
                    continue
                if label in hangingSlaves:
                    collisions.add(label)
                    continue
                if self._midsideMode == "curved":
                    labelsToCoord[label] = _projectOntoCylinder(
                        model.nodes[label].coordinates, self._origin, self._axis, self._radius
                    )
                else:
                    if edgeOf is None:
                        edgeOf = {
                            childEl.nodes[im].label: (childEl.nodes[ia].label, childEl.nodes[ib].label)
                            for ia, im, ib in self._topology.edges
                        }
                    a, b = edgeOf[label]
                    coordA = labelsToCoord.get(a, model.nodes[a].coordinates)
                    coordB = labelsToCoord.get(b, model.nodes[b].coordinates)
                    labelsToCoord[label] = 0.5 * (np.asarray(coordA, dtype=float) + np.asarray(coordB, dtype=float))

        if collisions:
            self._journal.message(
                f"surfaceSnap modifier {self._name!r}: {len(collisions)} new boundary node(s) are "
                f"also AMR hanging-node slaves and were NOT snapped (would break their MPC): "
                f"{sorted(collisions)}",
                "surfaceSnap",
                1,
            )

        # Quality safeguard: an affected element's min corner Jacobian must not invert or drop
        # below qualityDropThreshold of its pre-snap value. All-or-nothing per round, not per
        # node: a partial snap could leave a midside straight-mode mean referencing a corner that
        # was itself rolled back, which is not worth the complexity for what Phase 1 measured to
        # be a non-occurring case in practice. Either way, newWallNodes (topological membership)
        # is unaffected -- these nodes ARE on the wall face regardless of whether they got moved.
        affected = {childEl for childEl, _, newOnFace in wallFaceEntries if newOnFace & labelsToCoord.keys()}
        degraded = []
        for el in affected:
            before = self._minCornerJacobian([n.coordinates for n in el.nodes])
            afterCoords = [labelsToCoord.get(n.label, n.coordinates) for n in el.nodes]
            after = self._minCornerJacobian(afterCoords)
            if after <= 0.0 or (before > 0.0 and after / before < self._qualityDropThreshold):
                degraded.append((el.elNumber, before, after))
        if degraded:
            self._journal.message(
                f"surfaceSnap modifier {self._name!r}: {len(degraded)} affected element(s) would "
                f"drop below the quality threshold -- skipping ALL snapping this round (boundary-"
                f"face tracking still advances): {degraded}",
                "surfaceSnap",
                1,
            )
            labelsToCoord = {}

        return SnapPlan(
            labels=list(labelsToCoord.keys()), coords=list(labelsToCoord.values()), newWallNodes=newWallNodes
        )

    def apply(self, model: FEModel, plan: "SnapPlan") -> ModelChange:
        """Write the plan's coordinates and grow the tracked wall node set.

        Pure function of ``(model, plan)``: everything the decision depended on is already
        resolved in ``plan``. See
        :meth:`~edelweissfe.modelmodifiers.base.modelmodifierbase.ModelModifierBase.apply`.
        """

        for label, coord in zip(plan.labels, plan.coords):
            model.nodes[label].coordinates = np.array(coord, dtype=float)

        change = ModelChange(kind=ModelChangeType.GEOMETRY_CHANGE)
        change.movedNodes = set(plan.labels)

        if plan.newWallNodes:
            wallSet = model.nodeSets[self._wallSetName]
            existing = {n.label for n in wallSet}
            newLabels = [label for label in plan.newWallNodes if label not in existing]
            if newLabels:
                wallSet.add([model.nodes[label] for label in newLabels])
                # Growing this set is itself the meaningful, replay-worthy event: a round can
                # legitimately snap zero nodes (every candidate a hanging-node collision, or the
                # quality safeguard vetoed all of them) while still discovering real new wall-face
                # membership a later round needs. Marking changedNodeSets here -- not just when
                # movedNodes is non-empty -- is what makes updateTopology's isEmpty check record
                # (and therefore restart-replay) this round even when nothing moved.
                change.changedNodeSets.add(self._wallSetName)

        return change

    def encodePlan(self, plan: "SnapPlan") -> dict:
        """Serialize a :class:`SnapPlan` as flat numpy arrays for the topology history."""

        return {
            "labels": np.array(plan.labels, dtype=int),
            "coords": np.array(plan.coords, dtype=float).reshape(-1, 3),
            "newWallNodes": np.array(plan.newWallNodes, dtype=int),
        }

    def decodePlan(self, data: dict) -> "SnapPlan":
        """Inverse of :meth:`encodePlan`."""

        return SnapPlan(
            labels=[int(label) for label in data["labels"]],
            coords=[tuple(float(x) for x in row) for row in data["coords"]],
            newWallNodes=[int(label) for label in data["newWallNodes"]],
        )
