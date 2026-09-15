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
"""End-to-end test of the surfaceSnap model modifier (see the ``surfaceSnap`` section of
doc/source/documentation/modelmodifiers.rst) against a real (small) FEModel, driven through the
actual ``*modelModifier`` .inp pipeline together with hAdaptivity -- not just the geometry math in
isolation.

The coarse mesh here is deliberately built with genuinely FLAT inner-wall facets (edge midside
nodes placed at the Cartesian mean of their two corners, not on the true arc), reproducing the
"polygon, not circle" problem this feature exists to fix -- unlike a real Cubit export of an
actual cylindrical surface, whose own edge midsides already sit on the true arc.
"""

from pathlib import Path

import numpy as np
import pytest

from edelweissfe.adaptivity.hex20topology import Hex20Topology
from edelweissfe.helpers.inputfilehelpers import fillFEModelFromInputFile
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.utils.inputfileparser import parseInputFile

#: surfaceSnap's own tracked "confirmed exactly on the analytical surface" node set (private to the
#: modifier instance named "snap" in every .inp rendered below) -- checked here instead of the
#: user's original "wall" node set, because hAdaptivity's OWN node-set propagation is unreliable on
#: a genuinely curved face (its flat-coplanarity test silently drops face-interior new nodes) and
#: is not what this feature's correctness depends on.
TRACKED_WALL_SET = "__surfaceSnap_snap_wallNodes"

RADIUS_IN = 10.0
RADIUS_OUT = 16.0
Y0, Y1 = 0.0, 4.0


def _buildFacetedRingMesh(nSectors: int, angleSpan: float = np.pi / 2):
    """A strip of ``nSectors`` HEX20 "sector" elements between RADIUS_IN/RADIUS_OUT, spanning
    ``angleSpan`` radians total around the Y axis. Returns (nodes: {label: xyz},
    elements: {elNumber: [20 labels]}, wallLabels: set).

    The inner-wall (r=RADIUS_IN) face is built EXACTLY on the true cylinder -- both corners and
    midsides -- matching surfaceSnap's own documented precondition on its seed node set (mirrors a
    real Cubit export of an actual cylindrical surface, whose own boundary nodes are already
    geometrically exact).
    Every other face (outer wall, radial, non-wall axial) uses a flat Cartesian mean of its two
    corner endpoints for its midsides, which is irrelevant to correctness here but keeps the mesh
    well-shaped. The nonzero deviation this feature exists to fix appears one level down: even
    from this EXACT parent, ``Hex20Topology.subdivide``'s own quadratic interpolation of a
    wide-angle sector leaves its NEW wall nodes measurably off the true circle (~0.08mm at
    angleSpan/nSectors=90 degrees) -- see ``test_coarse_mesh_raw_subdivide_deviates_from_circle``.
    """

    topology = Hex20Topology()
    refParams = topology.reference_node_param()  # 20 x 3, values in {-1, 0, 1}

    def latticeAxis(v):
        return {-1.0: 0, 0.0: 1, 1.0: 2}[float(np.round(v))]

    dTheta = angleSpan / nSectors

    # Axis roles: lattice axis 0 (xi) = radial, axis 1 (eta) = axial (y), axis 2 (zeta) = angular
    # (the one multiple sectors extend along). This specific pairing is load-bearing, not
    # arbitrary: radial=xi with angular=eta (axial=zeta) gives a LEFT-handed map (negative
    # Jacobian) against Hex20Topology's reference-cube convention -- verified directly against
    # ``shape_functions_and_grad`` -- while radial=xi/axial=eta/angular=zeta is right-handed.

    def exactWallPoint(jAxial, kAngular):
        """Exact point on the true cylinder for ANY (possibly fractional-index/midside) lattice
        position on the i=0 (wall) face -- j/k need not be even."""
        y = Y0 + (jAxial / 2.0) * (Y1 - Y0)
        theta = (kAngular / 2.0) * dTheta
        return np.array([RADIUS_IN * np.cos(theta), y, RADIUS_IN * np.sin(theta)])

    def cornerPhysical(iGlobal, jGlobal, kGlobal):
        if iGlobal == 0:
            return exactWallPoint(jGlobal, kGlobal)
        y = Y0 + (jGlobal / 2.0) * (Y1 - Y0)
        theta = (kGlobal / 2.0) * dTheta
        return np.array([RADIUS_OUT * np.cos(theta), y, RADIUS_OUT * np.sin(theta)])

    # pass 1: all corner lattice points (i,j in {0,2}; k even in [0, 2*nSectors])
    corners = {}
    for i in (0, 2):
        for j in (0, 2):
            for k in range(0, 2 * nSectors + 1, 2):
                corners[(i, j, k)] = cornerPhysical(i, j, k)

    # pass 2: all midside lattice points (exactly one axis odd). On the wall (i=0) face: exact, per
    # the docstring above. Elsewhere: the CARTESIAN MEAN of the two flanking corners (a flat facet
    # -- fine off the tracked wall, since nothing there is checked against the true surface).
    def midsidePhysical(i, j, k):
        if i == 0:
            return exactWallPoint(j, k)
        if i % 2 == 1:
            return 0.5 * (corners[(i - 1, j, k)] + corners[(i + 1, j, k)])
        if j % 2 == 1:
            return 0.5 * (corners[(i, j - 1, k)] + corners[(i, j + 1, k)])
        return 0.5 * (corners[(i, j, k - 1)] + corners[(i, j, k + 1)])

    midsides = {}
    for i in range(0, 3):
        for j in range(0, 3):
            for k in range(0, 2 * nSectors + 1):
                odd = (i % 2) + (j % 2) + (k % 2)
                if odd != 1:
                    continue
                midsides[(i, j, k)] = midsidePhysical(i, j, k)

    allLattice = {**corners, **midsides}
    labelOf = {lat: idx + 1 for idx, lat in enumerate(sorted(allLattice))}
    nodes = {labelOf[lat]: xyz for lat, xyz in allLattice.items()}

    elements = {}
    for e in range(nSectors):
        conn = []
        for slot in range(20):
            xi, eta, zeta = refParams[slot]
            iLocal, jLocal, kLocal = latticeAxis(xi), latticeAxis(eta), latticeAxis(zeta)
            lat = (iLocal, jLocal, 2 * e + kLocal)
            conn.append(labelOf[lat])
        elements[e + 1] = conn

    wallLabels = {labelOf[lat] for lat in allLattice if lat[0] == 0}
    return nodes, elements, wallLabels


def _radialErrorsFromAxis(model, labels):
    return np.array(
        [abs(np.hypot(model.nodes[lab].coordinates[0], model.nodes[lab].coordinates[2]) - RADIUS_IN) for lab in labels]
    )


def _renderInp(nodes, elements, wallLabels, midsideNodes: str, maxLevel: int) -> str:
    nodeLines = "\n".join(f"{lab}, {x:.10f}, {y:.10f}, {z:.10f}" for lab, (x, y, z) in nodes.items())
    elementLines = "\n".join(f"{en}, " + ", ".join(str(n) for n in conn) for en, conn in elements.items())
    wallLines = "\n".join(str(lab) for lab in sorted(wallLabels))

    return f"""
*node
{nodeLines}

*element, type=C3D20, elset=ring
{elementLines}

*nset, nset=wall
{wallLines}

*nset, nset=fixed
{next(iter(elements.values()))[0]}

*material, name=linearelastic, id=mat
30000.0, 0.2

*section, name=sec, material=mat, type=solid
ring

*modelModifier, type=hAdaptivity, name=amr
>>marker, type=nodeSet, nSet=wall, initialOnly=True
maxLevel={maxLevel}
splitFactor=2

*modelModifier, type=surfaceSnap, name=snap
nodeSet=wall
originX=0.0, originY=0.0, originZ=0.0
axisX=0.0, axisY=1.0, axisZ=0.0
radius={RADIUS_IN}
midsideNodes={midsideNodes}

*job, name=surfaceSnapTest, domain=3d
*solver, solver=NIST, name=theSolver
*fieldOutput
>>perNode, elSet=ring, field=displacement, result=U, name=dispRing

*step, solver=theSolver
maxInc=1.0, minInc=1.0, maxNumInc=1, maxIter=25, stepLength=1
>>dirichlet, name=fix, nSet=fixed, field=displacement, 1=0.0, 2=0.0, 3=0.0
"""


def _buildModel(tmp_path: Path, midsideNodes: str, maxLevel: int = 1, nSectors: int = 1) -> tuple[FEModel, set]:
    nodes, elements, wallLabels = _buildFacetedRingMesh(nSectors)
    inpText = _renderInp(nodes, elements, wallLabels, midsideNodes, maxLevel)
    tmp_path.mkdir(parents=True, exist_ok=True)
    inpPath = tmp_path / "ring.inp"
    inpPath.write_text(inpText)
    inputfile = parseInputFile(str(inpPath))

    journal = Journal(verbose=False)
    model = FEModel(3)
    model = fillFEModelFromInputFile(model, inputfile, journal)
    model.prepareYourself(journal)
    model.advanceToTime(0.0)

    for nodeField in model.nodeFields.values():
        nodeField.createFieldValueEntry("U")
        nodeField.createFieldValueEntry("P")
    model._linkFieldVariableObjects(model.nodeSets["all"])

    return model, wallLabels


def test_coarse_mesh_raw_subdivide_deviates_from_circle():
    """Sanity check on the test fixture's premise, decoupled from the FEModel pipeline: even
    starting from an element whose OWN wall face is exactly on the true cylinder,
    ``Hex20Topology.subdivide`` -- the raw AMR refinement math, with no geometry-snap involved --
    produces new wall nodes that are measurably off the true circle. This is exactly the
    "polygon, not circle" premise surfaceSnap exists to fix; the rest of this module checks that
    it does, once wired into the real topology-update pipeline.
    """

    nodes, elements, wallLabels = _buildFacetedRingMesh(nSectors=1)
    topology = Hex20Topology()
    refParams = topology.reference_node_param()
    (elNumber, conn) = next(iter(elements.items()))
    coords = np.array([nodes[lab] for lab in conn])

    faceIdx = next(
        idx for idx, face in enumerate(topology.faces) if all(np.isclose(refParams[s][0], -1.0) for s in face)
    )
    faceSlots = topology.faces[faceIdx]
    childIdxOnFace = topology.face_child_indices(faceIdx, 2)

    children = topology.subdivide(coords, 2)
    maxErr = max(
        abs(np.hypot(children[ci][s][0], children[ci][s][2]) - RADIUS_IN) for ci in childIdxOnFace for s in faceSlots
    )
    assert maxErr > 0.01, "raw subdivide() of a wide-angle sector should leave a real, measurable wall-node deviation"


def test_surfacesnap_fixes_new_boundary_nodes_after_refinement(tmp_path):
    # 'curved' mode here: this check wants EVERY wall node (corners and midsides alike) exactly on
    # the true cylinder. 'straight' mode's own (much smaller, but nonzero) chord residual on
    # angular-direction midsides is checked separately and precisely in the mode-comparison test.
    model, wallLabels = _buildModel(tmp_path, midsideNodes="curved", maxLevel=1)
    preLabels = set(model.nodes)

    changed = model.updateTopology()
    assert changed, "hAdaptivity's initialOnly nodeSet marker should have refined the wall elements"

    newLabels = set(model.nodes) - preLabels
    assert newLabels, "refinement should have created new nodes"

    # every currently-tracked wall node (old, exact by construction, and new, snapped) must now be
    # within a tight numerical tolerance of the true cylinder -- the coarse-facet-scale error from
    # the sanity check above must be gone.
    finalWall = {n.label for n in model.nodeSets[TRACKED_WALL_SET]}
    assert finalWall >= wallLabels, "pre-existing wall nodes must never be dropped from the set"
    errors = _radialErrorsFromAxis(model, finalWall)
    assert (
        errors.max() < 1e-6
    ), f"wall nodes should sit exactly on the true cylinder after snapping, got max={errors.max()}"

    # and specifically: new nodes that the old (unfixed) subdivide() would have left on the flat
    # parent facet are among the ones now exact -- not just coincidentally already-exact originals
    newOnWall = newLabels & finalWall
    assert newOnWall, "at least one newly-created node should lie on the wall"
    assert _radialErrorsFromAxis(model, newOnWall).max() < 1e-6


def test_straight_mode_leaves_a_small_residual_curved_mode_does_not(tmp_path):
    """Both modes must land every wall CORNER exactly on the cylinder (radial projection is
    unconditional). The difference is in the new MIDSIDE nodes: an angular-direction midside's
    'straight' position is the Cartesian mean of two same-radius, same-y, different-angle
    corners -- the midpoint of a chord, whose radial distance from the axis is
    ``r*cos(dTheta/2)``, strictly less than ``r`` -- while 'curved' mode independently projects it
    back onto the true radius. This is exactly the kind of chord residual straight mode is expected
    to leave; here it must be reproduced (nonzero) in 'straight' mode and eliminated (~0) in
    'curved' mode.
    """

    modelStraight, wallLabels = _buildModel(tmp_path / "straight", midsideNodes="straight", maxLevel=1)
    preStraight = set(modelStraight.nodes)
    modelStraight.updateTopology()
    newStraightWall = (set(modelStraight.nodes) - preStraight) & {
        n.label for n in modelStraight.nodeSets[TRACKED_WALL_SET]
    }

    modelCurved, _ = _buildModel(tmp_path / "curved", midsideNodes="curved", maxLevel=1)
    preCurved = set(modelCurved.nodes)
    modelCurved.updateTopology()
    newCurvedWall = (set(modelCurved.nodes) - preCurved) & {n.label for n in modelCurved.nodeSets[TRACKED_WALL_SET]}

    assert newStraightWall and newCurvedWall

    straightErrors = _radialErrorsFromAxis(modelStraight, newStraightWall)
    curvedErrors = _radialErrorsFromAxis(modelCurved, newCurvedWall)

    assert curvedErrors.max() < 1e-9, "curved mode must land every new wall node exactly on the true cylinder"
    assert straightErrors.max() > 1e-4, (
        "straight mode should leave a real, measurable chord residual on angular-direction "
        "midsides -- if this is ~0 the mode distinction has silently stopped doing anything"
    )


def test_maxlevel_2_also_works(tmp_path):
    """Mirrors the Phase 1 prototype's own maxLevel=2 check, now through the real wired modifier.

    The fixture's marker is ``initialOnly=True``, so it can only ever mark on hAdaptivity's very
    first ``plan()`` call -- one ``updateTopology()`` call therefore only ever reaches level 1,
    regardless of ``maxLevel``. Reaching level 2 for real needs a second round: directly seed
    ``amr._pendingMarkedElements`` (the same "drive the mirror/marks directly" pattern
    ``tests/test_hadaptivity_cascade.py`` uses) with the now-active level-1 elements, then call
    ``updateTopology()`` again -- the same shape of thing a real deck's DYNAMIC (non-initialOnly)
    marker would do on a later increment.
    """

    model, wallLabels = _buildModel(tmp_path, midsideNodes="curved", maxLevel=2, nSectors=3)
    model.updateTopology()

    amr = model.modelModifiers["amr"]
    levelOfElement = {el: amr._mesh.elements[eid]["level"] for eid, el in amr._eidToEl.items()}
    assert set(levelOfElement.values()) == {
        1
    }, "sanity check: a single updateTopology() call must reach exactly level 1"

    # hAdaptivity guards against re-refining at the exact same model.time (a cutback retry) --
    # advance time first, exactly as a real multi-increment run would between two refinement
    # rounds, or this second, directly-seeded round is silently ignored.
    model.advanceToTime(1.0)
    amr._pendingMarkedElements = set(levelOfElement)
    changed = model.updateTopology()
    assert changed, "the second, directly-seeded round must also have refined something"

    levelOfElement = {el: amr._mesh.elements[eid]["level"] for eid, el in amr._eidToEl.items()}
    assert 2 in set(levelOfElement.values()), "level 2 must actually have been reached this time"

    finalWall = {n.label for n in model.nodeSets[TRACKED_WALL_SET]}
    assert len(finalWall) > len(wallLabels) * 2, "maxLevel=2 should have produced substantially more wall nodes"
    errors = _radialErrorsFromAxis(model, finalWall)
    assert errors.max() < 1e-6


def test_a_second_refinement_of_an_already_snapped_element_uses_corrected_geometry(tmp_path):
    """Regression test for a real bug found in review: hAdaptivity's own ``AdaptiveMesh`` mirror
    caches each element's node coordinates at the moment IT was created, and never re-reads
    ``model.nodes`` afterward. Without ``ModelModifier.syncNodeCoordinates`` (called from
    surfaceSnap's own ``apply()``), a level-1 element that surfaceSnap already corrected would,
    if refined AGAIN, generate its level-2 children from the mirror's stale, PRE-snap geometry --
    silently discarding the correction. Verified two ways: the mirror's own cached coordinates
    must match ``model.nodes`` immediately after the first snap, and after forcing a second
    refinement round (same direct-seeding technique as test_maxlevel_2_also_works), the resulting
    level-2 wall nodes must still be exactly on the true cylinder -- which they would NOT be if
    their parent's children were generated from stale geometry.
    """

    model, wallLabels = _buildModel(tmp_path, midsideNodes="curved", maxLevel=2, nSectors=1)
    amr = model.modelModifiers["amr"]
    model.updateTopology()

    mismatches = [
        label
        for label, coord in amr._mesh.registry.coordinates.items()
        if label in model.nodes and not np.allclose(coord, model.nodes[label].coordinates, atol=1e-9)
    ]
    assert not mismatches, f"hAdaptivity's mirror must be kept in sync with model.nodes, but diverged for {mismatches}"

    model.advanceToTime(1.0)  # hAdaptivity's cutback guard blocks re-refining at the same time
    amr._pendingMarkedElements = set(amr._eidToEl.values())
    model.updateTopology()

    finalWall = {n.label for n in model.nodeSets[TRACKED_WALL_SET]}
    assert len(finalWall) > len(wallLabels) * 2
    errors = _radialErrorsFromAxis(model, finalWall)
    assert errors.max() < 1e-6, (
        "level-2 wall nodes deviate from the true cylinder -- the second refinement must have used "
        "stale (pre-snap) parent geometry"
    )


def _buildAsymmetricRingModel(tmp_path, nSectors: int, refineElset: str):
    """Same C3D20 ring fixture as ``_buildModel``, but with an explicit ``refineMe`` element set
    (rather than a nodeSet marker) so a caller can choose exactly which sector(s) refine first --
    used to force an asymmetric refinement front, and the hanging-node interface it creates right
    at the wall's own rim.
    """

    nodes, elements, wallLabels = _buildFacetedRingMesh(nSectors=nSectors)
    inpText = f"""
*node
{chr(10).join(f"{lab}, {x:.10f}, {y:.10f}, {z:.10f}" for lab, (x, y, z) in nodes.items())}

*element, type=C3D20, elset=ring
{chr(10).join(f"{en}, " + ", ".join(str(n) for n in conn) for en, conn in elements.items())}

*nset, nset=wall
{chr(10).join(str(lab) for lab in sorted(wallLabels))}

*nset, nset=fixed
{next(iter(elements.values()))[0]}

*elset, elset=refineMe
{refineElset}

*material, name=linearelastic, id=mat
30000.0, 0.2

*section, name=sec, material=mat, type=solid
ring

*modelModifier, type=hAdaptivity, name=amr
>>marker, type=elementSet, elSet=refineMe, initialOnly=True
maxLevel=1
splitFactor=2

*modelModifier, type=surfaceSnap, name=snap
nodeSet=wall
originX=0.0, originY=0.0, originZ=0.0
axisX=0.0, axisY=1.0, axisZ=0.0
radius={RADIUS_IN}
midsideNodes=curved

*job, name=surfaceSnapAsymmetricTest, domain=3d
*solver, solver=NIST, name=theSolver
*fieldOutput
>>perNode, elSet=ring, field=displacement, result=U, name=dispRing

*step, solver=theSolver
maxInc=1.0, minInc=1.0, maxNumInc=1, maxIter=25, stepLength=1
>>dirichlet, name=fix, nSet=fixed, field=displacement, 1=0.0, 2=0.0, 3=0.0
"""
    tmp_path.mkdir(parents=True, exist_ok=True)
    inpPath = tmp_path / "ring.inp"
    inpPath.write_text(inpText)
    inputfile = parseInputFile(str(inpPath))
    journal = Journal(verbose=False)
    model = FEModel(3)
    model = fillFEModelFromInputFile(model, inputfile, journal)
    model.prepareYourself(journal)
    model.advanceToTime(0.0)
    for nodeField in model.nodeFields.values():
        nodeField.createFieldValueEntry("U")
        nodeField.createFieldValueEntry("P")
    model._linkFieldVariableObjects(model.nodeSets["all"])
    return model, wallLabels


def test_hanging_node_collisions_are_skipped_not_broken(tmp_path):
    """An asymmetric marker (only part of the ring refines) creates hanging nodes right on the
    wall's own rim -- those must be excluded from snapping, not silently moved off their MPC."""

    # only refine sector 1 (of 0..3), leaving neighbours coarse -- forces a hanging-node interface
    # directly adjacent to (and sharing corners with) the wall.
    model, wallLabels = _buildAsymmetricRingModel(tmp_path, nSectors=4, refineElset="2")

    changed = model.updateTopology()
    assert changed

    # no exception, and the model must still be geometrically sane: no NaN/inf coordinates
    for node in model.nodes.values():
        assert np.all(np.isfinite(node.coordinates))

    # the actual claim under test: nodes that are BOTH genuine AMR hanging-node slaves AND
    # topologically on the tracked wall face (surfaceSnap's own "collision" set) must (a) still be
    # correctly claimed by the hanging constraint -- proving the collision-skip didn't corrupt
    # the MPC bookkeeping -- and (b) NOT have been snapped onto the exact analytical radius, which
    # would only happen if the skip silently failed to take effect.
    from edelweissfe.constraints.hangingnode import Constraint as HangingNodeConstraint

    hangingConstraints = [c for c in model.multiPointConstraints.values() if isinstance(c, HangingNodeConstraint)]
    assert hangingConstraints, "this asymmetric refinement must have produced at least one hanging-node constraint"
    hangingSlaveLabels = {n.label for c in hangingConstraints for n in c.claimedSlaveNodes()}
    assert hangingSlaveLabels, "this asymmetric refinement must have produced at least one hanging slave"

    trackedWall = {n.label for n in model.nodeSets[TRACKED_WALL_SET]}
    collisionLabels = hangingSlaveLabels & trackedWall
    assert collisionLabels, "this fixture must produce at least one hanging slave that is also a wall candidate"

    for label in collisionLabels:
        assert any(
            label in {n.label for n in c.claimedSlaveNodes()} for c in hangingConstraints
        ), f"node {label} must still be claimed by the hanging-node constraint"

    # The direct, unambiguous check that the skip actually took effect: these labels must not be
    # among the coordinates surfaceSnap's own recorded plan(s) actually wrote. (A radial-error
    # check would be a false negative here: a node this close to an already-exact parent can end
    # up almost exactly on the true radius from raw, un-snapped subdivide() alone.)
    snappedLabels = set()
    for record in model.topologyHistory:
        if record.modifier != "snap":
            continue
        plan = model.modelModifiers["snap"].decodePlan(record.plan)
        snappedLabels.update(plan.labels)
    assert collisionLabels.isdisjoint(snappedLabels), (
        f"hanging-slave/wall-candidate node(s) {collisionLabels & snappedLabels} were snapped -- "
        "the collision skip failed to exclude them"
    )


def test_a_resolved_hanging_collision_is_retried_and_snapped(tmp_path):
    """Regression test for a real gap found in review: a node skipped this round because it was a
    hanging-node collision (or a quality-safeguard veto) was, before this fix, marked as
    permanently confirmed wall-face membership WITHOUT ever being retried -- since surfaceSnap's
    only trigger was ``change.addedNodes``, and a skipped node is never "added" again later, it
    would stay at its raw, un-snapped position forever, even after the very condition that
    blocked it resolved.

    Reproduced directly: refine one sector or a 4-sector ring (creating hanging-node collisions at
    its rim, exactly like test_hanging_node_collisions_are_skipped_not_broken), confirm they land
    in the modifier's own pending set, then refine the REMAINING sectors too (making the interface
    conforming again, so those nodes stop being hanging), and confirm they get retried, resolved,
    and end up exactly on the true cylinder.
    """

    PENDING_SET = "__surfaceSnap_snap_pendingNodes"

    model, wallLabels = _buildAsymmetricRingModel(tmp_path, nSectors=4, refineElset="2")
    model.updateTopology()

    pendingAfterRound1 = {n.label for n in model.nodeSets[PENDING_SET]}
    assert pendingAfterRound1, "the asymmetric refinement must have left at least one node pending"

    amr = model.modelModifiers["amr"]
    levelOf = {el: amr._mesh.elements[eid]["level"] for eid, el in amr._eidToEl.items()}
    stillCoarse = {el for el, level in levelOf.items() if level == 0}
    assert stillCoarse, "sanity check: the other three sectors must still be unrefined"

    model.advanceToTime(1.0)  # hAdaptivity's cutback guard blocks re-refining at the same time
    amr._pendingMarkedElements = stillCoarse
    model.updateTopology()

    pendingAfterRound2 = {n.label for n in model.nodeSets[PENDING_SET]}
    resolved = pendingAfterRound1 - pendingAfterRound2
    assert resolved, "refining the remaining sectors must have resolved at least one pending node"
    errorsAfter = _radialErrorsFromAxis(model, resolved)
    assert errorsAfter.max() < 1e-6, "a resolved (retried) node must end up exactly on the true cylinder"

    # and this two-round sequence -- with genuinely non-empty stillPendingLabels/
    # resolvedPendingLabels in the recorded plans, unlike the other replay test -- must itself
    # replay correctly onto an independently-built, fresh model.
    modelB, _ = _buildAsymmetricRingModel(tmp_path / "replay", nSectors=4, refineElset="2")
    modelB.replayTopologyHistory(model.topologyHistory)
    assert modelB.topologyFingerprint() == model.topologyFingerprint()
    assert {n.label for n in modelB.nodeSets[PENDING_SET]} == pendingAfterRound2
    assert {n.label for n in modelB.nodeSets[TRACKED_WALL_SET]} == {n.label for n in model.nodeSets[TRACKED_WALL_SET]}


def test_restart_replay_reproduces_the_snap(tmp_path):
    """The general restart-safety check: a model rebuilt from scratch and replayed through the
    recorded topology history must end up byte-identical to the one that made the decisions live
    -- mirroring tests/test_hadaptivity_restart.py's own check, extended to cover surfaceSnap.
    """

    modelA, wallLabels = _buildModel(tmp_path / "a", midsideNodes="curved", maxLevel=1)
    changed = modelA.updateTopology()
    assert changed
    assert modelA.topologyHistory
    assert any(
        rec.modifier == "snap" for rec in modelA.topologyHistory
    ), "surfaceSnap must have recorded at least one round of its own"

    modelB, _ = _buildModel(tmp_path / "b", midsideNodes="curved", maxLevel=1)
    assert len(modelB.elements) < len(modelA.elements), "model B must start unrefined"

    modelB.replayTopologyHistory(modelA.topologyHistory)

    assert modelB.topologyFingerprint() == modelA.topologyFingerprint()
    trackedA = {n.label for n in modelA.nodeSets[TRACKED_WALL_SET]}
    trackedB = {n.label for n in modelB.nodeSets[TRACKED_WALL_SET]}
    assert trackedB == trackedA, "the tracked wall node set must replay to the same membership"
    assert trackedB > wallLabels, "replay must have grown the tracked set beyond the original seed"


def test_a_round_that_snaps_nothing_still_replays(tmp_path):
    """The exact mechanism the restart-safety fix relies on, tested directly and deterministically
    (not depending on contriving a mesh where every single candidate on a face happens to collide
    with a hanging node): a round that discovers new wall-face membership but snaps ZERO node
    coordinates must still produce a non-empty ModelChange -- the same isEmpty gate
    FEModel.updateTopology uses to decide whether to record a round at all (see
    edelweissfe/models/femodel.py's updateTopology, and the module docstring of surfacesnap.py).
    A round that were empty here would never be recorded, and the tracked wall set's growth would
    be silently lost across a restart -- exactly the bug this test pins shut.
    """

    from edelweissfe.modelmodifiers.geometry.surfacesnap import SnapPlan

    model, wallLabels = _buildModel(tmp_path, midsideNodes="curved", maxLevel=1)
    snap = model.modelModifiers["snap"]

    # any node not yet in the tracked set stands in for "a node topologically confirmed to be on
    # the wall face this round" -- the mechanism under test doesn't care whether it's geometrically
    # meaningful, only whether apply() correctly persists membership growth with zero coordinate
    # writes.
    outsiderLabel = next(lab for lab in model.nodes if lab not in wallLabels)
    assert outsiderLabel not in {n.label for n in model.nodeSets[TRACKED_WALL_SET]}

    plan = SnapPlan(labels=(), coords=(), newWallNodes=(outsiderLabel,))
    change = snap.apply(model, plan)

    assert change.movedNodes == set(), "this plan snapped no coordinates, by construction"
    assert not change.isEmpty, (
        "a round that only grows the tracked wall set (zero coordinates moved) must NOT be "
        "reported as empty, or FEModel.updateTopology will never record it and a restart will "
        "silently lose this face's tracking"
    )
    assert TRACKED_WALL_SET in change.changedNodeSets
    assert outsiderLabel in {n.label for n in model.nodeSets[TRACKED_WALL_SET]}

    # and the encode/decode round-trip (what a real checkpoint actually stores) preserves it too
    encoded = snap.encodePlan(plan)
    decoded = snap.decodePlan(encoded)
    assert decoded.newWallNodes == plan.newWallNodes
    assert decoded.labels == () and decoded.coords == ()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
