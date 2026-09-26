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
"""Pins the exact conformity check :meth:`AdaptiveMesh.check_conformity` and the exact hanging-node
weights: a refined mesh is conforming (flat, curved and warped faces; split factors 2 and 3; a
two-level chain), every missing hanging-node constraint is caught, and a hanging node assigned to the
wrong edge or face is caught by the exact weight verification."""

from fractions import Fraction

import numpy as np
import pytest
from _adaptivemeshbuilder import AdaptiveMeshBuilder

from edelweissfe.adaptivity.hex20shapefunctions import hex20_box_coords
from edelweissfe.adaptivity.hex20topology import Hex20Topology
from edelweissfe.adaptivity.refinement import AdaptiveMesh
from edelweissfe.utils.exceptions import TopologyError

TOPOLOGY = Hex20Topology()


def _newNodesOnARefinedFace(n):
    """A face split n ways per direction carries (2n+1)^2 - n^2 serendipity nodes; 8 are the coarse
    face's own, 8n lie on its boundary, the rest in its interior."""
    total = (2 * n + 1) ** 2 - n * n
    return total - 8, total - 8 * n  # hanging nodes on the face, of which face-interior


def _coarseNextToRefined(splitFactor=2, curvedWarped=False, refineTwice=False):
    """A = [-2, 0] x [0, 2]^2 (coarse) next to B = [0, 2] x [0, 2]^2 (refined); optionally the shared
    face bulged and warped, optionally one child of B refined again (a level-2/level-1 interface)."""
    A = hex20_box_coords(-2.0, 0.0, 0.0, 2.0, 0.0, 2.0)
    B = hex20_box_coords(0.0, 2.0, 0.0, 2.0, 0.0, 2.0)
    if curvedWarped:
        for X in (A, B):
            for x in X:
                if abs(x[0]) < 1e-12:
                    isCorner = x[1] in (0.0, 2.0) and x[2] in (0.0, 2.0)
                    x[0] += 0.03 if (x[1], x[2]) == (2.0, 2.0) else (0.0 if isCorner else 0.05 + 0.01 * x[1])
    builder = AdaptiveMeshBuilder(splitFactor)
    builder.addRoot(A)
    kids = builder.mesh.refine(builder.addRoot(B))
    if refineTwice:
        builder.mesh.refine(kids[-1])  # a child away from A
    return builder.mesh


@pytest.mark.parametrize("splitFactor", [2, 3])
@pytest.mark.parametrize("curvedWarped", [False, True])
def test_aRefinedNeighbourIsConformingWithAllItsFaceNodesConstrained(splitFactor, curvedWarped):
    mesh = _coarseNextToRefined(splitFactor, curvedWarped)
    records = mesh.hanging_mpc_records()
    mesh.check_conformity(records)
    nHanging, nFaceInterior = _newNodesOnARefinedFace(splitFactor)
    assert len(records) == nHanging
    assert sum(1 for h in mesh.classify_hanging() if h.kind == "face") == nFaceInterior


def test_aTwoLevelChainIsConforming():
    mesh = _coarseNextToRefined(refineTwice=True)
    mesh.check_conformity(mesh.hanging_mpc_records())


def test_aSingleMissingConstraintIsCaught():
    mesh = _coarseNextToRefined()
    records = mesh.hanging_mpc_records()
    with pytest.raises(TopologyError, match="not\\s+conforming"):
        mesh.check_conformity(set(records) - {sorted(records)[0]})


def test_missingFaceInteriorConstraintsOnACurvedFaceAreCaught():
    """The check needs no geometry: leaving the face-interior hanging nodes of a curved and warped face
    unconstrained -- what a flat corner-plane test would do -- is caught, with exactly those nodes."""
    mesh = _coarseNextToRefined(curvedWarped=True)
    faceInterior = {h.slave for h in mesh.classify_hanging() if h.kind == "face"}
    with pytest.raises(TopologyError, match=rf"{len(faceInterior)} node\(s\) lie on the boundary"):
        mesh.check_conformity(set(mesh.hanging_mpc_records()) - faceInterior)


def test_theExactWeightCheckCatchesAWrongEntity():
    """Masters and weights are paired by node slot, so their order cannot diverge; what can go wrong is
    the choice of entity -- the wrong face, or an edge of the right face for a point inside it."""
    mesh = _coarseNextToRefined()
    reference = TOPOLOGY.reference_node_lattice - 1
    face = next(f for f in TOPOLOGY.faces if all(reference[i][0] == 1 for i in f))
    xi = (Fraction(1), Fraction(1, 2), Fraction(0))  # inside the face xi_1 = +1
    exact = TOPOLOGY.shape_functions_exact(*xi)
    mesh._verifyTraceWeights(0, 0, xi, list(face), exact)  # the correct trace
    oppositeFace = next(f for f in TOPOLOGY.faces if all(reference[i][0] == -1 for i in f))
    with pytest.raises(TopologyError, match="exact trace"):
        mesh._verifyTraceWeights(0, 0, xi, list(oppositeFace), exact)
    edgeOfTheFace = next(e for e in TOPOLOGY.edges if set(e) <= set(face))
    with pytest.raises(TopologyError, match="exact trace"):
        mesh._verifyTraceWeights(0, 0, xi, list(edgeOfTheFace), exact)


def test_rootsTakeTheModelsNodeLabels():
    """Roots use the labels they are given, as they are; a label that was not seeded is an error."""
    X = hex20_box_coords(0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
    labels = list(range(101, 121))
    mesh = AdaptiveMesh(splitFactor=2, topology=Hex20Topology())
    for label, x in zip(labels, X):
        mesh.registry.seed(label, x, 0)
    root = mesh.add_root(X, labels, 0)
    assert mesh.elements[root]["conn"] == labels
    with pytest.raises(ValueError, match="not seeded"):
        mesh.add_root(hex20_box_coords(1.0, 2.0, 0.0, 1.0, 0.0, 1.0), list(range(201, 221)), 0)


def test_refiningBeyondTheLatticeDepthRaises():
    """The exact reference lattice has a finite depth; exceeding it is an error, never an approximation."""
    from edelweissfe.adaptivity.refinement import MAXIMUM_REFINEMENT_DEPTH

    builder = AdaptiveMeshBuilder(2)
    eid = builder.addRoot(hex20_box_coords(0.0, 1.0, 0.0, 1.0, 0.0, 1.0))
    for _ in range(MAXIMUM_REFINEMENT_DEPTH):
        eid = builder.mesh.refine(eid)[0]
    with pytest.raises(TopologyError, match="maximum refinement depth"):
        builder.mesh.refine(eid)
    assert np.isfinite(builder.mesh.elements[eid]["coords"]).all()
