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
"""Cascaded refinement: the octree mirror can run several levels ahead of the model in one call, and
every one of those levels must reach the model.

``AdaptiveMesh.balance_2to1`` refines until the mesh is graded, so it can split a cell it created
earlier in the same call. The active leaf that comes out of that has a parent which is itself brand
new and was never turned into an element -- there is no materialised parent to split. Materialising
only the children of already-materialised cells would drop such a leaf from ``model.elements`` while
the mirror still lists it as active, which is a mesh the rest of the solver cannot see.
"""

from pathlib import Path

import numpy as np
import pytest

from edelweissfe.helpers.inputfilehelpers import fillFEModelFromInputFile
from edelweissfe.journal.journal import Journal
from edelweissfe.modelmodifiers.adaptivity.hadaptivity import RefinementPlan
from edelweissfe.models.femodel import FEModel
from edelweissfe.utils.exceptions import TopologyError
from edelweissfe.utils.inputfileparser import parseInputFile

_INP = """
*material, name=linearelastic, id=mat
18000, 0.0

*section, name=sec, material=mat, type=solid
bar_all

*modelGenerator, generator=boxGen, name=bar
nX      =4
nY      =1
nZ      =1
x0      =0
y0      =0
z0      =0
lX      =4
lY      =1
lZ      =1
elType  =C3D20

*modelModifier, type=hAdaptivity, name=amr
>>marker, type=nodeSet, nSet=bar_right, initialOnly=True
maxLevel=4

*job, name=hadaptivityCascadeTest, domain=3d
*solver, solver=NIST, name=theSolver
*fieldOutput
>>perNode, elSet=bar_all, field=displacement, result=U, name=dispBar

*step, solver=theSolver
maxInc=1.0, minInc=1.0, maxNumInc=1, maxIter=25, stepLength=1
>>dirichlet, name=fixLeft, nSet=bar_left, field=displacement, 1=0.0, 2=0.0, 3=0.0
"""


def _buildModel(tmp_path: Path) -> FEModel:
    inpPath = tmp_path / "cascade.inp"
    inpPath.write_text(_INP)
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

    return model


def _childTowards(mesh, eid, x):
    """The child of ``eid`` whose cell reaches furthest towards the plane ``x``."""

    return min(mesh.elements[eid]["children"], key=lambda k: abs(mesh.elements[k]["coords"][:, 0].min() - x))


def _driveMirrorAhead(model, modifier, levels: int):
    """Refine one corner of the bar down ``levels`` levels in the mirror only, leaving the model
    behind. This is the state 2:1 balancing produces for itself when it cascades; doing it directly
    keeps the test about what ``_materialize`` must cope with, not about how the mirror got there.

    Inside the model's topology window, since the mirror draws its new node labels from the model's
    allocator.
    """

    mesh = modifier._mesh
    deepest = max(mesh.active(), key=lambda e: mesh.elements[e]["coords"][:, 0].min())
    interface = mesh.elements[deepest]["coords"][:, 0].min()
    with model.topologyChanges():
        for _ in range(levels):
            mesh.refine(deepest)
            deepest = _childTowards(mesh, deepest, interface)


def test_a_cascade_materialises_every_level_it_produced(tmp_path):
    """The mirror is driven three levels ahead, then one apply() must bring the model all the way
    up to it -- including the leaves whose parents only came into existence during this same call.
    """

    model = _buildModel(tmp_path)
    amr = model.modelModifiers["amr"]
    mesh = amr._mesh

    _driveMirrorAhead(model, amr, 3)
    createdBefore = set(mesh.elements)

    with model.topologyChanges():
        change = amr.apply(model, RefinementPlan(eids=[]))

    # 2:1 balancing had to cascade: it refined at least one cell that did not exist when apply()
    # started -- the case a single-level materialisation cannot express.
    cascaded = [e for e in set(mesh.elements) - createdBefore if not mesh.elements[e]["active"]]
    assert cascaded, "expected balancing to split a cell it created in this same call"

    # the model and the mirror must describe the same mesh
    active = set(mesh.active())
    assert set(amr._eidToEl) == active
    assert {el.elNumber for el in amr._eidToEl.values()} == set(model.elements)

    # no element may be reported as both created and destroyed: the intermediate cells are internal
    # bookkeeping, and a consumer must never be handed one
    assert not (change.addedElements & change.removedElements)
    assert change.addedElements <= set(model.elements)
    assert not (change.removedElements & set(model.elements))

    # a refined parent's recorded children are the leaves that replaced it, not an intermediate
    for parentLabel, childLabels in change.parentToChildren.items():
        assert parentLabel not in model.elements
        assert set(childLabels) <= set(model.elements)


def test_a_fully_resplit_parent_leaves_nothing_behind(tmp_path):
    """The shape in which a dropped leaf is silently dropped rather than merely lost: when *every*
    child of a materialised cell is refined again, that cell has no active child left, so nothing
    downstream ever asks for the intermediates by name and the whole sub-tree can go missing without
    a single error. The model must still end up describing exactly the mirror's active mesh.
    """

    model = _buildModel(tmp_path)
    amr = model.modelModifiers["amr"]
    mesh = amr._mesh

    root = max(mesh.active(), key=lambda e: mesh.elements[e]["coords"][:, 0].min())
    with model.topologyChanges():  # the mirror mints its node labels from the model's allocator
        mesh.refine(root)
        for child in list(mesh.elements[root]["children"]):
            mesh.refine(child)
    assert not any(mesh.elements[c]["active"] for c in mesh.elements[root]["children"])

    with model.topologyChanges():
        amr.apply(model, RefinementPlan(eids=[]))

    assert set(amr._eidToEl) == set(mesh.active())
    assert {el.elNumber for el in amr._eidToEl.values()} == set(model.elements)


def test_an_orphaned_active_cell_is_reported_rather_than_dropped(tmp_path):
    """The backstop for the case the level loop cannot resolve: an active cell with no materialised
    ancestor at all. Silently leaving it out of the model is the one outcome that must not happen.
    """

    model = _buildModel(tmp_path)
    amr = model.modelModifiers["amr"]

    # drop a materialised root from the modifier's map while the mirror keeps it active
    orphan = sorted(amr._eidToEl)[0]
    amr._eidToEl.pop(orphan)

    with pytest.raises(TopologyError, match="no materialised ancestor"):
        with model.topologyChanges():
            amr.apply(model, RefinementPlan(eids=[]))


def test_the_warm_start_reaches_the_deepest_new_nodes(tmp_path):
    """A cascade interpolates level by level: the intermediate cells' own nodes are new, so the
    level below them has to warm-start from the values just computed, not from the pre-refinement
    snapshot -- otherwise the deepest nodes restart from zero and the first Newton iteration sees a
    spurious residual spike.
    """

    model = _buildModel(tmp_path)
    amr = model.modelModifiers["amr"]

    # a non-trivial converged state to warm-start from: u_x = x
    nodeField = model.nodeFields["displacement"]
    for node in nodeField.nodes:
        idx = nodeField._indicesOfNodesInArray[node]
        nodeField["U"][idx, 0] = node.coordinates[0]
        nodeField["P"][idx, 0] = node.coordinates[0]

    _driveMirrorAhead(model, amr, 3)
    with model.topologyChanges():
        amr.apply(model, RefinementPlan(eids=[]))

    nodeField = model.nodeFields["displacement"]
    for node in nodeField.nodes:
        idx = nodeField._indicesOfNodesInArray[node]
        assert np.isclose(nodeField["U"][idx, 0], node.coordinates[0], atol=1e-9), node.label
        assert np.isclose(nodeField["P"][idx, 0], node.coordinates[0], atol=1e-9), node.label
