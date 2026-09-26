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
"""Round-trip test for the recorded topology history of an h-adaptive model. Builds two independent
models from the same .inp via fillFEModelFromInputFile (mirroring the driver's own setup sequence,
minus solvers/steps, which this doesn't need), drives one real refinement on the first through the
live topology update, and asserts that replaying its recorded history onto the second -- freshly
rebuilt, unrefined -- reproduces the same topology.

Uses an `initialOnly` marker (edelweissfe.adaptivity.marking) so refinement triggers
deterministically on the very first topology update, regardless of field state -- which keeps the
test independent of a real solve's marker-evaluation timing.
"""

from dataclasses import replace
from pathlib import Path

import h5py
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
fixed_all

*modelGenerator, generator=boxGen, name=fixed
nX      =2
nY      =2
nZ      =2
x0      =0
y0      =0
z0      =0
lX      =1
lY      =1
lZ      =1
elType  =C3D20

*modelModifier, type=hAdaptivity, name=amr
>>marker, type=nodeSet, nSet=fixed_top, initialOnly=True
maxLevel=1

*job, name=hadaptivityRestartTest, domain=3d
*solver, solver=NIST, name=theSolver
*fieldOutput
>>perNode, elSet=fixed_all, field=displacement, result=U, name=dispFixed

*step, solver=theSolver
maxInc=1.0, minInc=1.0, maxNumInc=1, maxIter=25, stepLength=1
>>dirichlet, name=fixBack, nSet=fixed_back, field=displacement, 1=0.0, 2=0.0, 3=0.0
"""


def _buildModel(tmp_path: Path, name: str) -> FEModel:
    inpPath = tmp_path / name
    inpPath.write_text(_INP)
    inputfile = parseInputFile(str(inpPath))

    journal = Journal(verbose=False)
    job = inputfile["job"][0]
    from edelweissfe.config.phenomena import domainMapping

    model = FEModel(domainMapping[job["domain"]])
    model = fillFEModelFromInputFile(model, inputfile, journal)
    model.prepareYourself(journal)
    model.advanceToTime(job.get("startTime", 0.0))

    for nodeField in model.nodeFields.values():
        nodeField.createFieldValueEntry("U")
        nodeField.createFieldValueEntry("P")
    model._linkFieldVariableObjects(model.nodeSets["all"])

    return model


def _hangingRecordsByLabel(hangingConstraint) -> dict:
    """``_records`` is a ``list[(Node, [(Node, weight), ...])]`` -- Node identity differs between
    two independently-built models even when labels match, so compare by label instead."""
    return {
        slave.label: sorted((master.label, weight) for master, weight in masters)
        for slave, masters in hangingConstraint._records
    }


def test_topology_history_roundtrip_reproduces_the_refinement(tmp_path):
    """The replay contract, at the level of one modifier: a model that replays the recorded history
    ends up byte-identical to the one that made the decisions live.

    Compares by topologyFingerprint, which covers element numbers, connectivity and node
    coordinates -- not just the element-number set the old per-modifier round-trip checked.
    """

    modelA = _buildModel(tmp_path, "a.inp")
    amrA = modelA.modelModifiers["amr"]

    refined = modelA.updateTopology(step=None, timeStep=0.0)
    assert refined, "the initialOnly marker should have triggered a refinement on the first call"
    assert modelA.topologyHistory, "an applied decision must be recorded in the topology history"

    modelB = _buildModel(tmp_path, "b.inp")
    assert len(modelB.elements) < len(modelA.elements), "model B must start unrefined"

    modelB.replayTopologyHistory(modelA.topologyHistory)

    assert modelB.topologyFingerprint() == modelA.topologyFingerprint()
    assert _hangingRecordsByLabel(modelB.modelModifiers["amr"]._hanging) == _hangingRecordsByLabel(amrA._hanging)
    # A checkpoint only exists after an increment converged, so a replayed run is never truly making
    # its first call -- otherwise initialOnly markers would re-evaluate redundantly on the next one.
    assert modelB.modelModifiers["amr"]._isFirstCall is False


def test_replay_detects_a_tampered_plan_and_names_it(tmp_path):
    """A history that no longer reproduces its recorded topology must be reported, not silently
    applied. By default the replayed mesh is checked once, against the last record's fingerprint;
    with verifyTopologyFingerprintsPerRecord every record is checked as it is applied, which is what
    turns "the resumed run diverged" into "it diverged at THIS record"."""

    modelA = _buildModel(tmp_path, "a.inp")
    modelA.updateTopology(step=None, timeStep=0.0)
    assert modelA.topologyHistory

    tampered = [replace(record, fingerprint="0" * 32) for record in modelA.topologyHistory]

    modelB = _buildModel(tmp_path, "b.inp")
    with pytest.raises(TopologyError, match="restart replay diverged: after replaying all"):
        modelB.replayTopologyHistory(tampered)

    modelC = _buildModel(tmp_path, "c.inp")
    modelC.verifyTopologyFingerprintsPerRecord = True
    with pytest.raises(TopologyError, match="replay diverged at record 0"):
        modelC.replayTopologyHistory(tampered)


def test_replay_keeps_the_time_each_decision_was_originally_made_at(tmp_path):
    """A resumed run re-records everything it replays, and that re-recorded history is what its own
    next checkpoint holds. Stamping the resume time onto it would claim every past decision happened
    at the moment of resuming -- and hand the modifier a cutback guard set to the wrong time."""

    modelA = _buildModel(tmp_path, "a.inp")
    modelA.advanceToTime(3.5)
    modelA.updateTopology(step=None, timeStep=0.0)
    assert [record.time for record in modelA.topologyHistory] == [3.5]

    modelB = _buildModel(tmp_path, "b.inp")
    modelB.advanceToTime(9.0)
    modelB.replayTopologyHistory(modelA.topologyHistory)

    assert [record.time for record in modelB.topologyHistory] == [3.5]
    assert modelB.modelModifiers["amr"]._lastRefinedTime == 3.5


def test_pending_marks_are_not_checkpointed_but_re_derived(tmp_path):
    """Pending marks deliberately do NOT round-trip: they are a decision-side buffer, and the next
    plan() re-derives them from the restored solution state -- exactly as the live run would have.
    Checkpointing them would be a second source of truth for something already implied."""

    modelA = _buildModel(tmp_path, "a.inp")
    amrA = modelA.modelModifiers["amr"]
    modelA.updateTopology(step=None, timeStep=0.0)

    stillActive = [el for eid, el in amrA._eidToEl.items() if amrA._mesh.elements[eid]["active"]]
    amrA._pendingMarkedElements = set(stillActive[:1])

    modelB = _buildModel(tmp_path, "b.inp")
    modelB.replayTopologyHistory(modelA.topologyHistory)

    assert modelB.modelModifiers["amr"]._pendingMarkedElements == set()
    # what IS restored is the decision-side state the next plan() needs
    assert modelB.modelModifiers["amr"]._lastRefinedTime == modelA.topologyHistory[-1].time


# ---- multi-round replay: one fingerprint and one field bookkeeping pass for the whole history ----


def _innermostChild(mesh, eid: int) -> int:
    """The child of ``eid`` whose centroid is nearest the unit box's centre -- the one with the most
    coarse neighbours, so that refining it makes 2:1 balancing cascade."""

    centre = np.array([0.5, 0.5, 0.5])
    return min(
        mesh.elements[eid]["children"], key=lambda c: np.linalg.norm(mesh.elements[c]["coords"].mean(0) - centre)
    )


def _rounds(mesh):
    """Three refinement decisions on the 2x2x2 root mesh (roots are eids 1..8, in boxGen order): a
    root, another root, then a level-1 child of the first -- which 2:1 balancing answers by refining
    that child's coarse neighbours too, so the last round is a cascade rather than a single split.
    (apply() enforces no level cap; that is plan()'s job, which is never called here.)"""

    yield [1]
    yield [8]
    yield [_innermostChild(mesh, 1)]


def _driveRounds(model: FEModel) -> None:
    """Apply a sequence of refinement decisions live, one topology window per decision -- the same
    apply/record pair updateTopology issues per applied decision, fed explicit plans instead of a
    marker's, so that several rounds accumulate deterministically without a solve in between."""

    amr = model.modelModifiers["amr"]
    for roundIndex, eids in enumerate(_rounds(amr._mesh), start=1):
        model.advanceToTime(float(roundIndex))
        with model.topologyChanges():
            plan = RefinementPlan(eids=list(eids))
            change = amr.apply(model, plan)
            model.recordTopologyChange(1, "amr", amr, plan, change)


def _fieldBookkeepingByLabel(model: FEModel) -> dict:
    """Everything the deferred node-field bookkeeping decides, keyed by label rather than by Node
    identity: which fields each node carries, how each NodeField is laid out, and that every node's
    field variable is a view into its NodeField's 'U' row."""

    layouts = {name: [n.label for n in nf.nodes] for name, nf in model.nodeFields.items()}
    activation = {label: sorted(node.fields) for label, node in model.nodes.items()}
    for node in model.nodes.values():
        for field, fieldVariable in node.fields.items():
            nodeField = model.nodeFields[field]
            row = nodeField["U"][nodeField.indexOfNode(node)]
            assert np.shares_memory(fieldVariable.values, row), "field variable {:}/{:} is not linked".format(
                node.label, field
            )
    return {
        "layouts": layouts,
        "activation": activation,
        "scalarVariables": sorted(model.scalarVariables),
    }


def test_multi_round_replay_matches_the_live_run_exactly(tmp_path):
    """A replay does the whole-mesh work around apply() -- the fingerprint walk and the node-field
    bookkeeping -- once for the whole history instead of once per record. This pins that the
    shortcut changes nothing observable: the replayed model equals the live one in topology,
    numbering, node-field layout and activation, hanging-node records, and -- after the checkpoint
    is read back -- in every field value; and it equals a replay that does NOT defer (one run inside
    an already-open topology window, where deferral is documented to be a no-op)."""

    modelA = _buildModel(tmp_path, "a.inp")
    _driveRounds(modelA)
    assert len(modelA.topologyHistory) == 3
    assert modelA.topologyHistory[-1].nElementsAdded > 8, "the last round should have cascaded"

    # the octree mirror's active-set cache agrees with a scan of the hierarchy
    mesh = modelA.modelModifiers["amr"]._mesh
    assert mesh.active() == [eid for eid, e in mesh.elements.items() if e["active"]]

    # give every field value something to be restored to, so a wrong layout cannot hide behind zeros
    rng = np.random.default_rng(7)
    for nodeField in modelA.nodeFields.values():
        for entry in ("U", "P"):
            nodeField[entry][:] = rng.random(nodeField[entry].shape)
    checkpoint = tmp_path / "ckpt.h5"
    with h5py.File(checkpoint, "w") as f:
        modelA.writeRestart(f)

    referenceFingerprint = modelA.topologyFingerprint()
    referenceBookkeeping = _fieldBookkeepingByLabel(modelA)
    referenceHanging = _hangingRecordsByLabel(modelA.modelModifiers["amr"]._hanging)

    modelB = _buildModel(tmp_path, "b.inp")  # replays with the deferral (the default)
    modelC = _buildModel(tmp_path, "c.inp")  # replays without it: inner scope of an open window
    with h5py.File(checkpoint, "r") as f:
        modelB.readRestart(f)
        with modelC.topologyChanges():
            modelC.readRestart(f)

    for replayed in (modelB, modelC):
        assert replayed.topologyFingerprint() == referenceFingerprint
        assert set(replayed.elements) == set(modelA.elements)
        assert set(replayed.nodes) == set(modelA.nodes)
        assert _fieldBookkeepingByLabel(replayed) == referenceBookkeeping
        assert _hangingRecordsByLabel(replayed.modelModifiers["amr"]._hanging) == referenceHanging
        for name, nodeField in modelA.nodeFields.items():
            for entry in ("U", "P"):
                assert np.array_equal(replayed.nodeFields[name][entry], nodeField[entry])
        # the replayed history carries the digests that were verified, not freshly computed ones
        assert [r.fingerprint for r in replayed.topologyHistory] == [r.fingerprint for r in modelA.topologyHistory]
        assert [r.time for r in replayed.topologyHistory] == [1.0, 2.0, 3.0]


def test_replay_verifies_once_by_default_and_per_record_on_request(tmp_path):
    """The default check compares the final mesh against the LAST record's fingerprint -- a stale
    digest on an earlier record is not consulted (that is the cost of replaying in O(mesh) rather than
    O(records x mesh)) -- while the per-record mode names the first record that does not match."""

    modelA = _buildModel(tmp_path, "a.inp")
    _driveRounds(modelA)
    history = modelA.topologyHistory

    middleTampered = [replace(r, fingerprint="0" * 32) if i == 1 else r for i, r in enumerate(history)]
    modelB = _buildModel(tmp_path, "b.inp")
    modelB.replayTopologyHistory(middleTampered)
    assert modelB.topologyFingerprint() == modelA.topologyFingerprint()

    modelC = _buildModel(tmp_path, "c.inp")
    modelC.verifyTopologyFingerprintsPerRecord = True
    with pytest.raises(TopologyError, match="replay diverged at record 1 of 3"):
        modelC.replayTopologyHistory(middleTampered)

    lastTampered = [replace(r, fingerprint="0" * 32) if i == 2 else r for i, r in enumerate(history)]
    modelD = _buildModel(tmp_path, "d.inp")
    with pytest.raises(TopologyError, match="after replaying all 3 record"):
        modelD.replayTopologyHistory(lastTampered)

    # a record without a fingerprint (an older checkpoint) is replayed without complaint, and the
    # replayed history then carries a freshly computed digest for it
    bare = [replace(r, fingerprint="") for r in history]
    modelE = _buildModel(tmp_path, "e.inp")
    modelE.replayTopologyHistory(bare)
    assert [r.fingerprint for r in modelE.topologyHistory] == [r.fingerprint for r in history]
