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
"""Round-trip test for the recorded topology history of an h-adaptive model (unlike
tests/test_restart_integration.py's AMR case, which drives the same feature end-to-end through a
real solve). Builds two independent models from the same .inp via fillFEModelFromInputFile
(mirroring the driver's own setup sequence, minus solvers/steps, which this doesn't need), drives
one real refinement on the first through the live topology update, and asserts that replaying its
recorded history onto the second -- freshly rebuilt, unrefined -- reproduces the same topology.

Uses an `initialOnly` marker (edelweissfe.adaptivity.marking) so refinement triggers
deterministically on the very first topology update, regardless of field state -- avoids depending
on a real solve's marker-evaluation timing, which the full-solve integration test in
test_restart_integration.py already covers separately.
"""

from pathlib import Path

import pytest

from edelweissfe.helpers.inputfilehelpers import fillFEModelFromInputFile
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
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
    """The fingerprint recorded with each decision is what turns "the resumed run diverged" into
    "it diverged at THIS record" -- so a plan that no longer reproduces its recorded topology must
    be reported, not silently applied."""

    from dataclasses import replace

    from edelweissfe.utils.exceptions import TopologyError

    modelA = _buildModel(tmp_path, "a.inp")
    modelA.updateTopology(step=None, timeStep=0.0)
    assert modelA.topologyHistory

    tampered = [replace(record, fingerprint="0" * 32) for record in modelA.topologyHistory]

    modelB = _buildModel(tmp_path, "b.inp")
    with pytest.raises(TopologyError, match="replay diverged at record 0"):
        modelB.replayTopologyHistory(tampered)


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
