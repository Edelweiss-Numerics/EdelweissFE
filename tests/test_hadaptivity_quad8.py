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
"""h-adaptivity of a 2D (QUAD8) model, mirroring tests/test_hadaptivity_restart.py: the default refineable
set, a live refinement's node and element sets and surfaces, the hanging-node records, and the restart
replay -- of one decision and of a multi-round, 2:1-cascading history with a checkpoint read back."""

from pathlib import Path

import h5py
import numpy as np
import pytest

from edelweissfe.config.phenomena import domainMapping
from edelweissfe.helpers.inputfilehelpers import fillFEModelFromInputFile
from edelweissfe.journal.journal import Journal
from edelweissfe.modelmodifiers.adaptivity.hadaptivity import RefinementPlan
from edelweissfe.models.femodel import FEModel
from edelweissfe.utils.inputfileparser import parseInputFile

_INP = """
*material, name=linearelastic, id=mat
18000, 0.0

*section, name=sec, material=mat, type=plane, thickness=1.0
plate_all

*modelGenerator, generator=planeRectQuad, name=plate
nX      =3
nY      =3
x0      =0
y0      =0
l       =1
h       =1
elType  =CPS8R

*modelModifier, type=hAdaptivity, name=amr
>>marker, type=nodeSet, nSet=plate_top, initialOnly=True
maxLevel=1

*job, name=hadaptivityQuad8Test, domain=2d
*solver, solver=NIST, name=theSolver
*fieldOutput
>>perNode, elSet=plate_all, field=displacement, result=U, name=disp

*step, solver=theSolver
maxInc=1.0, minInc=1.0, maxNumInc=1, maxIter=25, stepLength=1
>>dirichlet, name=fixBottom, nSet=plate_bottom, field=displacement, 1=0.0, 2=0.0
"""


def _buildModel(tmp_path: Path, name: str) -> FEModel:
    inpPath = tmp_path / name
    inpPath.write_text(_INP)
    inputfile = parseInputFile(str(inpPath))
    journal = Journal(verbose=False)
    job = inputfile["job"][0]
    model = FEModel(domainMapping[job["domain"]])
    model = fillFEModelFromInputFile(model, inputfile, journal)
    model.prepareYourself(journal)
    model.advanceToTime(job.get("startTime", 0.0))
    model._linkFieldVariableObjects(model.nodeSets["all"])
    return model


def _hangingRecordsByLabel(hangingConstraint) -> dict:
    return {
        slave.label: sorted((master.label, weight) for master, weight in masters)
        for slave, masters in hangingConstraint._records
    }


def test_a_live_refinement_of_a_quad8_model_keeps_sets_surfaces_and_constraints_consistent(tmp_path):
    model = _buildModel(tmp_path, "a.inp")
    assert len(model.modelModifiers["amr"]._mesh.active()) == 9, "every 8-node element is a root by default"

    assert model.topology.update(step=None)
    amr = model.modelModifiers["amr"]
    # the three top elements, each split into 2 x 2
    assert len(model.elements) == 6 + 3 * 4
    assert all(len(el.nodes) == 8 for el in model.elements.values())

    coordinates = {label: node.coordinates for label, node in model.nodes.items()}
    top = model.nodeSets["plate_top"]
    assert len(top) == 2 * 3 * 2 + 1 and all(coordinates[n.label][1] == 1.0 for n in top)
    assert len(model.elementSets["plate_all"]) == len(model.elements)
    # the top surface is now tiled by the children's top edges
    for faceID, elements in model.surfaces["plate_top"].items():
        for el in elements:
            face = amr._topology.faces[amr._topology.faceid_to_face[faceID]]
            assert all(el.nodes[i].coordinates[1] == 1.0 for i in face)

    # the coarse/fine interface y = 2/3: two quarter points hang on each of the three coarse edges
    records = _hangingRecordsByLabel(amr._hanging)
    assert len(records) == 6
    for slave, masters in records.items():
        trace = sum(w * coordinates[m] for m, w in masters)
        assert np.allclose(coordinates[slave], trace)
        assert coordinates[slave][1] == pytest.approx(2.0 / 3.0)


def test_topology_history_roundtrip_reproduces_the_refinement(tmp_path):
    modelA = _buildModel(tmp_path, "a.inp")
    assert modelA.topology.update(step=None)
    modelB = _buildModel(tmp_path, "b.inp")
    modelB.topology.replayHistory(modelA.topology.history)
    assert modelB.topology.fingerprint() == modelA.topology.fingerprint()
    assert _hangingRecordsByLabel(modelB.modelModifiers["amr"]._hanging) == _hangingRecordsByLabel(
        modelA.modelModifiers["amr"]._hanging
    )


def _driveRounds(model: FEModel) -> None:
    """Three decisions, one topology window each: the centre root, a corner root, then a child of the
    centre root, which 2:1 balancing answers by refining coarse neighbours too."""
    amr = model.modelModifiers["amr"]
    mesh = amr._mesh

    def nearestToCentre(eids):
        return min(eids, key=lambda e: np.linalg.norm(mesh.elements[e]["coords"].mean(0) - 0.5))

    centre = nearestToCentre(mesh.active())
    rounds = [lambda: [centre], lambda: [mesh.active()[0]], lambda: [mesh.elements[centre]["children"][0]]]
    for roundIndex, eids in enumerate(rounds, start=1):
        model.advanceToTime(float(roundIndex))
        with model.topology.changes():
            plan = RefinementPlan(eids=list(eids()))
            change = amr.apply(model, plan)
            model.topology.recordChange(1, "amr", amr, plan, change)


def test_multi_round_replay_from_a_checkpoint_matches_the_live_run(tmp_path):
    modelA = _buildModel(tmp_path, "a.inp")
    _driveRounds(modelA)
    assert len(modelA.topology.history) == 3
    assert modelA.topology.history[-1].nElementsAdded > 4, "the last round should have cascaded"

    rng = np.random.default_rng(7)
    for nodeField in modelA.nodeFields.values():
        for entry in ("U", "P"):
            nodeField[entry][:] = rng.random(nodeField[entry].shape)
    checkpoint = tmp_path / "ckpt.h5"
    with h5py.File(checkpoint, "w") as f:
        modelA.writeRestart(f)

    modelB = _buildModel(tmp_path, "b.inp")
    with h5py.File(checkpoint, "r") as f:
        modelB.readRestart(f)
    assert modelB.topology.fingerprint() == modelA.topology.fingerprint()
    assert set(modelB.nodes) == set(modelA.nodes)
    assert _hangingRecordsByLabel(modelB.modelModifiers["amr"]._hanging) == _hangingRecordsByLabel(
        modelA.modelModifiers["amr"]._hanging
    )
    for name, nodeField in modelA.nodeFields.items():
        for entry in ("U", "P"):
            assert np.array_equal(modelB.nodeFields[name][entry], nodeField[entry])
