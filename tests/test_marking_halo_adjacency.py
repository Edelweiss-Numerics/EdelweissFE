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
"""The node adjacency a marker's halo grows over is kept across planning passes, and must give
exactly the marks a fresh build would -- before a refinement, after one, and after the same
refinement replayed on a fresh model (what a restart does).

The reference is the marking as it was before the adjacency was kept: a per-row threshold loop and
a halo whose node adjacency is rebuilt from the candidate pool on every call. Both run on a real
hAdaptivity model; only the fieldOutput result is synthetic, a deterministic function of the element
centroid, so that the live and the replayed model see the same field.
"""

from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

import edelweissfe.adaptivity.marking as marking
from edelweissfe.adaptivity.marking import FieldOutputMarker
from edelweissfe.helpers.inputfilehelpers import fillFEModelFromInputFile
from edelweissfe.journal.journal import Journal
from edelweissfe.modelmodifiers.adaptivity.hadaptivity import RefinementPlan
from edelweissfe.models.femodel import FEModel
from edelweissfe.utils.inputfileparser import parseInputFile

_INP = """
*material, name=linearelastic, id=mat
18000, 0.0

*section, name=sec, material=mat, type=solid
bar_all

*modelGenerator, generator=boxGen, name=bar
nX      =6
nY      =3
nZ      =2
x0      =0
y0      =0
z0      =0
lX      =6
lY      =3
lZ      =2
elType  =C3D20

*modelModifier, type=hAdaptivity, name=amr
>>marker, type=nodeSet, nSet=bar_right, initialOnly=True
maxLevel=4

*job, name=markingHaloAdjacencyTest, domain=3d
*solver, solver=NIST, name=theSolver

*step, solver=theSolver
maxInc=1.0, minInc=1.0, maxNumInc=1, maxIter=25, stepLength=1
>>dirichlet, name=fixLeft, nSet=bar_left, field=displacement, 1=0.0, 2=0.0, 3=0.0
"""


def _buildModel(tmp_path: Path) -> FEModel:
    inpPath = tmp_path / "markinghalo.inp"
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


def _syntheticField(model, monkeypatch):
    """Let every fieldOutput report, for each element of the model, a 2-entry row that depends on
    its centroid only: a band around x = 2.3 with a bump towards y = 0, which marks a few elements
    by one entry or the other but not the whole mesh."""

    def fieldOutputResult(model, fieldOutputName):
        elements = sorted(model.elements.values(), key=lambda element: element.elNumber)
        centroids = np.array([np.mean([node.coordinates for node in element.nodes], axis=0) for element in elements])
        band = np.exp(-((centroids[:, 0] - 2.3) ** 2))
        bump = np.exp(-(centroids[:, 1] ** 2)) * (centroids[:, 2] < 1.0)
        return elements, np.stack([band, bump], axis=1)

    monkeypatch.setattr(marking, "_perElementFieldOutputResult", fieldOutputResult)


def _referenceMark(model, marker, candidatePool):
    """The marking of ``marker`` as implemented before its adjacency was kept: the threshold row by
    row, and the halo grown over a node adjacency rebuilt from ``candidatePool`` for this one call."""

    elements, values = marking._perElementFieldOutputResult(model, marker.fieldOutputName)
    marked = set()
    for element, row in zip(elements, values):
        if bool(np.any(marker._compare(np.asarray(row), marker.threshold))):
            marked.add(element)
    if marker.halo <= 0 or not marked:
        return marked

    elementsAtNode = defaultdict(list)
    for element in candidatePool:
        for node in element.nodes:
            elementsAtNode[node.label].append(element)
    grown = set(marked)
    frontier = set(marked)
    for _ in range(marker.halo):
        nextFrontier = set()
        for element in frontier:
            for node in element.nodes:
                for neighbor in elementsAtNode[node.label]:
                    if neighbor not in grown:
                        grown.add(neighbor)
                        nextFrontier.add(neighbor)
        if not nextFrontier:
            break
        frontier = nextFrontier
    return grown


_MARKERS = [
    FieldOutputMarker("synthetic", threshold=0.9, operator=">", halo=1),
    FieldOutputMarker("synthetic", threshold=0.5, operator=">=", halo=2),
    FieldOutputMarker("synthetic", threshold=0.2, operator="<", halo=0),
]


def _assertMarksMatchReference(model, amr):
    for marker in _MARKERS:
        marked = marker.mark(model, amr._refineableElements, amr._mesh)
        reference = _referenceMark(model, marker, list(amr._eidToEl.values()))
        assert marked, "the synthetic field must mark something, or the comparison proves nothing"
        assert marked == reference


def _elementNumbers(elements):
    return sorted(element.elNumber for element in elements)


@pytest.mark.parametrize("nRefinements", [1, 2])
def test_kept_adjacency_marks_exactly_what_a_fresh_build_marks(tmp_path, monkeypatch, nRefinements):
    """Mark, refine where the halo marker says, and mark again: the adjacency built for the first
    marks must not survive into the refined mesh."""

    model = _buildModel(tmp_path)
    amr = model.modelModifiers["amr"]
    _syntheticField(model, monkeypatch)

    _assertMarksMatchReference(model, amr)
    for _ in range(nRefinements):
        marked = _MARKERS[0].mark(model, amr._refineableElements, amr._mesh)
        elForEid = {element: eid for eid, element in amr._eidToEl.items()}
        plan = RefinementPlan(eids=[elForEid[element] for element in sorted(marked, key=lambda e: e.elNumber)])
        with model.topologyChanges():
            amr.apply(model, plan)
        _assertMarksMatchReference(model, amr)


def test_replayed_refinement_marks_what_the_live_run_marked(tmp_path, monkeypatch):
    """A restart replays the recorded plans on a freshly built model; its marks must be those of the
    live run, element numbers included."""

    for directory in ("live", "replay"):
        (tmp_path / directory).mkdir()
    live = _buildModel(tmp_path / "live")
    replayed = _buildModel(tmp_path / "replay")
    # one synthetic field for both models: it reads the model it is asked about
    _syntheticField(live, monkeypatch)
    liveAmr = live.modelModifiers["amr"]
    replayedAmr = replayed.modelModifiers["amr"]

    plans = []
    for _ in range(2):
        marked = _MARKERS[0].mark(live, liveAmr._refineableElements, liveAmr._mesh)
        elForEid = {element: eid for eid, element in liveAmr._eidToEl.items()}
        plans.append(RefinementPlan(eids=[elForEid[element] for element in sorted(marked, key=lambda e: e.elNumber)]))
        with live.topologyChanges():
            liveAmr.apply(live, plans[-1])

    # the replayed modifier marks once on the unrefined mesh first, so it holds an adjacency the
    # replay has to discard
    _MARKERS[0].mark(replayed, replayedAmr._refineableElements, replayedAmr._mesh)
    for plan in plans:
        with replayed.topologyChanges():
            replayedAmr.apply(replayed, plan)

    for marker in _MARKERS:
        liveMarks = marker.mark(live, liveAmr._refineableElements, liveAmr._mesh)
        replayedMarks = marker.mark(replayed, replayedAmr._refineableElements, replayedAmr._mesh)
        assert _elementNumbers(replayedMarks) == _elementNumbers(liveMarks)
        assert replayedMarks == _referenceMark(replayed, marker, list(replayedAmr._eidToEl.values()))
