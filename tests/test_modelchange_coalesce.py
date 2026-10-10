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


"""``coalesce`` against the pairwise fold it replaces.

The net change of a history is *defined* by folding it pairwise, oldest first, through the
original ``ModelChange.mergedWith``; ``coalesce`` computes the same thing in linear time. The
reference fold lives only here, as the specification the production code is checked against.
"""

import random

import pytest

from edelweissfe.models.modelchange import ModelChange, coalesce
from edelweissfe.models.modelchangeobserver import ModelChangeType as _MCT

_SET_FIELDS = (
    "addedNodes",
    "removedNodes",
    "addedElements",
    "removedElements",
    "changedNodeSets",
    "changedElementSets",
    "changedSurfaces",
)


def _referenceMerge(older: ModelChange, newer: ModelChange) -> ModelChange:
    """The original pairwise ``ModelChange.mergedWith``, verbatim in behavior."""
    transientElements = older.addedElements & newer.removedElements
    transientNodes = older.addedNodes & newer.removedNodes

    def substituteChildren(children):
        resolved = []
        for label in children:
            resolved.extend(newer.parentToChildren.get(label, [label]))
        return resolved

    parentToChildren = {p: substituteChildren(children) for p, children in older.parentToChildren.items()}
    for p, children in newer.parentToChildren.items():
        parentToChildren.setdefault(p, list(children))

    def substituteFaces(pairs):
        resolved = []
        for elLabel, faceID in pairs:
            substituted = newer.faceMap.get((elLabel, faceID))
            resolved.extend(substituted if substituted is not None else [(elLabel, faceID)])
        return resolved

    faceMap = {key: substituteFaces(pairs) for key, pairs in older.faceMap.items()}
    for key, pairs in newer.faceMap.items():
        faceMap.setdefault(key, list(pairs))

    return ModelChange(
        kind=newer.kind,
        version=newer.version,
        addedNodes=(older.addedNodes | newer.addedNodes) - transientNodes,
        removedNodes=(older.removedNodes | newer.removedNodes) - transientNodes,
        addedElements=(older.addedElements | newer.addedElements) - transientElements,
        removedElements=(older.removedElements | newer.removedElements) - transientElements,
        parentToChildren=parentToChildren,
        faceMap=faceMap,
        changedNodeSets=older.changedNodeSets | newer.changedNodeSets,
        changedElementSets=older.changedElementSets | newer.changedElementSets,
        changedSurfaces=older.changedSurfaces | newer.changedSurfaces,
    )


def _referenceCoalesce(changes: list) -> ModelChange | None:
    """The original pairwise fold."""
    if not changes:
        return None
    result = changes[0]
    for change in changes[1:]:
        result = _referenceMerge(result, change)
    return result


def assertSameChange(actual: ModelChange, expected: ModelChange):
    """Field by field: sets by content, dicts by content AND key order, value lists by order."""
    assert actual.kind == expected.kind
    assert actual.version == expected.version
    for name in _SET_FIELDS:
        assert getattr(actual, name) == getattr(expected, name), name
    for name in ("parentToChildren", "faceMap"):
        actualMap, expectedMap = getattr(actual, name), getattr(expected, name)
        assert list(actualMap) == list(expectedMap), name
        for key, value in expectedMap.items():
            assert actualMap[key] == value, (name, key)
        # every value is a list of its own, so a consumer patching one cannot corrupt another
        assert len({id(value) for value in actualMap.values()}) == len(actualMap), name


def _snapshot(changes: list) -> list:
    """Deep enough a copy of a history to detect coalesce mutating its input."""
    return [
        (
            change.kind,
            change.version,
            [set(getattr(change, name)) for name in _SET_FIELDS],
            [(key, list(value)) for key, value in change.parentToChildren.items()],
            [(key, list(value)) for key, value in change.faceMap.items()],
        )
        for change in changes
    ]


def _check(changes: list):
    before = _snapshot(changes)
    expected = _referenceCoalesce(changes)
    assert _snapshot(changes) == before
    actual = coalesce(changes)
    assert _snapshot(changes) == before, "coalesce mutated its input"
    if expected is None:
        assert actual is None
    else:
        assertSameChange(actual, expected)
        for i in range(len(changes)):
            # a live consumer coalesces only the tail of the log it missed
            assertSameChange(coalesce(changes[i:]), _referenceCoalesce(changes[i:]))
    return actual


def _refinementHistory(seed: int, nChanges: int, levelsPerChange: int) -> list:
    """An AMR-like history: every change refines some active elements into 8 children, nested
    ``levelsPerChange`` deep, with the per-level changesets left separate (intermediates are
    created and removed within the history), plus a faceMap of 6 faces x 4 child faces and
    occasional set/surface changes. Labels are never reused, as in a real model."""
    rng = random.Random(seed)
    active = list(range(1, 41))
    nextElement, nextNode = 1000, 100000
    changes = []
    for _ in range(nChanges):
        refined = rng.sample(active, min(len(active), rng.randint(0, 4)))
        for _ in range(levelsPerChange):
            change = ModelChange(kind=rng.choice([_MCT.REFINEMENT, _MCT.TOPOLOGY_CHANGE]))
            children = []
            for parent in refined:
                kids = list(range(nextElement, nextElement + 8))
                nextElement += 8
                change.parentToChildren[parent] = kids
                change.addedElements |= set(kids)
                change.removedElements.add(parent)
                for faceID in range(1, 7):
                    change.faceMap[(parent, faceID)] = [(kid, faceID) for kid in rng.sample(kids, 4)]
                active.remove(parent)
                active.extend(kids)
                children.extend(kids)
            newNodes = rng.randint(0, 30)
            change.addedNodes |= set(range(nextNode, nextNode + newNodes))
            nextNode += newNodes
            if rng.random() < 0.3:
                change.changedNodeSets.add(rng.choice(["a", "b", "c"]))
                change.changedElementSets.add("all")
                change.changedSurfaces.add(rng.choice(["s1", "s2"]))
            changes.append(change)
            refined = rng.sample(children, min(len(children), rng.randint(0, 3)))
    for version, change in enumerate(changes, start=1):
        change.version = version
    return changes


def _arbitraryHistory(seed: int, nChanges: int) -> list:
    """Random maps over a small label universe, so that labels ARE reused: a parent refined
    twice, an item that is a key of its own map, a label removed and added again. Not what a model
    produces, but the fold's result is still well defined, and coalesce must still match it."""
    rng = random.Random(seed)
    labels = range(12)

    def labelSet():
        return set(rng.sample(labels, rng.randint(0, 4)))

    changes = []
    for version in range(1, nChanges + 1):
        change = ModelChange(kind=rng.choice(list(_MCT)), version=version)
        change.addedNodes, change.removedNodes = labelSet(), labelSet()
        change.addedElements, change.removedElements = labelSet(), labelSet()
        for key in rng.sample(labels, rng.randint(0, 4)):
            change.parentToChildren[key] = [rng.choice(labels) for _ in range(rng.randint(0, 3))]
        for key in rng.sample(labels, rng.randint(0, 3)):
            faceID = rng.randint(1, 2)
            change.faceMap[(key, faceID)] = [(rng.choice(labels), faceID) for _ in range(rng.randint(0, 3))]
        changes.append(change)
    return changes


def test_empty_history():
    assert coalesce([]) is None


def test_single_change_is_returned_itself():
    change = ModelChange(kind=_MCT.REFINEMENT, version=3, addedElements={1}, parentToChildren={0: [1]})
    assert coalesce([change]) is change


def test_empty_changes():
    _check([ModelChange(kind=_MCT.TOPOLOGY_CHANGE, version=v) for v in range(1, 5)])


def test_multilevel_chain_resolves_to_leaves():
    """1 -> 2..3, 2 -> 4..5, 4 -> 6..7: the net parent of everything is 1, intermediates are
    transient, and the faces of 1 resolve to the leaf faces."""
    changes = [
        ModelChange(
            kind=_MCT.REFINEMENT,
            version=1,
            addedElements={2, 3},
            removedElements={1},
            parentToChildren={1: [2, 3]},
            faceMap={(1, 1): [(2, 1), (3, 1)]},
        ),
        ModelChange(
            kind=_MCT.REFINEMENT,
            version=2,
            addedElements={4, 5},
            removedElements={2},
            parentToChildren={2: [4, 5]},
            faceMap={(2, 1): [(4, 1), (5, 1)]},
        ),
        ModelChange(
            kind=_MCT.REFINEMENT,
            version=3,
            addedElements={6, 7},
            removedElements={4},
            parentToChildren={4: [6, 7]},
            faceMap={(4, 1): [(6, 1), (7, 1)]},
        ),
    ]
    net = _check(changes)
    assert net.parentToChildren == {1: [6, 7, 5, 3], 2: [6, 7, 5], 4: [6, 7]}
    assert net.faceMap[(1, 1)] == [(6, 1), (7, 1), (5, 1), (3, 1)]
    assert net.addedElements == {3, 5, 6, 7} and net.removedElements == {1}


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("levelsPerChange", [1, 3])
def test_refinement_histories(seed, levelsPerChange):
    _check(_refinementHistory(seed, nChanges=15, levelsPerChange=levelsPerChange))


@pytest.mark.parametrize("seed", range(200))
def test_arbitrary_histories(seed):
    _check(_arbitraryHistory(seed, nChanges=random.Random(seed).randint(2, 12)))


def test_mergedWith_is_the_pairwise_merge():
    older, newer = _arbitraryHistory(7, nChanges=2)
    assertSameChange(older.mergedWith(newer), _referenceMerge(older, newer))
