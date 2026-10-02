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

"""The structured changeset describing *what* changed in a model mutation, as opposed to the bare
:class:`~edelweissfe.models.modelchangeobserver.ModelChangeType` marker. A modifier (e.g. AMR)
populates one from the delta it already computes; :meth:`~edelweissfe.models.topologypipeline.TopologyPipeline.notifyModelChanged`
records it (bumping ``model.topologyVersion``) and passes it to any registered push observers.

A pull-based consumer instead compares its own last-seen version against ``model.topologyVersion``
at its own next tick and, on a mismatch, reconciles from ``model.changesSince(lastSeenVersion)`` --
a single :class:`ModelChange`, coalesced across every mutation it missed. Cheap ``touches...()``
queries let it early-out when the change doesn't concern it, and ``parentToChildren``/``faceMap``
let it patch only what changed instead of rebuilding from scratch.
"""

from bisect import bisect_right
from dataclasses import dataclass, field

from edelweissfe.models.modelchangeobserver import ModelChangeType


@dataclass
class TopologyRecord:
    """One applied model-modifier decision, as recorded in
    :attr:`~edelweissfe.models.topologypipeline.TopologyPipeline.topologyHistory`.

    This is the authoritative record of how the model's topology came to be what it is -- not a
    debugging aid kept alongside one. A restart replays these through the modifier's own
    :meth:`~edelweissfe.modelmodifiers.base.modelmodifierbase.ModelModifierBase.apply`, which is the
    same code path the live run used, so there is no second implementation to drift from it.
    """

    modifier: str  #: name of the model modifier that made this decision
    roundNumber: int  #: which round of the topology update it was applied in
    time: float  #: model time at which it was applied
    plan: dict  #: the decision, encoded by the modifier (see ModelModifierBase.encodePlan)
    fingerprint: str = ""  #: model.topologyFingerprint() immediately after applying it
    #: summary fields, for the log and for forensics only -- never used to reconstruct anything
    nElementsAdded: int = 0
    nElementsRemoved: int = 0
    nNodesAdded: int = 0


@dataclass
class ModelChange:
    """One model mutation (or, from :meth:`~edelweissfe.models.topologypipeline.TopologyPipeline.changesSince`,
    several coalesced into one net change)."""

    kind: ModelChangeType
    version: int = 0
    addedNodes: set = field(default_factory=set)
    removedNodes: set = field(default_factory=set)
    addedElements: set = field(default_factory=set)
    removedElements: set = field(default_factory=set)
    parentToChildren: dict = field(default_factory=dict)  #: element label -> [child element labels]
    faceMap: dict = field(default_factory=dict)  #: (element label, faceID) -> [(child label, faceID)]
    changedNodeSets: set = field(default_factory=set)
    changedElementSets: set = field(default_factory=set)
    changedSurfaces: set = field(default_factory=set)

    @property
    def geometryChanged(self) -> bool:
        """True if any node or element was added or removed."""
        return bool(self.addedNodes or self.removedNodes or self.addedElements or self.removedElements)

    @property
    def isEmpty(self) -> bool:
        """True if this changeset records no mutation at all -- every collection is empty.

        A modifier can legitimately plan and then find nothing left to do, and returning an empty
        changeset is how it says so. ``kind`` is deliberately not consulted: it is always set, and a
        label alone is not a change.
        """
        return not (
            self.addedNodes
            or self.removedNodes
            or self.addedElements
            or self.removedElements
            or self.parentToChildren
            or self.faceMap
            or self.changedNodeSets
            or self.changedElementSets
            or self.changedSurfaces
        )

    def touchesSurface(self, name: str) -> bool:
        return name in self.changedSurfaces

    def touchesNodeSet(self, name: str) -> bool:
        return name in self.changedNodeSets

    def touchesElementSet(self, name: str) -> bool:
        return name in self.changedElementSets

    def childFacesOf(self, elLabel: int, faceID: int) -> list:
        """The child ``(elementLabel, faceID)`` pairs tiling the given parent face, or ``[]`` if
        that element/face wasn't refined by this change."""
        return list(self.faceMap.get((elLabel, faceID), []))

    def mergedWith(self, other: "ModelChange") -> "ModelChange":
        """Coalesce this (older) change with ``other`` (applied immediately after) into the single
        net change a consumer that missed both would need. Element/node labels are never reused, so
        a label added by ``self`` and removed again by ``other`` existed only within the window and
        is dropped from both the added and the removed set, rather than surfacing as a phantom
        create-then-delete.

        Every entry of ``self.parentToChildren`` / ``self.faceMap`` is resolved through ``other``'s
        (a child that ``other`` refined further is replaced by *its* children), and ``other``'s own
        entries are added for parents ``self`` did not already list. This is :func:`coalesce` of
        the two; folding a longer history pairwise gives the same result as coalescing it at once,
        only in quadratic instead of linear time.
        """
        return coalesce([self, other])


def _composeSubstitutions(maps: list) -> dict:
    """Compose a chronological list of substitution maps (``parentToChildren`` or ``faceMap``,
    one per change) into the single net map.

    The net map is, by definition, the pairwise fold: map ``k`` is merged into the net map of maps
    ``0 .. k-1`` by replacing every value item that map ``k`` has as a key with map ``k``'s value
    for it (an item it doesn't have stays itself), and by then appending map ``k``'s keys that are
    not keys already, in map ``k``'s order. Folding like that re-walks every list accumulated so far
    for every new map -- quadratic in the history, 368 s for the 372 changes of a long AMR run.

    Here every item is resolved exactly once instead. An item that appears in the value of map
    ``k`` is only ever substituted by maps ``k+1, k+2, ...``: by the *first* later map that has it
    as a key, whose value items are in turn resolved by the maps after *that* one. So, walking the
    maps from the newest to the oldest, the fully resolved value of every key of map ``k`` is the
    concatenation of the already resolved values its items have at their next occurrence as a key
    (or the item itself, if no later map has it). The cost is linear in the size of the result.

    Parameters
    ----------
    maps
        The per-change maps, oldest first; each maps a key to a list of items, and an item may be
        a key of a later map.

    Returns
    -------
    dict
        The net map: every key of any map, in order of first appearance, with its fully resolved
        value list (a new list, not one of the input lists).
    """
    # for every key: the indices of the maps it is a key in, ascending (nearly always just one,
    # since labels are never reused -- but a key appearing again must still be honored)
    occurrences = {}
    for k, substitution in enumerate(maps):
        for key in substitution:
            occurrences.setdefault(key, []).append(k)

    # resolved[k][key]: map k's value for key, with every item resolved through maps k+1, ...
    resolved = [None] * len(maps)
    for k in range(len(maps) - 1, -1, -1):
        resolvedHere = {}
        for key, items in maps[k].items():
            value = []
            for item in items:
                itemOccurrences = occurrences.get(item)
                nextOccurrence = None
                if itemOccurrences is not None:
                    i = bisect_right(itemOccurrences, k)
                    if i < len(itemOccurrences):
                        nextOccurrence = itemOccurrences[i]
                if nextOccurrence is None:
                    value.append(item)
                else:
                    value.extend(resolved[nextOccurrence][item])
            resolvedHere[key] = value
        resolved[k] = resolvedHere

    # a key keeps the value of the map it first appears in; a later map listing it again only
    # substitutes it where it appears as an item, which the resolution above already did
    net = {}
    for k, substitution in enumerate(maps):
        for key in substitution:
            if key not in net:
                net[key] = resolved[k][key]
    return net


def coalesce(changes: list) -> ModelChange | None:
    """Fold a chronological list of changes into the single net :class:`ModelChange` a consumer
    that missed all of them would need (see :meth:`ModelChange.mergedWith`). ``None`` for an empty
    list, and the change itself for a single one.

    Linear in the total size of the history: a resume coalesces every change recorded since the
    start of the run, so a pairwise fold, which re-walks the whole accumulated change per change,
    would be quadratic in it.
    """
    if not changes:
        return None
    if len(changes) == 1:
        return changes[0]

    first = changes[0]
    addedNodes, removedNodes = set(first.addedNodes), set(first.removedNodes)
    addedElements, removedElements = set(first.addedElements), set(first.removedElements)
    changedNodeSets, changedElementSets = set(first.changedNodeSets), set(first.changedElementSets)
    changedSurfaces = set(first.changedSurfaces)
    for change in changes[1:]:
        # labels added within the window and removed again existed only within it; "added" means
        # added by any EARLIER change, hence intersected before this change's additions join in
        transientNodes = addedNodes & change.removedNodes
        transientElements = addedElements & change.removedElements
        addedNodes |= change.addedNodes
        addedNodes -= transientNodes
        removedNodes |= change.removedNodes
        removedNodes -= transientNodes
        addedElements |= change.addedElements
        addedElements -= transientElements
        removedElements |= change.removedElements
        removedElements -= transientElements
        changedNodeSets |= change.changedNodeSets
        changedElementSets |= change.changedElementSets
        changedSurfaces |= change.changedSurfaces

    last = changes[-1]
    return ModelChange(
        kind=last.kind,
        version=last.version,
        addedNodes=addedNodes,
        removedNodes=removedNodes,
        addedElements=addedElements,
        removedElements=removedElements,
        parentToChildren=_composeSubstitutions([change.parentToChildren for change in changes]),
        faceMap=_composeSubstitutions([change.faceMap for change in changes]),
        changedNodeSets=changedNodeSets,
        changedElementSets=changedElementSets,
        changedSurfaces=changedSurfaces,
    )
