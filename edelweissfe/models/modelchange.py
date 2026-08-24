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
populates one from the delta it already computes; :meth:`~edelweissfe.models.femodel.FEModel.notifyModelChanged`
records it (bumping ``model.topologyVersion``) and passes it to any registered push observers.

A pull-based consumer instead compares its own last-seen version against ``model.topologyVersion``
at its own next tick and, on a mismatch, reconciles from ``model.changesSince(lastSeenVersion)`` --
a single :class:`ModelChange`, coalesced across every mutation it missed. Cheap ``touches...()``
queries let it early-out when the change doesn't concern it, and ``parentToChildren``/``faceMap``
let it patch only what changed instead of rebuilding from scratch.
"""

from dataclasses import dataclass, field

from edelweissfe.models.modelchangeobserver import ModelChangeType


@dataclass
class ModelChange:
    """One model mutation (or, from :meth:`~edelweissfe.models.femodel.FEModel.changesSince`,
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
        create-then-delete."""
        transientElements = self.addedElements & other.removedElements
        transientNodes = self.addedNodes & other.removedNodes

        def substituteChildren(children):
            resolved = []
            for label in children:
                resolved.extend(other.parentToChildren.get(label, [label]))
            return resolved

        parentToChildren = {p: substituteChildren(children) for p, children in self.parentToChildren.items()}
        for p, children in other.parentToChildren.items():
            parentToChildren.setdefault(p, list(children))

        def substituteFaces(pairs):
            resolved = []
            for elLabel, faceID in pairs:
                substituted = other.faceMap.get((elLabel, faceID))
                resolved.extend(substituted if substituted is not None else [(elLabel, faceID)])
            return resolved

        faceMap = {key: substituteFaces(pairs) for key, pairs in self.faceMap.items()}
        for key, pairs in other.faceMap.items():
            faceMap.setdefault(key, list(pairs))

        return ModelChange(
            kind=other.kind,
            version=other.version,
            addedNodes=(self.addedNodes | other.addedNodes) - transientNodes,
            removedNodes=(self.removedNodes | other.removedNodes) - transientNodes,
            addedElements=(self.addedElements | other.addedElements) - transientElements,
            removedElements=(self.removedElements | other.removedElements) - transientElements,
            parentToChildren=parentToChildren,
            faceMap=faceMap,
            changedNodeSets=self.changedNodeSets | other.changedNodeSets,
            changedElementSets=self.changedElementSets | other.changedElementSets,
            changedSurfaces=self.changedSurfaces | other.changedSurfaces,
        )


def coalesce(changes: list) -> ModelChange | None:
    """Fold a chronological list of changes into the single net :class:`ModelChange` a consumer
    that missed all of them would need. ``None`` for an empty list."""
    if not changes:
        return None
    result = changes[0]
    for change in changes[1:]:
        result = result.mergedWith(change)
    return result
