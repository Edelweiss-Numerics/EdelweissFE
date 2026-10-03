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
"""
Checkpointing of a frozen contact projection.

Between two contact searches, a small-sliding contact constraint keeps the projection frozen at
the last one (assigned master facet or rigid triangle, shape functions, normal). An explicit solver
searches only every ``contact-update-frequency`` increments, so this projection belongs to an older
configuration than the checkpointed one and cannot be recomputed on resume. It is therefore written
to the checkpoint together with the layout it indexes (contact points and master entities), and
restoring the constraint adopts it; a layout that does not match the restored model raises.
"""

from abc import ABC, abstractmethod

import numpy as np

from edelweissfe.journal.journal import Journal
from edelweissfe.utils.exceptions import RestartError

#: Restart entries describing the search layout carry this prefix, the projection itself does not.
_layoutPrefix = "searchLayout_"


def packPerPointArrays(arrays: list) -> tuple[np.ndarray, np.ndarray]:
    """Flatten one 1-D array (or ``None``) per contact point into (counts, values) for a checkpoint.

    A point without a projection is stored with count 0. Arrays of higher rank lose their shape."""

    counts = np.array([0 if a is None else np.size(a) for a in arrays], dtype=np.int64)
    present = [np.ravel(a) for a in arrays if a is not None]
    return counts, np.concatenate(present) if present else np.zeros(0)


def unpackPerPointArrays(counts: np.ndarray, values: np.ndarray) -> list:
    """Invert :func:`packPerPointArrays`."""

    ends = np.cumsum(counts)
    return [None if c == 0 else np.array(values[end - c : end]) for c, end in zip(counts, ends)]


def packAssignment(assignment: list) -> np.ndarray:
    """Assigned master index per contact point, -1 for ``None``."""

    return np.array([-1 if a is None else a for a in assignment], dtype=np.int64)


def unpackAssignment(packed: np.ndarray) -> list:
    """Invert :func:`packAssignment`."""

    return [None if a < 0 else int(a) for a in packed]


class FrozenContactSearch(ABC):
    """Mixin writing the frozen projection to the checkpoint and adopting it on resume.

    The restart data holds two parts: the *layout* (which contact points and master entities the
    projection refers to) and the *projection* itself. Restoring adopts the projection, after checking
    that the layout is the restored model's.

    Subclasses implement :meth:`_searchLayout`, :meth:`_frozenProjection` and
    :meth:`_adoptFrozenProjection`.
    """

    name: str
    journal: Journal

    @abstractmethod
    def _searchLayout(self) -> dict[str, np.ndarray]:
        """The contact points and master entities the projection refers to."""

    @abstractmethod
    def _frozenProjection(self) -> dict[str, np.ndarray]:
        """The frozen projection as flat arrays."""

    @abstractmethod
    def _adoptFrozenProjection(self, projection: dict[str, np.ndarray]) -> bool:
        """Install a projection from :meth:`_frozenProjection`; return as ``updateConnectivity``."""

    def getRestartData(self) -> dict[str, np.ndarray]:
        layout = {_layoutPrefix + key: value for key, value in self._searchLayout().items()}
        return layout | self._frozenProjection()

    def setRestartData(self, data: dict[str, np.ndarray]):
        # Called after the topology replay and after the mesh dependents caught up with it (see
        # FEModel.readRestart), so the restored layout is compared with the replayed one.
        if not self._layoutMatches(data):
            raise RestartError(
                "constraint {:}: the checkpointed contact search refers to contact points or master "
                "entities that the restored model does not have".format(self.name)
            )
        self._adoptFrozenProjection(data)

    def _layoutMatches(self, restored: dict[str, np.ndarray]) -> bool:
        """Whether a checkpoint's layout entries equal the layout of this constraint."""

        restoredLayout = {key: value for key, value in restored.items() if key.startswith(_layoutPrefix)}
        currentLayout = {_layoutPrefix + key: value for key, value in self._searchLayout().items()}

        if restoredLayout.keys() != currentLayout.keys():
            return False

        return all(np.array_equal(restoredLayout[key], currentLayout[key]) for key in currentLayout)
