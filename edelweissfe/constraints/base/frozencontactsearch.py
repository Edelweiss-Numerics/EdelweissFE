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
to the checkpoint together with the layout it indexes (contact points and master entities), and a
resumed step adopts it instead of searching, if the layout still matches.
"""

from abc import abstractmethod

import numpy as np

from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel

_layoutPrefix = "searchLayout_"


def packPerPointArrays(arrays: list) -> tuple[np.ndarray, np.ndarray]:
    """Flatten one array (or ``None``) per contact point into (counts, values)."""

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


class FrozenContactSearch:
    """Mixin writing the frozen projection to the checkpoint and adopting it on resume.

    Subclasses implement :meth:`_searchLayout`, :meth:`_frozenProjection` and
    :meth:`_adoptFrozenProjection`.
    """

    name: str
    journal: Journal
    _restoredProjection: dict[str, np.ndarray] | None = None

    @abstractmethod
    def _searchLayout(self) -> dict[str, np.ndarray]:
        """The contact points and master entities the projection indexes."""

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
        # Checked against the layout only in resumeConnectivity, after the topology replay.
        hasLayout = any(key.startswith(_layoutPrefix) for key in data)
        self._restoredProjection = dict(data) if hasLayout else None

    def resumeConnectivity(self, model: FEModel) -> bool:
        restored, self._restoredProjection = self._restoredProjection, None
        layout = {_layoutPrefix + key: value for key, value in self._searchLayout().items()}

        if restored is None:
            reason = "the checkpoint holds no frozen contact projection"
        elif {k for k in restored if k.startswith(_layoutPrefix)} != layout.keys() or not all(
            np.array_equal(restored[k], v) for k, v in layout.items()
        ):
            reason = "the checkpointed contact projection belongs to other contact points"
        else:
            return self._adoptFrozenProjection(restored)

        self.journal.message(f"{reason}; searching afresh", self.name)
        return self.updateConnectivity(model)
