"""Test helper: build a standalone :class:`~edelweissfe.adaptivity.refinement.AdaptiveMesh` from root
coordinates alone.

A model-backed mesh passes each root's node labels -- the model's connectivity. A test fixture often
has coordinates only; :class:`AdaptiveMeshBuilder` numbers its nodes by exact coordinate equality
*within the fixture* (nodes of two roots of one body are shared iff their coordinates are bitwise
identical) and seeds them into the mesh's registry. This is scaffolding for writing fixtures, not a
mechanism of the refinement itself.
"""

import numpy as np

from edelweissfe.adaptivity.hex20topology import Hex20Topology
from edelweissfe.adaptivity.refinement import AdaptiveMesh


class AdaptiveMeshBuilder:
    """An :class:`AdaptiveMesh` (``.mesh``) whose roots are added by coordinates."""

    def __init__(self, splitFactor: int = 2):
        self.mesh = AdaptiveMesh(splitFactor=splitFactor, topology=Hex20Topology())
        self._labelOfPoint = {}  # (componentId, coordinates) -> label

    def addRoot(self, coords, componentId: int = 0) -> int:
        """Add a root element from its 20 node coordinates; returns its eid."""
        labels = []
        for x in np.asarray(coords, dtype=float):
            point = (componentId, tuple(float(v) for v in x))
            if point not in self._labelOfPoint:
                self._labelOfPoint[point] = len(self._labelOfPoint) + 1
                self.mesh.registry.seed(self._labelOfPoint[point], x, componentId)
            labels.append(self._labelOfPoint[point])
        return self.mesh.add_root(coords, labels, componentId)
