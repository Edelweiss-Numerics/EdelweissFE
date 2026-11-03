"""Unit tests for the Ensight geometry writer."""

import os
import tempfile
import unittest

import numpy as np

from edelweissfe.outputmanagers.ensight import EnsightUnstructuredPart
from edelweissfe.points.node import Node


def _c80(string: str) -> bytes:
    return string.encode().ljust(80, b"\0")


class TestEnsightUnstructuredPartGeometry(unittest.TestCase):
    def test_geometry_bytes_of_a_part_with_two_element_shapes(self):
        """The file holds, in this order: the part header, the coordinates (all x, then all y, then all z, as float32), and per element shape its
        name, the number of elements, their labels and the connectivity table (node indices, 1-based) element after element. Checked against
        the bytes assembled here by hand, independent of the writer."""

        coordinates = [(float(i), 2.0 * i, 0.5 * i * i) for i in range(10)]
        nodes = [Node(100 + i, np.array(c)) for i, c in enumerate(coordinates)]
        hexas = {7: [0, 1, 2, 3, 4, 5, 6, 7], 3: [1, 2, 3, 4, 5, 6, 7, 8], 9: [2, 3, 4, 5, 6, 7, 8, 9]}
        tetras = {11: [0, 1, 2, 3], 5: [4, 5, 6, 7]}
        part = EnsightUnstructuredPart("test part", 4, nodes, {"hexa8": hexas, "tetra4": tetras})

        with tempfile.TemporaryDirectory() as directory:
            fileName = os.path.join(directory, "geo")
            with open(fileName, "wb") as f:
                part.writeToFile(f)
            with open(fileName, "rb") as f:
                written = f.read()

        expected = (
            _c80("part") + np.int32(4).tobytes() + _c80("test part") + _c80("coordinates") + np.int32(10).tobytes()
        )
        expected += np.arange(100, 110, dtype=np.int32).tobytes()
        for axis in range(3):
            expected += np.array([c[axis] for c in coordinates], dtype=np.float32).tobytes()
        for shape, elements in (("hexa8", hexas), ("tetra4", tetras)):
            expected += _c80(shape) + np.int32(len(elements)).tobytes()
            expected += np.array(list(elements.keys()), dtype=np.int32).tobytes()
            for indices in elements.values():
                expected += (np.array(indices, dtype=np.int32) + 1).tobytes()

        self.assertEqual(written, expected)


if __name__ == "__main__":
    unittest.main()
