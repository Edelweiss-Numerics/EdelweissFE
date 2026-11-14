"""Unit tests for the element sets the Abaqus-like model constructor builds from an input file."""

import os
import tempfile
import unittest

import edelweissfe.utils.inputfileparser  # noqa: F401 bootstrap input language
from edelweissfe.generators.abqmodelconstructor import AbqModelConstructor
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.utils.inputfileparser import parseInputFile

_N_ELEMENTS = 200


def _deck() -> str:
    """Eight nodes, 200 CPE8 elements on them numbered 1..200, two element sets that list them in an order unrelated to
    their numbers, and a set that is the union of the two (an ``*elSet`` whose data line names sets)."""

    lines = ["*node"]
    corners = [(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0), (1, 0.5), (0.5, 1), (0, 0.5)]
    lines += ["{:}, {:}, {:}".format(i + 1, x, y) for i, (x, y) in enumerate(corners)]
    lines.append("*element, type=CPE8")
    lines += ["{:}, 1, 2, 3, 4, 5, 6, 7, 8".format(n) for n in range(1, _N_ELEMENTS + 1)]
    scrambled = [(n * 73) % _N_ELEMENTS + 1 for n in range(_N_ELEMENTS)]
    assert sorted(scrambled) == list(range(1, _N_ELEMENTS + 1))
    for name, numbers in (("first", scrambled[: _N_ELEMENTS // 2]), ("second", scrambled[_N_ELEMENTS // 2 :])):
        lines.append("*elSet, elSet={:}".format(name))
        lines += [", ".join(str(n) for n in numbers[i : i + 10]) for i in range(0, len(numbers), 10)]
    lines += ["*elSet, elSet=union", "first, second"]
    return "\n".join(lines) + "\n"


class TestElementSetOrder(unittest.TestCase):
    def test_union_of_named_sets_keeps_the_order_of_its_members(self):
        """The union lists the elements of the first set, then those of the second, each in its own order. (An element
        set in memory-address order would make the numbers a generator hands out run-dependent, and with them the
        setup fingerprint a restart is checked against.)"""

        with tempfile.TemporaryDirectory() as directory:
            deck = os.path.join(directory, "deck.inp")
            with open(deck, "w") as f:
                f.write(_deck())
            inputFile = parseInputFile(deck)

        model = FEModel(2)
        AbqModelConstructor(Journal()).createGeometryFromInputFile(model, inputFile)

        expected = [el.elNumber for el in model.elementSets["first"]] + [
            el.elNumber for el in model.elementSets["second"]
        ]
        self.assertEqual([el.elNumber for el in model.elementSets["union"]], expected)


if __name__ == "__main__":
    unittest.main()
