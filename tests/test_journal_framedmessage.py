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
"""A framed message stands apart from the lines around it, and keeps its layout: a tree drawn with ``├─`` and ``└─`` is
wrapped below its own indentation, and every line of the frame is as wide as the message column, whatever it holds.
"""

import io

from edelweissfe.journal.journal import Journal, displayWidth

LEVEL = 2
TREE = """LGMRES(tolerance=ForcingTolerance(minimum=0.0001, maximum=0.001), maxIterations=1000, restart=50)
└─ SchurComplement(blockFactors=U, S̃=A_kk − A_kb A_bb⁻¹ A_bk, local)
   ├─ solveA_bb: DirectSolver(backend=superlu)
   └─ preconditionS: TwoLevel(coarse space=rigid-body modes, prolongation=smoothed, coarse solve=AMG V-cycle)

from solver.py (sha256 b88a93da5088)"""


def _messageColumn(line: str) -> str:
    """What a written line holds, without the indentation of the level and the sender."""
    return line[5:].rstrip().removesuffix("ComposedSolver").rstrip()


def _frame(message: str) -> list:
    output = io.StringIO()
    journal = Journal(verbose=False)
    journal.setFileOutput(output)
    journal.framedMessage(message, "ComposedSolver", "composed linear solver", level=LEVEL)
    return output.getvalue().splitlines()


def test_every_line_of_the_frame_is_as_wide_as_the_message_column():
    lines = _frame(TREE)
    width = Journal._messageWidth(LEVEL)
    for line in lines:
        column = _messageColumn(line)
        assert column[0] in "+|" and column[-1] in "+|"
        # the tilde of S̃ is a character of its own that takes no column
        assert displayWidth(column) == width


def test_the_frame_carries_the_title_and_the_sender_on_its_first_line_only():
    lines = _frame(TREE)
    assert "composed linear solver" in lines[0]
    assert lines[0].rstrip().endswith("ComposedSolver")
    assert all("ComposedSolver" not in line for line in lines[1:])


def test_a_long_line_of_a_tree_continues_below_its_own_text():
    lines = [_messageColumn(line) for line in _frame(TREE)]
    root = next(i for i, line in enumerate(lines) if line.startswith("| LGMRES("))
    assert lines[root + 1].startswith("|   ")
    # the last child's long line continues under the tree's own vertical line: there is none below "└─"
    twoLevel = next(i for i, line in enumerate(lines) if "preconditionS: TwoLevel" in line)
    assert lines[twoLevel + 1].startswith("|      ")
    # no text is lost by the wrapping
    text = " ".join(line[1:-1].strip() for line in lines[1:-1])
    assert "coarse solve=AMG V-cycle)" in text and "restart=50)" in text


def test_the_lines_of_a_wrapped_tree_continue_the_vertical_line_where_the_tree_goes_on():
    lines = [_messageColumn(line) for line in _frame("├─ " + "word " * 30 + "\n└─ last")]
    assert lines[2].startswith("| │    ")
