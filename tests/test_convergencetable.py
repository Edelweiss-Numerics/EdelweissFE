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
"""The rows of the iteration table, and the lines of linear solver detail below them, end in the same column: the sender
of a line never touches its last character.
"""

from edelweissfe.linsolve.base import LinearSolveSummary
from edelweissfe.solvers.base.convergencecriteria import FieldConvergence
from edelweissfe.solvers.base.convergencetable import ConvergenceTable


def test_a_detail_line_ends_where_a_row_ends_and_is_separated_from_the_sender():
    table = ConvergenceTable(["displacement", "rotation", "nonlocal damage"], showLinearSolve=True)
    columns = [FieldConvergence(1.0, True, 1.0, True, 0)] * 3
    summary = LinearSolveSummary(iters=31, residual=1e-3, residualMet=True, retries=0)
    row = table.row(columns, summary)
    detail = table.linearSolveDetailLine("rebuilt(first)")
    assert len(detail) == len(row)
    assert detail.endswith("rebuilt(first) ")
