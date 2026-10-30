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
"""The table of the Newton-Raphson iterations that the implicit solvers write to the journal.

One column per field (and one for the scalar variables, if any), each showing the largest flux residual
:math:`\\|R\\|_\\infty` and the largest correction :math:`\\|\\Delta\\Delta U\\|_\\infty` of the iteration, marked ``✓``
once the convergence criterion is met. Linear solvers that report per-solve diagnostics (see
:attr:`~edelweissfe.linsolve.base.LinearSolver.reportsSolveSummary`) get one more column with the solver's
iteration count and achieved residual.

The rows are written with :meth:`~edelweissfe.journal.journal.Journal.preformatted`, so they are never wrapped,
however many columns the model has.
"""

from edelweissfe.journal.journal import displayWidth
from edelweissfe.linsolve.base import LinearSolveSummary
from edelweissfe.solvers.base.convergencecriteria import FieldConvergence


class ConvergenceTable:
    """The column layout of the iteration table: the header lines, and one row per iteration.

    Parameters
    ----------
    columnNames
        The names of the field columns, in the order of the rows' entries.
    showLinearSolve
        Add the "linear solve" column.
    """

    #: The width of one column: two 12 characters wide cells (a value and its ``✓`` marker), and one space.
    columnWidth = 25
    #: The format of one field column in a row.
    fieldCell = "{:11.2e}{:1}{:11.2e}{:1} "
    #: The format of the two labels below a column name, each centered over its 12 characters wide cell.
    subHeaderCell = "{:^12}{:^12} "
    #: The space between the field columns and the "linear solve" column.
    linearSolveGap = "   "
    #: The marker of a linear solve that needed extra internal attempts, by the number of attempts.
    retriesSuperscript = {0: "", 1: "¹", 2: "²", 3: "³"}

    def __init__(self, columnNames: list[str], showLinearSolve: bool):
        self.columnNames = list(columnNames)
        self.showLinearSolve = showLinearSolve

    def headerLines(self) -> list[str]:
        """The two header lines: the column names, and the labels of the values below them.

        Returns
        -------
        list[str]
            The column-name line and the label line.
        """

        names = "".join("{:^{:}}".format(name, self.columnWidth) for name in self.columnNames)
        labels = self.subHeaderCell.format("||R||∞", "||ddU||∞") * len(self.columnNames)

        if self.showLinearSolve:
            names += self.linearSolveGap + "{:^{:}}".format("linear solve", self.columnWidth)
            labels += self.linearSolveGap + self.subHeaderCell.format("iters", "‖r‖")

        return [names, labels]

    def row(self, fieldColumns: list[FieldConvergence], linearSolve: LinearSolveSummary | None) -> str:
        """The row of one iteration.

        Parameters
        ----------
        fieldColumns
            The convergence state of each column, in the order of the column names.
        linearSolve
            The diagnostics of the linear solve that produced this iteration's correction, or None if there
            was none (the first iteration of an increment); the "linear solve" column is then left blank.

        Returns
        -------
        str
            The row.
        """

        row = "".join(
            self.fieldCell.format(
                column.fluxResidual,
                "✓" if column.fluxResidualConverged else " ",
                column.correction,
                "✓" if column.correctionConverged else " ",
            )
            for column in fieldColumns
        )

        if self.showLinearSolve:
            if linearSolve is None:
                row += self.linearSolveGap + " " * self.columnWidth
            else:
                iterations = "{:}{:}".format(linearSolve.iters, self.retriesSuperscript.get(linearSolve.retries, ""))
                row += self.linearSolveGap + "{:^12}{:11.2e}{:1} ".format(
                    iterations, linearSolve.residual, "✓" if linearSolve.residualMet else " "
                )

        return row

    def linearSolveDetailLine(self, text: str) -> str:
        """A line of linear solver detail, right-aligned inside the "linear solve" column, ending where the rows end.

        Parameters
        ----------
        text
            The detail, at most one column less one character wide to stay inside the column.

        Returns
        -------
        str
            The line, blank in front of the "linear solve" column.
        """

        fieldColumnsWidth = self.columnWidth * len(self.columnNames)
        padding = " " * (self.columnWidth - 1 - displayWidth(text))
        return " " * fieldColumnsWidth + self.linearSolveGap + padding + text + " "
