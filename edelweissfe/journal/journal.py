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
# Created on Sun Jan 15 11:30:59 2017

# @author: Matthias Neuner
"""
This module provides journaling capabilities,
which can be used troughout EdelweissFE.
"""

import math
import textwrap
import unicodedata

from prettytable import PrettyTable


def displayWidth(text: str) -> int:
    """The columns a text takes on the terminal: a combining character (the tilde of ``S̃``) takes none."""
    return sum(1 for character in text if not unicodedata.combining(character))


class Journal:
    """This class provides an interface to present messages to the user via console
    output and/or file output.
    Information messages can be sorted by the importance level.
    Suppressing certain levels of output is possible.

    Parameters
    ----------
    verbose
        Print to the terminal.
    outputFile
        Write to a file.
    suppressFromLevel
        Suppress certain levels.
    """

    # Class-level width settings to ensure all instances synchronize formatting
    linewidth = 120
    leftColumn = 100
    leftColumnMaxSize = 120
    outputWidths = {0: 99, 1: 96, 2: 95}
    _rightColumn = 20
    errorMessageTemplate = " > > > {:<88}{:>18} < < < "
    leveledOutput = {
        0: " {:<100}{:>18} ",
        1: "   {:<98}{:>18} ",
        2: "     {:<96}{:>18} ",
    }

    def __init__(
        self,
        verbose: bool = True,
        outputFile: str = None,
        suppressFromLevel: int = 3,
    ):
        self._suppressLvl = suppressFromLevel
        self._verbose = verbose
        self._fileHandle = None

    def setFileOutput(self, fileHandle=None):
        if fileHandle:
            self._fileHandle = fileHandle
        else:
            self._fileHandle = None

    @classmethod
    def setNewLineWidth(cls, newWidth: int = 100, leftColumn: int = 80):
        """Set the line width of the log file.

        Parameters
        ----------
        newWidth
            New total width.
        leftColumn
            Width of the main column.
        """

        cls.linewidth = newWidth
        cls.leftColumn = leftColumn
        cls.leftColumnMaxSize = 120
        cls.outputWidths = {}
        cls.outputWidths[0] = leftColumn - 1
        cls.outputWidths[1] = leftColumn - 4
        cls.outputWidths[2] = leftColumn - 5
        cls._rightColumn = cls.linewidth - leftColumn

        cls.errorMessageTemplate = " > > > {{:<{:}}}{{:>{:}}} < < < ".format(leftColumn - 12, cls._rightColumn - 2)
        cls.leveledOutput = {
            0: " {{:<{:}}}{{:>{:}}} ".format(leftColumn, cls._rightColumn - 2),
            1: "   {{:<{:}}}{{:>{:}}} ".format(leftColumn - 2, cls._rightColumn - 2),
            2: "     {{:<{:}}}{{:>{:}}} ".format(leftColumn - 4, cls._rightColumn - 2),
        }

    def message(self, message: str, senderIdentification: str, level: int = 1):
        """Write message to log. Lines longer than the message column are wrapped.

        Parameters
        ----------
        message
            The message.
        senderIdentification
            The name of the sender.
        level
            Level of message.
        """

        wrap_width = Journal._messageWidth(level)

        wrapped_lines = []
        for line in message.splitlines():
            if len(line) > wrap_width:
                wrapped_lines.extend(textwrap.wrap(line, width=wrap_width))
            else:
                wrapped_lines.append(line)

        self._writeLines(wrapped_lines, senderIdentification, level, senderOnEveryLine=False)

    @staticmethod
    def _messageWidth(level: int) -> int:
        """The characters a line of the given level may have before it is wrapped: the message column, less the
        indentation of the level (see :attr:`leveledOutput`)."""
        return Journal.leftColumn - {0: 0, 1: 2, 2: 4}.get(level, 0)

    def framedMessage(self, message: str, senderIdentification: str, title: str, level: int = 1):
        """Write a message inside a frame, a block that stands apart from the lines around it. The frame is as wide as
        the message column. A line longer than the frame is wrapped below itself, its continuation indented under its
        own text, and the ``│`` of a tree drawn with ``├─`` and ``└─`` carried on, so that such a tree stays readable.
        The sender is on the first line only.

        Parameters
        ----------
        message
            The message; a line break starts a new line of the frame.
        senderIdentification
            The name of the sender.
        title
            The title in the upper border of the frame.
        level
            Level of message.
        """

        frameWidth = Journal._messageWidth(level)
        innerWidth = frameWidth - 4

        lines = []
        for line in message.splitlines():
            lines.extend(Journal._wrapHanging(line, innerWidth))

        upperBorder = "+- {:} ".format(title).ljust(frameWidth - 1, "-") + "+"
        lowerBorder = "+" + "-" * (frameWidth - 2) + "+"
        framedLines = [upperBorder]
        for line in lines:
            padding = " " * (innerWidth - displayWidth(line))
            framedLines.append("| " + line + padding + " |")
        framedLines.append(lowerBorder)

        self._writeLines(framedLines, senderIdentification, level, senderOnEveryLine=False)

    @staticmethod
    def _wrapHanging(line: str, width: int) -> list[str]:
        """The line wrapped to the width, its continuation lines indented by two more characters than the text of the
        line starts at. What leads a line of a tree (spaces and ``│├└─``) continues as spaces, and as ``│`` where the
        tree goes on below."""
        if displayWidth(line) <= width:
            return [line]
        leading = line[: len(line) - len(line.lstrip(" │├└─"))]
        continuation = "".join("│" if character in "│├" else " " for character in leading) + "  "
        return textwrap.wrap(
            line, width=width, subsequent_indent=continuation, break_long_words=False, break_on_hyphens=False
        ) or [""]

    def preformatted(self, lines: list[str], senderIdentification: str, level: int = 1):
        """Write lines whose layout must be kept, such as the rows of a table, to log. They are never wrapped:
        a line longer than the message column pushes the sender to the right instead. Each line carries the
        sender.

        Parameters
        ----------
        lines
            The lines.
        senderIdentification
            The name of the sender.
        level
            Level of message.
        """

        self._writeLines(lines, senderIdentification, level, senderOnEveryLine=True)

    def _writeLines(self, lines: list[str], senderIdentification: str, level: int, senderOnEveryLine: bool):
        """Write lines to the terminal and the log file, with the sender in the right column.

        Parameters
        ----------
        lines
            The lines, already fitted to the message column (or deliberately not).
        senderIdentification
            The name of the sender.
        level
            Level of message.
        senderOnEveryLine
            Put the sender on every line, rather than on the first one only.
        """

        # Abbreviate senderIdentification if it is too long
        max_sender_len = Journal._rightColumn - 2
        if len(senderIdentification) > max_sender_len:
            senderIdentification = senderIdentification[: max_sender_len - 3] + "..."

        for i, line in enumerate(lines):
            sender = senderIdentification if i == 0 or senderOnEveryLine else ""
            theLine = Journal.leveledOutput[level].format(line, sender)
            if level < self._suppressLvl:
                if self._verbose:
                    print(theLine)

            if self._fileHandle:
                self._fileHandle.write(theLine + "\n")

    def errorMessage(self, errorMessage: str, senderIdentification: str):
        """Print an error message.

        Parameters
        ----------
        errorMessage
            The message.
        senderIdentification
            The name of the sender.
        """

        theLine = self.errorMessageTemplate.format(errorMessage, senderIdentification)
        print(theLine)

        if self._fileHandle:
            self._fileHandle.write(theLine + "\n")

    def printSeperationLine(
        self,
    ):
        """Write a seperation file to log."""

        theLine = "+" + "-" * (self.linewidth - 2) + "+"
        if self._verbose:
            print(theLine)

        if self._fileHandle:
            self._fileHandle.write(theLine + "\n")

    def printTable(
        self,
        table: list[list[str]],
        senderIdentification: str,
        level: int = 1,
        printHeaderRow: bool = True,
    ):
        """Print a pretty table.

        Parameters
        ----------
        table
            The table as nested list.
        senderIdentification
            The name of the sender.
        level
            The level of the mesage.
        printHeaderRow
            Special formatting for first row.
        """

        nCols = len(table[0])

        cellWidth = int(math.floor(self.outputWidths[level] / nCols)) - 3
        rowBar = "+" + (("-" * cellWidth) + "+") * nCols
        rowString = ("|{:}".format("{:" + str(cellWidth) + "}")) * nCols + "|"

        if printHeaderRow:
            self.message(rowBar, senderIdentification, level)
        for row in table:
            self.message(rowString.format(*row), senderIdentification, level)

        self.message(rowBar, senderIdentification, level)

    def printPrettyTable(self, prettyTable: PrettyTable, senderIdentification: str):
        prettyTable.min_table_width = self.leftColumn
        prettyTable.max_table_width = self.leftColumn
        self.message(str(prettyTable), senderIdentification, level=0)

    def setVerbose(
        self,
    ):
        """Set highest verbosity."""

        self._suppressLvl = 3

    def squelch(self, level):
        """Suppress all messages.

        Parameters
        ----------
        level
            The priority level of the message.
        """

        self._suppressLvl = level


# Initialize the class-level formats exactly once
Journal.setNewLineWidth(newWidth=100, leftColumn=80)
