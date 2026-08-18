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
#  Alexander Dummer alexander.dummer@uibk.ac.at
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


from edelweissfe.outputmanagers.base.outputmanagerbase import OutputManagerBase
from edelweissfe.utils.caseinsensitivedict import CaseInsensitiveDict
from edelweissfe.utils.inputlanguage import InputLanguage, Module
from edelweissfe.utils.misc import (
    caseInsensitiveKwargsChecker,
    castKwargsValuesAndAddDefaults,
)
from edelweissfe.utils.performancetiming import (
    extractIncrementTimeRows,
    makeIncrementTimesPrettyTable,
)

"""
Prints the compute times per increment to the screen and writes them into a file (optional).

.. code-block:: console
    :caption: Example:

    *output, type=computetimemonitor, name=mycomputetimes
        export=myComputeTimes

The exported file holds one whitespace-separated row per increment *and* timed category, so that
the dynamic, hierarchical set of categories the timing tree discovers at run time needs no fixed
columns and the file stays loadable as a plain table:

.. code-block:: console

    #
    # simulation step 1
    #
    #  increment  simulation time  inc compute time  level  function       time         calls
       1          1.00000e-01      2.31038e+00       0      elements       1.10214e+00  32
       1          1.00000e-01      2.31038e+00       0      linear solve   8.02311e-01  16
       1          1.00000e-01      2.31038e+00       1      pardiso ...    7.71204e-01  16

``inc compute time`` is the sum of the increment's level-0 categories, repeated on every row of
that increment so each row stands on its own.
"""

module = Module(
    "computetimemonitor", "A simple monitor to observe results (fieldOutputs) in the console during analysis."
)

inputLanguage = InputLanguage()

keyword = "output"
if keyword in inputLanguage:
    inputLanguage[keyword].addModule(module)

module.addOptionalArg("export", "Provide a filename to export the results.", str, None)

documentation = [module]

required = [kw.name for kw in module.requiredArgs]
required += [kw.name for kw in module.requiredKeywords]

optional = [kw.name for kw in module.optionalArgs]
optional += [kw.name for kw in module.optionalKeywords]


@caseInsensitiveKwargsChecker(required, optional)
@castKwargsValuesAndAddDefaults(module)
def outputManagerFactory(name, FEModel, fieldOutputController, moduleOptions, journal, plotter, **kwargs):
    kwargs = CaseInsensitiveDict(kwargs)

    filename = kwargs["export"]

    return OutputManager(name, FEModel, fieldOutputController, journal, plotter, filename)


class OutputManager(OutputManagerBase):
    identification = "ComputeTimeMonitor"

    def __init__(self, name, model, fieldOutputController, journal, plotter, filename):
        self.journal = journal
        self.stepcounter = 0

        self.exportFile = filename

        if self.exportFile:
            with open(self.exportFile, "w+") as f:
                f.write("# \n# EdelweissFE: computing times per increment\n#\n")

    def updateDefinition(self, **kwargs: dict):
        pass

    def initializeJob(self):
        pass

    def initializeStep(self, step):
        self.stepcounter += 1

        if self.exportFile:
            self._writeStepHeaderToFile()

    def finalizeIncrement(self, statusInfoDict: dict = {}, **kwargs):
        self._reportIncrementTimes(statusInfoDict)

    def finalizeFailedIncrement(self, statusInfoDict: dict = {}, **kwargs):
        self._reportIncrementTimes(statusInfoDict)

    def _reportIncrementTimes(self, statusInfoDict: dict):
        """Print this increment's times, and append them to the export file if one was requested.

        Parameters
        ----------
        statusInfoDict
            The solver's status information, supplying the increment number and simulation time the
            times belong to. May be empty or ``None`` -- not every solver provides one (the explicit
            dynamic solver passes ``None``), and the times are still worth reporting without it, so
            those two columns are then written as ``-`` rather than refusing to write anything.
        """

        # Extracted once, not once per consumer: extractIncrementTimeRows advances the snapshot the
        # next delta is measured against, so a second call within the same increment would report
        # zeros -- which is how writing to the file went missing-in-effect rather than just missing.
        rows = extractIncrementTimeRows()

        self.journal.printPrettyTable(makeIncrementTimesPrettyTable(rows), self.identification)

        if self.exportFile:
            self._writeIncrementTimesToFile(rows, statusInfoDict)

    def _writeStepHeaderToFile(self):
        """Append the column header of a new step's block to the export file."""

        with open(self.exportFile, "a") as f:
            f.write("#\n# simulation step {:}\n#\n".format(self.stepcounter))
            f.write(
                "# {:<11} {:<20} {:<20} {:<6} {:<50} {:<20} {:<10}\n".format(
                    "increment", "simulation time", "inc compute time", "level", "function", "time", "calls"
                )
            )

    def _writeIncrementTimesToFile(self, rows: list[tuple[int, str, float, int]], statusInfoDict: dict):
        """Append one row per timed category of the current increment to the export file.

        Parameters
        ----------
        rows
            The ``(level, function, time, calls)`` rows of this increment.
        statusInfoDict
            The solver's status information; see :meth:`_reportIncrementTimes`.
        """

        d = statusInfoDict if statusInfoDict else {}
        increment = d.get("inc")
        simulationTime = d.get("time end")

        incrementColumn = "{:<11}".format(int(increment) if increment is not None else "-")
        simulationTimeColumn = (
            "{:<20.5e}".format(simulationTime) if simulationTime is not None else "{:<20}".format("-")
        )
        # the sum over the top level of the tree: the nested levels are already part of their parents
        incrementComputeTime = sum(time for level, _, time, _ in rows if level == 0)

        with open(self.exportFile, "a") as f:
            for level, function, time, calls in rows:
                f.write(
                    "  {:} {:} {:<20.5e} {:<6} {:<50} {:<20.5e} {:<10}\n".format(
                        incrementColumn,
                        simulationTimeColumn,
                        incrementComputeTime,
                        level,
                        function,
                        time,
                        calls,
                    )
                )

    def finalizeStep(
        self,
    ):
        pass

    def finalizeJob(
        self,
    ):
        pass
