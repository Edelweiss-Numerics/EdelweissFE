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

import os
from collections import deque
from dataclasses import dataclass

import h5py

from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.outputmanagers.base.outputmanagerbase import OutputManagerBase
from edelweissfe.utils.fieldoutput import FieldOutputController
from edelweissfe.utils.plotter import Plotter
from edelweissfe.utils.schema import schemaField

"""
Writes restart checkpoints during the analysis (see ``*restart``), so a later run can resume from
the last converged increment written via ``*restart, readFrom=...``.

.. code-block:: console
    :caption: Example:

    *output, type=restart, name=restart
        writeInterval=10
        baseName=restart
        numberOfFilesToKeep=3
"""


@dataclass(frozen=True)
class RestartOutputManagerSchema:
    """L2: the options this output manager accepts, owned by this module and never mutated from
    outside it -- independent of the ``*restart`` keyword's own schema
    (``edelweissfe.keywords.restart.RestartSchema``), which only configures *resuming*
    (``readFrom``); writing is configured entirely here.
    """

    writeInterval: int = schemaField(
        description="write a checkpoint every N converged increments", dtype=int, default=1
    )
    baseName: str = schemaField(description="base file name for restart checkpoints", dtype=str, default="restart")
    numberOfFilesToKeep: int = schemaField(
        description="number of most recent restart checkpoints to keep (ring buffer)", dtype=int, default=3
    )


class _RestartFileRingBuffer(deque):
    """Rotates through ``numberOfFilesToKeep`` checkpoint file names, ported from
    EdelweissMeshfree's ``RestartHistoryManager``
    (``edelweissmeshfree/solvers/base/nonlinearsolverbase.py``).

    Resumes an existing ring buffer found on disk rather than always starting at index 0 --
    otherwise a resumed run that keeps writing checkpoints (the common case for a
    walltime-limited job chained across several resumes) would silently overwrite earlier
    checkpoints, including, in the common case of an unchanged ``baseName``, the very one it
    just resumed from.
    """

    def __init__(self, baseName: str, maxsize: int):
        super().__init__(maxlen=maxsize)
        self._baseName = baseName
        self._maxsize = maxsize
        self._nextIndex = self._resumeNextIndex()

    def _resumeNextIndex(self) -> int:
        """The index to write next, continuing whatever ring-buffer state already exists on disk:
        the first never-written slot if the buffer hasn't filled up yet, otherwise the slot
        holding the oldest checkpoint (by mtime), i.e. the one due to be overwritten next."""

        existingMTimeByIndex = {}
        for index in range(self._maxsize):
            fileName = "{:}_{:}.h5".format(self._baseName, index)
            if os.path.exists(fileName):
                existingMTimeByIndex[index] = os.path.getmtime(fileName)

        neverWritten = [index for index in range(self._maxsize) if index not in existingMTimeByIndex]
        if neverWritten:
            return min(neverWritten)
        return min(existingMTimeByIndex, key=existingMTimeByIndex.get)

    def nextFileName(self) -> str:
        """The file name for the next checkpoint to be written, rotating over
        ``[0, numberOfFilesToKeep)``."""

        fileName = "{:}_{:}.h5".format(self._baseName, self._nextIndex)
        self._nextIndex = (self._nextIndex + 1) % self._maxsize
        self.append(fileName)
        return fileName


class OutputManager(OutputManagerBase):
    """Writes restart checkpoints during the analysis."""

    identification = "Restart"

    #: L2 schema declared for the L3 registry, per OptionSchemaProvider.
    schema = RestartOutputManagerSchema

    def __init__(
        self,
        name: str,
        model: FEModel,
        fieldOutputController: FieldOutputController,
        journal: Journal,
        plotter: Plotter,
        *,
        configuration: RestartOutputManagerSchema = RestartOutputManagerSchema(),
    ):
        """L1: constructible standalone, with no parser involvement and no ``moduleOptions``.

        Parameters
        ----------
        name
            The name of this output manager.
        model
            The model tree.
        fieldOutputController
            The field output controller instance.
        journal
            The journal instance for logging.
        plotter
            The plotter instance.
        configuration
            The options this output manager accepts; defaults to all-defaults.
        """
        if configuration.writeInterval < 1:
            raise ValueError("writeInterval must be >= 1: a non-positive interval never converges to a write.")
        if configuration.numberOfFilesToKeep < 1:
            raise ValueError(
                "numberOfFilesToKeep must be >= 1: a non-positive ring buffer size cannot hold a checkpoint."
            )
        self.name = name
        self.model = model
        self.journal = journal
        self.writeInterval = configuration.writeInterval
        self._files = _RestartFileRingBuffer(configuration.baseName, configuration.numberOfFilesToKeep)
        self._incrementsSinceLastWrite = 0
        self._currentStep = None

    def initializeJob(self):
        pass

    def initializeStep(self, step):
        self._currentStep = step

    def finalizeIncrement(self, **kwargs):
        self._incrementsSinceLastWrite += 1
        if self._incrementsSinceLastWrite < self.writeInterval:
            return

        self._incrementsSinceLastWrite = 0
        fileName = self._files.nextFileName()
        with h5py.File(fileName, "w") as f:
            f.attrs["stepNumber"] = self._currentStep.number
            self.model.writeRestart(f)
            self._currentStep.timeStepper.writeRestart(f)

            # Sibling output managers' own sequence bookkeeping (e.g. Ensight's transient time/file
            # sets, whose file numbering a resumed run would otherwise restart from zero, orphaning
            # the pre-resume portion of the sequence). model.outputManagers is already populated by
            # the time any increment converges -- it's set right after construction, well before
            # stepping begins -- so it's safe to read here, unlike at readRestart time (see the
            # driver's own restore pass for why that side needs separate handling).
            outputManagersGroup = f.create_group("outputManagers")
            for name, manager in self.model.outputManagers.items():
                restartData = manager.getRestartData()
                if restartData is None:
                    continue
                managerGroup = outputManagersGroup.create_group(name)
                for entryName, entryValues in restartData.items():
                    managerGroup.create_dataset(entryName, data=entryValues)

        self.journal.message("Wrote restart checkpoint {:}".format(fileName), self.identification, 2)

    def finalizeFailedIncrement(self, **kwargs):
        pass

    def finalizeStep(self):
        pass

    def finalizeJob(self):
        pass
