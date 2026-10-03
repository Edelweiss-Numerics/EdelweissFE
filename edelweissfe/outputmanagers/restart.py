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
from edelweissfe.utils.checkpoint import writeCheckpoint
from edelweissfe.utils.fieldoutput import FieldOutputController
from edelweissfe.utils.plotter import Plotter
from edelweissfe.utils.schema import schemaField

"""
Writes restart checkpoints during the analysis (see ``*restart``), so a later run can resume from
the last converged increment written via ``*restart, readFrom=...``.

.. code-block:: console
    :caption: Example:

    *output, type=restart, name=restart
        writeInterval=10, baseName=restart, numberOfFilesToKeep=3

Put every option on ONE dataline. This manager does not aggregate datalines (its schema is not a
:class:`~edelweissfe.utils.schema.DatalineAggregatingSchema`), so the input-file helper builds one
manager per dataline -- three datalines silently produce three managers, each with its own ring
buffer, of which two would take the default ``writeInterval`` of 1 and checkpoint on every output.
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

    #: The slot written next, and the serial number of the next checkpoint. A restored writer
    #: continues from the checkpointed ones; only a writer the checkpoint does not know starts from
    #: what it finds on disk.
    checkpointedState = {"_nextIndex": int, "_nextSerial": int}

    def __init__(self, baseName: str, maxsize: int):
        super().__init__(maxlen=maxsize)
        self._baseName = baseName
        self._maxsize = maxsize
        self._nextIndex, self._nextSerial = self._continueFromDisk()

    def _fileName(self, index: int) -> str:
        return "{:}_{:}.h5".format(self._baseName, index)

    def _continueFromDisk(self) -> tuple[int, int]:
        """The slot and serial to write next, continuing whatever checkpoints already exist on
        disk: the first never-written slot if the buffer has not filled up yet, otherwise the slot
        holding the oldest checkpoint -- by the serial number each checkpoint carries, not by file
        times, which copying changes. A file without one (an older format) counts as oldest."""

        serialByIndex = {}
        for index in range(self._maxsize):
            if os.path.exists(self._fileName(index)):
                with h5py.File(self._fileName(index), "r") as f:
                    serialByIndex[index] = int(f.attrs.get("serial", -1))

        nextSerial = max(serialByIndex.values(), default=-1) + 1
        neverWritten = [index for index in range(self._maxsize) if index not in serialByIndex]
        if neverWritten:
            return min(neverWritten), nextSerial
        return min(serialByIndex, key=serialByIndex.get), nextSerial

    def nextCheckpoint(self) -> tuple[str, int]:
        """The file name and serial number for the next checkpoint to be written, rotating over
        ``[0, numberOfFilesToKeep)``.

        Returns
        -------
        tuple[str, int]
            The file name and the serial number.
        """

        fileName, serial = self._fileName(self._nextIndex), self._nextSerial
        self._nextIndex = (self._nextIndex + 1) % self._maxsize
        self._nextSerial += 1
        self.append(fileName)
        return fileName, serial


class OutputManager(OutputManagerBase):
    """Writes restart checkpoints during the analysis."""

    identification = "Restart"

    writesRestartCheckpoints = True

    #: L2 schema declared for the L3 registry, per OptionSchemaProvider.
    schema = RestartOutputManagerSchema

    #: How many increments passed since this writer last wrote a checkpoint (zero in its own
    #: checkpoints, but another restart writer with a different interval may be anywhere in its
    #: cycle), and the slot of its ring buffer it writes next.
    checkpointedState = {"_incrementsSinceLastWrite": int, "_files": _RestartFileRingBuffer}

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
        fileName, serial = self._files.nextCheckpoint()
        writeCheckpoint(fileName, self.model, self._currentStep, self.model.outputManagers, serial=serial)

        self.journal.message("Wrote restart checkpoint {:}".format(fileName), self.identification, 2)

    def finalizeFailedIncrement(self, **kwargs):
        pass

    def finalizeStep(self):
        pass

    def finalizeJob(self):
        pass
