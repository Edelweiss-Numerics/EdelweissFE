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
"""Restart checkpoints: the one place that knows how a checkpoint file is laid out.

A checkpoint is an HDF5 file holding

- the attribute ``stepNumber``, the step the run was in;
- the model's state, written by :meth:`~edelweissfe.models.femodel.FEModel.writeRestart`;
- the time stepper's and the solver's state, written by their own ``writeRestart``;
- a group ``outputManagers`` with the sequence bookkeeping of every output manager that has some
  (see :meth:`~edelweissfe.outputmanagers.base.outputmanagerbase.OutputManagerBase.getRestartData`).

:func:`writeCheckpoint` writes one, the restart output manager calls it; :class:`ResumeCheckpoint`
reads one back, the driver uses it.
"""

import h5py

from edelweissfe.utils.exceptions import RestartError

#: The layout of a checkpoint. Raise it whenever a checkpoint gains or changes state: a run resumes
#: only from checkpoints of its own layout, so a missing piece of state is refused up front instead
#: of surfacing as a lookup error deep inside some reader -- or as silently missing state.
CHECKPOINT_FORMAT_VERSION = 3


def writeRestartDataOf(group: h5py.Group, entities: dict):
    """Store the restart data of each entity in its own subgroup, named after it.

    For entities that hand over their restart state as a dict of arrays (constraints, output
    managers) rather than writing it themselves. An entity whose ``getRestartData`` returns None
    has nothing to store and gets no subgroup.

    Parameters
    ----------
    group
        The group to create the subgroups in.
    entities
        The entities, by name.
    """

    for name, entity in entities.items():
        restartData = entity.getRestartData()
        if restartData is None:
            continue
        entityGroup = group.create_group(name)
        for entryName, entryValues in restartData.items():
            entityGroup.create_dataset(entryName, data=entryValues)


def readRestartDataInto(group: h5py.Group, entities: dict):
    """Hand each entity the restart data :func:`writeRestartDataOf` stored for it, if any.

    Parameters
    ----------
    group
        The group holding the subgroups.
    entities
        The entities, by name.
    """

    for name, entity in entities.items():
        if name not in group:
            continue
        entity.setRestartData({entryName: values[:] for entryName, values in group[name].items()})


def writeCheckpoint(fileName: str, model, step, outputManagers: dict):
    """Write a restart checkpoint of the converged state.

    Parameters
    ----------
    fileName
        The file to write; overwritten if it exists.
    model
        The model tree.
    step
        The current step, whose time stepper and solver write their own state.
    outputManagers
        The output managers, by name.
    """

    with h5py.File(fileName, "w") as f:
        f.attrs["formatVersion"] = CHECKPOINT_FORMAT_VERSION
        f.attrs["stepNumber"] = step.number
        model.writeRestart(f)
        step.timeStepper.writeRestart(f)
        step.solver.writeRestart(f)
        writeRestartDataOf(f.create_group("outputManagers"), outputManagers)


class ResumeCheckpoint:
    """A restart checkpoint opened for resuming a run, restored piece by piece as the run is set up.

    The pieces are restored at different moments, because the objects they belong to are created at
    different moments: the model first (:meth:`restoreModel`), the output managers once they exist
    (:meth:`restoreOutputManagers`), and the time stepper and solver when the resumed step begins
    (:meth:`restoreStep`). Close the checkpoint right after that, see :meth:`close`.

    Parameters
    ----------
    fileName
        The checkpoint to resume from.
    """

    def __init__(self, fileName: str):
        self.fileName = fileName
        self._file = h5py.File(fileName, "r")
        formatVersion = int(self._file.attrs.get("formatVersion", 1))
        if formatVersion != CHECKPOINT_FORMAT_VERSION:
            self._file.close()
            raise RestartError(
                "checkpoint {:} has format version {:}, this version of EdelweissFE reads version {:} "
                "only".format(fileName, formatVersion, CHECKPOINT_FORMAT_VERSION)
            )
        #: The step the checkpointed run was in; the steps before it are not solved again.
        self.stepNumber = int(self._file.attrs["stepNumber"])

    def restoreModel(self, model, journal):
        """Restore the model's state, see :meth:`~edelweissfe.models.femodel.FEModel.readRestart`.

        Parameters
        ----------
        model
            The model tree, rebuilt from the same input file.
        journal
            The journal, for the progress messages of the topology replay.
        """

        model.readRestart(self._file, journal)

    def restoreOutputManagers(self, outputManagers: dict):
        """Restore the sequence bookkeeping of the output managers that stored some.

        Ensight is the motivating case: its transient file numbering is derived from the time
        values it has already written, and would restart from zero, orphaning the output written
        before the checkpoint.
        """

        readRestartDataInto(self._file["outputManagers"], outputManagers)

    def restoreStep(self, step):
        """Continue ``step`` from the checkpoint, when the resumed step begins: its time stepper's
        progress and its solver's state between increments; see
        :meth:`~edelweissfe.steps.base.stepbase.StepBase.readRestart`.
        """

        step.readRestart(self._file)

    def close(self):
        """Close the file. Nothing reads the checkpoint after :meth:`restoreStep`, and it must not
        stay open: the restart output manager's ring buffer overwrites the oldest of its files,
        which can be this very one, and HDF5 cannot truncate a file it still holds open. The
        resumed run would then fail mid-step with "unable to truncate a file which is already
        open", which reads like a solver failure and is not one."""

        if self._file is not None:
            self._file.close()
            self._file = None
