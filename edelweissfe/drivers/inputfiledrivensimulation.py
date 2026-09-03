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
# Created on Tue Jan  17 19:10:42 2017

# @author: Matthias Neuner
"""This is the main module of EdelweissFE.

Heart is the ``*job`` keyword, which defines the spatial dimension
A ``*job`` definition consists of multiple ``*steps``, associated with that job.
"""

from time import time as getCurrentTime

import h5py

from edelweissfe.config.configurator import loadConfiguration, updateConfiguration
from edelweissfe.config.phenomena import domainMapping
from edelweissfe.config.solvers import getSolverByName
from edelweissfe.helpers.inputfilehelpers import (
    createFieldOutputFromInputFile,
    createOutputManagersFromInputFile,
    createPlotterFromInputFile,
    createSolversFromInputFile,
    createStepManagerFromInputFile,
    fillFEModelFromInputFile,
)
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel, printPrettyModelSummary
from edelweissfe.utils.exceptions import StepFailed
from edelweissfe.utils.fieldoutput import FieldOutputController


def finiteElementSimulation(
    inputfile: dict, verbose: bool = False, suppressPlots: bool = False
) -> tuple[FEModel, FieldOutputController]:
    """This is core function of the finite element analysis.
    Based on the keyword ``*job``, the finite element model is defined.

    It assembles
     * the information on the job
     * the model tree
     * steps
     * field outputs
     * output managers

    and controls the respective solver based on the defined simulation steps.
    For each step, the step-actions (dirichlet, nodeforces) are collected by
    external modules.

    Parameters
    ----------
    inputfile
        The input file in dictionary form.
    verbose
        Be verbose during the simulation.
    suppressPlots
        Suppress plots at the end of simulation for batch runs.

    Returns
    -------
    tuple
        A tuple containing
            - The final model tree
            - The fieldoutput controller containing all processed results.
    """

    identification = "feCore"

    journal = Journal(verbose=verbose)

    job = inputfile["job"][0]
    jobName = job["name"]

    domainSize = domainMapping[job["domain"]]

    journal.printSeperationLine()

    journal.message(
        "Setting up finite element model",
        identification,
        0,
    )

    jobInfo = dict()

    tic = getCurrentTime()
    model = FEModel(domainSize)
    model = fillFEModelFromInputFile(model, inputfile, journal)
    model.prepareYourself(journal)
    model.advanceToTime(job.get("startTime", 0.0))
    toc = getCurrentTime()
    jobInfo["model setup time"] = toc - tic

    journal.printTable(
        [
            ("Model setup time ", "{:10.4f}s".format(jobInfo["model setup time"])),
        ],
        identification,
        level=0,
    )

    printPrettyModelSummary(model, journal)
    journal.printSeperationLine()

    jobInfo["computationTime"] = 0.0

    jobInfo.update(job)
    jobInfo = loadConfiguration(jobInfo)
    for updateConfig in inputfile["updateConfiguration"]:
        updateConfiguration(updateConfig, jobInfo, journal)

    # Create the default entries 'U' (flux) and 'P' (effort)
    for nodeField in model.nodeFields.values():
        nodeField.createFieldValueEntry("U")
        nodeField.createFieldValueEntry("P")

    model._linkFieldVariableObjects(model.nodeSets["all"])

    # *restart, readFrom=...: resume from a checkpoint. Reconstruct-then-overwrite, not full
    # serialization -- the model above was already rebuilt from this same .inp file, and
    # readRestart only overwrites its converged state (node fields, element history, scalar
    # variables, stateful constraints' history) plus model.time, so it must run after node fields
    # exist (createFieldValueEntry above) and after advanceToTime's cold-start bookkeeping, which
    # would otherwise clobber model.time back to job['startTime'].
    #
    # Limitation: resuming skips *solving* every step before the checkpoint's step entirely --
    # correct for the common case, but a `modelupdate` step action (or any other topology mutation)
    # in a skipped step's `solve()` never runs, so restart is only supported for analyses whose
    # topology is static across the resumed run.
    restartDefinitions = inputfile["restart"]
    resumeCheckpoint = None
    resumeStepNumber = None
    if restartDefinitions and restartDefinitions[0].get("readFrom"):
        checkpointPath = restartDefinitions[0]["readFrom"]
        resumeCheckpoint = h5py.File(checkpointPath, "r")
        resumeStepNumber = int(resumeCheckpoint.attrs["stepNumber"])
        model.readRestart(resumeCheckpoint)
        journal.message(
            "Resuming from restart checkpoint {:} (step {:}, time {:})".format(
                checkpointPath, resumeStepNumber, model.time
            ),
            identification,
            0,
        )

    plotter = createPlotterFromInputFile(inputfile, journal)
    stepManager = createStepManagerFromInputFile(inputfile)
    fieldOutputController = createFieldOutputFromInputFile(inputfile, model, journal)
    model.fieldOutputController = fieldOutputController
    fieldOutputController.initializeJob()

    outputManagers = createOutputManagersFromInputFile(
        inputfile, jobName, model, fieldOutputController, journal, plotter
    )
    for outputManager in outputManagers:
        outputManager.initializeJob()

    solvers = createSolversFromInputFile(inputfile, jobInfo, journal)

    if not solvers:
        from warnings import warn

        warn(
            "Warning, not defining a Solver is deprecated; Define solver using *solver keyword",
            DeprecationWarning,
            stacklevel=2,
        )

    defaultSolver = getSolverByName(job["solver"])
    solvers["default"] = defaultSolver(jobInfo, journal)

    # Looked up by name from a >>options block (edelweissfe.stepactions.options), which resolves
    # directly against these two rather than scanning step actions for a category tag.
    model.solvers = solvers
    model.outputManagers = {outputManager.name: outputManager for outputManager in outputManagers}

    # Output managers don't exist yet at the earlier model.readRestart(resumeCheckpoint) call
    # above (they're constructed here, well after) -- restore whichever of them wrote restart data
    # (see outputmanagers/restart.py's finalizeIncrement) now that they do, and while the
    # checkpoint is still open. Ensight is the motivating case: without this, its transient
    # sequence numbering (derived from its own history of already-written time values) would
    # restart from zero, orphaning the pre-resume portion of the sequence.
    if resumeCheckpoint is not None and "outputManagers" in resumeCheckpoint:
        for name, outputManager in model.outputManagers.items():
            if name not in resumeCheckpoint["outputManagers"]:
                continue
            restartData = {
                entryName: values[:] for entryName, values in resumeCheckpoint["outputManagers"][name].items()
            }
            outputManager.setRestartData(restartData)

    try:
        for step in stepManager.generateSteps(jobInfo, model, fieldOutputController, journal, solvers, outputManagers):
            if resumeStepNumber is not None:
                if step.number < resumeStepNumber:
                    # Constructed (so its StepActions register/accumulate normally, see the comment
                    # above) but not solved -- it already ran, in full, before the interrupted job
                    # wrote this checkpoint.
                    continue
                if step.number == resumeStepNumber:
                    step.timeStepper.readRestart(resumeCheckpoint)
                    resumeStepNumber = None

            tic = getCurrentTime()
            try:
                step.solve()
            finally:
                # Record inside finally so a step that raises (e.g. via a deliberate
                # maxNumInc cap) still counts its elapsed time -- previously this sat after
                # step.solve() outside any try and was skipped whenever a step failed.
                toc = getCurrentTime()
                stepTime = toc - tic
                jobInfo["computationTime"] += stepTime

                journal.printTable(
                    [
                        ("Step computation time", "{:10.4f}s".format(stepTime)),
                    ],
                    identification,
                    level=0,
                )

        if resumeStepNumber is not None:
            journal.errorMessage(
                "Restart checkpoint's step {:} was never reached -- nothing was resumed".format(resumeStepNumber),
                identification,
            )

    except KeyboardInterrupt:
        print("")
        journal.errorMessage("Interrupted by user", identification)

    except StepFailed as e:
        print("")
        # Report WHY. A StepFailed carries the reason it was raised with -- which solver, which
        # increment, which material -- and discarding it left the log with no explanation at all,
        # so a run that stopped for a diagnosable reason looked indistinguishable from one that
        # stopped for an unknown one.
        message = str(e)
        journal.errorMessage(
            "Simulation failed: {:}".format(message) if message else "Simulation failed", identification
        )

    except Exception as e:
        print("")
        journal.errorMessage("Simulation failed due to unhandled exception", identification)
        raise e

    finally:
        journal.printTable(
            [
                (
                    "Job computation time",
                    "{:10.4f}s".format(jobInfo["computationTime"]),
                ),
            ],
            identification,
            level=0,
            printHeaderRow=False,
        )

        fieldOutputController.finalizeJob()
        for manager in outputManagers:
            manager.finalizeJob()

        plotter.finalize()
        if not suppressPlots:
            plotter.show()

        if resumeCheckpoint is not None:
            resumeCheckpoint.close()

    return model, fieldOutputController
