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

from edelweissfe.config.configurator import loadConfiguration, updateConfiguration
from edelweissfe.config.phenomena import carriesLinearMomentum, domainMapping
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
from edelweissfe.utils.checkpoint import ResumeCheckpoint
from edelweissfe.utils.exceptions import RestartError, StepFailed
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

    # Create the default entries 'U' (flux) and 'P' (effort) on every node field, and the kinematic
    # entries 'V' (velocity) and 'A' (acceleration) on the ones a dynamic solver integrates in time
    # -- those whose inertia is a mass. They stay zero until such a solver publishes into them.
    #
    # Created here, before the field outputs take their sample at job initialisation, so that a
    # ``result=V``/``result=A`` output is valid from the start (a per-node field output reads
    # ``nodeField[result]`` directly and would otherwise raise) rather than only from the first
    # increment a dynamic solver has finished. Restricted to the mass-carrying fields because a
    # velocity of a non-local damage or a temperature field is not a kinematic quantity any solver
    # here publishes. On those fields they are created for every run, a quasi-static one included,
    # where they stay zero -- and, like every other entry, travel into its restart checkpoints and
    # through a refinement's warm start.
    for nodeField in model.nodeFields.values():
        nodeField.createFieldValueEntry("U")
        nodeField.createFieldValueEntry("P")
        if carriesLinearMomentum(nodeField.name):
            nodeField.createFieldValueEntry("V")
            nodeField.createFieldValueEntry("A")

    model._linkFieldVariableObjects(model.nodeSets["all"])

    # *restart, readFrom=...: resume from a checkpoint. Reconstruct-then-overwrite, not full
    # serialization -- the model above was already rebuilt from this same .inp file, and
    # readRestart only overwrites its converged state (node fields, element history, scalar
    # variables, stateful constraints' history) plus model.time, so it must run after node fields
    # exist (createFieldValueEntry above) and after advanceToTime's cold-start bookkeeping, which
    # would otherwise clobber model.time back to job['startTime'].
    #
    # Resuming skips every step before the checkpoint's step. Topology changes made by model
    # modifiers (e.g. AMR) are replayed from the checkpoint's topology history. A `modelupdate`
    # executes an arbitrary expression, whose effect no checkpoint records -- so a resume past one,
    # in a skipped step or at the start of the resumed step, is refused.
    restartDefinitions = inputfile["restart"]
    resumeCheckpoint = None
    resumeStepNumber = None
    if restartDefinitions and restartDefinitions[0].get("readFrom"):
        resumeCheckpoint = ResumeCheckpoint(restartDefinitions[0]["readFrom"])
        resumeStepNumber = resumeCheckpoint.stepNumber
        resumeCheckpoint.restoreModel(model, journal)
        journal.message(
            "Resuming from restart checkpoint {:} (step {:}, time {:})".format(
                resumeCheckpoint.fileName, resumeStepNumber, model.time
            ),
            identification,
            0,
        )

    plotter = createPlotterFromInputFile(inputfile, journal)
    stepManager = createStepManagerFromInputFile(inputfile)
    fieldOutputController = createFieldOutputFromInputFile(inputfile, model, journal)
    model.fieldOutputController = fieldOutputController
    fieldOutputController.initializeJob(resuming=resumeCheckpoint is not None)

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

    # The output managers exist only now, well after the model was restored above.
    try:
        for step in stepManager.generateSteps(jobInfo, model, fieldOutputController, journal, solvers, outputManagers):
            if resumeStepNumber is not None:
                if step.actions["modelupdate"]:
                    raise RestartError(
                        "step {:} has a modelupdate, whose effect is not recorded in a checkpoint, so "
                        "the run cannot be resumed after it".format(step.number)
                    )
                if step.number < resumeStepNumber:
                    # Constructed, so that its step actions exist, but not solved: it already ran
                    # before the interrupted job wrote this checkpoint, and the state its step
                    # actions carried over is restored with the resumed step.
                    continue
            resumeFrom = None
            if step.number == resumeStepNumber:
                resumeFrom, resumeStepNumber = resumeCheckpoint, None

            tic = getCurrentTime()
            try:
                step.solve(resumeFrom)
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
