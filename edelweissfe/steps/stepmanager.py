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

import textwrap

from edelweissfe.config import registry
from edelweissfe.config.stepactions import stepActionFactory
from edelweissfe.config.steps import getStepClassByType
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.steps.base.stepbase import StepBase
from edelweissfe.utils.caseinsensitivedict import CaseInsensitiveDict
from edelweissfe.utils.fieldoutput import FieldOutputController


class StepActionDefinition:
    """A step action definition, as parsed from the input file.

    Parameters
    ----------
    name
        The name of the step action.
    module
        The step action module (type), e.g., 'dirichlet'.
    kwargs
        The options defining the step action.
    """

    def __init__(self, name: str, module: str, kwargs: dict):
        self.name = name
        self.module = module
        self.kwargs = kwargs


class StepDefinition:
    """A step definition, as parsed from the input file.

    Parameters
    ----------
    stepType
        The type of the step, e.g., 'adaptive'.
    stepOptions
        The options defining the step.
    stepActionDefinitions
        The list of StepActionDefinitions for this step.
    """

    def __init__(
        self,
        stepType: str,
        stepOptions: dict,
        stepActionDefinitions: list[StepActionDefinition],
    ):
        self.type = stepType
        self.stepOptions = stepOptions
        self.stepActionDefinitions = stepActionDefinitions


class StepActionCollection:
    """A collection of step actions, grouped by step action module (type).

    Accessing a valid step action module without any defined actions yields an
    empty dictionary, whereas accessing an unknown step action module raises a
    KeyError. This ensures that typos in step action module names fail loudly
    instead of being silently treated as empty collections.

    What counts as "valid" comes from the L3 registry, via
    :func:`~edelweissfe.config.registry.isRegistered` -- deliberately the non-importing predicate
    rather than ``lookup``, because a consumer asking for a category it defines no actions in (a
    solver checking for ``indirectcontrol``, say) must not thereby import that step action's module.
    The check it replaces, ``importlib.util.find_spec("edelweissfe.stepactions." + module)``, could
    only ever see step actions shipped inside this package.
    """

    def __init__(self):
        self._actionsPerModule = dict()

    def __getitem__(self, module: str) -> dict:
        module = module.lower()

        if module not in self._actionsPerModule:
            if not registry.isRegistered("stepaction", module):
                raise KeyError(
                    f"'{module}' is not a known step action module. Available: "
                    + ", ".join(registry.availableNames("stepaction"))
                )
            self._actionsPerModule[module] = dict()

        return self._actionsPerModule[module]

    def __contains__(self, module: str) -> bool:
        return module.lower() in self._actionsPerModule

    def __iter__(self):
        return iter(self._actionsPerModule)

    def keys(self):
        return self._actionsPerModule.keys()

    def values(self):
        return self._actionsPerModule.values()

    def items(self):
        return self._actionsPerModule.items()


class StepManager:
    """This manager holds all step definitions of the simulation,
    and generates the Steps to be solved. Step actions are created
    (or updated, if already existing from previous steps) based on
    computed results, model info and job information.
    """

    identification = "StepManager"

    def __init__(
        self,
    ):
        self.stepActions = StepActionCollection()
        self.stepDefinitions = []

    def enqueueStepDefinition(self, stepDefinition: StepDefinition):
        """Enqueue a step definition.

        Parameters
        ----------
        stepDefinition
            The StepDefinition containing the step type, step options and list of StepActionDefinition.
        """
        self.stepDefinitions.append(stepDefinition)

    def generateSteps(
        self,
        jobInfo: dict,
        model: FEModel,
        fieldOutputController: FieldOutputController,
        journal: Journal,
        solvers: dict,
        outputManagers: list,
    ) -> StepBase:
        """Generate the Steps to be solved from the enqueued step definitions.

        The steps are generated lazily, i.e., every step (and its step actions)
        is created just when it is requested, such that it is created based on
        the current state of the model.

        Parameters
        ----------
        jobInfo
            A dictionary containing information on the job.
        model
            The model tree.
        fieldOutputController
            The field output controller.
        journal
            The journal instance for logging.
        solvers
            The dictionary of available solver instances.
        outputManagers
            The OutputManagers used.

        Yields
        ------
        StepBase
            The next Step to be solved.
        """

        def printActionDefinition(intro, options):
            for line in textwrap.wrap(
                intro + " [" + ", ".join(("{:}={:}".format(k, v) for k, v in options.items())) + "]",
                subsequent_indent=" " * (len(intro) + 1),
            ):
                journal.message(
                    line,
                    self.identification,
                    2,
                )

        for stepNumber, stepDefinition in enumerate(self.stepDefinitions):
            actionNamesInThisStep = set()

            if stepDefinition.stepActionDefinitions:
                journal.message(
                    "StepAction definitions:",
                    self.identification,
                    1,
                )

            for action in stepDefinition.stepActionDefinitions:
                if action.name in actionNamesInThisStep:
                    raise ValueError(
                        "StepAction {:} has multiple definitions in step {:}".format(action.name, stepNumber)
                    )
                actionNamesInThisStep.add(action.name)

                # Both paths go through the L4 seam on StepActionBase rather than calling
                # __init__/updateStepAction directly, so that a step action with a real typed
                # constructor and one still consuming the raw definition dict are driven
                # identically from here -- see StepActionBase.fromStepActionDefinition.
                if action.name in self.stepActions[action.module]:
                    self.stepActions[action.module][action.name].updateStepActionFromDefinition(
                        action.kwargs, jobInfo, model, fieldOutputController, journal
                    )
                    printActionDefinition('Updating "{:}"'.format(action.name), action.kwargs)

                else:
                    printActionDefinition('Creating "{:}"'.format(action.name), action.kwargs)

                    self.stepActions[action.module][action.name] = stepActionFactory(
                        action.module
                    ).fromStepActionDefinition(
                        action.name,
                        action.kwargs,
                        jobInfo,
                        model,
                        fieldOutputController,
                        journal,
                    )

            stepOptions = CaseInsensitiveDict(stepDefinition.stepOptions)
            solverName = stepOptions.pop("solver")

            try:
                solver = solvers[solverName]
            except KeyError:
                mssg = f"No definition found for solver {solverName}."
                availableSolvers = [key for key in solvers.keys() if not key == "default"]
                if availableSolvers:
                    mssg += " Available solvers: " + ", ".join(availableSolvers)
                else:
                    mssg += " Define solver using *solver keyword."
                raise KeyError(mssg)

            stepClass = getStepClassByType(stepDefinition.type)

            yield stepClass(
                stepNumber,
                model,
                fieldOutputController,
                journal,
                jobInfo,
                solver,
                outputManagers,
                self.stepActions,
                **stepOptions,
            )
