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

from edelweissfe.config.generators import getGeneratorFunction
from edelweissfe.config.outputmanagers import (
    getOutputManagerClass,
    getOutputManagerFactoryByName,
)
from edelweissfe.config.solvers import getSolverByName
from edelweissfe.generators.abqmodelconstructor import AbqModelConstructor
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.steps.adaptivestep import inputLanguage
from edelweissfe.steps.stepmanager import (
    StepActionDefinition,
    StepDefinition,
    StepManager,
)
from edelweissfe.utils.caseinsensitivedict import CaseInsensitiveDict
from edelweissfe.utils.fieldoutput import FieldOutputController
from edelweissfe.utils.inputfileparser import inputLanguage  # noqa: F811
from edelweissfe.utils.inputlanguage import (
    keywordIdentifier,
    moduleLevelKeywordIdentifier,
)
from edelweissfe.utils.math import createMathExpression, createModelAccessibleFunction
from edelweissfe.utils.misc import (
    convertLinesToStringDictionary,
    convertLineToStringDictionary,
    isInteger,
    strCaseCmp,
    strToRange,
)
from edelweissfe.utils.plotter import Plotter


def flattenDefinitions(ll):
    flat = []
    for item in ll:
        if isinstance(item, list):
            flat.extend(item)
        else:
            flat.append(item)
    return flat


def createFieldOutputFromInputFile(inputfile: dict, model: FEModel, journal: Journal) -> FieldOutputController:
    """Convenience helper function
    to create the FieldOutputController instance using the *fieldOutput keyword.

    Parameters
    ----------
    inputfile
        The processed inputfile in dictionary form.
    model
        The FEModel tree instance.
    journal
        The Journal for logging purposes.

    Returns
    -------
    FieldOutputController
        The configured FieldOutputController instance.
    """
    fieldOutputController = FieldOutputController(model, journal)
    for definition in inputfile["fieldOutput"]:
        moduleOptions = definition["moduleoptions"]
        perNodeDefs = moduleOptions.get("perNode", [])
        perElementDefs = moduleOptions.get("perElement", [])
        fromExpressionDefs = moduleOptions.get("fromExpression", [])

        for definition in perNodeDefs:
            field = definition["field"]
            nodeField = model.nodeFields[field]

            if bool(definition["nSet"]) and bool(definition["elSet"]):
                raise Exception(
                    f"During parsing of keyword {keywordIdentifier}fieldOutput ({moduleLevelKeywordIdentifier}perNode): Specify either nSet OR elSet."
                )

            subset = None
            if definition["nSet"]:
                subset = model.nodeSets[definition["nSet"]]
            elif definition["elSet"]:
                subset = model.elementSets[definition["elSet"]]

            if subset:
                nodeField = nodeField.subset(subset)

            f_of_x = definition["f(x)"]
            if f_of_x:
                f_of_x = createMathExpression(f_of_x)

            f_export_of_x = definition["f_export(x)"]
            if f_export_of_x:
                f_export_of_x = createMathExpression(f_export_of_x)

            fieldOutputController.addPerNodeFieldOutput(
                name=definition["name"],
                nodeField=nodeField,
                result=definition["result"],
                saveHistory=definition["saveHistory"],
                f_x=f_of_x,
                export=definition["export"],
                fExport_x=f_export_of_x,
            )

        for definition in perElementDefs:
            elSet = model.elementSets[definition["elSet"]]

            qp = definition["quadraturePoint"]
            quadraturePoints = strToRange(qp) if not isInteger(qp) else [int(qp)]

            f_of_x = definition["f(x)"]
            if f_of_x:
                f_of_x = createMathExpression(f_of_x)

            f_export_of_x = definition["f_export(x)"]
            if f_export_of_x:
                f_export_of_x = createMathExpression(f_export_of_x)

            fieldOutputController.addPerElementFieldOutput(
                name=definition["name"],
                elSet=elSet,
                result=definition["result"],
                quadraturePoints=quadraturePoints,
                saveHistory=definition["saveHistory"],
                f_x=f_of_x,
                export=definition["export"],
                fExport_x=f_export_of_x,
            )

        for definition in fromExpressionDefs:
            if definition["nSet"]:
                associatedSet = model.nodeSets[definition["nSet"]]
            elif definition["elSet"]:
                associatedSet = model.elementSets[definition["elSet"]]
            else:
                raise Exception(
                    f"During parsing of keyword {keywordIdentifier}fieldOutput ({moduleLevelKeywordIdentifier}fromExpression): All fieldOuputs must be associated with a set!"
                )

            theExpression = createModelAccessibleFunction(definition["expression"], model)

            f_of_x = definition["f(x)"]
            if f_of_x:
                f_of_x = createMathExpression(f_of_x)

            f_export_of_x = definition["f_export(x)"]
            if f_export_of_x:
                f_export_of_x = createMathExpression(f_export_of_x)

            fieldOutputController.addExpressionFieldOutput(
                associatedSet=associatedSet,
                theExpression=theExpression,
                name=definition["name"],
                saveHistory=definition["saveHistory"],
                f_x=f_of_x,
                export=definition["export"],
                fExport_x=f_export_of_x,
            )

    return fieldOutputController


def fillFEModelFromInputFile(model: FEModel, inputfile: dict, journal: Journal) -> FEModel:
    """Convenience helper function
    to fill an existing (possibly empty) FEModel using the input file and generators.

    Parameters
    ----------
    FEModel
        The model tree to be filled.
    input
        The processed inputfile in dictionary form.
    journal
        The Journal for logging purposes.

    Returns
    -------
    FEModel
        The updated, filled model tree.
    """

    # call individual optional model generators with executeAfterManualGeneration == True
    for definition in inputfile["modelGenerator"]:
        if definition.get("executeAfterManualGeneration", False):
            continue
        generatorDefinition = CaseInsensitiveDict(definition.copy())

        generatorType = generatorDefinition.pop("generator")
        data = generatorDefinition.pop("datalines")
        module = inputLanguage["modelGenerator"].getModule(generatorType)

        args, kwargs = module.parseDatalines(data)

        model = getGeneratorFunction(generatorType)(generatorDefinition, model, journal, *args, **kwargs)

    # the standard 'Abaqus like' model generator is invoked unconditionally, and it has direct access to the inputfile
    abqModelConstructor = AbqModelConstructor(journal)
    model = abqModelConstructor.createGeometryFromInputFile(model, inputfile)
    model = abqModelConstructor.createMaterialsFromInputFile(model, inputfile)
    model = abqModelConstructor.createAdvancedMaterialsFromInputFile(model, inputfile)
    model = abqModelConstructor.createAnalyticalFieldsFromInputFile(model, inputfile)
    model = abqModelConstructor.createSectionsFromInputFile(model, inputfile)

    # call individual optional model generators with executeAfterManualGeneration == False
    for definition in inputfile["modelGenerator"]:
        if not definition.get("executeAfterManualGeneration", False):
            continue
        generatorDefinition = CaseInsensitiveDict(definition.copy())

        generatorType = generatorDefinition.pop("generator")
        data = generatorDefinition.pop("datalines")
        module = inputLanguage["modelGenerator"].getModule(generatorType)

        args, kwargs = module.parseDatalines(data)

        if strCaseCmp(module.name, "executePythoncode"):
            args = data

        model = getGeneratorFunction(generatorType)(generatorDefinition, model, journal, *args, **kwargs)

    model = abqModelConstructor.createConstraintsFromInputFile(model, inputfile)
    return model


def createStepManagerFromInputFile(inputfile: dict):
    """Convenience helper function
    to create and fill a StepManager using *step definitions from the input file.

    Parameters
    ----------
    inputfile
        The processed inputfile in dictionary form.

    Returns
    -------
    StepManager
        The StepManager with enqueued StepDefinitions.
    """
    stepManager = StepManager()

    for stepDefinition in inputfile["step"]:
        stepType = stepDefinition.pop("type")
        stepActionLines = stepDefinition.pop("moduleOptions")

        inputFile = stepDefinition.pop("inputfile")  # noqa F841
        data = stepDefinition.pop("datalines")  # noqa F841

        # special treatment of *step/adaptive step module
        # arguments for adaptive step module must be provided in keyword line
        if not len(data) == 0:
            raise ValueError(
                f"Error during parsing of keyword {keywordIdentifier}step: {inputLanguage['step'].modules[0]} expects no data lines.\nProvide arguments to {inputLanguage['step'].modules[0]} in keyword line!\nUse the module-level keyword identifier {moduleLevelKeywordIdentifier} to define step actions."
            )

        stepActionDefinitions = []

        for module, definitions in stepActionLines.items():
            for definition in definitions:
                try:
                    name = definition.pop("name")
                except KeyError:
                    num = 0
                    for stepDef in stepManager.stepDefinitions:
                        num += len(stepDef.stepActionDefinitions)
                    num += len(stepActionDefinitions)
                    name = f"StepAction-{num}"
                stepActionDefinitions.append(StepActionDefinition(name, module, definition))

        stepDefinition = StepDefinition(stepType, stepDefinition, stepActionDefinitions)

        stepManager.enqueueStepDefinition(stepDefinition)

    return stepManager


def createSolversFromInputFile(inputfile: dict, jobInfo: dict, journal: Journal) -> dict:
    """Convenience helper function
    to create instances of nonlinear solvers using the *solver keyword.

    Parameters
    ----------
    inputfile
        The processed inputfile in dictionary form.
    jobInfo
        Additional informations about the job
    journal
        The Journal instance for logging purposes.

    Returns
    -------
    dict
        The dictionary containing the solver instances.
    """
    solvers = {}
    for solverDefinition in inputfile["solver"]:
        try:
            solverName = solverDefinition["name"]
        except KeyError:
            raise KeyError("Solver definition missing name")
        try:
            solverType = solverDefinition["solver"]
        except KeyError:
            raise KeyError(f"Missing type definition for solver {solverName}. Specify solver type with solver=...")

        solverData = solverDefinition["datalines"]

        Solver = getSolverByName(solverType)

        solverData = convertLinesToStringDictionary(solverData)

        solvers[solverName] = Solver(jobInfo, journal, **solverData)

    return solvers


def createOutputManagersFromInputFile(
    inputfile: dict,
    defaultName: str,
    model: FEModel,
    fieldOutputController: FieldOutputController,
    journal: Journal,
    plotter: Plotter,
) -> list:
    """Convenience helper function
    to create output managers using the *output keyword.

    Parameters
    ----------
    inputfile
        The processed inputfile in dictionary form.
    defaultName
        The default name prefix for output managers without specified name.
    fieldOutputController
        The FieldOutputController instance.
    journal
        The Journal instance for logging purposes.
    plotter
        The plotter instance.

    Returns
    -------
    list
        The list containing the OutputManager instances.
    """
    outputManagers = []

    for definition in inputfile["output"]:
        outputManagerKwargs = CaseInsensitiveDict(definition.copy())

        outputManagerType = outputManagerKwargs.pop("type")

        if outputManagerKwargs["name"] is not None:
            outputManagerName = outputManagerKwargs.pop("name")
        elif strCaseCmp(outputManagerType, "ensight"):
            outputManagerName = "esExport"
        else:
            outputManagerName = f"OutputManager-{len(outputManagers)}"

        datalines = outputManagerKwargs.pop("datalines")

        outputManagerKwargs.pop("inputfile")

        moduleOptions = outputManagerKwargs.pop("moduleOptions")

        # new input file parsing not yet implemented for meshplot
        if outputManagerType.casefold() in ["meshplot"]:
            OutputManager = getOutputManagerClass(definition["type"].lower())
            definitionLines = definition["datalines"]

            outputManager = OutputManager(outputManagerName, model, fieldOutputController, journal, plotter)

            for defLine in definitionLines:
                kwargs = convertLineToStringDictionary(defLine)
                if "elSet" in kwargs:
                    kwargs["elSet"] = model.elementSets[kwargs["elSet"]]
                if "nSet" in kwargs:
                    kwargs["nSet"] = model.nodeSets[kwargs["nSet"]]
                if "fieldOutput" in kwargs:
                    kwargs["fieldOutput"] = fieldOutputController.fieldOutputs[kwargs["fieldOutput"]]

                outputManager.updateDefinition(**kwargs)

            outputManagers.append(outputManager)
            continue

        if len(datalines) == 0:
            module = inputLanguage["output"].getModule(outputManagerType)
            args, kwargs = module.parseDatalines(datalines)

            outputManagerFactory = getOutputManagerFactoryByName(outputManagerType)
            try:
                outputManager = outputManagerFactory(
                    outputManagerName,
                    model,
                    fieldOutputController,
                    # args,
                    moduleOptions,
                    journal,
                    plotter,
                    **kwargs,
                )
            except ValueError as e:
                e.args = (
                    f"Error during parsing of keyword {keywordIdentifier}output (type={outputManagerType}): "
                    + e.args[0],
                )
                raise e

            outputManagers.append(outputManager)

        else:
            for dataline in datalines:
                module = inputLanguage["output"].getModule(outputManagerType)
                args, kwargs = module.parseDatalines(dataline)

                outputManagerFactory = getOutputManagerFactoryByName(outputManagerType)

                try:
                    outputManager = outputManagerFactory(
                        outputManagerName,
                        model,
                        fieldOutputController,
                        # args,
                        moduleOptions,
                        journal,
                        plotter,
                        **kwargs,
                    )
                except ValueError as e:
                    e.args = (
                        f"Error during parsing of keyword {keywordIdentifier}output (type={outputManagerType}): "
                        + e.args[0],
                    )
                    raise e

                outputManagers.append(outputManager)

    return outputManagers


def createPlotterFromInputFile(inputfile: dict, journal: Journal) -> Plotter:
    """Convenience helper function
    to create and configure the Plotter instance using the *configurePlots and *exportPlots keywords

    Parameters
    ----------
    inputfile
        The processed inputfile in dictionary form.
    journal
        The Journal instance for logging purposes.

    Returns
    -------
    Plotter
        The resulting plotter instance
    """
    plotConfigurations = [
        convertLineToStringDictionary(c) for configEntry in inputfile["configurePlots"] for c in configEntry["data"]
    ]

    exportJobs = [
        convertLineToStringDictionary(c) for configEntry in inputfile["exportPlots"] for c in configEntry["data"]
    ]

    plotter = Plotter(journal, plotConfigurations, exportJobs)

    return plotter
