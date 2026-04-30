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
# Created on Mon Apr 17 11:37:26 2017

# @author: Matthias Neuner

import numpy as np

from edelweissfe.outputmanagers.base.outputmanagerbase import OutputManagerBase
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.utils.caseinsensitivedict import CaseInsensitiveDict
from edelweissfe.utils.inputlanguage import InputLanguage
from edelweissfe.utils.math import createMathExpression
from edelweissfe.utils.misc import (
    caseInsensitiveKwargsChecker,
    castKwargsValuesAndAddDefaults,
)

"""
Plot result for a nodeSet or an elementSet along the true geometrical distance.
Corresponds to the plot along path functionality in Abaqus.
"""

inputLanguage = InputLanguage()
module = inputLanguage["output"].addModule(
    "plotAlongPath",
    "Plot result for a nodeSet or an elementSet along the true geometrical distance.",
)

module.addRequiredArg("fieldOutput", "Name of the field output.", str)

module.addOptionalArg("figure", "Figure of the plotter.", int, 1)
module.addOptionalArg("axSpec", "AxSpec (MATLAB syntax) in the figure.", int, 111)
module.addOptionalArg("normalize", "Normalize results.", int, 111)
module.addOptionalArg("label", "Label.", str, None)

module.addOptionalArg("f(x)", "Function to apply in each increment.", str, None)
module.addOptionalArg("nStages", "", int, 1)
module.addOptionalArg("export", "Export the field output to a file at the end of the job.", str, None)

documentation = [module]

required = [kw.name for kw in module.requiredArgs]
required += [kw.name for kw in module.requiredKeywords]

optional = [kw.name for kw in module.optionalArgs]
optional += [kw.name for kw in module.optionalKeywords]


@caseInsensitiveKwargsChecker(required, optional)
@castKwargsValuesAndAddDefaults(module)
def outputManagerFactory(name, FEModel, fieldOutputController, moduleOptions, journal, plotter, **kwargs):
    kwargs = CaseInsensitiveDict(kwargs)

    fieldOutputName = kwargs["fieldOutput"]
    figure = kwargs["figure"]
    axSpec = kwargs["axSpec"]
    normalize = kwargs["normalize"]
    label = kwargs["label"]
    fx = kwargs["f(x)"]
    if not fx:
        fx = "x"
    nStages = kwargs["nStages"]
    export = kwargs["export"]

    return OutputManager(
        name,
        FEModel,
        fieldOutputController,
        journal,
        plotter,
        fieldOutputName,
        figure,
        axSpec,
        normalize,
        label,
        fx,
        nStages,
        export,
    )


class OutputManager(OutputManagerBase):
    identification = "PathPlotter"

    def __init__(
        self,
        name,
        model,
        fieldOutputController,
        journal,
        plotter,
        fieldOutputName,
        figure,
        axSpec,
        normalize,
        label,
        # nSet,
        # elSet,
        fx,
        nStages,
        export,
    ):
        self.journal = journal
        self.monitorJobs = []
        self.plotter = plotter
        self.fieldOutputController = fieldOutputController
        self.model = model

        entry = dict()
        entry["fieldOutput"] = fieldOutputController.fieldOutputs[fieldOutputName]
        entry["f(x)"] = createMathExpression(fx)

        # compute distance(s), entity 0 is the reference entity in the 'origin'
        entry["pathDistances"] = [0.0]
        entry["nStages"] = nStages
        entry["export"] = export

        theSet = entry["fieldOutput"].associatedSet

        if type(theSet) is NodeSet:
            # try:  # nSet?
            nodes = theSet
            # 1) distances between nodes:
            distances = [np.linalg.norm(nodes[i + 1].coordinates - nodes[i].coordinates) for i in range(len(nodes) - 1)]
        elif type(theSet) is ElementSet:
            # except AttributeError:  # no, its an elSet!
            elements = entry["fieldOutput"].elSet
            # dirty computation of centroid by taking the mean (not correct, but fast)
            elCentroids = [np.asarray(el.nodeCoordinates).reshape(el.nNodes, -1).mean(axis=0) for el in elements]
            elCentroids = np.asarray(elCentroids)
            # 1) distances between elements:
            distances = [np.linalg.norm(elCentroids[i + 1, :] - elCentroids[i, :]) for i in range(len(elCentroids) - 1)]
        else:
            raise Exception("Invalid fieldoutput specified: Not operation on nSet or elSet!")

        # 2) distances with respect to entity 0
        for dist in distances:
            entry["pathDistances"].append(entry["pathDistances"][-1] + dist)

        entry["label"] = label

        entry["figure"] = figure
        entry["axSpec"] = axSpec

        entry["normalize"] = normalize
        self.monitorJobs.append(entry)

        if export:
            np.savetxt(export, np.asarray(entry["pathDistances"]))

    def initializeJob(self):
        pass

    def initializeStep(self, step):
        for nJob in self.monitorJobs:
            self.plotStages = np.linspace(0, step.length, nJob["nStages"])

    def finalizeIncrement(self, **kwargs):
        totalTime = self.model.time
        # totalTime = increment[3] + increment[4]
        if totalTime > self.plotStages[0]:
            for nJob in self.monitorJobs:
                nJob_ = nJob.copy()
                nJob_["label"] = None
                result = nJob["fieldOutput"].getLastResult()

                result = nJob["f(x)"](result)

                result = np.squeeze(result)

                if nJob["normalize"]:
                    result /= np.max(np.abs(result))

                if nJob["export"]:
                    exportData = np.column_stack((nJob["pathDistances"], result))
                    np.savetxt(
                        nJob["label"] + "stage_" + str(nJob["nStages"] - len(self.plotStages)) + ".csv",
                        exportData,
                    )

                self.plotter.plotXYData(nJob["pathDistances"], result, nJob["figure"], nJob["axSpec"], nJob_)

            self.plotStages = np.delete(self.plotStages, 0)

    def finalizeFailedIncrement(self, **kwargs):
        pass

    def finalizeStep(
        self,
    ):
        pass

    def finalizeJob(
        self,
    ):
        for nJob in self.monitorJobs:
            result = nJob["fieldOutput"].getLastResult()
            result = nJob["f(x)"](result)

            result = np.squeeze(result)

            if nJob["normalize"]:
                result /= np.max(np.abs(result))

            if nJob["export"]:
                exportData = np.column_stack((nJob["pathDistances"], result))
                np.savetxt(
                    nJob["label"] + "stage_" + str(nJob["nStages"] - len(self.plotStages)) + ".csv",
                    exportData,
                )

            self.plotter.plotXYData(nJob["pathDistances"], result, nJob["figure"], nJob["axSpec"], nJob)
