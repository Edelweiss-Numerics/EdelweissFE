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

from dataclasses import dataclass

import numpy as np

from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.outputmanagers.base.outputmanagerbase import OutputManagerBase
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.utils.fieldoutput import FieldOutputController
from edelweissfe.utils.math import createMathExpression
from edelweissfe.utils.plotter import Plotter
from edelweissfe.utils.schema import schemaField

"""
Plot result for a nodeSet or an elementSet along the true geometrical distance.
Corresponds to the plot along path functionality in Abaqus.
"""


@dataclass(frozen=True)
class PlotAlongPathSchema:
    """The options this output manager accepts, owned by this module and never mutated from
    outside it.

    ``fieldOutput`` is declared ``required=True`` explicitly, but is still given ``default=None``
    so that ``PlotAlongPathSchema()`` is constructible without an argument at import time. A
    caller going through ``buildSchemaFromOptions`` still must supply ``fieldOutput`` (a missing
    required field is rejected regardless of this default).

    ``f_x`` answers to the input-file option name ``f(x)``, which is not a valid Python
    identifier and therefore cannot be the field name itself -- see ``optionName`` on
    :func:`edelweissfe.utils.schema.schemaField`.
    """

    fieldOutput: str | None = schemaField(
        description="Name of the field output.", dtype=str, default=None, required=True
    )

    figure: int = schemaField(description="Figure of the plotter.", dtype=int, default=1)
    axSpec: int = schemaField(description="AxSpec (MATLAB syntax) in the figure.", dtype=int, default=111)
    normalize: int = schemaField(description="Normalize results.", dtype=int, default=111)
    label: str | None = schemaField(description="Label.", dtype=str, default=None)

    f_x: str | None = schemaField(
        description="Function to apply in each increment.", dtype=str, default=None, optionName="f(x)"
    )
    nStages: int = schemaField(description="", dtype=int, default=1)
    export: str | None = schemaField(
        description="Export the field output to a file at the end of the job.", dtype=str, default=None
    )


class OutputManager(OutputManagerBase):
    identification = "PathPlotter"

    #: Option schema for this output manager, per OptionSchemaProvider.
    schema = PlotAlongPathSchema

    def __init__(
        self,
        name: str,
        model: FEModel,
        fieldOutputController: FieldOutputController,
        journal: Journal,
        plotter: Plotter,
        *,
        configuration: PlotAlongPathSchema = PlotAlongPathSchema(),
    ):
        """Constructible standalone, with no parser involvement. Options arrive as an
        already-validated, already-typed schema instance.

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
            The options this output manager accepts; ``fieldOutput`` is required by the input
            language but defaults to ``None`` here so that the schema remains constructible with no
            arguments.
        """
        self.name = name
        self.journal = journal
        self.monitorJobs = []
        self.plotter = plotter
        self.fieldOutputController = fieldOutputController
        self.model = model

        fieldOutputName = configuration.fieldOutput
        figure = configuration.figure
        axSpec = configuration.axSpec
        normalize = configuration.normalize
        label = configuration.label

        # A falsy "f(x)" (i.e. None or "") is treated as "no transform requested" and substituted
        # with the identity expression "x", rather than passed on to createMathExpression as-is.
        # A schema default of "x" alone would not reproduce this for an explicitly-empty value.
        fx = configuration.f_x
        if not fx:
            fx = "x"

        nStages = configuration.nStages
        export = configuration.export

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
