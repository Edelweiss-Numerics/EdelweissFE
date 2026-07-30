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
#  Magdalena Schreter magdalena.schreter@uibk.ac.at
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
"""
@author: Magdalena

Module meshplot divided into classes:
    * Triangulation:
        creates triangles out of rectangles in order to use matplotlib plotting
        options
    * Plotter:
        creates figures, axes, grid, labels
    * Outputmanager:
        creates the plotting specific for the defined keyword lines
"""

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

import matplotlib.tri as mtri
import numpy as np
from matplotlib import colors

from edelweissfe.outputmanagers.base.outputmanagerbase import OutputManagerBase
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.utils.math import createMathExpression
from edelweissfe.utils.meshtools import (
    extractNodeCoordinatesFromElset,
    transferElsetResultsToElset,
)
from edelweissfe.utils.schema import (
    DatalineAggregatingSchema,
    buildSchemaFromOptions,
    schemaField,
)

documentation = {
    "figure": "figure number, (default=1)",
    "axSpec": "axis specification according to matplotlib syntax, (default=111)",
    "create=perNode": "result per node is plotted in a meshplot",
    "create=perElement": "result per element is plotted in a meshplot",
    "create=meshOnly": "plot the mesh only",
    "create=xyData": "2D Plot of results",
}


@dataclass(frozen=True)
class MeshPlotXYDataJob:
    """A 2D line/scatter plot of one field output against another (or against time)."""

    x: str = schemaField(description="Name of the x-axis field output, or 'time'.", dtype=str)
    y: str = schemaField(description="Name of the y-axis field output.", dtype=str)
    f_x: str = schemaField(
        description="Expression applied to the x values, e.g. 'sum(x[:,1])'.",
        dtype=str,
        default=None,
        optionName="f(x)",
    )
    f_y: str = schemaField(
        description="Expression applied to the y values, e.g. 'mean(y)'.",
        dtype=str,
        default=None,
        optionName="f(y)",
    )
    figure: str = schemaField(description="Figure number.", dtype=str, default="1")
    axSpec: str = schemaField(description="Axis specification in matplotlib syntax.", dtype=str, default="111")
    label: str = schemaField(description="Curve label; defaults to the y field output's name.", dtype=str, default=None)
    integral: bool = schemaField(
        description="Shade the area under the curve and label it with its integral.",
        dtype=bool,
        default=False,
    )
    c: str = schemaField(description="Matplotlib color.", dtype=str, default=None)
    ls: str = schemaField(description="Matplotlib line style.", dtype=str, default=None)
    plotType: str = schemaField(description="'linePlot' or 'scatter'.", dtype=str, default="linePlot")
    marker: str = schemaField(description="Matplotlib marker (scatter only).", dtype=str, default=None)
    ms: str = schemaField(description="Marker size (scatter only).", dtype=str, default=None)
    markerfacecolor: str = schemaField(description="Marker face color (scatter only).", dtype=str, default=None)


@dataclass(frozen=True)
class MeshPlotPerNodeJob:
    """A filled contour plot of a nodal result over the mesh."""

    fieldOutput: str = schemaField(
        description="Name of an element-based field output whose node set is contoured.", dtype=str
    )
    f_x: str = schemaField(
        description="Expression applied to the result values.", dtype=str, default=None, optionName="f(x)"
    )
    figure: str = schemaField(description="Figure number.", dtype=str, default="1")
    axSpec: str = schemaField(description="Axis specification in matplotlib syntax.", dtype=str, default="111")
    label: str = schemaField(
        description="Color bar label; defaults to the field output's name.", dtype=str, default=None
    )
    plotMeshGrid: str = schemaField(
        description="Draw the element grid beneath the contours ('undeformed' to draw it).",
        dtype=str,
        default="undeformed",
    )


@dataclass(frozen=True)
class MeshPlotPerElementJob:
    """A piecewise-constant contour plot of an element result over the mesh."""

    fieldOutput: str = schemaField(description="Name of an element-based field output.", dtype=str)
    f_x: str = schemaField(
        description="Expression applied to the result values.", dtype=str, default=None, optionName="f(x)"
    )
    figure: str = schemaField(description="Figure number.", dtype=str, default="1")
    axSpec: str = schemaField(description="Axis specification in matplotlib syntax.", dtype=str, default="111")
    label: str = schemaField(
        description="Color bar label; defaults to the field output's name.", dtype=str, default=None
    )


@dataclass(frozen=True)
class MeshPlotMeshOnlyJob:
    """A plot of the mesh itself, undeformed or warped by a field output."""

    configuration: str = schemaField(
        description="'undeformed', or 'deformed' to warp the mesh by 'warpBy'.", dtype=str, default="undeformed"
    )
    warpBy: str = schemaField(
        description="Name of the field output warping the mesh; required if configuration=deformed.",
        dtype=str,
        default=None,
    )
    scaleFactor: float = schemaField(description="Warp scaling factor.", dtype=float, default=1.0)
    figure: str = schemaField(description="Figure number.", dtype=str, default="1")
    axSpec: str = schemaField(description="Axis specification in matplotlib syntax.", dtype=str, default="111")
    plotNodeLabels: bool = schemaField(description="Annotate every node with its label.", dtype=bool, default=False)
    plotElementLabels: bool = schemaField(
        description="Annotate every element with its number.", dtype=bool, default=False
    )


@dataclass(frozen=True)
class MeshPlotSaveFigureJob:
    """Export a figure to a file at the end of the job."""

    figure: str = schemaField(description="Figure number to export.", dtype=str, default="1")
    fileName: str = schemaField(
        description="Output file name, without extension.", dtype=str, default="exportFigure", optionName="name"
    )
    width: float = schemaField(description="Figure width in points.", dtype=float, default=469.47)
    scale: float = schemaField(description="Scaling applied to the width.", dtype=float, default=1.0)
    heightRatio: float = schemaField(
        # 0.0 is the plotter's "use the golden ratio" sentinel (`float(heightRatio) or golden_mean`
        # in `Plotter._fancyFigSize`), and is what the legacy default of `False` evaluated to.
        description="Height/width ratio; 0 selects the golden ratio.",
        dtype=float,
        default=0.0,
    )
    png: bool = schemaField(description="Export as .png in addition to .pdf.", dtype=bool, default=True)


@dataclass(frozen=True)
class MeshPlotSchema(DatalineAggregatingSchema):
    """All plotting jobs of one ``*output, type=meshplot`` block, in file order.

    One meshplot instance aggregates a heterogeneous list of jobs rather than describing a single
    one, which is why this schema is a :class:`~edelweissfe.utils.schema.DatalineAggregatingSchema`
    -- see that class for why the arm dispatch lives here instead of in the generic L4 adapter.
    """

    #: Maps the value of the ``create`` tag option to the job schema it selects.
    _jobSchemasByTag: ClassVar[dict[str, type]] = {
        "pernode": MeshPlotPerNodeJob,
        "perelement": MeshPlotPerElementJob,
        "xydata": MeshPlotXYDataJob,
        "meshonly": MeshPlotMeshOnlyJob,
    }

    #: Maps a bare presence-flag option to the job schema it selects, independently of ``create``.
    _jobSchemasByFlag: ClassVar[dict[str, type]] = {"savefigure": MeshPlotSaveFigureJob}

    jobs: tuple = schemaField(
        description="The plotting jobs, one or more per dataline.",
        dtype=tuple,
        default=(),
    )

    @classmethod
    def fromDatalines(cls, datalines: list[Mapping[str, Any]]) -> "MeshPlotSchema":
        jobs = []

        for options in datalines:
            keysByFoldedName = {name.casefold(): name for name in options}
            selectors = cls._jobSchemasByFlag.keys() & keysByFoldedName.keys()

            tagKey = keysByFoldedName.get("create")
            if tagKey is not None:
                tag = str(options[tagKey]).casefold()
                if tag not in cls._jobSchemasByTag:
                    raise ValueError(
                        f"'{options[tagKey]}' is not a valid 'create' value; expected one of "
                        f"{', '.join(sorted(cls._jobSchemasByTag))}."
                    )
                selectors |= {tagKey.casefold()}

            if not selectors:
                raise ValueError(
                    "Every meshplot dataline must select a job, either with 'create=' or with "
                    f"{', '.join(sorted(cls._jobSchemasByFlag))}."
                )

            # A dataline selects exactly one job.
            if len(selectors) > 1:
                raise ValueError(
                    f"A meshplot dataline must select a single job, not {', '.join(sorted(selectors))}; "
                    "write one dataline per job."
                )

            (selector,) = selectors
            jobSchema = (
                cls._jobSchemasByTag[str(options[tagKey]).casefold()]
                if selector == "create"
                else cls._jobSchemasByFlag[selector]
            )
            jobOptions = {name: value for name, value in options.items() if name.casefold() != selector}
            jobs.append(buildSchemaFromOptions(jobSchema, jobOptions))

        return cls(jobs=tuple(jobs))


class Triangulation:
    """class that provides the division of quadrilateral elements into triangles"""

    def __init__(self, xCoord, yCoord, elNodesIdxList):
        self.triangleIdx, self.triang = self.quadIdxToTriIdx(xCoord, yCoord, elNodesIdxList)

    def quadIdxToTriIdx(self, xCoord, yCoord, elementIdxMatrix):
        triangleIdx = np.asarray(
            [[list(elIdx[:3]), [elIdx[2], elIdx[3], elIdx[0]]] for elIdx in elementIdxMatrix]
        ).reshape(-1, 3)
        triang = mtri.Triangulation(xCoord, yCoord, triangleIdx)
        return triangleIdx, triang

    def quadFieldToTriField(self, fieldValues):
        return np.asarray([2 * [fieldValues]]).reshape(-1, 1, order="f").flatten()


class MeshPlot:
    def __init__(self, coordinates, elNodesIdxList, elCoordinatesList):
        self.coordinates = coordinates
        self.elCoordinatesList = elCoordinatesList
        self.xCoord = coordinates[:, 0]
        self.yCoord = coordinates[:, 1]
        self.xLimits = [-self.xCoord.max() * 0.1, self.xCoord.max() * 1.1]
        self.yLimits = [-self.yCoord.max() * 0.1, self.yCoord.max() * 1.1]
        self.TriangObj = Triangulation(self.xCoord, self.yCoord, elNodesIdxList)
        self.contourPlotScaling = 50
        self.userColorMap = "coolwarm"

    def contourPlotFieldVariable(self, fieldValues, fig, ax, label):
        """divide quad elements into two triangles and apply a constant field
        value for both triangles"""
        resultPerTriElement = self.TriangObj.quadFieldToTriField(fieldValues)
        mapping = ax.tripcolor(
            self.xCoord,
            self.yCoord,
            self.TriangObj.triangleIdx,
            facecolors=resultPerTriElement,
            cmap=self.userColorMap,
            norm=colors.Normalize(vmax=np.nanmax(resultPerTriElement), vmin=np.nanmin(resultPerTriElement)),
        )
        cbar = fig.colorbar(mapping, fraction=0.046, pad=0.04)
        cbar.set_label(label)
        ax.set_xlim(self.xLimits)
        ax.set_ylim(self.yLimits)

    def contourPlotNodalValues(self, z, fig, ax, label, elements, nSet):
        """divide quads into two triangles and apply a nodal value to the corner nodes"""

        # PERFORMANCE IMPROVEMENT PENDING

        nSetList = [node.label for node in nSet]
        coordinates = np.asarray([node.coordinates for node in nSet])

        triangleNodes = []
        counter = 0
        for element in elements.values():
            nodeList = []
            for node in element.nodes:
                nodeList.append(node)

            if set([node.label for node in nodeList]) < set(nSetList):
                counter += 1
                triangleNodes.append([nSetList.index(node.label) for node in nodeList])

        triangObjTemp = Triangulation(coordinates[:, 0], coordinates[:, 1], triangleNodes)
        mapping = ax.tricontourf(
            triangObjTemp.triang,
            z,
            self.contourPlotScaling,
            cmap=self.userColorMap,
            norm=colors.Normalize(vmax=np.nanmax(z), vmin=np.nanmin(z)),
        )
        cbar = fig.colorbar(mapping, fraction=0.046, pad=0.04)
        cbar.set_label(label)
        ax.set_xlim(self.xLimits)
        ax.set_ylim(self.yLimits)

    def plotNodeLabels(self, labels, ax):
        """label nodes of elements"""
        for label in labels:
            ax.annotate(
                "%i" % label,
                xy=self.coordinates[label - 1, :],
                fontsize=6,
                textcoords="data",
            )

    def plotElementLabels(self, ax, elementList):
        """label nodes of elements"""
        for element in elementList:
            xCenter = 0
            yCenter = 0
            for node in element.nodes:
                xCenter += node.coordinates[0]
                yCenter += node.coordinates[1]
            ax.annotate(
                "%i" % element.elNumber,
                xy=[xCenter / 4, yCenter / 4],
                fontsize=6,
                textcoords="data",
            )

    def plotMeshGrid(self, ax, coordinateList):
        """plot grid of elements; so far only implemented for quads"""
        for element in coordinateList:
            ax.plot(
                np.append(element[:, 0], element[0, 0]),
                np.append(element[:, 1], element[0, 1]),
                "k",
                linewidth=0.3,
            )
        ax.set_xlim(self.xLimits)
        ax.set_ylim(self.yLimits)
        ax.grid(False)


class OutputManager(OutputManagerBase):
    identification = "meshPlot"
    schema = MeshPlotSchema

    def __init__(
        self,
        name,
        model,
        fieldOutputController,
        journal,
        plotter,
        *,
        configuration: MeshPlotSchema = MeshPlotSchema(),
    ):
        self.name = name
        self.domainSize = model.domainSize
        self.plotter = plotter
        self.journal = journal

        self.nodes = model.nodes
        self.elements = model.elements
        self.elSets = model.elementSets
        self.nSets = model.nodeSets

        # write List of nodeLabels
        self.labelList = np.asarray([nodeNumber for nodeNumber in self.nodes.keys()])
        # write List of node coordiantes
        self.coordinateList = np.asarray([node.coordinates for node in self.nodes.values()])
        # write list of element coordinates with 4x2 arrays (xCol, yCol)
        self.elCoordinatesList = []
        # write list of node indices for each element relevant for the meshplot output
        # in case of an 8-node element only the 4 first nodes are relevant
        self.elNodesIdxList = []

        self.perNodeJobs = []
        self.perElementJobs = []
        #        self.configJobs = []
        self.xyJobs = []
        self.saveJobs = []
        self.meshOnlyJobs = []

        self.fieldOutputController = fieldOutputController

        for job in configuration.jobs:
            self._addJob(job, fieldOutputController)

    def _addJob(self, job, fieldOutputController):
        """Turn one validated job schema into the internal job description consumed by
        :meth:`finalizeJob`, resolving field output *names* into live field output objects.

        Resolution happens here rather than in the input-file adapter, so a programmatic caller
        constructing a :class:`MeshPlotSchema` by hand gets the same treatment as an ``.inp`` file.

        Parameters
        ----------
        job
            One of the ``MeshPlot*Job`` schema instances.
        fieldOutputController
            The FieldOutputController holding the field outputs referenced by name.
        """
        fieldOutputs = fieldOutputController.fieldOutputs

        if isinstance(job, MeshPlotSaveFigureJob):
            self.saveJobs.append(
                {
                    "figure": job.figure,
                    "fileName": job.fileName,
                    "width": job.width,
                    "scale": job.scale,
                    "heightRatio": job.heightRatio,
                    "png": job.png,
                }
            )

        elif isinstance(job, MeshPlotPerNodeJob):
            fieldOutput = fieldOutputs[job.fieldOutput]

            if type(fieldOutput.associatedSet) is not ElementSet:
                raise Exception("perNode job must be defined on a perElement fieldOutput")

            perNodeJob = {
                "fieldOutput": fieldOutput,
                "nSet": fieldOutput.associatedSet.extractNodeSet(),
                "label": job.label if job.label is not None else job.fieldOutput,
                "axSpec": job.axSpec,
                "figure": job.figure,
                "plotMeshGrid": job.plotMeshGrid,
            }
            if job.f_x is not None:
                perNodeJob["f(x)"] = createMathExpression(job.f_x)
            self.perNodeJobs.append(perNodeJob)

        elif isinstance(job, MeshPlotPerElementJob):
            perElementJob = {
                "fieldOutput": fieldOutputs[job.fieldOutput],
                "label": job.label if job.label is not None else job.fieldOutput,
                "axSpec": job.axSpec,
                "figure": job.figure,
            }
            if job.f_x is not None:
                perElementJob["f(x)"] = createMathExpression(job.f_x)
            self.perElementJobs.append(perElementJob)

        elif isinstance(job, MeshPlotXYDataJob):
            y = fieldOutputs[job.y]
            xyJob = {
                "x": "time" if job.x == "time" else fieldOutputs[job.x],
                "y": y,
                "figure": job.figure,
                "axSpec": job.axSpec,
                "label": job.label if job.label is not None else y.name,
                "integral": job.integral,
                # Forwarded to `plotter.plotXYData`, which reads these keys directly out of the job dict.
                "plotType": job.plotType,
                **{
                    name: value
                    for name, value in (
                        ("c", job.c),
                        ("ls", job.ls),
                        ("marker", job.marker),
                        ("ms", job.ms),
                        ("markerfacecolor", job.markerfacecolor),
                    )
                    if value is not None
                },
            }
            if job.f_x is not None:
                xyJob["f(x)"] = createMathExpression(job.f_x)
            if job.f_y is not None:
                xyJob["f(y)"] = createMathExpression(job.f_y, symbol="y")
            self.xyJobs.append(xyJob)

        elif isinstance(job, MeshPlotMeshOnlyJob):
            meshOnlyJob = {
                "configuration": job.configuration,
                "scaleFactor": job.scaleFactor,
                "axSpec": job.axSpec,
                "figure": job.figure,
                "plotNodeLabels": job.plotNodeLabels,
                "plotElementLabels": job.plotElementLabels,
            }
            if job.configuration == "deformed":
                if job.warpBy is None:
                    raise ValueError("A meshOnly job with configuration=deformed requires 'warpBy'.")
                meshOnlyJob["warpBy"] = fieldOutputs[job.warpBy]
            self.meshOnlyJobs.append(meshOnlyJob)

        else:
            raise ValueError(f"{type(job).__name__} is not a valid meshplot job.")

    def initializeJob(self):
        pass

    def initializeStep(self, step):
        if self.perElementJobs or self.perNodeJobs or self.meshOnlyJobs:
            self.elCoordinatesList = extractNodeCoordinatesFromElset(self.elements.values())
            for element in self.elements.values():
                nodeIdxArray = [nodeNumber.label - 1 for nodeNumber in element.nodes[:]][:4]
                self.elNodesIdxList.append(nodeIdxArray)

            self.meshPlot = MeshPlot(self.coordinateList, self.elNodesIdxList, self.elCoordinatesList)

    def finalizeIncrement(self, **kwargs):
        pass

    def finalizeFailedIncrement(self, **kwargs):
        pass

    def finalizeStep(
        self,
    ):
        pass

    def finalizeJob(
        self,
    ):
        for xyJob in self.xyJobs:
            y = xyJob["y"].getResultHistory()

            if xyJob["x"] == "time":
                x = xyJob["y"].getTimeHistory()
            else:
                x = xyJob["x"].getResultHistory()

            if "f(x)" in xyJob:
                x = xyJob["f(x)"](x)
            if "f(y)" in xyJob:
                y = xyJob["f(y)"](y)

            self.plotter.plotXYData(x, y, xyJob["figure"], xyJob["axSpec"], xyJob)
            ax = self.plotter.getAx(xyJob["figure"], xyJob["axSpec"])
            if xyJob["integral"]:
                integral = np.trapezoid(y.flatten(), x=x.flatten())
                ax.fill_between(x.flatten(), 0, y.flatten(), color="gray", label=str(integral))

        for perNodeJob in self.perNodeJobs:
            result = perNodeJob["fieldOutput"].getLastResult()
            fig = self.plotter.getFig(perNodeJob["figure"])
            ax = self.plotter.getAx(perNodeJob["figure"], perNodeJob["axSpec"])
            ax.set_axis_off()
            ax.set_aspect("equal")

            if "f(x)" in perNodeJob:
                result = perNodeJob["f(x)"](result)

            if perNodeJob["plotMeshGrid"] == "undeformed":
                self.meshPlot.plotMeshGrid(ax, self.elCoordinatesList)

            result = np.squeeze(result)

            self.meshPlot.contourPlotNodalValues(
                result, fig, ax, perNodeJob["label"], self.elements, perNodeJob["nSet"]
            )

        for perElementJob in self.perElementJobs:
            fig = self.plotter.getFig(perElementJob["figure"])
            ax = self.plotter.getAx(perElementJob["figure"], perElementJob["axSpec"])
            ax.set_axis_off()
            ax.set_aspect("equal")

            #            print(self.elCoordinatesList)
            #            if perElementJob['configuration'] == 'deformed':
            #                elCoordinatesListDeformed = extractNodeCoordinatesFromElset(self.elSets['all'], perElementJob['fieldOutput'].getLastResult(), perElementJob['scaleFactor'])
            #                self.meshPlot.plotMeshGrid( ax, elCoordinatesListDeformed)

            #            self.meshPlot.plotMeshGrid(ax,  self.elCoordinatesList)

            resultArray = perElementJob["fieldOutput"].getLastResult()

            if "f(x)" in perElementJob:
                resultArray = perElementJob["f(x)"](resultArray)

            if perElementJob["fieldOutput"].associatedSet.name != "all":
                shape = (
                    (len(self.elSets["all"]), resultArray.shape[-1])
                    if resultArray.ndim >= 2
                    else len(self.elSets["all"])
                )
                resultsTarget = np.empty(shape)
                resultsTarget[:] = np.nan
                transferElsetResultsToElset(
                    self.elSets["all"],
                    perElementJob["fieldOutput"].elSet,
                    resultsTarget,
                    resultArray,
                )
                resultArray = resultsTarget

            self.meshPlot.contourPlotFieldVariable(resultArray, fig, ax, perElementJob["label"])
        for meshOnlyJob in self.meshOnlyJobs:
            fig = self.plotter.getFig(meshOnlyJob["figure"])
            ax = self.plotter.getAx(meshOnlyJob["figure"], meshOnlyJob["axSpec"])
            ax.set_axis_off()
            ax.set_aspect("equal")

            if meshOnlyJob["plotNodeLabels"]:
                self.meshPlot.plotNodeLabels(self.nodes.keys(), ax)

            if meshOnlyJob["plotElementLabels"]:
                self.meshPlot.plotElementLabels(ax, self.elSets["all"])

            if meshOnlyJob["configuration"] == "deformed":
                elCoordinatesListDeformed = extractNodeCoordinatesFromElset(
                    self.elSets["all"],
                    meshOnlyJob["warpBy"].getLastResult(),
                    meshOnlyJob["scaleFactor"],
                )
                self.meshPlot.plotMeshGrid(ax, elCoordinatesListDeformed)

            else:
                elCoordinatesListUnDeformed = extractNodeCoordinatesFromElset(self.elSets["all"])
                self.meshPlot.plotMeshGrid(ax, elCoordinatesListUnDeformed)

        #        for configJob in self.configJobs:
        #            self.plotter.configAxes(**configJob)

        for saveJob in self.saveJobs:
            self.plotter.exportFigure(
                saveJob["fileName"],
                saveJob["figure"],
                saveJob["width"],
                saveJob["scale"],
                saveJob["heightRatio"],
                saveJob["png"],
            )


##
