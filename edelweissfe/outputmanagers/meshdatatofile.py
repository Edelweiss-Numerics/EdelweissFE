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
#  Alexander Dummer alexander.dummer@uibk.ac.at
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

from dataclasses import dataclass

from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.outputmanagers.base.outputmanagerbase import OutputManagerBase
from edelweissfe.utils.fieldoutput import FieldOutputController
from edelweissfe.utils.plotter import Plotter
from edelweissfe.utils.schema import schemaField

"""
Writes the (generated) mesh data to a file.

.. code-block:: console
    :caption: Example:

    *output, type=meshdatatofile, name=meshdata
        filename=myMesh.inc
"""


@dataclass(frozen=True)
class MeshDataToFileSchema:
    """The options this output manager accepts, owned by this module and never mutated from
    outside it.
    """

    filename: str | None = schemaField(description="Name of file for writing output.", dtype=str, default=None)


class OutputManager(OutputManagerBase):
    """Simple status file writer for step, incrementation and iterations"""

    identification = "Meshdatatofile"
    printTemplate = "{:}, {:}: {:}"

    #: Option schema for this output manager, per OptionSchemaProvider.
    schema = MeshDataToFileSchema

    def __init__(
        self,
        name: str,
        model: FEModel,
        fieldOutputController: FieldOutputController,
        journal: Journal,
        plotter: Plotter,
        *,
        configuration: MeshDataToFileSchema = MeshDataToFileSchema(),
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
            The options this output manager accepts; defaults to all-defaults.
        """
        self.name = name
        self.journal = journal
        self.model = model

        # An unspecified filename (schema default None) resolves to "<name>_mesh.inc", which
        # cannot be expressed as a static schema default since it depends on the instance's name.
        if configuration.filename is not None:
            self.filename = configuration.filename
        else:
            self.filename = f"{name}_mesh.inc"

        self.writeMeshDataToFile(self.model)

    def initializeJob(self):
        pass

    def initializeStep(self, step):
        pass

    def finalizeIncrement(self, statusInfoDict: dict = {}, **kwargs):
        pass

    def finalizeFailedIncrement(self, statusInfoDict: dict = {}, **kwargs):
        pass

    def finalizeStep(self):
        pass

    def finalizeJob(
        self,
    ):
        pass

    def writeMeshDataToFile(self, model: FEModel):
        """Write the mesh data to a file.

        Parameters
        ----------
        model
            The model dictionary containing the mesh data.
        """

        with open(self.filename, "w+") as f:
            # write nodes
            f.write("*NODE\n")
            for nodeID in model.nodes:
                f.write("{:},".format(nodeID))
                [f.write(" {:},".format(coord)) for coord in model.nodes[nodeID].coordinates]
                f.write("\n")

            # write elements
            # first, get all element types
            elementTypes = set()
            [elementTypes.add(element.elType) for element in model.elements.values()]

            for elementType in elementTypes:
                f.write("*ELEMENT, TYPE={:}\n".format(elementType))
                for elementID in model.elements:
                    if model.elements[elementID].elType == elementType:
                        f.write("{:>5},".format(elementID))
                        [f.write(" {:>5},".format(node.label)) for node in model.elements[elementID].nodes]
                        f.write("\n")

            # write node sets
            for nodeSetName in model.nodeSets:
                f.write("*NSET, NSET={:}\n".format(nodeSetName))
                counter = 0
                for node in model.nodeSets[nodeSetName]:
                    counter += 1
                    f.write(" {:>5},".format(node.label))
                    if counter % 16 == 0:
                        f.write("\n")
                f.write("\n")

            # write element sets
            for elementSetName in model.elementSets:
                f.write("*ELSET, ELSET={:}\n".format(elementSetName))
                counter = 0
                for element in model.elementSets[elementSetName]:
                    counter += 1
                    f.write(" {:>5},".format(element.elNumber))
                    if counter % 16 == 0:
                        f.write("\n")
                f.write("\n")

            # write surfaces
            for surfaceName in model.surfaces:
                f.write("*SURFACE, TYPE=ELEMENT, NAME={:}\n".format(surfaceName))
                for faceID in model.surfaces[surfaceName].keys():
                    elset = model.surfaces[surfaceName][faceID]
                    f.write("{elset}, {faceID}\n".format(elset=elset.name, faceID=faceID))
