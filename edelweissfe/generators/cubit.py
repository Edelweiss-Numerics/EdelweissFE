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
#  Paul Hofer Paul.Hofer@uibk.ac.at
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
# Created on 2022-03-08

# @author: Paul Hofer
"""
Interface to Cubit. Generate mesh using Cubit .jou files.
"""

import os
import shlex
from dataclasses import dataclass

from edelweissfe.generators.base.generatorbase import GeneratorBase
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.utils.schema import schemaField


@dataclass(frozen=True)
class CubitSchema:
    """L2: the options this generator accepts, owned by this module and never mutated from
    outside it.

    ``jouFile`` is declared ``required=True`` explicitly, but
    is still given a ``default=None`` so the schema remains constructible for the L1 constructor's
    default argument.
    """

    cubitCmd: str = schemaField(description="Cubit executable.", dtype=str, default="cubit")
    jouFile: str | None = schemaField(
        description="Path to Cubit journal (.jou) file.", dtype=str, default=None, required=True
    )
    outFile: str = schemaField(description="Path to output mesh file.", dtype=str, default="mesh.inc")
    APREPROVars: str | None = schemaField(
        description="APREPRO variables as comma-separated <key>=<value> pairs.", dtype=str, default=None
    )
    overwrite: bool = schemaField(description="Overwrite existing output files.", dtype=bool, default=True)
    runCubit: bool = schemaField(description="Run Cubit GUI for debugging purposes.", dtype=bool, default=False)
    silent: bool = schemaField(description="Hide Cubit output.", dtype=bool, default=False)
    elType: str | None = schemaField(description="Specify element type for all sections.", dtype=str, default=None)
    elTypePerBlock: str | None = schemaField(
        description="Specify element type per block as comma-separated <key>=<value> pairs.",
        dtype=str,
        default=None,
    )
    elProvider: str | None = schemaField(description="Element provider.", dtype=str, default=None)


class Generator(GeneratorBase):
    """Interface to Cubit. Generate mesh using Cubit .jou files."""

    #: L2 schema declared for the L3 registry, per OptionSchemaProvider.
    schema = CubitSchema

    def __init__(self, name: str, model: FEModel, journal: Journal, *, configuration: CubitSchema = CubitSchema()):
        """L1: constructible standalone, with no parser involvement.
        Populates ``model`` directly; construction *is* the generation.

        Parameters
        ----------
        name
            Unused: this generator names no sets of its own; sections/constraints created from the
            Cubit-exported mesh come from that mesh's own ``.inc`` definitions.
        model
            The model tree to populate. Mutated in place.
        journal
            The journal object for logging.
        configuration
            The options this generator accepts; ``jouFile`` is still required, see
            :class:`CubitSchema`.
        """
        from edelweissfe.generators.abqmodelconstructor import AbqModelConstructor
        from edelweissfe.utils.inputfileparser import parseInputFile

        cubitCmd = configuration.cubitCmd
        jouFile = configuration.jouFile
        outFile = configuration.outFile
        APREPROVars = configuration.APREPROVars
        elType = configuration.elType
        elTypePerBlock = configuration.elTypePerBlock
        overwrite = configuration.overwrite
        runCubit = configuration.runCubit
        silent = configuration.silent

        generate = False
        if not os.path.exists(outFile) or overwrite:
            generate = True

        if generate:
            cubitOptns = []
            cubitOptns.append("-information off")
            cubitOptns.append("-nojournal")

            if not runCubit:
                cubitOptns.append("-batch")
                cubitOptns.append("-nographics")

            if APREPROVars:
                varStr = ""
                s = shlex.shlex(APREPROVars.replace(" ", ""), posix=True)
                s.whitespace_split = True
                s.whitespace = ","
                varDict = dict(item.split("=", 1) for item in s)

                for key, value in varDict.items():
                    varStr += "{}={} ".format(key, value)
                cubitOptns.append(varStr)

            cubitOptns.append(jouFile)

            optnStr = " ".join(cubitOptns)
            cmd = " ".join([cubitCmd, optnStr])

            exportFile = "./exportAbaqus.jou"
            with open(exportFile, "w+") as f:
                f.write('export abaqus "{}" partial overwrite\n'.format(outFile))
            cmd = " ".join([cmd, exportFile])

            if silent:
                cmd = " ".join([cmd, "> /dev/null"])

            os.system(cmd)
            os.remove(exportFile)

        fileDict = parseInputFile(outFile)

        if elType:
            for elDef in fileDict["element"]:
                elDef["type"] = elType

        if elTypePerBlock:
            s = shlex.shlex(elTypePerBlock.replace(" ", ""), posix=True)
            s.whitespace_split = True
            s.whitespace = ","
            elDict = dict(item.split("=", 1) for item in s)
            for elDef in fileDict["element"]:
                elSet = elDef["elset"]
                elDef["type"] = elDict[elSet]

        abqModelConstructor = AbqModelConstructor(journal)
        model = abqModelConstructor.createGeometryFromInputFile(model, fileDict)
        model = abqModelConstructor.createSectionsFromInputFile(model, fileDict)
        model = abqModelConstructor.createConstraintsFromInputFile(model, fileDict)
