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

from edelweissfe.utils.caseinsensitivedict import CaseInsensitiveDict
from edelweissfe.utils.inputlanguage import InputLanguage, Module
from edelweissfe.utils.misc import (
    caseInsensitiveKwargsChecker,
    castKwargsValuesAndAddDefaults,
)

module = Module("cubit", "Interface to Cubit. Generate mesh using Cubit .jou files.")

inputLanguage = InputLanguage()

keyword = "modelGenerator"
if keyword in inputLanguage:
    inputLanguage[keyword].addModule(module)

module.addOptionalArg("cubitCmd", "Cubit executable.", str, "cubit")
module.addRequiredArg("jouFile", "Path to Cubit journal (.jou) file.", str)
module.addOptionalArg("outFile", "Path to output mesh file.", str, "mesh.inc")
module.addOptionalArg("APREPROVars", "APREPRO variables as comma-separated <key>=<value> pairs.", str, None)
module.addOptionalArg("overwrite", "Overwrite existing output files.", bool, True)
module.addOptionalArg("runCubit", "Run Cubit GUI for debugging purposes.", bool, False)
module.addOptionalArg("silent", "Hide Cubit output.", bool, False)

module.addOptionalArg("elType", "Specify element type for all sections.", str, None)
module.addOptionalArg(
    "elTypePerBlock", "Specify element type per block as comma-separated <key>=<value> pairs.", str, None
)
module.addOptionalArg("elProvider", "Element provider.", str, None)

documentation = [module]


@caseInsensitiveKwargsChecker([kw.name for kw in module.requiredArgs], [kw.name for kw in module.optionalArgs])
@castKwargsValuesAndAddDefaults(module)
def generateModelData(generatorDefinition, model, journal, *args, **kwargs):
    from edelweissfe.generators.abqmodelconstructor import AbqModelConstructor
    from edelweissfe.utils.inputfileparser import parseInputFile

    kwargs = CaseInsensitiveDict(kwargs)

    # options = generatorDefinition["datalines"]
    # options = convertLinesToStringDictionary(options)
    # name = generatorDefinition.get("name", "cubit")

    cubitCmd = kwargs["cubitCmd"]
    jouFile = kwargs["jouFile"]
    outFile = kwargs["outFile"]
    APREPROVars = kwargs["APREPROVars"]
    elType = kwargs["elType"]
    elTypePerBlock = kwargs["elTypePerBlock"]
    overwrite = kwargs["overwrite"]
    runCubit = kwargs["runCubit"]
    silent = kwargs["silent"]

    # getElementClass(options["elType"], options.get("elProvider", None))

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

    return model
