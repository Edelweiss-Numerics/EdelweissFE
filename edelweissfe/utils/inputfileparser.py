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
"""
Inputfileparser for inputfiles employing an Abaqus-like syntax.
"""

import textwrap
from os.path import dirname, join

from edelweissfe.utils.caseinsensitivedict import CaseInsensitiveDict
from edelweissfe.utils.inputlanguage import InputLanguage
from edelweissfe.utils.misc import (
    caseInsensitiveKwargsChecker,
    convertAssignmentsToCaseInsensitiveStringDictionary,
    splitLineAtCommas,
    strCaseCmp,
)


def parseKeywordLine(line, fileName):
    lineElements = splitLineAtCommas(line)

    keyword = lineElements[0]
    optionAssignments = lineElements[1:]

    options = convertAssignmentsToCaseInsensitiveStringDictionary(optionAssignments)

    kw = inputLanguage[keyword]

    @caseInsensitiveKwargsChecker([kw.name for kw in kw.requiredArgs], [kw.name for kw in kw.optionalArgs])
    def checkKeywordInput(*args, **kwargs):
        """this is a dummy function needed to apply kwargsChecker"""
        return

    try:
        checkKeywordInput(**options)
    except ValueError as e:
        e.args = (f"Error during parsing of keyword {keyword}: " + e.args[0],)
        raise e

    for optKey, optVal in options.items():
        try:
            options[optKey] = kw[optKey].dtype(optVal)
        except ValueError:
            raise ValueError(f"{keyword}, option {optKey}: cannot convert {optVal} to {kw[optKey].dtype}")
        # except Exception as e:
        #     raise e

    options["inputFile"] = fileName  # save also the filename of the original inputfile!

    if kw.expectsRequiredDatalines or kw.expectsOptionalDatalines:
        options["data"] = []

    return keyword, options


inputLanguage = InputLanguage()

kw = inputLanguage.addKeyword("*element", "definition of element(s)")
kw.addRequiredArg("type", "assign one of the types defined in the elementlibrary", str)
kw.addOptionalArg("elSet", "name of elSet to be created", str, None)
kw.addOptionalArg("provider", "provider (library) for the element type. Default: Marmot", str, "Marmot")
kw.addRequiredDatalines("Abaqus like element definition lines", "")

kw = inputLanguage.addKeyword("*elSet", "definition of an element set")
kw.addRequiredArg("elSet", "name", str)
kw.addOptionalArg("generate", "set True to generate from data line 1: start-element, end-element, step", bool, False)
kw.addRequiredDatalines("Abaqus like element set definition lines", "")

kw = inputLanguage.addKeyword("*node", "definition of nodes")
kw.addOptionalArg("nSet", "name of nSet to be created", str, None)
kw.addRequiredDatalines("Abaqus like node definition lines: label, x, [y], [z]", "")

kw = inputLanguage.addKeyword("*nSet", "definition of an element set")
kw.addRequiredArg("nSet", "name", str)
kw.addOptionalArg("generate", "set True to generate from data line 1: start-node, end-node, step", bool, False)
kw.addRequiredDatalines("Abaqus like node set definition lines", "")

kw = inputLanguage.addKeyword("*surface", "definition of surface set")
kw.addRequiredArg("name", "name", str)
kw.addRequiredArg("type", "type of surface (currently 'element' only)", str)
kw.addRequiredDatalines("Abaqus like definition. Type 'element': elSet, faceID", "")

kw = inputLanguage.addKeyword("*section", "definition of a section")
kw.addRequiredArg("name", "name", str)
kw.addRequiredArg("material", "associated id of defined material", str)
kw.addRequiredArg("type", "type of the section", str)
kw.addRequiredDatalines("list of associated element sets", "")

kw.addOptionalArg("thickness", "associated element set", float, 1.0)

kw = inputLanguage.addKeyword("*material", "definition of a material")
kw.addRequiredArg("name", "name of material", str)
kw.addRequiredArg("id", "id of material", str)
kw.addRequiredDatalines("material properties", "")

# kw.addOptionalArg("statevars", , , None)

kw = inputLanguage.addKeyword("*fieldOutput", "define fieldoutput, which is used by outputmanagers")
kw.addRequiredDatalines("definition lines for the output module", "")

kw = inputLanguage.addKeyword("*analyticalField", "define an analytical field")
kw.addRequiredArg("name", "name of analytical field", str)
kw.addRequiredArg("type", "type of analytical field", str)
kw.addRequiredDatalines("definition lines", "")

kw = inputLanguage.addKeyword("*output", "define an output module")
kw.addOptionalArg("name", "name of output manager", str, None)
kw.addRequiredArg("type", "output module", str)
kw.addRequiredDatalines("definition lines for the output module", "")

kw = inputLanguage.addKeyword("*job", "definition of an analysis job")
kw.addRequiredArg("domain", "define spatial domain: 1d, 2d, 3d", str)

kw.addOptionalArg("startTime", "(optional) start time of job", float, 0.0)
kw.addOptionalArg("name", "(optional) name of job, standard = defaultJob", str, None)
kw.addOptionalArg("solver", "(deprecated) define the solver to be used", str, None)

kw = inputLanguage.addKeyword("*solver", "define a solver")
kw.addRequiredArg("name", "solver name", str)
kw.addRequiredArg("solver", "solver type", str)
kw.addOptionalDatalines("define options which are passed to the respective solver instance.", "")

kw = inputLanguage.addKeyword("*step", "define steps")
kw.addRequiredArg("solver", "solver to be used", str)

kw.addOptionalArg("stepLength", "time period of step", float, None)
kw.addOptionalArg("startInc", "size of the start increment", float, None)
kw.addOptionalArg("maxInc", "maximum size of increment", float, None)
kw.addOptionalArg("minInc", "minimum size of increment", float, None)
kw.addOptionalArg("maxNumInc", "maximum number of increments", int, None)
kw.addOptionalArg("maxIter", "maximum number of iterations", int, None)
kw.addOptionalArg("type", "define step type, default = AdaptiveStep", str, None)
kw.addOptionalArg("criticalIter", "maximum number of iterations to prevent from increasing the increment", int, None)
kw.addOptionalDatalines("define step actions, which are handled by the corresponding stepaction modules", "")

kw = inputLanguage.addKeyword("*updateConfiguration", "update a configuration")
kw.addRequiredArg("configuration", "name of configuration to be changed", str)
kw.addRequiredDatalines("keyword arguments", "")

kw = inputLanguage.addKeyword("*modelGenerator", "define a model generator, loaded from a module")
kw.addRequiredArg("name", "name of the generator", str)
kw.addRequiredArg("generator", "name of generator module", str)
kw.addOptionalArg(
    "executeAfterManualGeneration", "Delay the execution of the generator after model generation", bool, False
)
kw.addRequiredDatalines("keyword arguments", "")

kw = inputLanguage.addKeyword("*constraint", "define a constraint")
kw.addRequiredArg("type", "constraint type", str)
kw.addRequiredDatalines("definition of the constraint", "")
kw.addOptionalArg("name", "name of the constraint", str, None)

kw = inputLanguage.addKeyword("*configurePlots", "customize the figures and axes")
kw.addRequiredDatalines("key=value pairs for configuration of figures and axes", "")

kw = inputLanguage.addKeyword("*exportPlots", "export your figures")
kw.addRequiredDatalines("key=value pairs for exporting of figures and axes", "")

kw = inputLanguage.addKeyword("*include", "load contents of extra file")
kw.addRequiredArg("input", "path to file (use relative path to current .inp)", str)


def parseInputFile(
    fileName: str,
    currentKeyword: str = None,
    existingFileDict: CaseInsensitiveDict = None,
) -> CaseInsensitiveDict:
    """Parse an Abaqus-like input file to generate a dictionary with its content.

    Parameters
    ----------
    fileName
        The name of the file to parse.
    currentKeyword
        If nested parsing is performed by using ``*include``, this option tells which
        keyword is currently active.
    existingFileDict
        An existing dictionary to append. If Nonde, a new dictionary is created.

    Returns
    -------
    CaseInsensitiveDict
        The parsed input file.
    """

    if not existingFileDict:
        fileDict = CaseInsensitiveDict({kw.name: [] for kw in inputLanguage})
    else:
        fileDict = existingFileDict

    keyword = currentKeyword
    with open(fileName) as f:
        # filter out empty lines and comments
        lines = (line.strip() for line in f)
        lines = (line for line in lines if line and not line.startswith("**"))

        for line in lines:
            if line.startswith("*"):  # line is keywordline
                lastkeyword = keyword
                keyword, options = parseKeywordLine(line, fileName)

                # special treatment for *include:
                if strCaseCmp(keyword, "*include"):
                    includeFile = options["input"]
                    parseInputFile(
                        join(dirname(fileName), includeFile),
                        currentKeyword=lastkeyword,
                        existingFileDict=fileDict,
                    )
                    keyword = lastkeyword
                else:
                    fileDict[keyword].append(options)

            else:  # line is a dataline
                try:
                    assert (
                        inputLanguage[keyword].expectsOptionalDatalines
                        or inputLanguage[keyword].expectsRequiredDatalines
                    )
                except AssertionError:
                    raise ValueError(f"{keyword} expects no data lines")
                else:
                    fileDict[keyword][-1]["data"].append(line)

    return fileDict


def printKeywords():
    """Print the input file language set."""

    kwString = "    {:}    "
    kwDataString = "        {:22}{:20}"

    wrapper = textwrap.TextWrapper(width=80, replace_whitespace=False)
    for kw, (kwDoc, optiondict) in sorted(inputLanguage.items()):
        wrapper.initial_indent = kwString.format(str(kw))
        wrapper.subsequent_indent = " " * len(wrapper.initial_indent)
        print(wrapper.fill(kwDoc))
        print("")
        for key in sorted(optiondict.keys()):
            optionName = key
            dType, description = optiondict[key]
            wrapper.initial_indent = kwDataString.format(str(optionName), dType)
            wrapper.subsequent_indent = " " * len(wrapper.initial_indent)
            print(wrapper.fill(description))
        print("\n")


def printKeywordsRST():
    """Print the input file language set in an RST conform format."""

    for kw, (kwDoc, optiondict) in sorted(inputLanguage.items()):
        print(".. list-table:: " + "``{:}`` : {:}".format(kw, kwDoc))
        print("    :width: 100%")
        print("    :widths: 25 25 40")
        print("    :header-rows: 1")
        print(" ")
        print("    * - Option")
        print("      - Type")
        print("      - Description")
        for key in sorted(optiondict.keys()):
            optionName = key
            dType, description = optiondict[key]

            print("    * - ``{:}``".format(optionName))
            print("      - ``{:}``".format(dType))
            print("      - " + description)
        print(" ")
