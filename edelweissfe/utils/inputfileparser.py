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
from edelweissfe.utils.inputlanguage import (
    InputLanguage,
    keywordIdentifier,
    moduleLevelKeywordIdentifier,
)
from edelweissfe.utils.misc import (
    caseInsensitiveKwargsChecker,
    castKwargsValuesAndAddDefaults,
    convertAssignmentsToCaseInsensitiveStringDictionary,
    splitLineAtCommas,
    strCaseCmp,
    typeString,
)


def parseKeywordLine(line, fileName):
    lineElements = splitLineAtCommas(line.removeprefix(keywordIdentifier))

    keyword = lineElements[0]
    optionAssignments = lineElements[1:]

    try:
        options = convertAssignmentsToCaseInsensitiveStringDictionary(optionAssignments)
    except ValueError as e:
        e.args = (f"Error during parsing of keyword {keywordIdentifier}{keyword}: " + e.args[0],)
        raise e

    kw = inputLanguage[keyword]

    @castKwargsValuesAndAddDefaults(kw)
    def checkKeywordInput(*args, **kwargs):
        """this is a dummy function needed to apply kwargsChecker"""
        return CaseInsensitiveDict(kwargs)

    options = checkKeywordInput(**options)

    @caseInsensitiveKwargsChecker([kw.name for kw in kw.requiredArgs], [kw.name for kw in kw.optionalArgs])
    def checkKeywordInput(*args, **kwargs):
        """this is a dummy function needed to apply kwargsChecker"""
        return

    try:
        checkKeywordInput(**options)
    except ValueError as e:
        try:
            module = None  # in some cases, module-specific kwArgs arguments can be given in the keyword line
            if strCaseCmp(kw.name, "section") and strCaseCmp(options.get("type", ""), "plane"):
                module = kw.getModule("plane")

            if module is not None:
                addRequired = [kw.name for kw in module.requiredArgs]
                addOptional = [kw.name for kw in module.optionalArgs]

                @caseInsensitiveKwargsChecker(
                    [kw.name for kw in kw.requiredArgs], [kw.name for kw in kw.optionalArgs] + addOptional + addRequired
                )
                def checkKeywordInput(*args, **kwargs):
                    """this is a dummy function needed to apply kwargsChecker"""
                    return

                checkKeywordInput(**options)

            else:
                e.args = (f"Error during parsing of keyword {keywordIdentifier}{keyword}: " + e.args[0],)
                raise e
        except ValueError as e:
            raise e

    options["inputFile"] = fileName  # save also the filename of the original inputfile!

    options["datalines"] = []

    return keyword, options


def parseModuleKeywordLine(line, fileName, topLevelKeyword, topLevelOptions, fileDict):
    lineElements = splitLineAtCommas(line.removeprefix(moduleLevelKeywordIdentifier))

    keyword = lineElements[0]
    optionAssignments = lineElements[1:]

    try:
        options = convertAssignmentsToCaseInsensitiveStringDictionary(optionAssignments)
    except ValueError as e:
        e.args = (f"Error during parsing of keyword {keywordIdentifier}{keyword}: " + e.args[0],)
        raise e

    if "type" in inputLanguage[topLevelKeyword].argNames:
        module = inputLanguage[topLevelKeyword].getModule(
            inputLanguage[topLevelKeyword].getArg("type").getValueFromKwargs(topLevelOptions)
        )
    else:
        module = inputLanguage[topLevelKeyword].getModule(topLevelKeyword)

    kw = module.getKeyword(keyword)

    @caseInsensitiveKwargsChecker([kw.name for kw in kw.requiredArgs], [kw.name for kw in kw.optionalArgs])
    @castKwargsValuesAndAddDefaults(kw)
    def checkKeywordInput(*args, **kwargs):
        """this is a dummy function needed to apply kwargsChecker"""
        return CaseInsensitiveDict(kwargs)

    try:
        options = checkKeywordInput(**options)
    except ValueError as e:
        if (  # stepaction update
            topLevelKeyword == "step"
            and module.name == "adaptive"
            and kw.name
            in [
                # "bodyforce",
                "dirichlet",
                "distributedload",
                # "geostatic",
                "nodeforces",
            ]  # these stepactions can be updated by repeating the module level keyword and using the same name as previously defined
            and "name" in options
            and options["name"].casefold()
            in [  # check if a step action with the same name already exists
                item["name"].casefold()
                for step in fileDict["step"]
                if keyword in step["moduleoptions"]
                for item in step["moduleoptions"][keyword]
            ]
        ):
            kw = module.getKeyword("update" + keyword)

            @caseInsensitiveKwargsChecker([kw.name for kw in kw.requiredArgs], [kw.name for kw in kw.optionalArgs])
            @castKwargsValuesAndAddDefaults(kw)
            def checkUpdateKeywordInput(*args, **kwargs):
                """this is a dummy function needed to apply kwargsChecker"""
                return CaseInsensitiveDict(kwargs)

            try:
                options = checkUpdateKeywordInput(**options)
            except ValueError as e2:
                e2.args = (
                    f"Error during updating stepaction {moduleLevelKeywordIdentifier}{keyword}, name={options['name']}: "
                    + e2.args[0],
                )
                raise e2
        else:
            e.args = (
                f"Error during parsing of module level keyword {moduleLevelKeywordIdentifier}{keyword}: " + e.args[0],
            )
            raise e

    for opt in kw.optionalArgs:
        if opt.name not in options:
            options[opt.name] = opt.default

    options["inputFile"] = fileName  # save also the filename of the original inputfile!

    if kw.expectsRequiredDatalines or kw.expectsOptionalDatalines:
        options["datalines"] = []

    return keyword, options


inputLanguage = InputLanguage()

kw = inputLanguage.addKeyword("element", "definition of element(s)")
kw.addRequiredArg("type", "assign one of the types defined in the elementlibrary", str)
kw.addOptionalArg("elSet", "name of elSet to be created", str, None)
kw.addOptionalArg(
    "provider",
    "provider (library) for the element type. Default: Marmot",
    str,
    "Marmot",
)
kw.addRequiredDatalines("Abaqus like element definition lines", "")

kw = inputLanguage.addKeyword("elSet", "definition of an element set")
kw.addRequiredArg("elSet", "name", str)
kw.addOptionalArg(
    "generate",
    "set True to generate from data line 1: start-element, end-element, step",
    bool,
    False,
)
kw.addRequiredDatalines("Abaqus like element set definition lines", "")

kw = inputLanguage.addKeyword("node", "definition of nodes")
kw.addOptionalArg("nSet", "name of nSet to be created", str, None)
kw.addRequiredDatalines("Abaqus like node definition lines: label, x, [y], [z]", "")

kw = inputLanguage.addKeyword("nSet", "definition of an element set")
kw.addRequiredArg("nSet", "name", str)
kw.addOptionalArg(
    "generate",
    "set True to generate from data line 1: start-node, end-node, step",
    bool,
    False,
)
kw.addRequiredDatalines("Abaqus like node set definition lines", "")

kw = inputLanguage.addKeyword("surface", "definition of surface set")
kw.addRequiredArg("name", "name", str)
kw.addRequiredArg("type", "type of surface (currently 'element' only)", str)
kw.addRequiredDatalines("Abaqus like definition. Type 'element': elSet, faceID", "")

"""
*section
"""
kw = inputLanguage.addKeyword("section", "definition of a section")
kw.addRequiredArg("name", "name", str)
kw.addRequiredArg("material", "associated id of defined material", str)
kw.addRequiredArg("type", "type of the section", str)
kw.addRequiredDatalines("list of associated element sets", "")

# kw.addOptionalArg("thickness", "associated element thickness", float, 1.0)
# kw.addOptionalArg("density", "associated element density", float, 1.0)

# isort: off
from edelweissfe.sections.solid import inputLanguage  # noqa: F811,E402
from edelweissfe.sections.plane import inputLanguage  # noqa: F811,E402

# isort: on

"""
*material
"""
kw = inputLanguage.addKeyword("material", "definition of a material")
kw.addRequiredArg("name", "name of material", str)
kw.addRequiredArg("id", "id of material", str)
kw.addOptionalArg("provider", "material provider", str, "marmotmaterial")
kw.addRequiredDatalines("material properties", "")
# kw.addOptionalArg("statevars", , , None)

"""
*advancedmaterial
"""
kw = inputLanguage.addKeyword("advancedmaterial", "definition of an advanced material")
kw.addRequiredArg("name", "name of material", str)
kw.addRequiredArg("id", "id of material", str)
kw.addOptionalArg("provider", "material provider", str, "marmotmaterial")
kw.addRequiredDatalines("material properties", "")

"""
*fieldOutput
"""
kw = inputLanguage.addKeyword("fieldOutput", "define fieldoutput, which is used by outputmanagers")

# isort: off
from edelweissfe.utils.fieldoutput import inputLanguage  # noqa: F811,E402

# isort: on

"""
*analyticalField
"""
kw = inputLanguage.addKeyword("analyticalField", "define an analytical field")
kw.addRequiredArg("name", "name of analytical field", str)
kw.addRequiredArg("type", "type of analytical field", str)
# kw.addRequiredDatalines("definition lines", "")

# isort: off
from edelweissfe.analyticalfields.randomscalar import inputLanguage  # noqa: F811,E402
from edelweissfe.analyticalfields.fromvtk import inputLanguage  # noqa: F811,E402
from edelweissfe.analyticalfields.scalarexpression import inputLanguage  # noqa: F811,E402

# isort: on

"""
*job
"""
kw = inputLanguage.addKeyword("job", "definition of an analysis job")
kw.addRequiredArg("domain", "define spatial domain: 1d, 2d, 3d", str)
kw.addOptionalArg("startTime", "(optional) start time of job", float, 0.0)
kw.addOptionalArg("name", "Name of job.", str, "defaultJob")
kw.addOptionalArg("solver", "(deprecated) define the solver to be used", str, "NIST")

"""
*solver
"""
kw = inputLanguage.addKeyword("solver", "define a solver")
kw.addRequiredArg("name", "solver name", str)
kw.addRequiredArg("solver", "solver type", str)
kw.addOptionalDatalines("define options which are passed to the respective solver instance.", "")

"""
*step
"""
kw = inputLanguage.addKeyword("step", "define steps")
kw.addRequiredArg("solver", "solver to be used", str)
kw.addOptionalArg("type", "step type", str, "adaptive")

# isort: off
from edelweissfe.steps.adaptivestep import inputLanguage  # noqa: F811,E402
from edelweissfe.steps.adaptivestepforexplicitsimulations import inputLanguage  # noqa: F811,E402

from edelweissfe.stepactions.bodyforce import inputLanguage  # noqa: F811,E402
from edelweissfe.stepactions.changematerialproperty import inputLanguage  # noqa: F811,E402
from edelweissfe.stepactions.dirichlet import inputLanguage  # noqa: F811,E402
from edelweissfe.stepactions.distributedload import inputLanguage  # noqa: F811,E402
from edelweissfe.stepactions.geostatic import inputLanguage  # noqa: F811,E402
from edelweissfe.stepactions.indirectcontractioncontrol import inputLanguage  # noqa: F811,E402
from edelweissfe.stepactions.indirectcontrol import inputLanguage  # noqa: F811,E402
from edelweissfe.stepactions.initializematerial import inputLanguage  # noqa: F811,E402
from edelweissfe.stepactions.modelupdate import inputLanguage  # noqa: F811,E402
from edelweissfe.stepactions.nodeforces import inputLanguage  # noqa: F811,E402
from edelweissfe.stepactions.setfield import inputLanguage  # noqa: F811,E402
from edelweissfe.stepactions.setinitialconditions import inputLanguage  # noqa: F811,E402

from edelweissfe.stepactions.options import inputLanguage  # noqa: F811,E402
from edelweissfe.solvers.nonlinearimplicitstatic import inputLanguage  # noqa: F811,E402
from edelweissfe.solvers.nonlinearimplicitstaticparallelarclength import inputLanguage  # noqa: F811,E402

# isort: on

"""
*output
"""
kw = inputLanguage.addKeyword("output", "define an output module")
kw.addRequiredArg("type", "output module", str)
kw.addOptionalArg("name", "name of output manager", str, None)
# kw.addOptionalDatalines("definition lines", "")

# isort: off
from edelweissfe.outputmanagers.computetimemonitor import inputLanguage  # noqa: F811,E402
from edelweissfe.outputmanagers.conditionalstop import inputLanguage  # noqa: F811,E402
from edelweissfe.outputmanagers.ensight import inputLanguage  # noqa: F811,E402
from edelweissfe.outputmanagers.fractureenergyintegrator import inputLanguage  # noqa: F811,E402
from edelweissfe.outputmanagers.meshdatatofile import inputLanguage  # noqa: F811,E402
from edelweissfe.outputmanagers.meshplot import inputLanguage  # noqa: F811,E402
from edelweissfe.outputmanagers.monitor import inputLanguage  # noqa: F811,E402
from edelweissfe.outputmanagers.plotalongpath import inputLanguage  # noqa: F811,E402
from edelweissfe.outputmanagers.statusfile import inputLanguage  # noqa: F811,E402
from edelweissfe.outputmanagers.timemonitor import inputLanguage  # noqa: F811,E402

# isort: on

"""
*updateConfiguration
"""
kw = inputLanguage.addKeyword("updateConfiguration", "update a configuration")
kw.addRequiredArg("configuration", "name of configuration to be changed", str)
kw.addRequiredDatalines("keyword arguments", "")

"""
*modelGenerator
"""
kw = inputLanguage.addKeyword("modelGenerator", "define a model generator, loaded from a module")
kw.addRequiredArg("name", "name of the generator", str)
kw.addRequiredArg("generator", "name of generator module", str)
kw.addOptionalArg(
    "executeAfterManualGeneration",
    "Delay the execution of the generator after model generation",
    bool,
    False,
)
# kw.addRequiredDatalines("keyword arguments", "")

# isort: off
# from edelweissfe.generators.abqmodelconstructor import inputLanguage  # noqa: F811,E402
from edelweissfe.generators.boxgen import inputLanguage  # noqa: F811,E402
from edelweissfe.generators.cubit import inputLanguage  # noqa: F811,E402
from edelweissfe.generators.executepythoncode import inputLanguage  # noqa: F811,E402
from edelweissfe.generators.findclosestnode import inputLanguage  # noqa: F811,E402
from edelweissfe.generators.pipegen import inputLanguage  # noqa: F811,E402
from edelweissfe.generators.planerectquad import inputLanguage  # noqa: F811,E402
from edelweissfe.generators.cuboidlatticegenerator import inputLanguage  # noqa: F811,E402

# isort: on

"""
*constraint
"""
kw = inputLanguage.addKeyword("constraint", "define a constraint")
kw.addRequiredArg("type", "constraint type", str)
kw.addRequiredDatalines("definition of the constraint", "")
kw.addOptionalArg("name", "name of the constraint", str, None)

# isort: off
from edelweissfe.constraints.equalvaluelagrangian import inputLanguage  # noqa: F811,E402
from edelweissfe.constraints.equalvaluepenalty import inputLanguage  # noqa: F811,E402
from edelweissfe.constraints.linearizedrigidbody import inputLanguage  # noqa: F811,E402
from edelweissfe.constraints.penaltyindirectcontrol import inputLanguage  # noqa: F811,E402
from edelweissfe.constraints.rigidbody import inputLanguage  # noqa: F811,E402
from edelweissfe.constraints.directionalspringpenalty import inputLanguage  # noqa: F811,E402
from edelweissfe.constraints.nodetorigidsurfacepenalty import inputLanguage  # noqa: F811,E402

# isort: on

"""
*configurePlots
"""
kw = inputLanguage.addKeyword("configurePlots", "customize the figures and axes")
kw.addRequiredDatalines("key=value pairs for configuration of figures and axes", "")

"""
*exportPlots
"""
kw = inputLanguage.addKeyword("exportPlots", "export your figures")
kw.addRequiredDatalines("key=value pairs for exporting of figures and axes", "")

"""
*include
"""
kw = inputLanguage.addKeyword("include", "load contents of extra file")
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
                lastKeyword = keyword
                keyword, options = parseKeywordLine(line, fileName)
                options["moduleOptions"] = dict()

                # special treatment for *include:
                if strCaseCmp(keyword, "include"):
                    includeFile = options["input"]
                    parseInputFile(
                        join(dirname(fileName), includeFile),
                        currentKeyword=lastKeyword,
                        existingFileDict=fileDict,
                    )
                    keyword = lastKeyword
                else:
                    fileDict[keyword].append(options)

            elif line.startswith(moduleLevelKeywordIdentifier):  # line is a module level keyword line
                moduleKeyword, moduleOptions = parseModuleKeywordLine(line, fileName, keyword, options, fileDict)

                if moduleKeyword in fileDict[keyword][-1]["moduleOptions"]:
                    fileDict[keyword][-1]["moduleOptions"][moduleKeyword].append(moduleOptions)
                else:
                    fileDict[keyword][-1]["moduleOptions"].update({moduleKeyword: [moduleOptions]})

            # else splitLineAtCommas(line)[]  # line is a module level keyword line

            else:  # line is a data line
                # # module kw parsing backward compatibility
                # try:
                #     module = inputLanguage[keyword].getModule(options["type"])
                # except:
                #     module = None
                # moduleKeyword = ">>"+splitLineAtCommas(line)[0]
                #
                # if module and moduleKeyword in [kw.name for kw in module.keywords]:
                #     moduleKeyword, moduleOptions = parseModuleKeywordLine(">>"+line, fileName, keyword, options)
                #
                #     if moduleKeyword in fileDict[keyword][-1]["moduleOptions"]:
                #         fileDict[keyword][-1]["moduleOptions"][moduleKeyword].append(moduleOptions)
                #     else:
                #         fileDict[keyword][-1]["moduleOptions"] = {moduleKeyword: [moduleOptions]}
                #     continue
                # # module kw parsing backward compatibility

                inputFileKeyword = inputLanguage[keyword]
                if not inputFileKeyword.modules:  # for keywords with no modules:
                    if not (
                        inputLanguage[keyword].expectsOptionalDatalines
                        or inputLanguage[keyword].expectsRequiredDatalines
                    ):
                        raise ValueError(
                            f"Error during parsing of keyword {keywordIdentifier}{keyword}: {keywordIdentifier}{keyword} expects no data lines"
                        )
                    fileDict[keyword][-1]["datalines"].append(line)
                else:  # for keywords with modules
                    # module = inputLanguage[keyword].getModule(options["type"] if "type" in options else keyword)
                    if "type" in inputLanguage[keyword].argNames:
                        module = inputLanguage[keyword].getModule(
                            inputLanguage[keyword].getArg("type").getValueFromKwargs(options)
                        )
                    elif "generator" in inputLanguage[keyword].argNames:
                        module = inputLanguage[keyword].getModule(
                            inputLanguage[keyword].getArg("generator").getValueFromKwargs(options)
                        )
                    else:
                        module = inputLanguage[keyword].getModule(keyword)
                    if not (module.expectsOptionalDatalines or module.expectsRequiredDatalines):
                        raise ValueError(
                            f"Error during parsing of keyword {keywordIdentifier}{keyword}: {module} expects no data lines"
                        )
                    fileDict[keyword][-1]["datalines"].append(line)

    return fileDict


def printKeywords():
    """Print the input file language set."""

    kwString = "    {:}    "
    kwDataString = "        {:22}{:20}"

    wrapper = textwrap.TextWrapper(width=80, replace_whitespace=False)
    # for kw, (kwDoc, optiondict) in sorted(inputLanguage.items()):
    for kw in inputLanguage:
        wrapper.initial_indent = kwString.format(kw.name)
        wrapper.subsequent_indent = " " * len(wrapper.initial_indent)
        print(wrapper.fill(kw.description))
        # print("")
        for arg in kw.requiredArgs:
            wrapper.initial_indent = kwDataString.format(arg.name, typeString(arg.dtype))
            wrapper.subsequent_indent = " " * len(wrapper.initial_indent)
            print(wrapper.fill(arg.description))
        for arg in kw.optionalArgs:
            wrapper.initial_indent = kwDataString.format(arg.name, typeString(arg.dtype))
            wrapper.subsequent_indent = " " * len(wrapper.initial_indent)
            print(wrapper.fill(arg.description))
        print("\n")
