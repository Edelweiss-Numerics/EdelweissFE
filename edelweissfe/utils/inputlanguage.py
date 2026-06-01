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

from edelweissfe.utils.caseinsensitivedict import CaseInsensitiveDict
from edelweissfe.utils.misc import (
    caseInsensitiveKwargsChecker,
    convertAssignmentsToCaseInsensitiveStringDictionary,
    convertAssignmentsToStringDictionary,
    findSimilarString,
    splitLineAtCommas,
    strtobool,
)

keywordIdentifier = "*"
moduleLevelKeywordIdentifier = ">>"

indent0 = " " * 0
indent1 = " " * 2
indent2 = " " * 4
indent3 = " " * 6
indent4 = " " * 8


def singleton(class_):
    """class decorated with this function will only be instantiated once"""
    instances = {}

    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]

    return getinstance


@singleton
class InputLanguage:
    def __init__(self):
        self.keywords = []

        return

    def __contains__(self, item) -> bool:
        return item.casefold() in [kw.name.casefold() for kw in self.keywords]

    def __repr__(self) -> str:
        return str(self.keywords)

    def __getitem__(self, keyword: str):
        casefoldedKeywords = [kw.name.casefold() for kw in self.keywords]
        try:
            idx = casefoldedKeywords.index(keyword.casefold())
            return self.keywords[idx]
        except ValueError:
            similarKeyword = findSimilarString(keyword, [kw.name for kw in self.keywords])
            raise ValueError(
                f"{keywordIdentifier}{keyword} is not a valid keyword. Did you mean {keywordIdentifier}{similarKeyword}?"
            )

    def __iter__(self):
        return self.keywords.__iter__()

    def addKeyword(self, name: str, description: str):
        kw = InputFileKeyword(name, description)
        self.keywords.append(kw)
        return kw


class InputFileKeyword:
    def __init__(self, name, description):
        self.name = name
        self.description = description
        self.requiredArgs = []
        self.optionalArgs = []
        self.dtype = []

        self.expectsDatalines = False
        self.datalines = None

        self.expectsRequiredDatalines = False
        self.requiredDatalines = None

        self.expectsOptionalDatalines = False
        self.optionalDatalines = None

        self.modules = []

        return

    def addModule(self, module):
        self.modules.append(module)

        return

    def addModuleHelper(self, name: str, description: str):
        module = Module(name, description)
        self.addModule(module)

        return module

    @property
    def args(self):
        return self.requiredArgs + self.optionalArgs

    @property
    def argNames(self):
        return [arg.name for arg in self.args]

    def getArg(self, arg: str):
        casefoldedArgs = [arg_.name.casefold() for arg_ in self.args]
        try:
            idx = casefoldedArgs.index(arg.casefold())
            return self.args[idx]
        except ValueError:
            similarArg = findSimilarString(arg, [arg_.name for arg_ in self.args])
            raise ValueError(f"{arg} is not a valid argument. Did you mean {similarArg}?")

    def __getitem__(self, arg: str):
        casefoldedArgs = [arg_.name.casefold() for arg_ in self.args]
        try:
            idx = casefoldedArgs.index(arg.casefold())
            return self.args[idx]
        except ValueError:
            similarKeyword = findSimilarString(arg, [arg_.name for arg_ in self.args])
            raise ValueError(
                f"{arg} is not a valid argument for {keywordIdentifier}{self.name}. Did you mean {similarKeyword}?"
            )

    def parseDatalines(self, datalines):
        args = []
        kwargs = CaseInsensitiveDict()

        if not isinstance(datalines, list):
            datalines = [datalines]

        datalineOptions = []
        for line in datalines:
            datalineOptions += splitLineAtCommas(line)

        for option in datalineOptions:
            if "=" in option:
                kwargs.update(convertAssignmentsToStringDictionary([option]))
            else:
                args.append(option)

        return args, kwargs

    # @property
    # def datalines(self):
    #     pass

    def getModule(self, module: str):
        casefoldedModules = [module_.name.casefold() for module_ in self.modules]
        try:
            idx = casefoldedModules.index(module.casefold())
            return self.modules[idx]
        except ValueError:
            similarModule = findSimilarString(module, [module_.name for module_ in self.modules])
            raise ValueError(
                f"{module} is not a valid argument for {keywordIdentifier}{self.name}. Did you mean {similarModule}?"
            )

    def __repr__(self) -> str:
        return f"< {self.name} >"

    def __doc__(self) -> str:
        lines = []
        lines.append(self.__repr__() + " " + self.description)
        if self.requiredArgs:
            lines.append(indent1 + "required arguments")
            for item in self.requiredArgs:
                lines.append(indent2 + item.__doc__())
        if self.optionalArgs:
            lines.append(indent1 + "optional arguments")
            for item in self.optionalArgs:
                lines.append(indent2 + item.__doc__())
        if self.optionalDatalines:
            lines.append(indent1 + "required datalines")
            lines.append(indent2 + self.requiredDatalines.__repr__())
        if self.optionalDatalines:
            lines.append(indent1 + "optional datalines")
            lines.append(indent2 + self.optionalDatalines.__repr__())
        return "\n".join(lines)

    def addRequiredArg(self, name: str, description: str, dataType: type):
        arg = KeywordArg(name, description, dataType)
        self.requiredArgs.append(arg)

        return arg

    def addOptionalArg(self, name: str, description: str, dataType: type, defaultValue):
        arg = OptionalKeywordArg(name, description, dataType, default=defaultValue)
        self.optionalArgs.append(arg)

        return arg

    def addRequiredDatalines(self, description: str, dtype: str):
        datalines = DataLines(description, dtype)

        self.expectsDatalines = True
        self.datalines = datalines

        self.expectsRequiredDatalines = True
        self.requiredDatalines = datalines

        return datalines

    def addOptionalDatalines(self, description: str, dtype: str):
        datalines = DataLines(description, dtype)

        self.expectsDatalines = True
        self.datalines = datalines

        self.expectsOptionalDatalines = True
        self.optionalDatalines = datalines

        return datalines


class Module:
    def __init__(self, name: str, description: str):
        # self.inputFileKeyword = inputFileKeyword

        self.name = name
        self.description = description

        self.requiredArgs = []
        self.optionalArgs = []

        self.requiredKeywords = []
        self.optionalKeywords = []

        # self.requiredDatalineKwArgs = []
        # self.optionalDatalineKwArgs = []

        self.expectsRequiredDatalines = False
        self.requiredDatalines = None

        self.expectsOptionalDatalines = False
        self.optionalDatalines = None

        return

    @property
    def args(self):
        return self.requiredArgs + self.optionalArgs

    @property
    def argNames(self):
        return [arg.name for arg in self.args]

    def getArg(self, arg: str):
        casefoldedArgs = [arg_.name.casefold() for arg_ in self.args]
        try:
            idx = casefoldedArgs.index(arg.casefold())
            return self.args[idx]
        except ValueError:
            similarArg = findSimilarString(arg, [arg_.name for arg_ in self.args])
            raise ValueError(f"{arg} is not a valid argument. Did you mean {similarArg}?")

    @property
    def keywords(self):
        return self.requiredKeywords + self.optionalKeywords

    def getKeyword(self, keyword: str):
        casefoldedKeywords = [keyword_.name.casefold() for keyword_ in self.keywords]
        try:
            idx = casefoldedKeywords.index(keyword.casefold())
            return self.keywords[idx]
        except ValueError:
            similarKeyword = findSimilarString(keyword, [keyword_.name for keyword_ in self.keywords])
            raise ValueError(f"{keyword} is not a valid argument. Did you mean {similarKeyword}?")

    def __getitem__(self, arg: str):
        casefoldedArgs = [arg_.name.casefold() for arg_ in self.args]
        try:
            idx = casefoldedArgs.index(arg.casefold())
            return self.args[idx]
        except ValueError:
            similarKeyword = findSimilarString(arg, [arg_.name for arg_ in self.args])
            raise ValueError(f"{arg} is not a valid argument. Did you mean {similarKeyword}?")

    def addRequiredArg(self, name: str, description: str, dataType: type):
        arg = KeywordArg(name, description, dataType)
        self.requiredArgs.append(arg)

        self.expectsRequiredDatalines = True

        return arg

    def addOptionalArg(self, name: str, description: str, dataType: type, defaultValue):
        arg = OptionalKeywordArg(name, description, dataType, default=defaultValue)
        self.optionalArgs.append(arg)

        self.expectsOptionalDatalines = True

        return arg

    def addRequiredKeyword(self, name: str, description: str):
        kw = InputFileKeyword(name, description)
        self.requiredKeywords.append(kw)

        return kw

    def addOptionalKeyword(self, name: str, description: str):
        kw = InputFileKeyword(name, description)
        self.optionalKeywords.append(kw)

        return kw

    def addRequiredDatalines(self, description: str, dtype: str):
        datalines = DataLines(description, dtype)

        self.expectsDatalines = True
        self.datalines = datalines

        self.expectsRequiredDatalines = True
        self.requiredDatalines = datalines

        return datalines

    def addOptionalDatalines(self, description: str, dtype: str):
        datalines = DataLines(description, dtype)

        self.expectsDatalines = True
        self.datalines = datalines

        self.expectsOptionalDatalines = True
        self.optionalDatalines = datalines

        return datalines

    def __repr__(self) -> str:
        return f"[{self.name}]"

    def __doc__(self) -> str:
        lines = []
        lines.append(f"[{self.name}]" + " " + self.description)
        if self.requiredArgs:
            lines.append(indent1 + "required arguments")
            for item in self.requiredArgs:
                lines.append(indent2 + item.__doc__())
        if self.optionalArgs:
            lines.append(indent1 + "optional arguments")
            for item in self.optionalArgs:
                lines.append(indent2 + item.__doc__())
        if self.requiredKeywords:
            lines.append(indent1 + "required keywords")
            for item in self.requiredKeywords:
                lines.append(indent0 + item.__doc__())
        if self.optionalKeywords:
            lines.append(indent1 + "optional keywords")
            for item in self.optionalKeywords:
                lines += [indent2 + line for line in item.__doc__().split("\n")]
        if self.requiredDatalines:
            lines.append(indent1 + "required datalines")
            lines.append(indent2 + self.requiredDatalines.__doc__())
        if self.optionalDatalines:
            lines.append(indent1 + "optional datalines")
            lines.append(indent2 + self.optionalDatalines.__doc__())
        return "\n".join(lines)

    def parseKeywordLine(self, line):
        lineElements = splitLineAtCommas(line)

        keyword = lineElements[0]
        optionAssignments = lineElements[1:]

        options = convertAssignmentsToCaseInsensitiveStringDictionary(optionAssignments)

        moduleKw = self.getKeyword(keyword)

        requiredArgs = [kw.name for kw in moduleKw.requiredArgs]
        optionalArgs = [kw.name for kw in moduleKw.optionalArgs]

        @caseInsensitiveKwargsChecker(requiredArgs, optionalArgs)
        def checkKeywordInput(*args, **kwargs):
            """this is a dummy function needed to apply kwargsChecker"""
            return

        try:
            checkKeywordInput(**options)
        except ValueError as e:
            e.args = (f"Error during parsing of keyword {keywordIdentifier}{keyword}: " + e.args[0],)
            raise e

        for optKey, optVal in options.items():
            try:
                options[optKey] = moduleKw[optKey].dtype(optVal)
            except ValueError:
                raise ValueError(f"{keyword}, option {optKey}: cannot convert {optVal} to {moduleKw[optKey].dtype}")

        return moduleKw.name, options

    def parseDatalines(self, datalines):
        args = []
        kwargs = CaseInsensitiveDict()

        if not isinstance(datalines, list):
            datalines = [datalines]

        datalineOptions = []
        for line in datalines:
            datalineOptions += splitLineAtCommas(line)

        for option in datalineOptions:
            if "=" in option:
                kwargs.update(convertAssignmentsToStringDictionary([option]))
            else:
                args.append(option)

        return args, kwargs


class KeywordArg:
    def __init__(self, name: str, description: str, dtype: type):
        self.name = name
        self.description = description
        self.dtype = dtype

        return

    def getValueFromKwargs(self, kwargs: dict):
        kwargs = CaseInsensitiveDict(kwargs)

        try:
            if self.dtype == bool:
                kwargs[self.name] = strtobool(kwargs[self.name])
            else:
                kwargs[self.name] = self.dtype(kwargs[self.name])
            return kwargs[self.name]
        except ValueError:
            raise ValueError(f"Cannot convert {kwargs[self.name]} to {self.dtype}")

    def __repr__(self) -> str:
        return f"[{self.name}]"

    def __doc__(self) -> str:
        return self.__repr__() + " " + self.description + " " + f"({self.dtype})"


class OptionalKeywordArg(KeywordArg):
    def __init__(self, name: str, description: str, dtype: type, default):
        self.default = default
        super().__init__(name, description, dtype)

        return

    def getValueFromKwargs(self, kwargs: dict):
        try:
            return super().getValueFromKwargs(kwargs)
        except KeyError:
            return self.default

    def __doc__(self) -> str:
        return self.__repr__() + " " + self.description + " " + f"({self.dtype}, default = {self.default})"


class ModuleKeywordArg:
    def __init__(self, name: str, description: str, dtype: type):
        self.name = name
        self.description = description
        self.dtype = dtype

        self.requiredArgs = []
        self.optionalArgs = []

        return


class DataLines:
    def __init__(self, description: str, dtype: str):
        self.name = "datalines"
        self.description = description
        self.dtype = dtype

        return

    def __repr__(self) -> str:
        return self.name

    def __doc__(self) -> str:
        return self.description
