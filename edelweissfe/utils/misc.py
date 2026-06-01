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
"""
Created on Mon Apr 18 17:36:07 2016

@author: Matthias Neuner
"""

import difflib
import shlex
from collections import Counter
from importlib.resources import files

import numpy as np

from edelweissfe.utils.caseinsensitivedict import CaseInsensitiveDict


def getSuccessfulExtensions():
    log_file = files("edelweissfe") / "built_extensions.log"

    if not log_file.is_file():
        return set()

    return {line.strip() for line in log_file.read_text(encoding="utf-8").splitlines() if line.strip()}


def checkSuccessfulExtension(name):
    return name in getSuccessfulExtensions()


def flagDict(configLine):
    parts = [x.strip() for x in configLine.split("=")]
    opt = parts[0]
    val = True if (len(parts) > 1 and parts[1] == "True") else False
    return {opt: val}


def splitLineAtCommas(line: str) -> list:
    """Split a line at commas and strip the individual parts.

    Parameters
    ----------
    line
        The line to be split.

    Returns
    -------
    list
        The list of parts.
    """

    lexer = shlex.shlex(line, posix=True)
    lexer.whitespace_split = True
    lexer.whitespace = ","

    lineElements = [x.strip() for x in lexer]

    return lineElements


def splitLinesAtCommas(lines: list[str]) -> list:
    """Split multiple lines at commas, strip the individual parts and return a list of all items.

    Parameters
    ----------
    lines
        The lines to be split.

    Returns
    -------
    list
        The list of parts.
    """

    lineElements = []
    for line in lines:
        lineElements += splitLineAtCommas(line)

    return lineElements


def convertAssignmentsToStringDictionary(assignments: list) -> dict:
    """Create a dictionary from a list of assignments in
    the form a=b.

    Parameters
    ----------
    assignments
        The list of assignments.

    Returns
    -------
    dict
        The resulting dictionary.
    """

    resultDict = dict()
    keys = []
    vals = []
    for entry in assignments:
        parts = [x.strip() for x in entry.split("=")]
        keys.append(parts[0])
        vals.append("=".join(parts[1:]) if len(parts) > 1 else "True")
    duplicates = [k for k, c in Counter(keys).items() if c > 1]
    if duplicates:
        raise ValueError(f"Key{'s'[:len(duplicates) ^ 1]} {", ".join(duplicates)} used more than once.")
    resultDict = dict(zip(keys, vals))

    return resultDict


def convertAssignmentsToCaseInsensitiveStringDictionary(assignments: list) -> dict:
    """Create a case insensitive dictionary from a list of assignments in
    the form a=b.

    Parameters
    ----------
    assignments
        The list of assignments.

    Returns
    -------
    dict
        The resulting case insensitive dictionary.
    """

    # to do: avoid redundant code -> convertAssignmentsToStringDictionary

    resultDict = dict()
    keys = []
    vals = []
    for entry in assignments:
        parts = [x.strip() for x in entry.split("=")]
        keys.append(parts[0].casefold())
        vals.append("=".join(parts[1:]) if len(parts) > 1 else "True")
    duplicates = [k for k, c in Counter(keys).items() if c > 1]
    if duplicates:
        raise ValueError(f"Key{'s'[:len(duplicates) ^ 1]} {", ".join(duplicates)} used more than once.")
    resultDict = dict(zip(keys, vals))

    return CaseInsensitiveDict(resultDict)


def convertLinesToMixedDictionary(lines: list) -> dict:
    """Create a mixed dictionary from a list of strings containing multiple assignments in
    the form a=b. All strings containing evaluatable values will be evaluated.
    All other dictionary values stay the same.

    Parameters
    ----------
    lines
        The list of strings containing the assignments.

    Returns
    -------
    dict
        The resulting dictionary.
    """

    dictionary = convertLineToStringDictionary(",".join(lines)).copy()
    for key, value in dictionary.items():
        try:
            dictionary[key] = eval(value)
        except NameError:
            pass

    return dictionary


def convertLineToStringDictionary(line: str) -> dict:
    """Create a dictionary from a string containing multiple assignments in
    the form a=b.

    Parameters
    ----------
    line
        The string containing the assignments.

    Returns
    -------
    dict
        The resulting dictionary.
    """

    lineElements = splitLineAtCommas(line)

    return convertAssignmentsToStringDictionary(lineElements)


def convertLinesToStringDictionary(lines: list) -> dict:
    """Create a dictionary from a list of strings containing multiple assignments in
    the form a=b.

    Parameters
    ----------
    lines
        The list of strings containing the assignments.

    Returns
    -------
    dict
        The resulting dictionary.
    """

    return convertLineToStringDictionary(",".join(lines))


def convertLinesToFlatArray(lines: list, dtype: type = float) -> np.ndarray:
    """Create a 1D numpy array from a list of lines with elements separated by commas.

    Parameters
    ----------
    lines
        The list of strings containing the elements.

    Returns
    -------
    np.ndarray
        The resulting 1D array.
    """

    theLines = [np.asarray(splitLineAtCommas(line), dtype=dtype) for line in lines]
    return np.hstack(theLines)


def strCaseCmp(str1, str2):
    return str1.casefold() == str2.casefold()


def strToSlice(string):
    if ":" in string:
        idcs = [int(i) for i in string.split(":")]
        a, b = idcs
        return slice(a, b)
    else:
        return slice(int(string), int(string) + 1)


def strToRange(string):
    if ":" in string:
        idcs = [int(i) for i in string.split(":")]
        a, b = idcs
        return range(a, b)
    else:
        return range(int(string))


def isInteger(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def filterByJobName(canditates, jobName):
    return [cand for cand in canditates if "jobName" not in cand or cand["jobName"] == jobName]


def mergeNumpyDataLines(multiLineData: np.ndarray) -> np.ndarray:
    """Flatten a numpy array."""
    flattenedMatProps = [p for row in multiLineData for p in row]
    return np.array(flattenedMatProps, dtype=float)


def strtobool(val: str) -> bool:
    """Convert a string representation of truth to true or false.
    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.

    Parameters
    ----------
    val
        The string representing the truth value:

    Returns
    -------
    bool
        The truth value.
    """
    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return True
    elif val in ("n", "no", "f", "false", "off", "0"):
        return False
    else:
        raise ValueError("invalid truth value %r" % (val,))


def typeString(dtype: type or str) -> str:
    """.

    Parameters
    ----------
    dtype
        data type or string

    Returns
    -------
    str
        string representing data type
    """
    dtypeMapping = {
        str: "string",
        bool: "boolean",
        int: "integer",
        float: "float",
    }
    return dtypeMapping.get(dtype, str(dtype))


def findSimilarString(s: str, ll: list[str], threshold=0):
    if not len(ll) > 0:
        raise Exception(f"You tried to find a string similar to {s} in an empty list.")
    result = [difflib.SequenceMatcher(a=s.casefold(), b=item.casefold()).ratio() for item in ll]
    if not max(result) > threshold:
        raise ValueError(f"No similar string to {s} was found in list {ll}.")

    return ll[np.argmax(result)]


def kwargsChecker(kwargsRequired: list[str], kwargsOptional: list[str]):
    def wrapper(fun, *args, **kwargs):
        def wrapped(*args, **kwargs):
            missing_kwargs = []
            for kwarg in kwargsRequired:
                if kwarg not in kwargs:
                    missing_kwargs.append(kwarg)

            nMissing = len(missing_kwargs)
            if not nMissing == 0:
                raise ValueError(
                    f"Function call to {fun} missing {nMissing} required keyword argument{'s'[:nMissing ^ 1]}: "
                    + ", ".join(missing_kwargs)
                )

            unexpected_kwargs = []
            for kwarg in kwargs:
                if not (kwarg in kwargsRequired or kwarg in kwargsOptional):
                    unexpected_kwargs.append(kwarg)

            nUnexpected = len(unexpected_kwargs)
            if not nUnexpected == 0:
                if nUnexpected == 1 and len(kwargsOptional) > 0:
                    try:  # try to find a matching optional keyword
                        similarKeyword = findSimilarString(unexpected_kwargs[0], [item for item in kwargsOptional], 0.1)
                        hint = f" Did you mean {similarKeyword}?"
                    except ValueError:
                        hint = ""
                else:
                    hint = ""
                raise ValueError(
                    f"Function call to {fun} got {nUnexpected} unexpected keyword argument{'s'[:nUnexpected ^ 1]}: "
                    + ", ".join(unexpected_kwargs)
                    + "."
                    + hint
                )

            return fun(*args, **kwargs)

        return wrapped

    return wrapper


def caseInsensitiveKwargsChecker(kwargsRequired: list[str], kwargsOptional: list[str]):
    casefoldedKwargsRequired = [kw.casefold() for kw in kwargsRequired]
    casefoldedKwargsOptional = [kw.casefold() for kw in kwargsOptional]

    def wrapper(fun, *args, **kwargs):
        def wrapped(*args, **kwargs):
            casefoldedKwargs = {key.casefold(): val for key, val in kwargs.items()}

            return kwargsChecker(casefoldedKwargsRequired, casefoldedKwargsOptional)(fun)(*args, **casefoldedKwargs)

        return wrapped

    return wrapper


def castKwargsValuesAndAddDefaults(module):
    def wrapper(fun, *args, **kwargs):
        def wrapped(*args, **kwargs):
            kwargs = CaseInsensitiveDict(kwargs)
            for arg in module.requiredArgs:
                if arg.name in kwargs:
                    kwargs[arg.name] = arg.getValueFromKwargs(kwargs)
            for arg in module.optionalArgs:
                if arg.name in kwargs:
                    kwargs[arg.name] = arg.getValueFromKwargs(kwargs)
                else:
                    kwargs[arg.name] = arg.default

            return fun(*args, **kwargs)

        return wrapped

    return wrapper
