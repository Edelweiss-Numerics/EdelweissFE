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
# Created on Wed Jun 14 21:40:55 2017

# @author: Matthias Neuner
"""The regression tests: every subdirectory of the test directory holding a ``test.inp`` is a test (a deck), whose
solution after its last step is compared with the reference solution in its ``U.ref``.

A deck declares what it needs beyond EdelweissFE itself in comment lines of its own, e.g.::

    ** requires: marmot-module VonMises
    ** requires: extension edelweissfe.linsolve.amgcl.amgcl
    ** requires: python-module num_dual

- ``marmot-module``: a module of Marmot, installed in the Marmot installation that EdelweissFE was built against (its
  header ``Marmot/<module>.h``).
- ``extension``: an optional extension of EdelweissFE, built (``edelweissfe/built_extensions.log``).
- ``python-module``: a Python module, installed.

A deck with a missing requirement is skipped, with the requirement as the reason. A deck fails if it cannot be parsed,
if its simulation raises any exception (a failed step included), or if its solution differs from the reference by more
than :data:`RELATIVE_TOLERANCE` times the largest magnitude of the reference.

``--expected-skips <file>`` names the decks that may be skipped, one ``<test directory name>/<deck>`` per line (``#``
starts a comment): the run fails if any other deck is skipped, or if a listed deck of the test directory runs or does
not exist.
"""

import argparse
import importlib.util
import os
import sys
from importlib.resources import files
from pathlib import Path
from timeit import default_timer as timer

import matplotlib
import numpy as np
from rich import print

from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.utils.inputfileparser import parseInputFile
from edelweissfe.utils.misc import checkSuccessfulExtension

matplotlib.use("Agg")

TEST_FILE = "test.inp"
REFERENCE_SOLUTION_FILE = "U.ref"
REQUIREMENT_MARKER = "** requires:"

#: The largest difference to the reference solution, relative to the largest magnitude of the reference.
RELATIVE_TOLERANCE = 1e-6

#: The smallest largest magnitude of a reference solution: a smaller one, which is zero up to round-off, cannot be
#: compared relative to its magnitude, and is refused by ``--create``.
SMALLEST_REFERENCE_MAGNITUDE = 1e-8


def missingMarmotModule(module: str) -> str | None:
    """Why the Marmot module ``module`` is missing, or None if it is installed.

    Parameters
    ----------
    module
        The name of the Marmot module, e.g. ``VonMises``.

    Returns
    -------
    str | None
        The reason, or None.
    """

    if not checkSuccessfulExtension("edelweissfe.elements.marmotelement.element"):
        return "EdelweissFE is built without Marmot"
    marmotInstallDirectory = Path((files("edelweissfe") / "marmot_install_dir.txt").read_text(encoding="utf-8").strip())
    if not (marmotInstallDirectory / "include" / "Marmot" / "{:}.h".format(module)).is_file():
        return "Marmot module {:} is not installed".format(module)
    return None


def missingExtension(extension: str) -> str | None:
    """Why the optional extension ``extension`` of EdelweissFE is missing, or None if it was built.

    Parameters
    ----------
    extension
        The name of the extension, e.g. ``edelweissfe.linsolve.amgcl.amgcl``.

    Returns
    -------
    str | None
        The reason, or None.
    """

    return None if checkSuccessfulExtension(extension) else "extension {:} is not built".format(extension)


def missingPythonModule(module: str) -> str | None:
    """Why the Python module ``module`` is missing, or None if it is installed.

    Parameters
    ----------
    module
        The name of the module, e.g. ``num_dual``.

    Returns
    -------
    str | None
        The reason, or None.
    """

    return None if importlib.util.find_spec(module) is not None else "Python module {:} is not installed".format(module)


#: The check of each kind of requirement.
REQUIREMENT_CHECKS = {
    "marmot-module": missingMarmotModule,
    "extension": missingExtension,
    "python-module": missingPythonModule,
}


def missingRequirements(testFile: Path) -> list[str]:
    """Why the requirements a deck declares are missing.

    Parameters
    ----------
    testFile
        The deck.

    Returns
    -------
    list[str]
        The reasons why requirements are missing, each once; empty if nothing is missing.

    Raises
    ------
    ValueError
        If a requirement line does not name a known kind and one name.
    """

    reasons = []
    for line in testFile.read_text(encoding="utf-8").splitlines():
        if not line.startswith(REQUIREMENT_MARKER):
            continue
        kindAndName = line[len(REQUIREMENT_MARKER) :].split()
        if len(kindAndName) != 2 or kindAndName[0] not in REQUIREMENT_CHECKS:
            raise ValueError(
                "'{:}': a requirement is one of {:} and a name".format(line, ", ".join(REQUIREMENT_CHECKS))
            )
        kind, name = kindAndName
        reason = REQUIREMENT_CHECKS[kind](name)
        if reason is not None and reason not in reasons:
            reasons.append(reason)
    return reasons


def readExpectedSkips(expectedSkipsFile: Path, testDirectory: Path) -> set[str]:
    """The decks of ``testDirectory`` that ``expectedSkipsFile`` lists as expected to be skipped.

    Parameters
    ----------
    expectedSkipsFile
        The file, one ``<test directory name>/<deck>`` per line, ``#`` starting a comment.
    testDirectory
        The test directory.

    Returns
    -------
    set[str]
        The names of the decks.
    """

    decks = set()
    for line in expectedSkipsFile.read_text(encoding="utf-8").splitlines():
        entry = line.split("#", 1)[0].strip()
        if entry:
            directoryName, deck = entry.split("/")
            if directoryName == testDirectory.name:
                decks.add(deck)
    return decks


def solutionOf(model) -> np.ndarray:
    """The solution of a model that is compared with the reference: every node field, then every scalar variable.

    Parameters
    ----------
    model
        The model tree.

    Returns
    -------
    numpy.ndarray
        The solution.
    """

    return np.hstack(
        [f["U"].flatten() for f in model.nodeFields.values()] + [v.value for v in model.scalarVariables.values()]
    ).flatten()


def runTest(deckDirectory: Path, create: bool) -> tuple[str, str]:
    """Run the deck of ``deckDirectory`` and compare its solution with the reference, or create the reference.

    Parameters
    ----------
    deckDirectory
        The directory of the deck.
    create
        Write the solution as the reference instead of comparing with it.

    Returns
    -------
    tuple[str, str]
        The outcome (``PASSED``, ``CREATED``, ``SKIPPED`` or ``FAILED``) and what to report with it.
    """

    os.chdir(deckDirectory)
    testFile = deckDirectory / TEST_FILE
    try:
        reasons = missingRequirements(testFile)
    except ValueError as e:
        return "FAILED", str(e)
    if reasons:
        return "SKIPPED", "; ".join(reasons)

    try:
        inputFile = parseInputFile(str(testFile))
    except Exception as e:
        return "FAILED", "during parsing: {:}".format(e)

    try:
        tic = timer()
        model, _ = finiteElementSimulation(inputFile, verbose=False, suppressPlots=True)
        toc = timer()
    except Exception as e:
        return "FAILED", str(e)

    U = solutionOf(model)
    referenceSolutionFile = deckDirectory / REFERENCE_SOLUTION_FILE
    if create:
        if np.max(np.abs(U)) < SMALLEST_REFERENCE_MAGNITUDE:
            return "FAILED", "the solution is zero up to {:}, too small for a reference".format(
                SMALLEST_REFERENCE_MAGNITUDE
            )
        np.savetxt(referenceSolutionFile, U)
        return "CREATED", ""

    UReference = np.atleast_1d(np.loadtxt(referenceSolutionFile))
    if U.shape != UReference.shape:
        return "FAILED", "{:} values, but the reference has {:}".format(U.size, UReference.size)
    largestDifference = np.max(np.abs(U - UReference))
    largestMagnitude = np.max(np.abs(UReference))
    if largestDifference > RELATIVE_TOLERANCE * largestMagnitude:
        return "FAILED", "differs from the reference by {:.1e}, whose largest magnitude is {:.1e}".format(
            largestDifference, largestMagnitude
        )
    return "PASSED", "[{:2.1f}]".format(toc - tic)


def main():
    parser = argparse.ArgumentParser(description="validation script for FE analyses")
    parser.add_argument("testdirectory", help="The directory containing the testfiles")
    parser.add_argument(
        "--create",
        dest="create",
        action="store_true",
        help="create reference solutions",
    )
    parser.add_argument(
        "--tests",
        help="comma-separated list (without whitespace inbetween) with names of analyzed test files, "
        "e.g. MeshPlot,NodeForces, or simply type all. The names are case-sensitive.",
        type=str,
        default="all",
    )
    parser.add_argument(
        "--expected-skips",
        dest="expectedSkips",
        type=Path,
        default=None,
        help="a file of the decks that may be skipped, one <test directory name>/<deck> per line; any other skip, and "
        "a listed deck that runs or does not exist, fails the run",
    )
    args = parser.parse_args()

    tests = args.tests.split(",")

    testDirectory = Path(args.testdirectory).resolve()
    expectedSkips = readExpectedSkips(args.expectedSkips, testDirectory) if args.expectedSkips else set()
    # keep the sorted order: a test case may consume a file another one produces
    testsDirs = sorted((path.name for path in testDirectory.iterdir() if path.is_dir()), key=str.casefold)

    if "all" not in tests:
        requested = set(tests)
        testsDirs = [directory for directory in testsDirs if directory in requested]

    failedTests = 0
    skippedTests = 0
    unexpectedSkips = []
    decksThatRan = []
    directoriesWithoutTest = []

    for directory in testsDirs:
        deckDirectory = testDirectory / directory

        # directories without a test.inp are listed at the end, so that no deck silently drops out
        if not (deckDirectory / TEST_FILE).exists():
            directoriesWithoutTest.append(directory)
            continue

        print("Test {:50}".format(directory), end="\r")
        outcome, message = runTest(deckDirectory, args.create)
        color = {"PASSED": "green", "CREATED": "green", "SKIPPED": "grey", "FAILED": "red"}[outcome]
        print("Test {:50} [{:}]{:}[/] {:}".format(directory, color, outcome, message))

        if outcome == "FAILED":
            failedTests += 1
        if outcome == "SKIPPED":
            skippedTests += 1
            if args.expectedSkips and directory not in expectedSkips:
                unexpectedSkips.append(directory)
        else:
            decksThatRan.append(directory)

    if directoriesWithoutTest:
        print("[grey]Directories without a {}: {}[/]".format(TEST_FILE, ", ".join(directoriesWithoutTest)))
    print("[blue]Tests skipped: {:3}[/]".format(skippedTests))
    print("[blue]Tests failed: {:3}[/]".format(failedTests))
    # a listed deck that runs, or does not exist, makes the list stale
    staleSkips = sorted(
        deck for deck in expectedSkips if deck in decksThatRan or not (testDirectory / deck / TEST_FILE).is_file()
    )
    if unexpectedSkips:
        print("[red]Skipped, but not listed in {}: {}[/]".format(args.expectedSkips, ", ".join(unexpectedSkips)))
    if staleSkips:
        print("[red]Listed in {}, but run or not a deck: {}[/]".format(args.expectedSkips, ", ".join(staleSkips)))
    if failedTests > 0 or unexpectedSkips or staleSkips:
        sys.exit(1)


if __name__ == "__main__":
    main()
