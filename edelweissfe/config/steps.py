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
"""
Steps are defined via the ``*step`` keyword, and the step type is chosen
via the ``type`` option:

.. code-block:: edelweiss

    *step, solver=mySolver, type=adaptive
"""

import importlib

stepLibrary = {
    "adaptive": ("edelweissfe.steps.adaptivestep", "AdaptiveStep"),
    "adaptiveForExplicitSimulations": (
        "edelweissfe.steps.adaptivestepforexplicitsimulations",
        "AdaptiveStepForExplicitSimulations",
    ),
}


def getStepClassByType(stepType: str) -> type:
    """Get the class type of the requested step type.

    Parameters
    ----------
    stepType
        The type of the step to load (case insensitive).

    Returns
    -------
    type
        The step class type.
    """

    casefoldedStepLibrary = {key.casefold(): value for key, value in stepLibrary.items()}

    try:
        moduleName, className = casefoldedStepLibrary[stepType.casefold()]
    except KeyError:
        raise KeyError(
            f"Step type {stepType} not found in library. Available step types: " + ", ".join(stepLibrary.keys())
        )

    module = importlib.import_module(moduleName)
    return getattr(module, className)
