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

from edelweissfe.config import registry


def getStepClassByType(stepType: str) -> type:
    """Get the class type of the requested step type.

    Resolved through the L3 registry (``step`` category) rather than through this module's own
    ``stepLibrary`` table of ``(module, class)`` pairs. That table could only ever list steps living
    *inside* this package, so an external package -- EdelweissMeshfree, a plugin -- had no way to
    contribute one; going through the registry means a built-in, an entry point and an in-process
    :func:`~edelweissfe.config.registry.register` call are all equally reachable here. Names remain
    case-insensitive, as ``stepLibrary`` already made them by casefolding its keys per call. An
    unknown type now raises :class:`~edelweissfe.config.registry.RegistryLookupError` naming the
    available step types, instead of a ``KeyError``.

    Parameters
    ----------
    stepType
        The type of the step to load (case insensitive).

    Returns
    -------
    type
        The step class type.
    """

    stepClass, _ = registry.lookup("step", stepType)

    return stepClass
