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

"""ModelModifiers dynamically alter mesh topology, element/node sets, material states,
or field allocations during an analysis step.
"""

from edelweissfe.config import registry


def getModelModifierClass(name: str) -> type:
    """Return the class implementing model modifier ``name``.

    Resolved through the registry (``modelmodifier`` category), which names the modifier's module
    explicitly rather than guessing a subpackage.

    Parameters
    ----------
    name
        The name of the model modifier to load (e.g. 'hadaptivity').

    Returns
    -------
    type
        The model modifier class type.

    Raises
    ------
    edelweissfe.config.registry.RegistryLookupError
        If no model modifier is registered under ``name``.
    """

    modelModifierClass, _ = registry.lookup("modelmodifier", name)

    return modelModifierClass
