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
    """Get the class type of the requested model modifier.

    Resolved through the L3 registry (``modelmodifier`` category) rather than by importing
    ``edelweissfe.modelmodifiers.<name>`` directly. Besides being unreachable for an external package
    -- EdelweissMeshfree, a plugin -- that import-by-convention had to *guess* the subpackage: it
    tried ``edelweissfe.modelmodifiers.<name>`` and fell back to
    ``edelweissfe.modelmodifiers.adaptivity.<name>`` on ``ModuleNotFoundError``, so a modifier whose
    own module raised ``ModuleNotFoundError`` for an unrelated missing dependency was silently
    re-looked-up in the wrong package and then reported as absent. The registry names the module
    explicitly, so there is nothing to guess and no exception to swallow. An unknown name now raises
    :class:`~edelweissfe.config.registry.RegistryLookupError` naming the available model modifiers,
    instead of a bare ``ModuleNotFoundError``.

    Parameters
    ----------
    name
        The name of the model modifier to load (e.g. 'hadaptivity').

    Returns
    -------
    type
        The model modifier class type.
    """

    modelModifierClass, _ = registry.lookup("modelmodifier", name)

    return modelModifierClass
