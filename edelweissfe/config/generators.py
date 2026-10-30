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
# Created on Wed Apr 12 15:37:28 2017

# @author: Matthias Neuner
"""
Mesh generators are used to create meshes,
using different methods.

Keyword: ``*generator``

.. code-block:: edelweiss
    :caption: Example:

    *modelGenerator, generator=theGeneratorType, name=myGeneratorName
        multiple lines of defintion ...
        multiple lines of defintion ...
        multiple lines of defintion ...
"""

from edelweissfe.config import registry


def getGeneratorClass(name: str) -> type:
    """Return the class implementing generator ``name``.

    Resolved through the registry (``generator`` category), which lets built-in generators,
    entry points, and in-process :func:`~edelweissfe.config.registry.register` calls all be
    reached the same way.

    Parameters
    ----------
    name
        The name of the generator class type to load.

    Returns
    -------
    type
        The generator class type.

    Raises
    ------
    edelweissfe.config.registry.RegistryLookupError
        If no generator is registered under ``name``.
    """

    generatorClass, _ = registry.lookup("generator", name)

    return generatorClass
