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
# Created on Tue Jan  17 19:10:42 2017

# @author: Matthias Neuner
"""
EdelweissFE currently supports finite element implementations provided by the Marmot library.
In future, elements by other providers or elements directly implemented in EdelweissFE may be added here.

.. code-block:: edelweiss
    :caption: Example:

    *element, type=C3D8, provider=marmot
        ** el_label, node1, node2, node3, node4, ...
        1000,        1,     2,     3,     4,     ...
"""

from edelweissfe.config import registry
from edelweissfe.utils.misc import strCaseCmp


def getElementClass(elType: str, provider: str = None) -> type:
    """Return the class implementing element type ``elType`` for the given ``provider``.

    ``provider`` selects a namespace, not a variant of one lookup, and is dispatched via an
    explicit table rather than the registry: ``marmot`` and ``marmotsingleqpelement`` ignore
    ``elType`` entirely and return a single wrapper class that resolves the type at the Marmot
    boundary.

    The ``edelweiss`` provider is resolved through the registry (``element`` category), keyed by
    element type, which lets a third party contribute an element type through an entry point.

    Parameters
    ----------
    elType
        A string identifying the requested element formulation.
    provider
        The name of the element provider to load.

    Returns
    -------
    type
        The element provider class type.

    Raises
    ------
    edelweissfe.config.registry.RegistryLookupError
        If ``provider`` is ``edelweiss`` and no element is registered under ``elType``.
    """

    if provider is None:
        provider = "marmot"

    if strCaseCmp(provider, "edelweiss"):

        elementClass, _ = registry.lookup("element", elType)

        return elementClass

    elif provider.lower() == "marmot":
        from edelweissfe.elements.marmotelement.element import MarmotElementWrapper

        return MarmotElementWrapper

    elif provider.lower() == "marmotsingleqpelement":
        from edelweissfe.elements.marmotsingleqpelement.element import (
            MarmotMaterialWrappingElement,
        )

        return MarmotMaterialWrappingElement

    else:
        raise Exception("This element provider doesn't exist!")
