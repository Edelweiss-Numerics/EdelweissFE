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
    """Get the class type of the requested element provider.

    The ``provider`` dispatch below is deliberately an explicit table and **not** a registry
    lookup: a provider selects a *namespace*, not a variant of one lookup. Only ``edelweiss``
    addresses anything by name -- ``marmot`` and ``marmotsingleqpelement`` ignore ``elType``
    entirely and return a single wrapper class that then reads the type at the Marmot boundary. There
    is nothing per-name to register for those two, so folding them in would mean inventing 42
    identical entries per provider (see ``PLAN_INPUT_SYSTEM.md`` §9).

    The ``edelweiss`` branch *is* resolved through the L3 registry (``element`` category), keyed by
    element type. It used to read a class *name* out of
    :data:`~edelweissfe.elements.library.elLibrary`'s ``elClass`` field and ``eval`` it -- which is
    why this module used to import ``DisplacementElement`` and ``DisplacementTLElement`` behind
    ``# noqa: F401``: the imports looked unused but were load-bearing as the ``eval``'s scope. Both
    the ``eval`` and the ``elClass`` field are now gone, so those two imports were genuinely dead and
    have been deleted -- do not "restore" them. The registry is the single source of truth for
    type -> class, which also means a third party can contribute an element type through an entry
    point instead of having to edit ``elLibrary``. An unknown type now raises
    :class:`~edelweissfe.config.registry.RegistryLookupError` naming the available types, instead of
    ``Exception("Edelweiss element not found in library.")``.

    Parameters
    ----------
    elType
        A string identifying the requested element formulation.
    provider
        The name of the element provider ot load.

    Returns
    -------
    type
        The element provider class type.
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
