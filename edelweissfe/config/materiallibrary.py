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
#  Daniel Reitmair daniel.reitmair@uibk.ac.at
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

from edelweissfe.config import registry
from edelweissfe.utils.misc import strCaseCmp


def getMaterialClass(materialName: str, provider: str = None) -> type:
    """Get the the requested material class.

    The ``provider`` dispatch below is deliberately an explicit table and **not** a registry lookup:
    a provider selects a *namespace*, not a variant of one lookup. Only ``edelweiss`` addresses
    anything by name; ``marmotmaterial`` ignores ``materialName`` and returns ``None`` (see below).
    There is nothing per-name to register for it, so it stays here (``PLAN_INPUT_SYSTEM.md`` §9).

    **Returning ``None`` for the ``marmotmaterial`` provider is deliberate, not a missing case.** A
    Marmot material has no Python class at all -- it is instantiated inside the C++/Cython element
    wrapper from its name plus its property array -- so there is nothing for this function to hand
    back. ``None`` is the caller's signal to keep the material as a ``{"name": ..., "properties":
    ...}`` record instead of constructing an object -- see
    ``AbqModelConstructor.createMaterialsFromInputFile``, which branches on exactly that.

    The ``edelweiss`` branch is resolved through the L3 registry (``material`` category), which
    replaces an eleven-arm ``if/elif`` chain of local imports. Besides being one table instead of
    eleven branches, that chain could only ever name materials living *inside* this package, so an
    external package -- EdelweissMeshfree, a plugin -- had no way to contribute one. An unknown name
    now raises :class:`~edelweissfe.config.registry.RegistryLookupError` naming the available
    materials, instead of ``Exception("This material type doesn't exist (yet)...")``.

    Parameters
    ----------
    materialName
        The name of the requested material.
    provider
        The name of the material provider.

    Returns
    -------
    type
        The material provider class type, or ``None`` for the ``marmotmaterial`` provider.
    """

    if provider is None:
        provider = "MarmotMaterial"

    if strCaseCmp(provider, "marmotmaterial"):

        return None

    if strCaseCmp(provider, "edelweiss"):

        materialClass, _ = registry.lookup("material", materialName)

        return materialClass
