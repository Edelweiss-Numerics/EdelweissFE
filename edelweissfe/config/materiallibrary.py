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

from importlib import import_module

from edelweissfe.utils.misc import strCaseCmp

# materialName (lowercase) -> (module path, class name)
_EDELWEISS_MATERIALS = {
    "linearelastic": ("edelweissfe.materials.linearelastic.linearelastic", "LinearElasticMaterial"),
    "vonmises": ("edelweissfe.materials.vonmises.vonmises", "VonMisesMaterial"),
    "neohooke": ("edelweissfe.materials.neohooke.neohooke", "NeoHookeanMaterial"),
    "hyperelasticadvanced": (
        "edelweissfe.materials.hyperelasticadvanced.hyperelasticadvanced",
        "HyperelasticAdvancedMaterial",
    ),
    "hyperelasticadvancedi2extended": (
        "edelweissfe.materials.hyperelasticadvanced.hyperelasticadvancedi2extended",
        "HyperelasticAdvancedI2ExtendedMaterial",
    ),
    "neohookeplastic": ("edelweissfe.materials.neohookeplastic.neohookeplastic", "NeoHookeanPlasticMaterial"),
    "hyperplasticadvanced": (
        "edelweissfe.materials.hyperplasticadvanced.hyperplasticadvanced",
        "HyperplasticAdvancedMaterial",
    ),
}


def getMaterialClass(materialName: str, provider: str = None) -> type:
    """Get the the requested material class.

    Parameters
    ----------
    materialName
        The name of the requested material.
    provider
        The name of the material provider.

    Returns
    -------
    type
        The material provider class type.
    """

    if provider is None:
        provider = "MarmotMaterial"

    if strCaseCmp(provider, "marmotmaterial"):
        # The material is created and owned by the Marmot element itself.

        return None

    if strCaseCmp(provider, "marmotmaterialpoint"):
        # A Marmot material evaluated point-wise, without an element in between.
        # Which class applies follows from the base class the material is registered for
        # in Marmot, not from its name, so the caller states the base class.
        if strCaseCmp(materialName, "hypoelastic"):
            from edelweissfe.materials.marmot.marmothypoelastic import (
                MarmotHypoElasticMaterial,
            )

            return MarmotHypoElasticMaterial

        if strCaseCmp(materialName, "gradientenhancedhypoelastic"):
            from edelweissfe.materials.marmot.marmotgradientenhancedhypoelastic import (
                MarmotGradientEnhancedHypoElasticMaterial,
            )

            return MarmotGradientEnhancedHypoElasticMaterial

        if strCaseCmp(materialName, "gradientplasticityhypoelastic"):
            from edelweissfe.materials.marmot.marmotgradientplasticityhypoelastic import (
                MarmotGradientPlasticityHypoElasticMaterial,
            )

            return MarmotGradientPlasticityHypoElasticMaterial

        raise Exception(
            "Unknown Marmot material point base class '{:}'; expected 'hypoelastic', "
            "'gradientEnhancedHypoElastic' or 'gradientPlasticityHypoElastic'".format(materialName)
        )

    if strCaseCmp(provider, "edelweiss"):
        modulePath, className = _EDELWEISS_MATERIALS.get(materialName.lower(), (None, None))
        if modulePath is None:
            raise Exception("This material type doesn't exist (yet). Chosen material was: " + materialName)

        return getattr(import_module(modulePath), className)
