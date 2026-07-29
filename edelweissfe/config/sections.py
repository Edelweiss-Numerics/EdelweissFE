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
# Created on Sun Apr 03 11:27:14 2022

# @author: Matthias Neuner
"""
Keyword: ``*section``

Sections combine domains (expressed in terms of element sets)
with properties, such as materials.

.. code-block:: edelweiss
    :caption: Example:

    *section, name=mySection, thickness=1.0, material=myMaterial, type=plane
        all
"""

# This module deliberately holds no code any more. The L1/L2/L4 split replaced its construction
# protocol (``getSectionFactoryByName``) with the L3 registry's ``section`` category, see
# :mod:`edelweissfe.config.registry`. The module survives for the docstring above, which is
# user-facing prose rendered by ``doc/source/documentation/sections.rst`` via
# ``.. automodule:: edelweissfe.config.sections``; deleting the file would break that build.
