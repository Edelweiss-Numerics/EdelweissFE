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


"""The compiled Marmot element wrapper: the element number reaches Marmot, and Marmot's exceptions reach Python.

Skipped where Marmot is not built.
"""

import numpy as np
import pytest

elementModule = pytest.importorskip("edelweissfe.elements.marmotelement.element")


def _elementOrSkip(elementType: str, elNumber: int):
    try:
        return elementModule.MarmotElementWrapper(elementType, elNumber)
    except NotImplementedError:
        pytest.skip(f"Marmot has no {elementType} element")


def test_a_misspelled_result_raises_instead_of_aborting():
    """Marmot throws for an unknown state name; undeclared as `except +`, that reached std::terminate."""
    element = _elementOrSkip("CPE4", 42)
    element.setMaterial("LINEARELASTIC", np.array([30000.0, 0.2]))

    with pytest.raises(ValueError, match="nonsense"):
        element.getResultArray("nonsense", 0)


def test_the_element_number_reaches_marmot():
    """__cinit__ runs before __init__: it passed the still unset self._elNumber, so every Marmot element was 0.

    The embedded bond element names itself in its errors (Marmot >= the bond-slip elements)."""
    element = _elementOrSkip("EB2D2Q4", 42)

    with pytest.raises(ValueError, match="EmbeddedBondElement 42:"):
        element.setProperties(np.array([-1.0, -1.0, 1.0]))
