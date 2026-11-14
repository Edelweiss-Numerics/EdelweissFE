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

"""The compiled Marmot element wrapper implements the protocol of :class:`BaseNodeCouplingEntity` without deriving from
it: a Cython extension type cannot derive from a Python class. Every public member of the base must be there.

Skipped where Marmot is not built.
"""

import pytest

from edelweissfe.nodecouplingentity.base.nodecouplingentity import (
    BaseNodeCouplingEntity,
)


def _publicMembers(cls) -> list[str]:
    return [name for name in dir(cls) if not name.startswith("_")]


def test_the_marmot_element_wrapper_has_every_public_member_of_the_entity_protocol():
    wrapper = pytest.importorskip("edelweissfe.elements.marmotelement.element").MarmotElementWrapper

    missing = [name for name in _publicMembers(BaseNodeCouplingEntity) if name not in dir(wrapper)]

    assert missing == []
