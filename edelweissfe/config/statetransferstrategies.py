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

"""Registry of quadrature-point state-transfer strategies used by adaptive mesh refinement.

A strategy decides how the material history stored at a refined parent element's quadrature points
is handed down to its children. Selectable by name from the ``*modelModifier`` input; see
:mod:`edelweissfe.adaptivity.statetransfer`.
"""

from edelweissfe.adaptivity.statetransfer import (
    NearestQuadraturePointCopy,
    PolynomialProjection,
    VirginState,
)

_STRATEGIES = {
    "nearestqp": NearestQuadraturePointCopy,
    "projection": PolynomialProjection,
    "virgin": VirginState,
}


def getStateTransferStrategyClass(name: str) -> type:
    """Get the class of the requested state-transfer strategy.

    Parameters
    ----------
    name
        Strategy name (case-insensitive): ``nearestQp``, ``projection`` or ``virgin``.

    Returns
    -------
    type
        The :class:`~edelweissfe.adaptivity.statetransfer.base.StateTransferStrategy` subclass.
    """
    key = name.lower()
    if key not in _STRATEGIES:
        raise KeyError(
            "unknown state-transfer strategy '{:}'; available: {:}".format(name, ", ".join(sorted(_STRATEGIES)))
        )
    return _STRATEGIES[key]
