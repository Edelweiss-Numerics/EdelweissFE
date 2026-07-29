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

from edelweissfe.config import registry


def getStateTransferStrategyClass(name: str) -> type:
    """Get the class of the requested state-transfer strategy.

    Resolved through the L3 registry (``statetransferstrategy`` category) rather than through this
    module's own ``_STRATEGIES`` table. That table could only ever list strategies living *inside*
    this package, so an external package -- EdelweissMeshfree, a plugin -- had no way to contribute
    one; going through the registry means a built-in, an entry point and an in-process
    :func:`~edelweissfe.config.registry.register` call are all equally reachable here. An unknown
    name now raises :class:`~edelweissfe.config.registry.RegistryLookupError` naming the available
    strategies, instead of a ``KeyError``.

    It also removes an eager import that had nothing to do with resolving a name: ``_STRATEGIES``
    held the three classes themselves, so it had to ``from edelweissfe.adaptivity.statetransfer
    import ...`` at module scope, and every importer of this config module -- including anything that
    merely wanted to know *whether* a strategy name is valid -- paid for importing the whole
    ``statetransfer`` subpackage. The registry holds dotted strings instead and imports the one
    strategy actually asked for, on first use.

    Parameters
    ----------
    name
        Strategy name (case-insensitive): ``nearestQp``, ``projection`` or ``virgin``.

    Returns
    -------
    type
        The :class:`~edelweissfe.adaptivity.statetransfer.base.StateTransferStrategy` subclass.
    """

    strategyClass, _ = registry.lookup("statetransferstrategy", name)

    return strategyClass
