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

"""Virgin (no-history) state transfer (tier F0)."""

from edelweissfe.adaptivity.statetransfer.base import StateTransferStrategy


class VirginState(StateTransferStrategy):
    """Children keep their freshly-initialised (virgin) state; no history is inherited.

    Sound only when refining *ahead of* the process zone. As a whole-block strategy it discards all
    history; as a per-state-variable override (see
    :class:`~edelweissfe.adaptivity.statetransfer.perstatevar.PerStateVarStateTransfer`) it resets
    just the selected variables to their initial values while the rest are transferred normally --
    useful for rate/step counters or trial-state flags that must not carry over.
    """

    def _transferColumns(self, parentValues, parentRefCoords, childRefCoords, childInitValues, columns):
        return childInitValues[:, columns]
