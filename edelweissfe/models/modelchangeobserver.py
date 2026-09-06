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

"""The kinds of model mutation a :class:`~edelweissfe.models.modelchange.ModelChange` can describe.

This module once also defined a push-based ``ModelChangeObserver``, notified synchronously at the
instant of each mutation. It was removed in favour of a single pull mechanism
(:class:`~edelweissfe.models.meshdependent.MeshDependent`, driven by
:meth:`~edelweissfe.models.femodel.FEModel.refreshMeshDependents`): model modifiers now run to a
fixed point in rounds, so a per-mutation callback necessarily fires mid-pipeline -- handing the
consumer a state that no longer exists by the time the solve begins, and letting a consumer that
mutates in response do so re-entrantly, inside the modifier's own loop.
"""

from enum import Enum, auto


class ModelChangeType(Enum):
    REFINEMENT = auto()  # elements subdivided / nodes added
    COARSENING = auto()  # elements merged / nodes removed
    ELEMENT_EROSION = auto()  # elements deleted
    TOPOLOGY_CHANGE = auto()  # boundary / surface / set changes
