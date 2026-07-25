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

"""Observer interface for modules that cache mesh topology, node/element references, node sets, or
surfaces and must re-synchronize when the :class:`~edelweissfe.models.femodel.FEModel` is mutated
mid-analysis (e.g. by adaptive mesh refinement). The FEModel is the subject: it holds a list of
observers and calls :meth:`ModelChangeObserver.onModelChanged` after a mutation.
"""

from abc import ABC, abstractmethod
from enum import Enum, auto


class ModelChangeType(Enum):
    REFINEMENT = auto()  # elements subdivided / nodes added
    COARSENING = auto()  # elements merged / nodes removed
    ELEMENT_EROSION = auto()  # elements deleted
    TOPOLOGY_CHANGE = auto()  # boundary / surface / set changes


class ModelChangeObserver(ABC):
    """Interface for any module that caches mesh topology, node/element references, or DOF indices
    and needs re-synchronization when the FEModel changes (e.g. Dirichlet BCs re-resolving their
    node set after refinement adds boundary nodes)."""

    @abstractmethod
    def onModelChanged(self, model, changeType: ModelChangeType, details: dict = None):
        """Callback invoked immediately after the FEModel topology or sets are mutated."""
