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

"""Mixin for any component that caches data derived from the mesh (contact facets, tie records,
DOF numbering, ...) and must patch it up after a model mutation (e.g. an AMR refinement) -- the
pull-based counterpart to :class:`~edelweissfe.models.modelchangeobserver.ModelChangeObserver`'s
push-based ``onModelChanged``. Rather than registering to be notified, a :class:`MeshDependent`
reconciles lazily at its own next per-increment tick (call :meth:`reconcileIfChanged` there), which
needs no registration and hence has no observer lifecycle to leak.
"""

from abc import ABC, abstractmethod


class MeshDependent(ABC):
    """Interface for mesh-derived data that must stay consistent across a model mutation."""

    _lastSeenTopologyVersion = 0

    @abstractmethod
    def reconcile(self, model, change) -> bool:
        """Patch cached mesh-derived state to account for ``change`` (a
        :class:`~edelweissfe.models.modelchange.ModelChange`). Called only when the model's
        ``topologyVersion`` actually advanced since this consumer last checked.

        Returns
        -------
        bool
            True if ``change`` was actually relevant to this consumer (e.g. touched one of its
            watched surfaces/sets) and it patched its cached state; False if it was a no-op.
        """

    def reconcileIfChanged(self, model) -> bool:
        """Pull-by-version entry point: call this at the consumer's own per-increment tick.

        Returns
        -------
        bool
            True if the model changed since this consumer last checked AND :meth:`reconcile` found
            that change relevant (e.g. this is a direct, correct value to return from a
            :meth:`~edelweissfe.constraints.base.constraintbase.ConstraintBase.updateConnectivity`
            override -- a mesh mutation this consumer didn't care about shouldn't, on its own,
            force an equation-system rebuild).
        """
        if model.topologyVersion == self._lastSeenTopologyVersion:
            return False
        change = model.changesSince(self._lastSeenTopologyVersion)
        self._lastSeenTopologyVersion = model.topologyVersion
        return change is not None and self.reconcile(model, change)
