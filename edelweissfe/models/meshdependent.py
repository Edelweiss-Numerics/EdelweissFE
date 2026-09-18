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
DOF numbering, ...) and must patch it up after the model mutates (e.g. an AMR refinement).

This is the *only* mechanism for learning that the mesh changed. A consumer registers once via
:meth:`~edelweissfe.models.femodel.FEModel.registerMeshDependent`, and
:meth:`~edelweissfe.models.femodel.FEModel.refreshMeshDependents` calls it once per increment, after
the model modifiers have run to a fixed point -- so it sees the *net* change across every round,
never a half-finished intermediate state, and it never runs while another component is mid-mutation.

The synchronous push alternative (an observer notified at the instant of each mutation) was removed:
with a multi-round pipeline it fires at moments that are by construction mid-pipeline, so a consumer
is handed a state that no longer exists when the solve begins, and a consumer that mutates in
response does so re-entrantly, inside the modifier's own loop.
"""

from abc import ABC, abstractmethod


class MeshDependent(ABC):
    """Interface for mesh-derived data that must stay consistent across a model mutation."""

    _lastSeenTopologyVersion = 0

    @abstractmethod
    def refresh(self, model, change) -> bool:
        """Patch cached mesh-derived state to account for ``change`` (a
        :class:`~edelweissfe.models.modelchange.ModelChange`). Called only when the model's
        ``topologyVersion`` actually advanced since this consumer last checked.

        **Must not create or delete elements or nodes.** Topology is a model modifier's business,
        and the topology window is closed by the time this runs (see
        :meth:`~edelweissfe.models.femodel.FEModel.topologyChanges`), so an attempt raises.

        Returns
        -------
        bool
            True if ``change`` was actually relevant to this consumer (e.g. touched one of its
            watched surfaces/sets) and it patched its cached state; False if it was a no-op.
        """

    def refreshIfMeshChanged(self, model) -> bool:
        """Pull-by-version entry point, called by
        :meth:`~edelweissfe.models.femodel.FEModel.refreshMeshDependents`.

        Returns
        -------
        bool
            True if the model changed since this consumer last checked AND :meth:`refresh` found
            that change relevant (e.g. this is a direct, correct value to return from a
            :meth:`~edelweissfe.constraints.base.constraintbase.ConstraintBase.updateConnectivity`
            override -- a mesh mutation this consumer didn't care about shouldn't, on its own,
            force an equation-system rebuild).
        """
        if model.topologyVersion == self._lastSeenTopologyVersion:
            return False
        change = model.changesSince(self._lastSeenTopologyVersion)
        self._lastSeenTopologyVersion = model.topologyVersion
        return change is not None and self.refresh(model, change)
