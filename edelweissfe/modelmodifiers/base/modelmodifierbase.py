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

"""Abstract base class for all ModelModifier entities in EdelweissFE."""

from abc import ABC, abstractmethod

from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.utils.schema import OptionSchemaProvider


class ModelModifierBase(OptionSchemaProvider, ABC):
    """Abstract base class for entities that dynamically mutate the FEModel topology,
    mesh, or state variables during analysis steps.
    """

    #: Whether this modifier can start a topology change by itself. A purely reactive modifier sets
    #: this to False: it only ever plans in response to another modifier's mutation, so in a solver
    #: that never runs the topology update it cannot do anything, and its presence alone must not
    #: disqualify that solver. See NonlinearSolverBase.validateModelCapabilities.
    initiatesTopologyChanges = True

    def __init__(self, name: str, model: FEModel, journal: Journal, **kwargs):
        self._name = name
        self._model = model
        self._journal = journal

    @property
    def name(self) -> str:
        """Name of the model modifier."""
        return self._name

    @abstractmethod
    def plan(self, model: FEModel, change, step, timeStep: float):
        """Decide what, if anything, this modifier wants to change about the model -- without
        changing it.

        This half is allowed to read solution state: markers, node fields, the current time, the
        step. It must return a **serializable** description of the decision, which
        :meth:`apply` can carry out on its own, or ``None`` if there is nothing to do.

        A restart replay never calls this. It reads the recorded plan and calls :meth:`apply`
        directly, which is what makes a resumed run reproduce the original topology exactly instead
        of re-deciding from state that may not be identical.

        Parameters
        ----------
        model
            The FEModel object. Read only -- do not mutate it here.
        change
            The net :class:`~edelweissfe.models.modelchange.ModelChange` since this modifier last
            planned within the current topology update, or ``None`` on the first round. **Return
            ``None`` when it does not touch this modifier's domain** -- that is what lets the
            pipeline reach a fixed point instead of looping (see
            :meth:`~edelweissfe.models.femodel.FEModel.updateTopology`).
        step
            The current step.
        timeStep
            The current timeStep.

        Returns
        -------
        object | None
            A plan for :meth:`apply`, or ``None``.
        """

    @abstractmethod
    def apply(self, model: FEModel, plan):
        """Carry out ``plan``, mutating the model.

        **Must not read solution state.** A pure function of ``(model, plan)`` -- everything the
        decision depended on belongs in the plan. This is the single code path shared by a live run
        and a restart replay, and the reason a replayed mesh is numbered identically: there is no
        second implementation to drift apart from this one.

        Runs inside an open topology window (see
        :meth:`~edelweissfe.models.femodel.FEModel.topologyChanges`), so it may create and delete
        elements -- through :meth:`~edelweissfe.models.femodel.FEModel.reserveElementNumbers` and
        :meth:`~edelweissfe.models.femodel.FEModel.createElement`, never by writing
        ``model.elements`` directly.

        Parameters
        ----------
        model
            The FEModel object, mutated in place.
        plan
            A plan previously returned by :meth:`plan` (live) or restored from the recorded
            topology history (replay).

        Returns
        -------
        ModelChange
            The changeset this mutation produced.
        """

    @abstractmethod
    def encodePlan(self, plan) -> dict:
        """Serialize a plan for the topology history, as a flat mapping of name to
        :class:`numpy.ndarray` (or scalar).

        Everything :meth:`apply` needs must be here: on a restart, the plan is rebuilt from this and
        nothing else -- :meth:`plan` is not called.
        """

    @abstractmethod
    def decodePlan(self, data: dict):
        """Inverse of :meth:`encodePlan`."""

    def declaredDomain(self, model: FEModel) -> set:
        """The element numbers this modifier claims exclusive authority over.

        Two modifiers that both own an element will fight over it: each mutates ``model.elements``
        directly, so one deleting or subdividing an element the other still tracks leaves the second
        holding a stale reference, which later corrupts element-set membership and can surface as a
        node that is simultaneously Dirichlet-prescribed and a multi-point-constraint slave.

        :meth:`~edelweissfe.models.femodel.FEModel.checkModelModifierDomains` compares these
        pairwise once, at the end of setup, and refuses the model rather than letting the conflict
        appear deep in the solve loop. The default claims nothing, which is correct for a modifier
        that only ever *adds* entities.
        """

        return set()

    def restoreDecisionState(self, records):
        """Restore whatever *decision-side* state :meth:`plan` needs, from this modifier's own
        records, after a restart replay.

        Optional -- the default does nothing, which is correct for any modifier whose next decision
        depends only on the model and the solution state, both of which the restart restores anyway.
        Deliberately separate from :meth:`apply`: this is about how the *next* decision is made, not
        about reconstructing the model, so getting it wrong cannot corrupt the mesh.

        Parameters
        ----------
        records
            This modifier's :class:`~edelweissfe.models.modelchange.TopologyRecord` entries, in
            order. Empty if it never changed anything.
        """

    def onStepStart(self, model: FEModel, step):
        """Optional lifecycle hook called at the start of an analysis step."""

    def onIncrementEnd(self, model: FEModel, step, timeStep: float):
        """Optional lifecycle hook called after an increment converges."""

    # getRestartData/setRestartData are gone. A modifier no longer serializes its own history, nor
    # implements its own replay: FEModel records every applied plan in model.topologyHistory and
    # replays it through this class's apply(). The previous arrangement had each modifier
    # reimplementing the mutation for the replay path, which is precisely why a resumed run could
    # rebuild a differently-numbered mesh -- two implementations of one mutation always drift.
    # Decision-side state that plan() needs goes through restoreDecisionState() instead.
