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

from dataclasses import dataclass

import numpy as np

from edelweissfe.constraints.base.constraintbase import ConstraintBase
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.schema import buildSchemaFromOptions, schemaField

"""
Acceptance test double for the "topological containers have stable identity" contract
(see ``PLAN_TRANSPARENT_AMR.md``): caches a node set reference exactly the way an ordinary,
AMR-unaware constraint would -- ``self._nodes = model.nodeSets[nSet]`` at construction, never
re-fetched -- and implements neither :class:`~edelweissfe.models.modelchangeobserver.
ModelChangeObserver` nor :class:`~edelweissfe.models.meshdependent.MeshDependent`. It contributes
zero degrees of freedom and touches no field, so it never affects the converged solution; its only
purpose is to raise if its cached node set ever fails to reflect a mid-run mesh refinement, which
would mean AMR silently reintroduced replacing a set instead of mutating it in place. It has no use
outside test suites and is not intended as a template for an actual boundary condition.
"""


@dataclass(frozen=True)
class AmrTransparencyProbeSchema:
    """L2: the options this constraint accepts, owned by this module and never mutated from
    outside it.

    Its only option is the structural ``nSet`` it watches -- a node set *name*, resolved to the
    actual node set in :meth:`Constraint.fromConstraintDefinition`. It is declared ``required=True``
    explicitly, but still given a ``default=None`` so the schema remains constructible for the L1
    constructor's default argument."""

    nSet: str | None = schemaField(
        description="The node set whose growth under AMR this probe verifies.",
        dtype=str,
        default=None,
        required=True,
    )


class Constraint(ConstraintBase):
    """A zero-DOF constraint that caches a node set reference at construction time and checks,
    lazily at its own :meth:`updateConnectivity` tick, that the reference already reflects any
    mesh refinement the model reports via :attr:`~edelweissfe.models.femodel.FEModel.topologyVersion`
    -- without ever re-fetching the set from the model or registering as an observer.

    Parameters
    ----------
    name
        The name of the constraint.
    model
        The full finite element model.
    nSet
        The node set to watch; typically one AMR is expected to grow (e.g. a tracked boundary set).
    configuration
        The options this constraint accepts; there are none beyond ``nSet``.
    """

    #: L2 schema declared for the L3 registry, per OptionSchemaProvider.
    schema = AmrTransparencyProbeSchema

    def __init__(
        self,
        name: str,
        model: FEModel,
        nSet: NodeSet,
        *,
        configuration: AmrTransparencyProbeSchema = AmrTransparencyProbeSchema(),
    ):
        super().__init__(name, model)
        self.name = name
        self._nodes = nSet
        self._initialNodeCount = len(self._nodes)
        self._lastSeenTopologyVersion = model.topologyVersion

    @classmethod
    def fromConstraintDefinition(
        cls, name: str, definition: dict, model: FEModel, journal: "Journal" = None
    ) -> "Constraint":
        """Build this constraint from a parsed ``*constraint`` definition. See
        :class:`~edelweissfe.constraints.base.constraintbase.ConstraintBase` for why this is
        separate from ``__init__``."""
        configuration = buildSchemaFromOptions(cls.schema, definition)
        return cls(name, model, model.nodeSets[configuration.nSet], configuration=configuration)

    @property
    def nodes(self) -> list:
        return self._nodes

    @property
    def fieldsOnNodes(self) -> list:
        return [[] for _ in self._nodes]

    @property
    def nDof(self) -> int:
        return 0

    def updateConnectivity(self, model: FEModel) -> bool:
        """Called once per increment, before the equation system is (re)built -- exactly the tick
        at which a freshly refined mesh's ``topologyVersion`` has already advanced. Raises if the
        cached node set has not grown accordingly, i.e. if it is still the pre-refinement object
        (a stale reference) rather than the same, in-place-mutated one."""
        if model.topologyVersion != self._lastSeenTopologyVersion:
            self._lastSeenTopologyVersion = model.topologyVersion
            if len(self._nodes) <= self._initialNodeCount:
                raise RuntimeError(
                    "AMR transparency probe failed: constraint '{:}' still sees {:} nodes (the "
                    "construction-time count) after the model topologyVersion advanced -- the "
                    "cached NodeSet reference went stale.".format(self.name, self._initialNodeCount)
                )
        return False

    def applyConstraint(
        self,
        U_np: np.ndarray,
        dU: np.ndarray,
        PExt: np.ndarray,
        V: np.ndarray,
        timeStep: TimeStep,
    ):
        pass
