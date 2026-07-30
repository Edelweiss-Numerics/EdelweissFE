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
#  Konstantin Basche konstantin.basche@uibk.ac.at
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
# Created on Thu Mar 26 10:21:35 2026

# @author: Konstantin Basche

from dataclasses import dataclass

import numpy as np

from edelweissfe.config.phenomena import getFieldSize
from edelweissfe.constraints.base.constraintbase import ConstraintBase
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.models.meshdependent import MeshDependent
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.inputlanguage import InputLanguage, Module
from edelweissfe.utils.schema import buildSchemaFromOptions, schemaField

"""
A penalty based unilateral constraint used for preventing the nodes of a node set from penetrating a defined rigid boundary.

This constraint is a :class:`~edelweissfe.models.meshdependent.MeshDependent`: if an AMR refinement
adds nodes to the watched ``nSet``, the new nodes are picked up and protected from penetrating the
boundary at the constraint's own next :meth:`updateConnectivity` tick -- no separate wiring needed.
"""

module = Module(
    "nodeToRigidSurfacePenalty",
    "A penalty based unilateral constraint used for preventing the nodes of a node set from penetrating a defined rigid boundary.",
)

inputLanguage = InputLanguage()

keyword = "constraint"
if keyword in inputLanguage:
    inputLanguage[keyword].addModule(module)

module.addRequiredArg("field", "The field this constraint acts on.", str)
module.addRequiredArg("component", "The component of the field.", int)
module.addRequiredArg("penalty", "The numerical penalty value.", float)
module.addRequiredArg("nSet", "The node set to be constrained.", str)

module.addOptionalArg(
    "value",
    "The prescribed distance to the rigid boundary. A value of 0.0 implies no initial gap between the node set and the boundary.",
    float,
    0.0,
)
module.addOptionalArg(
    "direction", "The normal direction outward from the continuum towards the boundary (1.0 or -1.0).", float, 1.0
)
module.addOptionalArg(
    "type",
    "The formulation type: 'linear' (linear force, constant stiffness with jump) or 'quadratic' (quadratic force, linear stiffness).",
    str,
    "linear",
)

documentation = [module]


@dataclass(frozen=True)
class NodeToRigidSurfacePenaltySchema:
    """L2: the options this constraint accepts, owned by this module and never mutated from
    outside it.

    Mirrors the ``module.addRequiredArg``/``module.addOptionalArg(...)`` declarations above
    one-for-one. The two declarations coexist while the migration is in progress; the ``Module``
    one goes away with the ``InputLanguage`` singleton in P5.

    The update-type option is spelled ``type`` in the input file but the field is named
    ``contactType`` here -- a dataclass field literally called ``type`` would shadow the builtin,
    which this project's conventions avoid. Each required field is declared ``required=True``
    explicitly, but is still given a ``default=None`` so the schema remains constructible for the
    L1 constructor's default argument; the L4 adapter (``buildSchemaFromOptions``) still enforces
    that an ``.inp`` file supplies it.
    """

    field: str | None = schemaField(
        description="The field this constraint acts on.", dtype=str, default=None, required=True
    )
    component: int | None = schemaField(
        description="The component of the field.", dtype=int, default=None, required=True
    )
    penalty: float | None = schemaField(
        description="The numerical penalty value.", dtype=float, default=None, required=True
    )
    nSet: str | None = schemaField(
        description="The node set to be constrained.", dtype=str, default=None, required=True
    )
    value: float = schemaField(
        description="The prescribed distance to the rigid boundary. A value of 0.0 implies no "
        "initial gap between the node set and the boundary.",
        dtype=float,
        default=0.0,
    )
    direction: float = schemaField(
        description="The normal direction outward from the continuum towards the boundary (1.0 or -1.0).",
        dtype=float,
        default=1.0,
    )
    contactType: str = schemaField(
        description="The formulation type: 'linear' (linear force, constant stiffness with jump) "
        "or 'quadratic' (quadratic force, linear stiffness).",
        dtype=str,
        default="linear",
        optionName="type",
    )


class Constraint(ConstraintBase, MeshDependent):
    """A penalty based unilateral constraint used for preventing the nodes of a node set from
    penetrating a defined rigid boundary.

    Parameters
    ----------
    name
        The name of the constraint.
    model
        The model tree.
    nSet
        The node set to be constrained.
    configuration
        The options this constraint accepts; ``field``/``component``/``penalty`` are still
        required, see :class:`NodeToRigidSurfacePenaltySchema`.
    """

    #: L2 schema declared for the L3 registry, per OptionSchemaProvider.
    schema = NodeToRigidSurfacePenaltySchema

    def __init__(
        self,
        name: str,
        model: FEModel,
        nSet: NodeSet,
        journal: Journal,
        *,
        configuration: NodeToRigidSurfacePenaltySchema = NodeToRigidSurfacePenaltySchema(),
    ):
        super().__init__(name, model)

        self.journal = journal

        self._field = configuration.field
        self.sizeField = getFieldSize(self._field, model.domainSize)
        self.component = configuration.component
        self.penalty = configuration.penalty
        self.value = configuration.value
        self.direction = configuration.direction

        self.type = configuration.contactType.lower()
        if self.type not in ["linear", "quadratic"]:
            raise ValueError(f"Constraint type '{self.type}' is not supported. Use 'linear' or 'quadratic'.")

        self._nSetName = nSet.name
        self._lastSeenTopologyVersion = model.topologyVersion
        self._nodes = nSet
        self._rebuildFromNodes()

        self.active = True

        self.journal.message(
            f"{self._nNodes} nodes, field={self._field}, component={self.component}, "
            f"type={self.type}, direction={self.direction:+.1f}",
            name,
        )

    @classmethod
    def fromConstraintDefinition(
        cls, name: str, definition: dict, model: FEModel, journal: "Journal" = None
    ) -> "Constraint":
        """Build this constraint from a parsed ``*constraint`` definition. See
        :class:`~edelweissfe.constraints.base.constraintbase.ConstraintBase` for why this is
        separate from ``__init__``."""
        configuration = buildSchemaFromOptions(cls.schema, definition)
        return cls(name, model, model.nodeSets[configuration.nSet], journal, configuration=configuration)

    def _rebuildFromNodes(self) -> None:
        """(Re)derive every quantity that depends on the node set/count."""

        self._nNodes = len(self._nodes)
        self._nDof = self.sizeField * self._nNodes
        self.indices_component = np.arange(self.component, self._nDof + self.component, self.sizeField)
        self._fieldsOnNodes = [[self._field]] * self._nNodes

    def reconcile(self, model: FEModel, change) -> bool:
        """Refresh the node list from the (possibly grown) watched ``nSet``."""

        if not change.touchesNodeSet(self._nSetName):
            return False
        self._nodes = model.nodeSets[self._nSetName]
        self._rebuildFromNodes()
        return True

    def updateConnectivity(self, model: FEModel) -> bool:
        return self.reconcileIfChanged(model)

    @property
    def nodes(self) -> list:
        return self._nodes

    @property
    def fieldsOnNodes(self) -> list:
        return self._fieldsOnNodes

    @property
    def nDof(self) -> int:
        return self._nDof

    def applyConstraint(
        self,
        U_np: np.ndarray,
        dU: np.ndarray,
        PExt: np.ndarray,
        K: np.ndarray,
        timeStep: TimeStep,
    ):
        if not self.active:
            return

        values = U_np[self.indices_component]
        gap = (values - self.value) * self.direction

        active_mask = gap > 0

        if not np.any(active_mask):
            return

        active_indices = self.indices_component[active_mask]
        active_gaps = gap[active_mask]

        if self.type == "linear":
            force_magnitude = self.penalty * active_gaps
            stiffness = self.penalty
        elif self.type == "quadratic":
            force_magnitude = 0.5 * self.penalty * active_gaps**2
            stiffness = self.penalty * active_gaps

        PExt[active_indices] -= force_magnitude * self.direction
        K[active_indices, active_indices] += stiffness
