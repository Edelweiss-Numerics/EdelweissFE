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
# Created on Fri Sep 9 11:34:35 2022

# @author: Matthias Neuner

from dataclasses import dataclass

import numpy as np

from edelweissfe.config.phenomena import getFieldSize
from edelweissfe.constraints.base.constraintbase import ConstraintBase
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.inputlanguage import InputLanguage, Module
from edelweissfe.utils.schema import buildSchemaFromOptions, schemaField

"""
A lagrangian multiplier based constraint used for constraining nodal values
of a node set to be equal.
"""

# documentation = {
#     "field": "The field this constraint acts on.",
#     "component": "The component of the field.",
#     "nSet": "The node set to be constrained.",
# }

module = Module(
    "equalvaluelagrangian",
    "A lagrangian multiplier based constraint used for constraining nodal values of a node set to be equal.",
)

inputLanguage = InputLanguage()

keyword = "constraint"
if keyword in inputLanguage:
    inputLanguage[keyword].addModule(module)

module.addRequiredArg("field", "The field this constraint acts on.", str)
module.addRequiredArg("component", "The component of the field.", int)
module.addRequiredArg("nSet", "The node set to be constrained.", str)

documentation = [module]


@dataclass(frozen=True)
class EqualValueLagrangianSchema:
    """L2: the options this constraint accepts, owned by this module and never mutated from
    outside it.

    Mirrors the ``module.addRequiredArg(...)`` declarations above one-for-one. The two declarations
    coexist while the migration is in progress; the ``Module`` one goes away with the
    ``InputLanguage`` singleton in P5.

    Each field is declared ``required=True`` explicitly, mirroring ``addRequiredArg`` above, but is
    still given a ``default=None`` so that ``EqualValueLagrangianSchema()`` remains constructible
    for the L1 constructor's default argument; the L4 adapter (``buildSchemaFromOptions``) still
    enforces that an ``.inp`` file supplies each.
    """

    field: str | None = schemaField(
        description="The field this constraint acts on.", dtype=str, default=None, required=True
    )
    component: int | None = schemaField(
        description="The component of the field.", dtype=int, default=None, required=True
    )
    nSet: str | None = schemaField(
        description="The node set to be constrained.", dtype=str, default=None, required=True
    )


class Constraint(ConstraintBase):
    """A Lagrangian multiplier based constraint enforcing equal nodal values on a node set.

    .. note::
       This constraint requires a node set of *fixed* size. It needs ``len(nSet) - 1`` Lagrange
       multipliers, and those scalar variables are allocated exactly once, when the equation system
       is first set up. A node set that grows in-place during the simulation (e.g. by adaptive mesh
       refinement) can therefore not be supported, and is rejected in :meth:`updateConnectivity`.
       Use the penalty variant ``equalValuePenalty`` instead, which re-sizes itself per increment.

    Parameters
    ----------
    name
        The name of the constraint.
    model
        The model tree.
    nSet
        The node set to be constrained.
    configuration
        The options this constraint accepts; both are still required, see
        :class:`EqualValueLagrangianSchema`.
    """

    #: L2 schema declared for the L3 registry, per OptionSchemaProvider.
    schema = EqualValueLagrangianSchema

    def __init__(
        self,
        name: str,
        model: FEModel,
        nSet: NodeSet,
        *,
        configuration: EqualValueLagrangianSchema = EqualValueLagrangianSchema(),
    ):
        super().__init__(name, model)

        theField = configuration.field
        self.sizeField = getFieldSize(theField, model.domainSize)
        self.component = configuration.component
        self._name = name
        self._nodes = nSet
        self.nNodes = len(self._nodes)
        self.nMultipliers = len(self._nodes) - 1

        self._nDof = self.sizeField * self.nNodes + self.nMultipliers

        self.index_master = self.component
        self.indices_slaves = slice(
            self.sizeField + self.component,
            self.sizeField * self.nNodes,
            self.sizeField,
        )
        self.indices_multipliers = slice(self.sizeField * self.nNodes, self._nDof)

        self._fieldsOnNodes = [
            [
                theField,
            ]
        ] * self.nNodes

        self.active = True

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
        return self._fieldsOnNodes

    @property
    def nDof(self) -> int:
        return self._nDof

    def updateConnectivity(self, model) -> bool:
        """Called once per increment, before the equation system is (re)built.

        This constraint cannot adapt to a node set that changed size: it would need
        ``len(nSet) - 1`` Lagrange multipliers, but its scalar variables were already allocated for
        the original size and are not re-allocated per increment. Rebuilding the derived state
        would silently emit more DOF indices than ``nDof`` accounts for, so we fail loudly instead.

        Parameters
        ----------
        model
            The current model.

        Returns
        -------
        bool
            Always False; this constraint's DOF footprint is fixed at construction.

        Raises
        ------
        Exception
            If the constrained node set was mutated in-place since the last check.
        """

        if self._checkSetChanged(self._nodes):
            raise Exception(
                "Constraint '{:}' (equalValueLagrangian) does not support a node set that changes "
                "size during the simulation: its {:} Lagrange multipliers are allocated once, when "
                "the equation system is first set up, and cannot grow with the node set (e.g. under "
                "adaptive mesh refinement). Use the penalty variant 'equalValuePenalty' on this node "
                "set instead, or constrain a node set that is not refined.".format(self._name, self.nMultipliers)
            )
        return False

    def getNumberOfAdditionalNeededScalarVariables(self):
        return self.nNodes - 1

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

        val_master = U_np[self.index_master]
        vals_slaves = U_np[self.indices_slaves]
        multipliers = U_np[self.indices_multipliers]

        gs = vals_slaves - val_master

        PExt[self.index_master] -= -np.sum(multipliers)
        PExt[self.indices_slaves] -= multipliers
        PExt[self.indices_multipliers] -= gs

        K[self.index_master, self.indices_multipliers] += -1.0

        diag_sm = np.diag(K[self.indices_slaves, self.indices_multipliers])
        diag_sm.setflags(write=True)  # bug in numpy
        diag_sm[:] += 1.0

        K[self.indices_multipliers, self.index_master] = K[self.index_master, self.indices_multipliers].T
        K[self.indices_multipliers, self.indices_slaves] = K[self.indices_slaves, self.indices_multipliers].T
