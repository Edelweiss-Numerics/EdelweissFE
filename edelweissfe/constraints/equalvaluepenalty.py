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
from edelweissfe.utils.schema import buildSchemaFromOptions, schemaField

"""
A penalty based constraint used for constraining nodal values
of a node set to be equal.
"""


@dataclass(frozen=True)
class EqualValuePenaltySchema:
    """The options this constraint accepts, owned by this module and never mutated from outside
    it.

    Each field is declared ``required=True``, but is still given a ``default=None`` so that
    ``EqualValuePenaltySchema()`` remains constructible on its own; ``buildSchemaFromOptions``
    still enforces that an ``.inp`` file supplies each.
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


class Constraint(ConstraintBase):
    """A penalty based constraint used for constraining nodal values of a node set to be equal.

    Parameters
    ----------
    name
        The name of the constraint.
    model
        The model tree.
    nSet
        The node set to be constrained.
    configuration
        The options this constraint accepts; all are still required, see
        :class:`EqualValuePenaltySchema`.
    """

    #: Option schema for this constraint, per OptionSchemaProvider.
    schema = EqualValuePenaltySchema

    def __init__(
        self,
        name: str,
        model: FEModel,
        nSet: NodeSet,
        *,
        configuration: EqualValuePenaltySchema = EqualValuePenaltySchema(),
    ):
        super().__init__(name, model)

        self.theField = configuration.field
        self.sizeField = getFieldSize(self.theField, model.domainSize)
        self.component = configuration.component
        self.penalty = configuration.penalty
        self._nodes = nSet

        self.active = True

        self._rebuildDerivedState()

    @classmethod
    def fromConstraintDefinition(
        cls, name: str, definition: dict, model: FEModel, journal: "Journal" = None
    ) -> "Constraint":
        """Build this constraint from a parsed ``*constraint`` definition. See
        :class:`~edelweissfe.constraints.base.constraintbase.ConstraintBase` for why this is
        separate from ``__init__``."""
        configuration = buildSchemaFromOptions(cls.schema, definition)
        return cls(name, model, model.nodeSets[configuration.nSet], configuration=configuration)

    def _rebuildDerivedState(self):
        """(Re)derive every quantity sized to the constrained node set -- the node count, ``nDof``,
        the component index slice and the field list -- from its *current* size. Called once at
        construction and again, lazily, from :meth:`updateConnectivity` whenever the node set was
        mutated in-place (e.g. by AMR) since the last increment."""

        self._nNodes = len(self._nodes)
        self._nDof = self.sizeField * self._nNodes

        self.indices_component = slice(self.component, self._nDof + self.component, self.sizeField)

        self._fieldsOnNodes = [[self.theField]] * self._nNodes

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
        """Called once per increment, before the equation system is (re)built. Recomputes the
        node-set-sized derived state (see :meth:`_rebuildDerivedState`) if the constrained node set
        was mutated in-place since the last check, and reports the change so the caller rebuilds
        the equation system even on an increment where nothing else did."""

        if self._checkSetChanged(self._nodes):
            self._rebuildDerivedState()
            return True
        return False

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

        mean = np.sum(values) / self._nNodes

        PExt[self.indices_component] -= self.penalty * (values - mean)

        diag = np.diag(K)
        diag.setflags(write=True)  # bug in numpy
        diag[self.indices_component] += self.penalty
        K[self.indices_component, self.indices_component] += -self.penalty * 1.0 / self._nNodes
