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
# Created on Sun May 21 11:34:35 2017

# @author: Matthias Neuner

from dataclasses import dataclass

import numpy as np

from edelweissfe.constraints.base.constraintbase import ConstraintBase
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.stepactions.base.amplitude import (
    amplitudeFromExpression,
    linearAmplitude,
)
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.caseinsensitivedict import CaseInsensitiveDict
from edelweissfe.utils.inputlanguage import InputLanguage, Module
from edelweissfe.utils.schema import buildSchemaFromOptions, schemaField

"""
A penalty based constraint used for indirect (displacement) control.
"""

module = Module("penaltyindirectcontrol", "A penalty based constraint used for indirect (displacement) control.")

inputLanguage = InputLanguage()

keyword = "constraint"
if keyword in inputLanguage:
    inputLanguage[keyword].addModule(module)

module.addOptionalArg("field", "The field this constraint acts on.", str, "displacement")
module.addRequiredArg("cVector", "The projection vector for the constrained nodes (e.g., CMOD).", str)
module.addRequiredArg("constrainedNSet", "The node set for determining the constraint (e.g., CMOD).", str)
module.addRequiredArg("loadNSet", "The node set for application of the controlled load.", str)
module.addRequiredArg(
    "loadVector", "The vector (in correct) dimensions and tensorial order  determining the load.", str
)
module.addRequiredArg("length", "The value of the constraint (e.g., CMOD).", float)
module.addRequiredArg("penaltyStiffness", "The stiffness for formulating the constraint.", float)
module.addOptionalArg(
    "offset", "A correction value for the computation of the constraint (e.g, initial displacement).", float, 0.0
)
module.addOptionalArg(
    "normalizeLoad",
    "Normalize the applied force per node w. r. t. the number of nodes, i.e., apply a load irrespective of the total number of nodes in ``loadNSet``.",
    bool,
    True,
)
module.addOptionalArg("f(t)", "Amplitude function.", str, None)

documentation = [module]


@dataclass(frozen=True)
class PenaltyIndirectControlSchema:
    """L2: the options this constraint accepts, owned by this module and never mutated from
    outside it.

    Mirrors the ``module.addRequiredArg``/``module.addOptionalArg(...)`` declarations above
    one-for-one. The two declarations coexist while the migration is in progress; the ``Module``
    one goes away with the ``InputLanguage`` singleton in P5.

    ``f_t`` is spelled ``f(t)`` in the input file, which is not a valid Python identifier -- hence
    the ``optionName`` indirection, see :func:`edelweissfe.utils.schema.schemaField`. Each required
    field is declared ``required=True`` explicitly, but is still given a ``default=None`` so the
    schema remains constructible for the L1 constructor's default argument.
    """

    field: str = schemaField(description="The field this constraint acts on.", dtype=str, default="displacement")
    cVector: str | None = schemaField(
        description="The projection vector for the constrained nodes (e.g., CMOD).",
        dtype=str,
        default=None,
        required=True,
    )
    loadVector: str | None = schemaField(
        description="The vector (in correct) dimensions and tensorial order  determining the load.",
        dtype=str,
        default=None,
        required=True,
    )
    length: float | None = schemaField(
        description="The value of the constraint (e.g., CMOD).", dtype=float, default=None, required=True
    )
    penaltyStiffness: float | None = schemaField(
        description="The stiffness for formulating the constraint.", dtype=float, default=None, required=True
    )
    offset: float = schemaField(
        description="A correction value for the computation of the constraint (e.g, initial displacement).",
        dtype=float,
        default=0.0,
    )
    normalizeLoad: bool = schemaField(
        description="Normalize the applied force per node w. r. t. the number of nodes, i.e., "
        "apply a load irrespective of the total number of nodes in ``loadNSet``.",
        dtype=bool,
        default=True,
    )
    f_t: str | None = schemaField(description="Amplitude function.", dtype=str, default=None, optionName="f(t)")


class Constraint(ConstraintBase):
    """A penalty based constraint used for indirect (displacement) control.

    Parameters
    ----------
    name
        The name of the constraint.
    model
        The model tree.
    constrainedNSet
        The node set for determining the constraint (e.g., CMOD).
    loadNSet
        The node set for application of the controlled load.
    configuration
        The options this constraint accepts; ``cVector``/``loadVector``/``length``/
        ``penaltyStiffness`` are still required, see :class:`PenaltyIndirectControlSchema`.
    """

    #: L2 schema declared for the L3 registry, per OptionSchemaProvider.
    schema = PenaltyIndirectControlSchema

    def __init__(
        self,
        name,
        model: FEModel,
        constrainedNSet: NodeSet,
        loadNSet: NodeSet,
        *,
        configuration: PenaltyIndirectControlSchema = PenaltyIndirectControlSchema(),
    ):
        super().__init__(name, model)

        self.theField = configuration.field

        self.cVector = np.fromstring(configuration.cVector, dtype=float, sep=",")
        self.constrainedNSet = constrainedNSet
        self.loadNSet = loadNSet

        # the target total load, never mutated -- the per-node share (self.unitResidual, built in
        # _rebuildDerivedState) is (re)derived from this against the *current* size of loadNSet, so
        # a mid-run AMR growth of loadNSet keeps the documented normalizeLoad semantics (constant
        # total load, or a fixed per-node load) instead of freezing them at construction-time size
        self._targetLoadVector = np.fromstring(configuration.loadVector, dtype=float, sep=",")
        self._normalizeLoad = configuration.normalizeLoad

        self.penaltyStiffness = configuration.penaltyStiffness
        self.length = configuration.length

        self.amplitude = amplitudeFromExpression(configuration.f_t) or linearAmplitude

        self.offset = configuration.offset

        self._nDim = model.domainSize

        self.active = True

        self.constrainedValue = 0.0

        self._rebuildDerivedState()

    @classmethod
    def fromConstraintDefinition(
        cls, name: str, definition: dict, model: FEModel, journal: "Journal" = None
    ) -> "Constraint":
        """Build this constraint from a parsed ``*constraint`` definition. See
        :class:`~edelweissfe.constraints.base.constraintbase.ConstraintBase` for why this is
        separate from ``__init__``."""
        definition = CaseInsensitiveDict(definition)
        constrainedNSetName = definition.pop("constrainedNSet")
        loadNSetName = definition.pop("loadNSet")
        configuration = buildSchemaFromOptions(cls.schema, definition)
        return cls(
            name,
            model,
            model.nodeSets[constrainedNSetName],
            model.nodeSets[loadNSetName],
            configuration=configuration,
        )

    def _rebuildDerivedState(self):
        """(Re)derive every array/index sized to ``loadNSet``/``constrainedNSet`` -- the node
        list, the field list, the DOF-block boundaries, ``nDof`` and the per-node unit residual --
        from their *current* size. Called once at construction and again, lazily, from
        :meth:`updateConnectivity` whenever either watched node set was mutated in-place (e.g. by
        AMR) since the last increment."""
        self._nodes = list(self.loadNSet) + list(self.constrainedNSet)

        self._fieldsOnNodes = [
            [
                self.theField,
            ]
        ] * len(self._nodes)

        sizeBlock_loadNodes = self._nDim * len(self.loadNSet)
        self.startBlock_loadNodes = 0
        self.endBlock_loadNodes = sizeBlock_loadNodes

        sizeBlock_constrainedNodes = self._nDim * len(self.constrainedNSet)
        self.startBlock_constrainedNodes = self.endBlock_loadNodes
        self.endBlock_constrainedNodes = self.startBlock_constrainedNodes + sizeBlock_constrainedNodes

        self._nDof = self.endBlock_constrainedNodes

        # we may normalize in order to end up with an identical total load irrespective of the
        # number of nodes in the load node set
        perNodeLoad = self._targetLoadVector
        if self._normalizeLoad:
            perNodeLoad = perNodeLoad / len(self.loadNSet)
        self.unitResidual = np.tile(perNodeLoad, len(self.loadNSet))

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
        node lists, DOF-block boundaries and unit residual (see :meth:`_rebuildDerivedState`) if
        either watched node set was mutated in-place since the last check, and reports the change
        so the caller rebuilds the equation system even on an increment where nothing else did."""
        loadChanged = self._checkSetChanged(self.loadNSet)
        constrainedChanged = self._checkSetChanged(self.constrainedNSet)
        if loadChanged or constrainedChanged:
            self._rebuildDerivedState()
            return True
        return False

    def applyConstraint(self, U_np, dU, PExt, K, timeStep: TimeStep):
        if not self.active:
            return

        sBL = self.startBlock_loadNodes
        eBL = self.endBlock_loadNodes
        sBC = self.startBlock_constrainedNodes
        eBC = self.endBlock_constrainedNodes

        U_c = U_np[sBC:eBC]

        L = self.length * self.amplitude(timeStep.stepProgress)

        cVector = self.cVector

        self.constrainedValue = cVector.dot(U_c)

        loadFactor = self.penaltyStiffness * (self.constrainedValue - self.offset - L)
        dLoadFactor_ddU = self.penaltyStiffness * cVector

        t = self.unitResidual

        PExt[sBL:eBL] = -t * loadFactor
        K[sBL:eBL, sBC:eBC] = np.outer(t, dLoadFactor_ddU)
