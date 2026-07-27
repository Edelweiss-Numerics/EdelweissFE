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

"""
Hanging-node constraint for adaptive-mesh-refinement, enforced by master-slave DOF elimination
(multi-point constraint), Abaqus-style -- no Lagrange multipliers, no extra DOFs, no saddle point.

Each hanging (slave) node on a 2:1-refined HEX20 face is tied to the coarse serendipity trace it
hangs on:

.. math::
    u_s = \\sum_a N_a \\, u_{m_a}

for EVERY field active on the node (displacement, nonlocal damage, ...) and every component. Because
the QUAD8 face-trace / quadratic-edge spaces are nested under octree refinement, these constraints
are exact. The records are precomputed (masters + weights, with multi-level chains already flattened
to independent masters) by the adaptivity machinery -- see
:meth:`edelweissfe.adaptivity.refinement.AdaptiveMesh.hanging_mpc_records` -- and written to a plain
records file, one line per slave node::

    <slaveLabel>  <masterLabel> <weight>  <masterLabel> <weight>  ...
"""

from edelweissfe.constraints.base.multipointconstraintbase import (
    MultiPointConstraintBase,
)
from edelweissfe.models.femodel import FEModel
from edelweissfe.utils.caseinsensitivedict import CaseInsensitiveDict
from edelweissfe.utils.inputlanguage import InputLanguage, Module
from edelweissfe.utils.misc import (
    caseInsensitiveKwargsChecker,
    castKwargsValuesAndAddDefaults,
)

module = Module(
    "hangingnode",
    "Hanging-node MPC (DOF elimination) tying refined-side nodes to the coarse serendipity trace.",
)

inputLanguage = InputLanguage()

keyword = "constraint"
if keyword in inputLanguage:
    inputLanguage[keyword].addModule(module)

module.addOptionalArg(
    "recordsFile",
    "Path (relative to the input file) to the flattened hanging-node records: one line per slave "
    "'<slaveLabel> <masterLabel> <weight> ...' with independent masters. Omit for dynamic AMR, where "
    "the adaptivity manager sets the records in memory via updateRecords().",
    str,
    None,
)

documentation = [module]


class Constraint(MultiPointConstraintBase):
    """Hanging-node multi-point constraint enforced by DOF elimination.

    Constrains ALL fields active on each slave node (auto-detected), which is required: every field
    must stay continuous across a hanging node. Weights are field-independent (equal-order
    interpolation), so the same coarse-trace weights apply to each field's components.
    """

    @caseInsensitiveKwargsChecker([kw.name for kw in module.requiredArgs], [kw.name for kw in module.optionalArgs])
    @castKwargsValuesAndAddDefaults(module)
    def __init__(self, name: str, model: FEModel, **kwargs):
        kwargs = CaseInsensitiveDict(kwargs)
        self._name = name
        self._model = model

        # (slaveNode, [(masterNode, weight), ...]); slave node first, so the base class'
        # claimedSlaveNodes() accessor works unmodified for this constraint
        self._records = []
        if kwargs["recordsFile"] is not None:
            with open(kwargs["recordsFile"]) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    tok = line.split()
                    records = {int(tok[0]): [(int(tok[i]), float(tok[i + 1])) for i in range(1, len(tok), 2)]}
                    self.updateRecords(records)

    def updateRecords(self, records: dict):
        """(Re)set the hanging-node records from a {slaveLabel: [(masterLabel, weight), ...]} dict of
        already-flattened (independent-master) records. Used by the dynamic adaptivity manager after
        each refinement, and by the file loader at construction. Appends to any existing records."""
        for slaveLabel, masters in records.items():
            slave = self._model.nodes[slaveLabel]
            masterNodes = [(self._model.nodes[m], w) for m, w in masters]
            self._records.append((slave, masterNodes))

    def setRecords(self, records: dict):
        """Replace all records (dynamic AMR: full regeneration after a mesh change)."""
        self._records = []
        self.updateRecords(records)

    def getMultiPointConstraints(self, dofManager) -> list:
        fieldVariableIndices = dofManager.idcsOfFieldVariablesInDofVector
        records = []
        for slaveNode, masters in self._records:
            # constrain every field the slave shares with all of its masters (all of them, for
            # equal-order elements)
            fields = [f for f in slaveNode.fields if all(f in mNode.fields for mNode, _ in masters)]
            for field in fields:
                slaveDofs = fieldVariableIndices[slaveNode.fields[field]]
                masterDofs = [(fieldVariableIndices[mNode.fields[field]], w) for mNode, w in masters]
                for component in range(len(slaveDofs)):
                    records.append(
                        (
                            slaveDofs[component],
                            [(mDofs[component], w) for mDofs, w in masterDofs if w != 0.0],
                        )
                    )
        return records
