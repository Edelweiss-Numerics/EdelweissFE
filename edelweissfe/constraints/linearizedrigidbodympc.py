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
Linearized rigid body constraint enforced by master-slave DOF elimination (multi-point constraint):
the kinematics of :mod:`~edelweissfe.constraints.linearizedrigidbody`, without its Lagrange
multipliers, so without extra DOFs and without a saddle-point system (iterative linear solvers such
as ``blockamg`` need a nonzero diagonal).

The displacement of every slave node :math:`s` follows the displacement and the (small) rotation of
the reference point (RP),

.. math::
    \\mathbf{u}_s = \\mathbf{u}_{RP} + \\boldsymbol{\\phi}_{RP} \\times \\mathbf{d}_s, \\quad
    \\mathbf{d}_s = \\mathbf{x}_s - \\mathbf{x}_{RP},

in 3D (rotation field with 3 components), and in 2D (one rotation :math:`\\phi` about z)

.. math::
    u_{s,x} = u_{RP,x} - \\phi \\, d_{s,y}, \\quad u_{s,y} = u_{RP,y} + \\phi \\, d_{s,x}.

The slave displacements are eliminated; the displacement and the rotation of the RP are the masters.
Prescribe Dirichlet conditions on the RP only, not on the slave nodes: a slave degree of freedom that
is also prescribed loses its record to the Dirichlet condition (reported by the solver), so the body
is no longer rigid at that node.

This constraint is a :class:`~edelweissfe.models.meshdependent.MeshDependent`: if adaptive refinement
changes the slave node set (e.g. by :mod:`~edelweissfe.modelmodifiers.adaptivity.hadaptivity`), the
records are rebuilt from the current members, so new nodes are tied as well.

Limitation: static solvers only. The RP carries no mass of its own, and both dynamic solvers check
the assembled inertia before the multi-point constraint enters, so they reject the RP's degrees of
freedom (NID: no mass on a time-integrated DOF, as for
:mod:`~edelweissfe.constraints.linearizedrigidbody`; NED: an inertia on only some DOFs of a field).
For NED, folding the slave masses onto the RP would not suffice: the row-sum lumping used for ties
gives the RP rotation the signed static moment of the slave masses instead of a rotational inertia.
"""

from dataclasses import dataclass

from edelweissfe.constraints.base.multipointconstraintbase import (
    MultiPointConstraintBase,
)
from edelweissfe.constraints.linearizedrigidbody import rigidBodyKinematicsCoefficients
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.models.meshdependent import MeshDependent
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.utils.exceptions import WrongDomain
from edelweissfe.utils.schema import buildSchemaFromOptions, schemaField


@dataclass(frozen=True)
class LinearizedRigidBodyMPCSchema:
    """The options this constraint accepts, owned by this module and never mutated from outside
    it: the node set *names*, resolved in :meth:`Constraint.fromConstraintDefinition`."""

    nSet: str | None = schemaField(description="Node set to tie.", dtype=str, default=None, required=True)
    referencePoint: str | None = schemaField(
        description="Node set containing only the reference point.", dtype=str, default=None, required=True
    )


class Constraint(MultiPointConstraintBase, MeshDependent):
    """Linearized rigid body constraint as a multi-point constraint.

    Parameters
    ----------
    name
        The name of the constraint.
    model
        The model tree.
    nSet
        The slave nodes (the RP is removed if it is part of it).
    referencePoint
        The node set of the reference point, a single node.
    configuration
        The options this constraint accepts.
    """

    schema = LinearizedRigidBodyMPCSchema

    # nothing else ties a slave to the rigid body, so no other constraint may take it over
    mayYieldSlaveToEarlierClaim = False

    def __init__(
        self,
        name: str,
        model: FEModel,
        nSet: NodeSet,
        referencePoint: NodeSet,
        *,
        configuration: LinearizedRigidBodyMPCSchema = LinearizedRigidBodyMPCSchema(),
    ):
        if model.domainSize not in (2, 3):
            raise WrongDomain("linearizedRigidBodyMPC: domain size 2 or 3 required")
        if len(referencePoint) != 1:
            raise ValueError(f"node set for reference point '{referencePoint.name}' must contain exactly one node")
        self._name = name
        self._model = model
        self.nDim = model.domainSize
        self.rp = referencePoint[0]
        self._nSetName = nSet.name
        self._buildRecords(nSet)

        # gets the records rebuilt after topology changes, via TopologyPipeline.refreshMeshDependents
        model.topology.registerMeshDependent(self)

    @classmethod
    def fromConstraintDefinition(
        cls, name: str, definition: dict, model: FEModel, journal: "Journal" = None
    ) -> "Constraint":
        """Build this constraint from a parsed ``*constraint`` definition."""
        configuration = buildSchemaFromOptions(cls.schema, definition)
        return cls(
            name,
            model,
            model.nodeSets[configuration.nSet],
            model.nodeSets[configuration.referencePoint],
            configuration=configuration,
        )

    def _buildRecords(self, nSet: NodeSet):
        # (slaveNode, d_s); slave node first, as claimedSlaveNodes() of the base class expects
        self._records = [(s, s.coordinates - self.rp.coordinates) for s in nSet if s is not self.rp]

    def refresh(self, model: FEModel, change) -> bool:
        """Rebuild the records from the current members of the slave node set, if ``change``
        touched it."""
        if not change.touchesNodeSet(self._nSetName):
            return False
        self._buildRecords(model.nodeSets[self._nSetName])
        return True

    # the RP carries no element: its displacement and rotation are activated here; the slaves are
    # element nodes, which carry their displacement already
    @property
    def nodes(self) -> list:
        return [self.rp]

    @property
    def fields(self) -> list:
        return [["displacement", "rotation"]]

    def getMultiPointConstraints(self, numbering) -> list:
        idcs = numbering.idcsOfFieldVariablesInDofVector
        uRP = idcs[self.rp.fields["displacement"]]
        phi = idcs[self.rp.fields["rotation"]]
        records = []
        for slave, d in self._records:
            uS = idcs[slave.fields["displacement"]]
            # u_s = u_RP + phi x d
            rotationTerms = rigidBodyKinematicsCoefficients(d, self.nDim)
            for i in range(self.nDim):
                masters = [(uRP[i], 1.0)] + [(phi[j], c) for j, c in rotationTerms[i] if c != 0.0]
                records.append((uS[i], masters))
        return records
