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
Prescribe Dirichlet conditions on the RP only, not on the slave nodes.
"""

from dataclasses import dataclass

from edelweissfe.constraints.base.multipointconstraintbase import (
    MultiPointConstraintBase,
)
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
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


class Constraint(MultiPointConstraintBase):
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
        # (slaveNode, d_s); slave node first, as claimedSlaveNodes() of the base class expects
        self._records = [(s, s.coordinates - self.rp.coordinates) for s in nSet if s is not self.rp]

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

    @property
    def activatedNodeFields(self) -> list:
        # the RP carries no element: its displacement and rotation are activated here
        return [(self.rp, ["displacement", "rotation"])] + [(s, ["displacement"]) for s, _ in self._records]

    def getMultiPointConstraints(self, dofManager) -> list:
        idcs = dofManager.idcsOfFieldVariablesInDofVector
        uRP = idcs[self.rp.fields["displacement"]]
        phi = idcs[self.rp.fields["rotation"]]
        records = []
        for slave, d in self._records:
            uS = idcs[slave.fields["displacement"]]
            if self.nDim == 3:
                dx, dy, dz = d
                # u_s = u_RP + phi x d
                rotationTerms = (
                    [(phi[1], dz), (phi[2], -dy)],
                    [(phi[2], dx), (phi[0], -dz)],
                    [(phi[0], dy), (phi[1], -dx)],
                )
            else:
                dx, dy = d[:2]
                rotationTerms = ([(phi[0], -dy)], [(phi[0], dx)])
            for i in range(self.nDim):
                masters = [(uRP[i], 1.0)] + [(dof, c) for dof, c in rotationTerms[i] if c != 0.0]
                records.append((uS[i], masters))
        return records
