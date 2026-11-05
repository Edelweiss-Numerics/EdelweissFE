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
from edelweissfe.utils.exceptions import WrongDomain
from edelweissfe.utils.schema import buildSchemaFromOptions, schemaField


def rigidBodyKinematicsCoefficients(d: np.ndarray, nDim: int) -> list[list[tuple[int, float]]]:
    """The linearized rigid body kinematics of a point at distance ``d`` from the reference point,
    :math:`\\mathbf{u} = \\mathbf{u}_{RP} + \\boldsymbol{\\phi} \\times \\mathbf{d}`, as the
    rotation terms of each displacement component.

    Parameters
    ----------
    d
        The distance vector :math:`\\mathbf{x} - \\mathbf{x}_{RP}`.
    nDim
        The domain size; in 2D, the rotation has the one component :math:`\\phi_z`.

    Returns
    -------
    list[list[tuple[int, float]]]
        Per displacement component :math:`i`, the pairs ``(j, c)`` with
        :math:`u_i = u_{RP,i} + \\sum c \\, \\phi_j`.
    """
    if nDim == 3:
        dx, dy, dz = d
        return [[(1, dz), (2, -dy)], [(2, dx), (0, -dz)], [(0, dy), (1, -dx)]]
    dx, dy = d[:2]
    return [[(0, -dy)], [(0, dx)]]


@dataclass(frozen=True)
class LinearizedRigidBodySchema:
    """The options this constraint accepts, owned by this module and never mutated from outside
    it.

    Its only options are the structural ``nSet``/``referencePoint`` it ties -- node set *names*,
    resolved to the actual node sets in :meth:`Constraint.fromConstraintDefinition`. Each is
    declared ``required=True``, but still given a ``default=None`` so the schema remains
    constructible on its own."""

    nSet: str | None = schemaField(description="Node set to tie.", dtype=str, default=None, required=True)
    referencePoint: str | None = schemaField(
        description="Node set containing only the reference point.", dtype=str, default=None, required=True
    )


class Constraint(ConstraintBase):
    """Linearized Rigid Body constraint.

    An implementation of a linearized rigid body constraint for displacements.


    Defining the constraints
    ========================

    2D: 2 constraints per slave node
    1) 0 = ux1 - uxRP + R*dy1
    2) 0 = uy1 - uyRP - R*dx1
    1) 0 = ux2 - uxRP + R*dy2
    2) 0 = uy2 - uyRP - R*dx2
    ...

            | ux1 - uxRP + R*dy1 |   | 0 |
            | uy1 - uyRP - R*dx1 |   | 0 |
    --> g = | ux2 - uxRP + R*dy2 | = | 0 |
            | uy2 - uyRP - R*dx2 |   | 0 |
            | ...                |   | 0 |


    3D: 3 constraints per slave node
    1) 0 = ux1 - uxRP + Rz*dy1 - Ry*dz1
    2) 0 = uy1 - uyRP - Rz*dx1 + Rx*dz1
    3) 0 = uz1 - uzRP - Rx*dy1 + Ry*dx1
    1) 0 = ux2 - uxRP + Rz*dy2 - Ry*dz2
    2) 0 = uy2 - uyRP - Rz*dx2 + Rx*dz2
    3) 0 = uz2 - uzRP - Rx*dy2 + Ry*dx2
    ...

            | ux1 - uxRP + Rz*dy1 - Ry*dz1 |   | 0 |
            | uy1 - uyRP - Rz*dx1 + Rx*dz1 |   | 0 |
            | uz1 - uzRP - Rx*dy1 + Ry*dx1 |   | 0 |
    --> g = | ux2 - uxRP + Rz*dy2 - Ry*dz2 | = | 0 |
            | uy2 - uyRP - Rz*dx2 + Rx*dz2 |   | 0 |
            | uz2 - uzRP - Rx*dy2 + Ry*dx2 |   | 0 |
            | ...                          |   | 0 |


    Create dg/du matrix
    ===================

    2D
            | d_dux1 d_duy1 d_dux2 d_duy2 ... d_duxRP d_duyRP d_dR |
    --------+---------------------------- ... -------------------- +
    dg1_dU  | 1      0      0      0      ... -1       0       dy1 |
    dg2_dU  | 0      1      0      0      ...  0      -1      -dx1 |
    dg1_dU  | 0      0      1      0      ... -1       0       dy2 |
    dg2_dU  | 0      0      0      1      ...  0      -1      -dx2 |
    ...     |......................................................|


    3D
            | d_dux1 d_duy1 d_duz1 d_dux2 d_duy2 d_duz2 ... d_duxRP d_duyRP d_duzRP d_dRx d_dRy d_dRz |
    --------+------------------------------------------ ... ----------------------------------------- +
    dg1_dU  | 1      0      0      0      0      0      ... -1       0       0       0    -dz1   dy1  |
    dg2_dU  | 0      1      0      0      0      0      ...  0      -1       0       dz1   0    -dx1  |
    dg3_dU  | 0      0      1      0      0      0      ...  0       0      -1      -dy1   dx1   0    |
    dg1_dU  | 0      0      0      1      0      0      ... -1       0       0       0    -dz2   dy2  |
    dg2_dU  | 0      0      0      0      1      0      ...  0      -1       0       dz2   0    -dx2  |
    dg3_dU  | 0      0      0      0      0      1      ...  0       0      -1      -dy2   dx2   0    |
    ...     |.........................................................................................|


    Create K matrix
    ===============

    K =     | 0       dg_du.T |
            | dg_du   0       |


    """

    #: Option schema for this constraint, per OptionSchemaProvider.
    schema = LinearizedRigidBodySchema

    #: Carries nothing from one increment to the next.
    checkpointedState = {}

    def __init__(
        self,
        name,
        model: FEModel,
        nSet: NodeSet,
        referencePoint: NodeSet,
        *,
        configuration: LinearizedRigidBodySchema = LinearizedRigidBodySchema(),
    ):
        super().__init__(name, model)

        if model.domainSize not in [2, 3]:
            raise WrongDomain("Wrong domain size!")

        if len(referencePoint) > 1:
            raise Exception(f"node set for reference point '{referencePoint.name}' contains more than one node")

        self.rp = referencePoint[0]

        # slave node set may contain the reference point
        # RP is removed (if present) and node set is converted to list
        self.slaveNodes = [node for node in nSet if not node == self.rp]

        # list of all nodes including RP at end
        self._nodes = self.slaveNodes + [self.rp]

        nSlaves = len(self.slaveNodes)

        self.slaveNodesFields = [["displacement"]] * nSlaves
        self.referencePointFields = [["displacement", "rotation"]]
        self._fieldsOnNodes = self.slaveNodesFields + self.referencePointFields

        nDim = model.domainSize

        nRp = 1

        # one rotational dof for 2D, three for 3D
        nRot = 1 if nDim == 2 else 3

        nConstraints = nSlaves * nDim
        self.nConstraints = nConstraints

        nAffectedDofs = nDim * (nSlaves + nRp) + nRot

        self._nDof = nAffectedDofs + nConstraints

        dG_dU = np.zeros((nConstraints, nAffectedDofs))
        rpDisplacementCol = nDim * nSlaves
        rotationCol = nDim * (nSlaves + nRp)

        # g = u_s - u_RP - phi x d_s, row s * nDim + i
        for s, slave in enumerate(self.slaveNodes):
            rotationTerms = rigidBodyKinematicsCoefficients(slave.coordinates - self.rp.coordinates, nDim)
            for i in range(nDim):
                row = s * nDim + i
                dG_dU[row, row] = 1.0
                dG_dU[row, rpDisplacementCol + i] = -1.0
                for j, c in rotationTerms[i]:
                    dG_dU[row, rotationCol + j] = -c

        self.dG_dU = dG_dU

        K = np.zeros((self._nDof, self._nDof))

        K[0 : dG_dU.shape[1], -dG_dU.shape[0] :] = dG_dU.T
        K[-dG_dU.shape[0] :, 0 : dG_dU.shape[1] :] = dG_dU

        self.K = K

    @classmethod
    def fromConstraintDefinition(
        cls, name: str, definition: dict, model: FEModel, journal: "Journal" = None
    ) -> "Constraint":
        """Build this constraint from a parsed ``*constraint`` definition. See
        :class:`~edelweissfe.constraints.base.constraintbase.ConstraintBase` for why this is
        separate from ``__init__``."""
        configuration = buildSchemaFromOptions(cls.schema, definition)
        return cls(
            name,
            model,
            model.nodeSets[configuration.nSet],
            model.nodeSets[configuration.referencePoint],
            configuration=configuration,
        )

    @property
    def nodes(self) -> list:
        return self._nodes

    @property
    def fieldsOnNodes(self) -> list:
        return self._fieldsOnNodes

    @property
    def nDof(self) -> int:
        return self._nDof

    def getNumberOfAdditionalNeededScalarVariables(self):
        return self.nConstraints

    def applyConstraint(self, U_np, dU, PExt, K, timeStep):
        LambdaN1 = U_np[-self.nConstraints :]

        PExt[: -self.nConstraints] -= LambdaN1.dot(self.dG_dU)

        K += self.K
