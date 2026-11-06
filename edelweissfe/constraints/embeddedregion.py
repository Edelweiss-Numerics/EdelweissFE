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
Embedded region constraint: perfect bond of embedded elements (e.g., reinforcement bars) to host continuum
elements, enforced by master-slave DOF elimination (multi-point constraint), like Abaqus' ``*EMBEDDED ELEMENT``.

Every node of the embedded elements is tied to the host element it lies in,

.. math::
    \\boldsymbol{u}_s = \\sum_a N_a(\\boldsymbol{\\xi}_s) \\, \\boldsymbol{u}_a ,

with the parametric coordinates :math:`\\boldsymbol{\\xi}_s` of the node in the host element, found once in the
reference configuration. The embedded elements are meshed independently of the host mesh; they keep their own
stiffness, which is added to the host's (the volume of the embedded material is not subtracted from the host).

Embedded beams (elements with a ``rotation`` field, e.g., Marmot ``B2D2``, ``B3D3``): their displacements are tied
as above. Their rotations are left free by default (``rotations=free``): the beam then resists bending only by the
tied displacements of its nodes, i.e., by the curvature of its nodal polygon. With ``rotations=axis``, the rotation
of an embedded beam node is tied to the rotation of the host material line along the beam axis
:math:`\\boldsymbol{t}`,

.. math::
    \\boldsymbol{\\theta}_s = \\boldsymbol{t} \\times ( \\nabla \\boldsymbol{u} \\, \\boldsymbol{t} )
    + ( \\boldsymbol{t} \\cdot \\boldsymbol{\\omega} ) \\, \\boldsymbol{t}
    \\qquad \\text{(2D: } \\theta_s = \\boldsymbol{t} \\times ( \\nabla \\boldsymbol{u} \\, \\boldsymbol{t} )
    \\cdot \\boldsymbol{e}_z \\text{)},

with the host displacement gradient :math:`\\nabla \\boldsymbol{u} = \\sum_a \\boldsymbol{u}_a \\otimes \\nabla N_a`
at the node and the infinitesimal host rotation :math:`\\boldsymbol{\\omega}` (axial vector of the skew part of
:math:`\\nabla \\boldsymbol{u}`), which sets only the twist. This is consistent with the Euler-Bernoulli kinematics
of a beam whose axis is embedded in the host: a host in uniform strain (also in shear) leaves an embedded straight
beam straight and unbent. With ``rotations=host``, the rotation is tied to :math:`\\boldsymbol{\\omega}` itself, the
rotation of the host material, which bends beams in sheared hosts. At a kink of a beam (two elements with different
axes) the operators of both elements are averaged. Both are linear in the host displacements (infinitesimal
rotations, consistent with the geometrically linear beams). Trusses in the embedded set carry no rotation and are
unaffected.

.. code-block:: edelweiss
    :caption: Example:

    *constraint, type=embeddedRegion, name=rebarsInConcrete
        embeddedElSet=rebars, hostElSet=concrete
"""

from dataclasses import dataclass

import numpy as np

from edelweissfe.constraints.base.multipointconstraintbase import (
    MultiPointConstraintBase,
)
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.utils.embedding import HostElementLocator
from edelweissfe.utils.isoparametricshapes import shapeFunctionsAndDerivatives
from edelweissfe.utils.schema import buildSchemaFromOptions, schemaField

#: The modes of tying the rotations of embedded beam nodes.
ROTATION_MODES = ("free", "axis", "host")


def infinitesimalRotation(gradU: np.ndarray) -> np.ndarray:
    """The axial vector of the skew part of a displacement gradient (2D: the scalar rotation about z)."""
    if gradU.shape[0] == 2:
        return np.array([0.5 * (gradU[1, 0] - gradU[0, 1])])
    return 0.5 * np.array([gradU[2, 1] - gradU[1, 2], gradU[0, 2] - gradU[2, 0], gradU[1, 0] - gradU[0, 1]])


def axisRotation(gradU: np.ndarray, t: np.ndarray) -> np.ndarray:
    """The rotation of the material line along the unit vector ``t`` for a displacement gradient (infinitesimal),
    with the twist about ``t`` from the infinitesimal rotation (2D: the scalar rotation about z)."""
    dt = gradU @ t
    if gradU.shape[0] == 2:
        return np.array([t[0] * dt[1] - t[1] * dt[0]])
    return np.cross(t, dt) + np.dot(t, infinitesimalRotation(gradU)) * t


def rotationOperator(hostPoint, mode: str, axes: list) -> np.ndarray:
    """The linear operator from the host nodal displacements to the tied rotation of an embedded beam node.

    Parameters
    ----------
    hostPoint
        The located point in the host element.
    mode
        ``axis`` or ``host``, see the module documentation.
    axes
        The unit axes of the beam elements at the node (averaged for ``axis``).

    Returns
    -------
    np.ndarray
        ``C[r, a, i]``: the rotation component ``r`` per displacement component ``i`` of host node ``a``.
    """
    element = hostPoint.element
    X = np.array([n.coordinates for n in element.nodes], dtype=float)
    _, dNdXi = shapeFunctionsAndDerivatives(element.ensightType, hostPoint.xi)
    dNdX = dNdXi @ np.linalg.inv(X.T @ dNdXi)  # (nNodes, nDim)
    nNodes, nDim = dNdX.shape
    nRot = 1 if nDim == 2 else 3
    C = np.zeros((nRot, nNodes, nDim))
    for a in range(nNodes):
        for i in range(nDim):
            gradU = np.zeros((nDim, nDim))
            gradU[i, :] = dNdX[a]
            if mode == "host":
                C[:, a, i] = infinitesimalRotation(gradU)
            else:
                C[:, a, i] = np.mean([axisRotation(gradU, t) for t in axes], axis=0)
    return C


@dataclass(frozen=True)
class EmbeddedRegionSchema:
    """The options this constraint accepts, owned by this module and never mutated from outside it."""

    embeddedElSet: str | None = schemaField(
        description="The element set of the embedded elements (e.g., truss elements); all their nodes are tied to "
        "the host elements.",
        dtype=str,
        default=None,
        required=True,
    )
    hostElSet: str | None = schemaField(
        description="The element set of the host (continuum) elements.",
        dtype=str,
        default=None,
        required=True,
    )
    field: str = schemaField(description="The field to be constrained.", dtype=str, default="displacement")
    allowUnembedded: bool = schemaField(
        description="Whether embedded nodes lying outside of all host elements are left free (with a warning) "
        "instead of raising an error, e.g., for bars sticking out of the host.",
        dtype=bool,
        default=False,
    )
    rotations: str = schemaField(
        description="The rotations of embedded beam nodes: 'free' (default), 'axis' (tied to the rotation of the host "
        "material line along the beam axis) or 'host' (tied to the infinitesimal rotation of the host).",
        dtype=str,
        default="free",
    )
    tolerance: float = schemaField(
        description="Tolerance of the parametric coordinates of an embedded node on the boundary of a host " "element.",
        dtype=float,
        default=1e-6,
    )


class Constraint(MultiPointConstraintBase):
    """Embedded region multi-point constraint enforced by DOF elimination.

    Parameters
    ----------
    name
        The name of the constraint.
    model
        The model tree.
    journal
        The journal.
    configuration
        The options this constraint accepts.
    """

    #: Option schema for this constraint, per OptionSchemaProvider.
    schema = EmbeddedRegionSchema

    def __init__(
        self,
        name: str,
        model: FEModel,
        journal: Journal = None,
        *,
        configuration: EmbeddedRegionSchema = EmbeddedRegionSchema(),
    ):
        self._name = name
        self._model = model
        self._field = configuration.field
        self._rotationMode = configuration.rotations.lower()
        if self._rotationMode not in ROTATION_MODES:
            raise ValueError(f"{name}: unknown rotations mode '{configuration.rotations}', use one of {ROTATION_MODES}")

        embeddedElements = list(model.elementSets[configuration.embeddedElSet])
        hostElements = list(model.elementSets[configuration.hostElSet])
        hostNodes = {n for el in hostElements for n in el.nodes}

        locator = HostElementLocator(hostElements, tolerance=configuration.tolerance)

        # (slaveNode, [(masterNode, weight), ...]); slave node first, see claimedSlaveNodes()
        self._records = []
        #: The embedded nodes outside of all host elements, left free.
        self.unembeddedNodes = []

        # (slaveNode, [(masterNode, component, [coefficient per rotation component]), ...]) of the beam rotations
        self._rotationRecords = []
        # the axes of the beam elements (with a rotation field) at each of their nodes
        beamAxes = {}
        for el in embeddedElements:
            if self._rotationMode != "free" and el.ensightType in ("bar2", "bar3") and "rotation" in el.fields[0]:
                X = np.array([n.coordinates for n in el.nodes[:2]], dtype=float)
                t = (X[1] - X[0]) / np.linalg.norm(X[1] - X[0])
                for node in el.nodes:
                    beamAxes.setdefault(node, []).append(t)

        embeddedNodes = list(dict.fromkeys(n for el in embeddedElements for n in el.nodes))
        for node in embeddedNodes:
            shared = node in hostNodes  # shared with the host: no displacement to tie
            if shared and node not in beamAxes:
                continue
            hostPoint = locator.locate(node.coordinates)
            if hostPoint is None:
                self.unembeddedNodes.append(node)
                continue
            if not shared:
                N = hostPoint.shapeFunctions()
                self._records.append((node, [(m, w) for m, w in zip(hostPoint.element.nodes, N) if w != 0.0]))
            if node in beamAxes:
                C = rotationOperator(hostPoint, self._rotationMode, beamAxes[node])
                scale = np.abs(C).max()
                masters = [
                    (m, i, C[:, a, i])
                    for a, m in enumerate(hostPoint.element.nodes)
                    for i in range(C.shape[2])
                    if np.abs(C[:, a, i]).max() > 1e-14 * scale
                ]
                self._rotationRecords.append((node, masters))

        if self.unembeddedNodes:
            labels = [n.label for n in self.unembeddedNodes]
            message = f"{name}: {len(labels)} embedded node(s) lie outside of all host elements: {labels[:20]}"
            if not configuration.allowUnembedded:
                raise ValueError(message + " (use allowUnembedded=True to leave them free)")
            if journal:
                journal.message(message + "; they are left free", name)

        if journal:
            journal.message(f"{len(self._records)} node(s) embedded in {len(hostElements)} host element(s)", name)
            if self._rotationRecords:
                journal.message(
                    f"rotations of {len(self._rotationRecords)} beam node(s) tied to the host ({self._rotationMode})",
                    name,
                )

    @classmethod
    def fromConstraintDefinition(
        cls, name: str, definition: dict, model: FEModel, journal: "Journal" = None
    ) -> "Constraint":
        """Build this constraint from a parsed ``*constraint`` definition."""
        configuration = buildSchemaFromOptions(cls.schema, definition)
        return cls(name, model, journal, configuration=configuration)

    def getMultiPointConstraints(self, dofManager) -> list:
        fieldVariableIndices = dofManager.idcsOfFieldVariablesInDofVector
        records = []
        field = self._field
        for slaveNode, masters in self._records:
            if field not in slaveNode.fields:
                continue
            slaveDofs = fieldVariableIndices[slaveNode.fields[field]]
            masterDofs = [(fieldVariableIndices[mNode.fields[field]], w) for mNode, w in masters]
            for component in range(len(slaveDofs)):
                records.append((slaveDofs[component], [(mDofs[component], w) for mDofs, w in masterDofs]))

        for slaveNode, masters in self._rotationRecords:
            if "rotation" not in slaveNode.fields:
                continue
            slaveDofs = fieldVariableIndices[slaveNode.fields["rotation"]]
            masterDofs = [(fieldVariableIndices[mNode.fields[field]][i], c) for mNode, i, c in masters]
            for r in range(len(slaveDofs)):
                records.append((slaveDofs[r], [(dof, c[r]) for dof, c in masterDofs if c[r] != 0.0]))
        return records

    def claimedSlaveNodes(self) -> set:
        return {record[0] for record in self._records} | {record[0] for record in self._rotationRecords}
