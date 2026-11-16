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
reference configuration (inverse isoparametric mapping, :func:`~edelweissfe.utils.isoparametricshapes.inverseMapping`).
Each dependency is a record of a :class:`~edelweissfe.constraints.base.multipointconstraintbase.MultiPointConstraintBase`:
a row of the transformation :math:`\\mathbf{T}` that condenses the system to :math:`\\mathbf{T}^\\mathsf{T}
\\mathbf{K} \\mathbf{T}` (:class:`~edelweissfe.numerics.mpctransformation.MultiPointConstraintTransformation`). The
embedded elements are meshed independently of the host mesh; they keep their own stiffness, which is added to the
host's.

For bond-slip instead of perfect bond, see the ``embeddedBond`` generator
(:mod:`edelweissfe.generators.embeddedbond`).

**Embedded beams** (elements with a ``rotation`` field, e.g., Marmot ``B2D2``, ``B3D3``): their displacements are
tied as above; their rotations as chosen by ``rotations``. The rotation of an Euler-Bernoulli beam is the slope of
its axis, :math:`\\boldsymbol{\\theta}_\\perp = \\boldsymbol{t} \\times \\mathrm{d}\\boldsymbol{u}/\\mathrm{d}s`. An embedded
axis moves with the host, :math:`\\mathrm{d}\\boldsymbol{u}/\\mathrm{d}s = (\\nabla \\boldsymbol{u}) \\, \\boldsymbol{t}`,
hence :math:`\\boldsymbol{\\theta}_\\perp = \\boldsymbol{t} \\times (\\nabla \\boldsymbol{u} \\, \\boldsymbol{t})`. A material
line carries no twist, so the twist comes from the infinitesimal host rotation :math:`\\boldsymbol{\\omega} =
\\tfrac{1}{2} \\nabla \\times \\boldsymbol{u}`:

.. math::
    \\boldsymbol{\\theta}_s = \\boldsymbol{t} \\times ( \\nabla \\boldsymbol{u} \\, \\boldsymbol{t} )
    + ( \\boldsymbol{t} \\cdot \\boldsymbol{\\omega} ) \\, \\boldsymbol{t}
    \\qquad \\text{(2D: } \\theta_s = \\boldsymbol{t} \\times ( \\nabla \\boldsymbol{u} \\, \\boldsymbol{t} )
    \\cdot \\boldsymbol{e}_z \\text{)} .

Which mode?

==========  =======================================================================================================
``free``    displacements only (default; as Abaqus' embedded elements): the beam resists bending only by the tied
            displacements of its nodes, i.e., by the curvature of its nodal polygon -- an embedded cantilever is
            softer than a clamped one (about 1.11 times in the test of this module). Right for rebar-like beams.
``axis``    the rotation of the host material line along the beam axis (above): clamps the beam, consistent with
            the Euler-Bernoulli kinematics; a host in uniform strain (also in shear) leaves a straight embedded beam
            straight. For members carrying moments into the host.
``host``    the host rotation :math:`\\boldsymbol{\\omega}` itself: differs from ``axis`` where the host is strained --
            a uniform strain rotates every inclined material line (:math:`\\boldsymbol{t} \\times (\\boldsymbol{\\varepsilon}
            \\boldsymbol{t}) \\neq \\mathbf{0}`) while :math:`\\boldsymbol{\\omega} = \\mathbf{0}`, so inclined beams bend
            with the material rotation only.
==========  =======================================================================================================

At a beam node on a face, an edge or a node shared by host elements, :math:`\\nabla \\boldsymbol{u}` jumps between
them: the rotation operator is the average of those of all hosts containing the node (independent of the host order),
and at a kink of a beam (two elements with different axes) the average of the operators of both axes. The axis of a
3-node beam is its tangent at the node.

Limitations: the host is located in the reference configuration; the rotation ties are linear (infinitesimal
rotations, for the geometrically linear beams: the co-rotational beams ``B2D2CR``, ``B3D2CR`` may only be embedded
with ``rotations=free``); the volume of the embedded material is not subtracted from the host; embedded nodes outside
of all hosts raise an error, or are left free with ``allowUnembedded=True``. The ties work with every solver that
supports multi-point constraints (NIST, NID, NED; not NEST or the parallel arc-length solver). With explicit
dynamics, the masses of the slave nodes are folded onto the host nodes; Marmot beams have no rotary inertia, so the
rotation records fold none. Under adaptive mesh refinement of the host, the ties are rebuilt on the refined mesh.

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
from edelweissfe.models.meshdependent import MeshDependent
from edelweissfe.utils.embedding import HostElementLocator, HostPoint
from edelweissfe.utils.isoparametricshapes import (
    INSIDE_TOLERANCE,
    shapeFunctionsAndDerivatives,
)
from edelweissfe.utils.schema import buildSchemaFromOptions, schemaField

#: The modes of tying the rotations of embedded beam nodes.
ROTATION_MODES = ("free", "axis", "host")

#: Coefficients below this fraction of the largest one of a record are round-off zeros and dropped (e.g., the
#: weight of a host node on whose face the embedded node lies)
NEGLIGIBLE_COEFFICIENT = 1e-14


def infinitesimalRotation(gradU: np.ndarray) -> np.ndarray:
    """The axial vector of the skew part of a displacement gradient (2D: the scalar rotation about z)."""
    if gradU.shape[0] == 2:
        return np.array([0.5 * (gradU[1, 0] - gradU[0, 1])])
    return 0.5 * np.array([gradU[2, 1] - gradU[1, 2], gradU[0, 2] - gradU[2, 0], gradU[1, 0] - gradU[0, 1]])


def axisRotation(gradU: np.ndarray, t: np.ndarray) -> np.ndarray:
    """The rotation of the material line along the unit vector ``t`` for a displacement gradient (infinitesimal),
    with the twist about ``t`` from the infinitesimal rotation (2D: the scalar rotation about z)."""
    dAxis_dS = gradU @ t
    if gradU.shape[0] == 2:
        return np.array([t[0] * dAxis_dS[1] - t[1] * dAxis_dS[0]])
    return np.cross(t, dAxis_dS) + np.dot(t, infinitesimalRotation(gradU)) * t


def dTheta_dU(dN_dX: np.ndarray, mode: str, t: np.ndarray = None) -> np.ndarray:
    """The derivative of the tied rotation w.r.t. the host nodal displacements, in closed form.

    With :math:`\\nabla \\boldsymbol{u} = \\sum_a \\boldsymbol{u}_a \\otimes \\nabla N_a`:
    ``host``: :math:`\\partial \\boldsymbol{\\omega} / \\partial u_{ai} = \\tfrac{1}{2} \\nabla N_a \\times \\boldsymbol{e}_i`;
    ``axis``: :math:`\\partial \\boldsymbol{\\theta} / \\partial u_{ai} = (\\nabla N_a \\cdot \\boldsymbol{t}) \\, \\boldsymbol{t}
    \\times \\boldsymbol{e}_i + (\\boldsymbol{t} \\cdot \\tfrac{1}{2} \\nabla N_a \\times \\boldsymbol{e}_i) \\, \\boldsymbol{t}`
    (2D: the z components, the twist term vanishes).

    Parameters
    ----------
    dN_dX
        The derivatives of the host shape functions w.r.t. the coordinates, ``(nNodes, nDim)``.
    mode
        ``axis`` or ``host``.
    t
        The unit axis (``axis``).

    Returns
    -------
    np.ndarray
        ``[r, a, i]``: the rotation component ``r`` per displacement component ``i`` of host node ``a``.
    """
    nNodes, nDim = dN_dX.shape
    if nDim == 2:
        result = np.zeros((1, nNodes, 2))
        if mode == "host":
            result[0, :, 0], result[0, :, 1] = -0.5 * dN_dX[:, 1], 0.5 * dN_dX[:, 0]
        else:
            dN_dS = dN_dX @ t
            result[0, :, 0], result[0, :, 1] = -t[1] * dN_dS, t[0] * dN_dS
        return result
    result = np.zeros((3, nNodes, 3))
    for i in range(3):
        e = np.eye(3)[i]
        dOmega = 0.5 * np.cross(dN_dX, e)  # (nNodes, 3)
        if mode == "host":
            result[:, :, i] = dOmega.T
        else:
            result[:, :, i] = (np.outer(dN_dX @ t, np.cross(t, e)) + np.outer(dOmega @ t, t)).T
    return result


def _dN_dX(hostPoint: HostPoint) -> np.ndarray:
    """The derivatives of the host shape functions w.r.t. the coordinates at a located point,
    :math:`\\partial N / \\partial \\boldsymbol{X} = \\partial N / \\partial \\boldsymbol{\\xi} \\, \\mathbf{J}^{-1}` with
    :math:`\\mathbf{J} = \\partial \\boldsymbol{X} / \\partial \\boldsymbol{\\xi}`."""
    X = np.array([n.coordinates for n in hostPoint.element.nodes], dtype=float)
    _, dN_dXi = shapeFunctionsAndDerivatives(hostPoint.element.ensightType, hostPoint.xi)
    J = X.T @ dN_dXi
    return dN_dXi @ np.linalg.inv(J)


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
        description="The rotations of embedded beam nodes: 'free' (default, displacements only), 'axis' (tied to the "
        "rotation of the host material line along the beam axis: clamps the beam) or 'host' (tied to the "
        "infinitesimal rotation of the host). Small rotations only.",
        dtype=str,
        default="free",
    )
    tolerance: float = schemaField(
        description="Tolerance of the parametric coordinates of an embedded node on the boundary of a host element.",
        dtype=float,
        default=INSIDE_TOLERANCE,
    )


class Constraint(MultiPointConstraintBase, MeshDependent):
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
        self._configuration = configuration
        self._journal = journal
        self._field = configuration.field
        self._rotationMode = configuration.rotations.lower()
        if self._rotationMode not in ROTATION_MODES:
            raise ValueError(f"{name}: unknown rotations mode '{configuration.rotations}', use one of {ROTATION_MODES}")

        self._build(model)
        # rebuilt after a refinement of the host (or the embedded elements), see refresh()
        model.topology.registerMeshDependent(self)

    def _build(self, model: FEModel):
        """Locate the embedded nodes in the current host elements and build the displacement and rotation records."""
        configuration = self._configuration
        embeddedElements = list(model.elementSets[configuration.embeddedElSet])
        hostElements = list(model.elementSets[configuration.hostElSet])
        self._hostElementIds = {id(el) for el in hostElements}
        self._embeddedElementIds = {id(el) for el in embeddedElements}
        hostNodes = {n for el in hostElements for n in el.nodes}
        locator = HostElementLocator(hostElements, tolerance=configuration.tolerance)

        #: (slaveNode, [(masterNode, weight), ...]); slave node first, see claimedSlaveNodes()
        self._records = []
        #: (slaveNode, [(masterNode, component, [coefficient per rotation component]), ...]) of the beam rotations
        self._rotationRecords = []
        #: The embedded nodes outside of all host elements, left free.
        self.unembeddedNodes = []

        beamAxes = self._beamAxesAtNodes(embeddedElements) if self._rotationMode != "free" else {}
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
                self._records.append((node, self._displacementMasters(hostPoint)))
            if node in beamAxes:
                self._rotationRecords.append(
                    (node, self._rotationMasters(locator.locateAll(node.coordinates), beamAxes[node]))
                )

        if self.unembeddedNodes:
            labels = [n.label for n in self.unembeddedNodes]
            message = (
                f"{self._name}: {len(labels)} embedded node(s) lie outside of all host elements"
                f" (or the inverse mapping did not converge): {labels[:20]}"
            )
            if not configuration.allowUnembedded:
                raise ValueError(message + " (use allowUnembedded=True to leave them free)")
            if self._journal:
                self._journal.message(message + "; they are left free", self._name)

        if self._journal:
            self._journal.message(
                f"{len(self._records)} node(s) embedded in {len(hostElements)} host element(s)", self._name
            )
            if self._rotationRecords:
                self._journal.message(
                    f"rotations of {len(self._rotationRecords)} beam node(s) tied to the host ({self._rotationMode})",
                    self._name,
                )

    def _beamAxesAtNodes(self, embeddedElements) -> dict:
        """The unit axes of the embedded beams at each of their nodes (a node of several elements has several).

        An element is a beam if its nodes carry a ``rotation`` field. The axis is the tangent at the node: the chord
        for 2 nodes, the tangent of the quadratic interpolation for 3 (node order end, end, mid)."""
        axes = {}
        for el in embeddedElements:
            if el.ensightType not in ("bar2", "bar3") or "rotation" not in el.fields[0]:
                continue  # trusses carry no rotation
            if el.elType.upper().endswith("CR"):
                raise ValueError(
                    f"{self._name}: the co-rotational beam {el.elNumber} ({el.elType}) cannot be embedded with "
                    f"rotations={self._rotationMode}: the rotation ties are infinitesimal; use rotations=free"
                )
            X = np.array([n.coordinates for n in el.nodes], dtype=float)
            nodeXi = (-1.0, 1.0, 0.0)
            for a, node in enumerate(el.nodes):
                _, dN_dXi = shapeFunctionsAndDerivatives(el.ensightType, [nodeXi[a]])
                dX_dXi = X.T @ dN_dXi[:, 0]
                axes.setdefault(node, []).append(dX_dXi / np.linalg.norm(dX_dXi))
        return axes

    @staticmethod
    def _displacementMasters(hostPoint: HostPoint) -> list:
        """The masters of an embedded node's displacement: the host nodes with their shape function values (round-off
        zeros dropped)."""
        N = hostPoint.shapeFunctions()
        cutoff = NEGLIGIBLE_COEFFICIENT * np.abs(N).max()
        return [(m, w) for m, w in zip(hostPoint.element.nodes, N) if abs(w) > cutoff]

    def _rotationMasters(self, hostPoints: list, axes: list) -> list:
        """The masters of an embedded beam node's rotation: the averaged operator dTheta/dU of all hosts containing
        the node and all beam axes at it, per host node and displacement component."""
        operator = {}  # (masterNode, component) -> coefficients per rotation component
        for hostPoint in hostPoints:
            dN_dX = _dN_dX(hostPoint)
            if self._rotationMode == "host":
                C = dTheta_dU(dN_dX, "host")
            else:
                C = np.mean([dTheta_dU(dN_dX, "axis", t) for t in axes], axis=0)
            for a, m in enumerate(hostPoint.element.nodes):
                for i in range(C.shape[2]):
                    operator[(m, i)] = operator.get((m, i), 0.0) + C[:, a, i] / len(hostPoints)
        cutoff = NEGLIGIBLE_COEFFICIENT * max(np.abs(c).max() for c in operator.values())
        return [(m, i, c) for (m, i), c in operator.items() if np.abs(c).max() > cutoff]

    @classmethod
    def fromConstraintDefinition(
        cls, name: str, definition: dict, model: FEModel, journal: "Journal" = None
    ) -> "Constraint":
        """Build this constraint from a parsed ``*constraint`` definition."""
        configuration = buildSchemaFromOptions(cls.schema, definition)
        return cls(name, model, journal, configuration=configuration)

    def refresh(self, model: FEModel, change) -> bool:
        """Rebuild the records on the current mesh if the host or the embedded elements changed (e.g., after an
        adaptive refinement of the host: the embedded nodes are then tied to the refined elements).

        Returns
        -------
        bool
            Whether the records were rebuilt.
        """
        hostIds = {id(el) for el in model.elementSets[self._configuration.hostElSet]}
        embeddedIds = {id(el) for el in model.elementSets[self._configuration.embeddedElSet]}
        if hostIds == self._hostElementIds and embeddedIds == self._embeddedElementIds:
            return False
        self._build(model)
        return True

    def getMultiPointConstraints(self, numbering) -> list:
        """The records of the displacements, :math:`u_{s,i} = \\sum_a N_a u_{a,i}`, and of the beam rotations,
        :math:`\\theta_{s,r} = \\sum_a \\sum_i \\partial\\theta_r / \\partial u_{a,i} \\, u_{a,i}`.

        Parameters
        ----------
        numbering
            The current degree-of-freedom numbering.

        Returns
        -------
        list[tuple[int, list[tuple[int, float]]]]
            One record per slave degree of freedom.
        """
        fieldVariableIndices = numbering.idcsOfFieldVariablesInDofVector
        records = []
        for slaveNode, masters in self._records:
            if self._field in slaveNode.fields:
                records += self.nodalDofRecords(fieldVariableIndices, self._field, slaveNode, masters)

        for slaveNode, masters in self._rotationRecords:
            if "rotation" not in slaveNode.fields:
                continue
            slaveDofs = fieldVariableIndices[slaveNode.fields["rotation"]]
            masterDofs = [(fieldVariableIndices[mNode.fields[self._field]][i], c) for mNode, i, c in masters]
            for r in range(len(slaveDofs)):
                records.append((slaveDofs[r], [(dof, c[r]) for dof, c in masterDofs if c[r] != 0.0]))
        return records

    def claimedSlaveNodes(self) -> set:
        return {record[0] for record in self._records} | {record[0] for record in self._rotationRecords}
