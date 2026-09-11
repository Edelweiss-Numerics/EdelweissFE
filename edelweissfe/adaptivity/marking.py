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

"""Refinement marking (WS-G): the markers that decide which elements an adaptivity mechanism refines.

Each marker is a small, registry-resolved policy object (see :mod:`edelweissfe.config.markerlibrary`)
implementing :meth:`MarkerBase.mark`. Topological markers select from a set/surface; the
:class:`FieldOutputMarker` thresholds a per-element quantity. Deliberately, a marker never re-derives
a field quantity itself: the reduction (a magnitude, a principal stress via ``eigVal``, a norm, ...)
lives in the referenced fieldOutput's own ``f(x)`` -- one first-class, reusable quantity -- and the
marker only applies the refinement decision (a threshold) on top of it.
"""

import operator
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from edelweissfe.adaptivity.hex20shapefunctions import (
    EDGES,
    LOCAL_COORDS,
    N_NODES,
    hex20_shape,
    hex20_shape_grad,
)
from edelweissfe.utils.fieldoutput import ElementFieldOutput
from edelweissfe.utils.schema import (
    OptionSchemaProvider,
    buildSchemaFromOptions,
    schemaField,
)

# 2x2x2 Gauss points on the reference cube [-1, 1]^3, with unit weights. These are the reduced-
# integration points of GC3D20R and, for a serendipity HEX20, the superconvergent points of the
# gradient -- so evaluating the FE gradient here (and averaging it to nodes) is the classic
# Zienkiewicz-Zhu recovery in its sweet spot. Shape-function values/derivatives are the same for
# every element, so they are tabulated once at import time (the marker runs every increment).
_GP_1D = 1.0 / np.sqrt(3.0)
_GAUSS_POINTS = np.array(
    [(sx * _GP_1D, sy * _GP_1D, sz * _GP_1D) for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)],
    dtype=float,
)
_GP_N = np.array([hex20_shape(*gp) for gp in _GAUSS_POINTS])  # (8, 20)  shape-fn values at Gauss points
_GP_DNDXI = np.array([hex20_shape_grad(*gp)[1] for gp in _GAUSS_POINTS])  # (8, 20, 3)  dN/dxi at Gauss points
_NODE_DNDXI = np.array([hex20_shape_grad(*p)[1] for p in LOCAL_COORDS])  # (20, 20, 3)  dN/dxi at each node

# HEX20 corner (vertex) local indices and edge topology, used by superconvergent patch recovery
# (SPR): the 8 corners are the patch-assembly nodes, and each of the 12 edge midsides is recovered
# from the polynomial fits of its two corner-endpoint patches.
_MID_LOCALS = {edge[1] for edge in EDGES}  # 12 edge-midside local indices
_CORNER_LOCALS = [i for i in range(N_NODES) if i not in _MID_LOCALS]  # 8 corner local indices
_MID_EDGES = [(edge[1], edge[0], edge[2]) for edge in EDGES]  # (midLocal, cornerA_local, cornerB_local)


@dataclass(frozen=True)
class MarkerOptionsBase:
    """L2 options common to every ``>>marker`` block, regardless of ``type``.

    Each concrete marker's schema derives from this so that ``initialOnly`` -- the one option every
    marker accepts -- is declared once. The ``type`` dispatch key itself is deliberately *not* a
    field here: it is consumed by the adaptivity mechanism to resolve the marker class through the
    :mod:`~edelweissfe.config.markerlibrary` registry, and is stripped before a marker validates the
    remaining options against its own schema.
    """

    initialOnly: bool = schemaField(
        description="Evaluate the marker only once, at simulation start, instead of every increment.",
        dtype=bool,
        default=False,
    )


class MarkerBase(OptionSchemaProvider):
    """Base class for AMR refinement markers.

    A marker decides which elements to refine. It is resolved by ``type`` through the L3 registry
    (category ``marker``; see :mod:`edelweissfe.config.markerlibrary`) and constructed from its
    ``>>marker`` block options via :meth:`fromOptions`, so *any* adaptivity mechanism -- not only the
    HEX20 h-adaptivity model modifier -- can reuse the same marker library, and third-party packages
    can contribute markers through entry points. Each concrete marker owns its option :attr:`schema`
    (an :class:`MarkerOptionsBase` subclass), which both validates and documents its ``>>marker``
    line, replacing the former flat union schema that jammed every marker's options together.
    """

    #: The L2 option schema for this marker's ``>>marker`` block, overridden per subclass.
    schema = None

    def __init__(self, initialOnly=False):
        self.initialOnly = initialOnly

    def mark(self, model, refineElements, mesh):
        raise NotImplementedError()

    @classmethod
    def fromOptions(cls, options: Mapping[str, Any]) -> "MarkerBase":
        """Construct this marker from its raw parsed ``>>marker`` options.

        ``options`` is the ``>>marker`` block's option mapping with the ``type`` dispatch key already
        removed (the adaptivity mechanism uses ``type`` to look the class up before calling this).
        The options are validated and coerced against :attr:`schema` via
        :func:`~edelweissfe.utils.schema.buildSchemaFromOptions`, then mapped onto the constructor.
        """
        raise NotImplementedError()


def _perElementFieldOutputResult(model, fieldOutputName):
    """Resolve an already-declared ``perElement`` fieldOutput and return ``(elements, values)`` --
    the element list (its associated set) and the last per-element result array ``(nElem, ...)``.

    The fieldOutput *may* carry an ``f(x)``: that is precisely how a marker consumes a shaped
    quantity -- a magnitude ``abs(x)``, a principal stress ``eigVal(...)``, a norm -- computed once in
    the fieldOutput layer (where it is also reusable for output/monitoring) rather than re-derived
    here. The only requirement is that it keeps one result row per element; an ``f(x)`` that collapses
    the per-element axis is caught by the row-count check below. Raises with an actionable message if
    the fieldOutput is missing, is not a ``perElement`` one, or reports a row count that does not match
    its element set. Shared by every marker that drives refinement off a fieldOutput.
    """
    fieldOutputController = model.fieldOutputController
    if fieldOutputController is None or fieldOutputName not in fieldOutputController.fieldOutputs:
        raise KeyError(
            f"hAdaptivity marker references fieldOutput {fieldOutputName!r}, which is not "
            "defined. Declare it under '*fieldOutput' (before the increment loop starts) so the "
            "marker can look it up by name."
        )
    fieldOutput = fieldOutputController.fieldOutputs[fieldOutputName]

    if not isinstance(fieldOutput, ElementFieldOutput):
        raise TypeError(
            f"hAdaptivity marker references fieldOutput {fieldOutputName!r}, which is a "
            f"{type(fieldOutput).__name__}; only a 'perElement' fieldOutput exposes one result "
            "row per element, as required for marking."
        )

    elements = list(fieldOutput.associatedSet)
    values = np.asarray(fieldOutput.getLastResult())

    if values.ndim < 1 or values.shape[0] != len(elements):
        rowCount = values.shape[0] if values.ndim >= 1 else 0
        raise ValueError(
            f"hAdaptivity marker: fieldOutput {fieldOutputName!r} reports {rowCount} "
            f"result row(s) (shape {values.shape}) for {len(elements)} element(s) in its associated set; refusing to mark "
            "against a mismatched fieldOutput."
        )
    return elements, values


# Comparison operators a FieldOutputMarker may apply against its threshold, keyed by the string a
# user writes in the input file. numpy broadcasts these element-wise over a result row.
_COMPARISON_OPERATORS = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}


@dataclass(frozen=True)
class FieldOutputMarkerSchema(MarkerOptionsBase):
    """L2 options of a ``>>marker, type=fieldOutput`` block."""

    fieldOutput: str | None = schemaField(
        description=(
            "Name of an already-declared 'perElement' *fieldOutput to threshold. The *quantity* is "
            "shaped by that fieldOutput's own 'f(x)' (e.g. 'abs(x)' for a magnitude, "
            "'eigVal(x.reshape(-1,6))[:,0]...' for a principal stress); the marker only compares it."
        ),
        dtype=str,
        default=None,
        required=True,
    )
    threshold: float | None = schemaField(
        description="Value the fieldOutput result is compared against (element-wise, via 'operator').",
        dtype=float,
        default=None,
        required=True,
    )
    operator: str = schemaField(
        description="Comparison against 'threshold', applied element-wise: one of >, >=, <, <=, ==, !=.",
        dtype=str,
        default=">=",
    )


class FieldOutputMarker(MarkerBase):
    """Marks every element whose fieldOutput result satisfies ``value <operator> threshold`` at any
    entry (any quadrature point / component).

    The marker is a pure *threshold decision*: the quantity it compares is produced by the referenced
    ``perElement`` fieldOutput -- either raw, or shaped by that fieldOutput's own ``f(x)`` (``abs(x)``
    for a magnitude, ``eigVal`` for a principal stress, a norm, ...). Keeping the reduction in the
    fieldOutput (where it is a first-class, reusable/visualizable quantity) and only the threshold
    policy here is the division of labour -- the fieldOutput says *what* the quantity is, the marker
    says *when* it is large enough to refine. Marking is driven by the exact numbers the fieldOutput
    reports, and an element is marked if ``np.any`` of the comparison over its result row holds, so a
    multi-QP / multi-component row is decided by its worst entry.
    """

    schema = FieldOutputMarkerSchema

    def __init__(self, fieldOutputName, threshold, operator=">=", initialOnly=False):
        super().__init__(initialOnly)
        self.fieldOutputName = fieldOutputName
        self.threshold = float(threshold)
        if operator not in _COMPARISON_OPERATORS:
            raise ValueError(
                f"FieldOutputMarker: unknown operator {operator!r}; expected one of "
                f"{', '.join(_COMPARISON_OPERATORS)}."
            )
        self.operator = operator
        self._compare = _COMPARISON_OPERATORS[operator]

    @classmethod
    def fromOptions(cls, options):
        opts = buildSchemaFromOptions(cls.schema, options)
        return cls(opts.fieldOutput, opts.threshold, operator=opts.operator, initialOnly=opts.initialOnly)

    def mark(self, model, refineElements, mesh):
        elements, values = _perElementFieldOutputResult(model, self.fieldOutputName)
        marked = set()
        for element, row in zip(elements, values):
            if bool(np.any(self._compare(np.asarray(row), self.threshold))):
                marked.add(element)
        return marked


@dataclass(frozen=True)
class ElementSetMarkerSchema(MarkerOptionsBase):
    """L2 options of a ``>>marker, type=elementSet`` block."""

    elSet: str | None = schemaField(
        description="Element set whose members are marked for refinement.",
        dtype=str,
        default=None,
        required=True,
    )


class ElementSetMarker(MarkerBase):
    schema = ElementSetMarkerSchema

    def __init__(self, elSetName, initialOnly=False):
        super().__init__(initialOnly)
        self.elSetName = elSetName

    @classmethod
    def fromOptions(cls, options):
        opts = buildSchemaFromOptions(cls.schema, options)
        return cls(opts.elSet, initialOnly=opts.initialOnly)

    def mark(self, model, refineElements, mesh):
        if self.elSetName not in model.elementSets:
            return set()
        return set(model.elementSets[self.elSetName])


@dataclass(frozen=True)
class NodeSetMarkerSchema(MarkerOptionsBase):
    """L2 options of a ``>>marker, type=nodeSet`` block."""

    nSet: str | None = schemaField(
        description="Node set whose touching elements are marked for refinement.",
        dtype=str,
        default=None,
        required=True,
    )


class NodeSetMarker(MarkerBase):
    schema = NodeSetMarkerSchema

    def __init__(self, nSetName, initialOnly=False):
        super().__init__(initialOnly)
        self.nSetName = nSetName

    @classmethod
    def fromOptions(cls, options):
        opts = buildSchemaFromOptions(cls.schema, options)
        return cls(opts.nSet, initialOnly=opts.initialOnly)

    def mark(self, model, refineElements, mesh):
        if self.nSetName not in model.nodeSets:
            return set()
        ns_nodes = set(model.nodeSets[self.nSetName].nodes)
        marked = set()
        for el in refineElements:
            if any(n in ns_nodes for n in el.nodes):
                marked.add(el)
        return marked


@dataclass(frozen=True)
class SurfaceMarkerSchema(MarkerOptionsBase):
    """L2 options of a ``>>marker, type=surface`` block."""

    surface: str | None = schemaField(
        description="Surface whose elements are marked for refinement.",
        dtype=str,
        default=None,
        required=True,
    )


class SurfaceMarker(MarkerBase):
    schema = SurfaceMarkerSchema

    def __init__(self, surfaceName, initialOnly=False):
        super().__init__(initialOnly)
        self.surfaceName = surfaceName

    @classmethod
    def fromOptions(cls, options):
        opts = buildSchemaFromOptions(cls.schema, options)
        return cls(opts.surface, initialOnly=opts.initialOnly)

    def mark(self, model, refineElements, mesh):
        if self.surfaceName not in model.surfaces:
            return set()
        marked = set()
        # model.surfaces[name] is a dict of faceID -> list of elements
        for elements in model.surfaces[self.surfaceName].values():
            for el in elements:
                marked.add(el)
        return marked


# A predictive Rankine (max-principal-stress) criterion needs no dedicated marker: the expression
# stack (edelweissfe.utils.math.mathModules) already ships eigVal(), which turns a Marmot Voigt
# stress row into its principal stresses in descending order. Compute the quantity in the stress
# fieldOutput's own f(x) and let a plain FieldOutputMarker threshold it:
#     >>perElement, ..., result=stress, f(x)='eigVal(x.reshape(-1,6))[:,0].reshape(x.shape[0],x.shape[1])'
#     >>marker, type=fieldOutput, fieldOutput=<that output>, operator='>=', threshold=<factor*f_t>
# refines every element whose largest principal stress reaches the threshold at any quadrature point.
# See examples/WinklerL_AMR.


def _sprPolynomialBasis(localCoords, order):
    """Complete polynomial basis (3D) evaluated at ``localCoords`` (shape ``(n, 3)``, relative to a
    patch centre). ``order=0`` -> ``[1]`` (1 term); ``order=1`` -> that plus ``[x, y, z]`` (4 terms);
    ``order=2`` -> that plus ``[x^2, y^2, z^2, xy, yz, zx]`` (10 terms)."""
    ones = np.ones((len(localCoords), 1))
    if order == 0:
        return ones
    x, y, z = localCoords[:, 0:1], localCoords[:, 1:2], localCoords[:, 2:3]
    if order == 1:
        return np.concatenate([ones, x, y, z], axis=1)
    return np.concatenate([ones, x, y, z, x * x, y * y, z * z, x * y, y * z, z * x], axis=1)


def _recoverAveraging(coordsAll, valuesAll, connectivity, nGlobalNodes, dim, volume):
    """Volume-weighted nodal averaging recovery (ZZ 1987): the recovered nodal gradient is the
    volume-weighted mean of the FE gradient sampled at that node by every element sharing it.
    Returns ``(recovered (nGlobalNodes, dim, 3), hasRecovery (nGlobalNodes,))``."""
    jacobianNode = np.einsum("ebi,abj->eaij", coordsAll, _NODE_DNDXI)  # (nElem, 20, 3, 3)
    paramGradNode = np.einsum("ebd,abj->eadj", valuesAll, _NODE_DNDXI)  # (nElem, 20, dim, 3)
    gradAtNode = np.einsum("eadj,eaji->eadi", paramGradNode, np.linalg.inv(jacobianNode))  # (nElem, 20, dim, 3)

    weight = np.where(volume > 0.0, volume, 0.0)  # (nElem,) drop any inverted/degenerate element
    flatConnectivity = connectivity.ravel()
    recoveredSum = np.zeros((nGlobalNodes, dim, 3))
    recoveredWeight = np.zeros(nGlobalNodes)
    np.add.at(recoveredSum, flatConnectivity, (weight[:, None, None, None] * gradAtNode).reshape(-1, dim, 3))
    np.add.at(recoveredWeight, flatConnectivity, np.repeat(weight, N_NODES))
    hasRecovery = recoveredWeight > 0.0
    recovered = np.zeros((nGlobalNodes, dim, 3))
    recovered[hasRecovery] = recoveredSum[hasRecovery] / recoveredWeight[hasRecovery][:, None, None]
    return recovered, hasRecovery


def _recoverSpr(coordsAll, connectivity, nGlobalNodes, dim, gaussCoords, gradGauss):
    """Superconvergent patch recovery (Zienkiewicz-Zhu 1992). For each corner (vertex) node a least-
    squares polynomial is fitted to the FE gradient at the superconvergent Gauss points of the patch
    of elements sharing that node; the recovered corner value is the fit at the node. Each edge
    midside node is recovered by averaging the two corner-endpoint patch fits evaluated at its
    location. Returns ``(recovered (nGlobalNodes, dim, 3), hasRecovery (nGlobalNodes,))``.

    Parameters
    ----------
    gaussCoords
        ``(nElem, 8, 3)`` physical coordinates of each element's Gauss points.
    gradGauss
        ``(nElem, 8, dim, 3)`` FE gradient at those Gauss points.
    """
    nComponents = dim * 3
    gaussCoords = gaussCoords.reshape(-1, 8, 3)
    gaussValues = gradGauss.reshape(-1, 8, nComponents)

    # global node coordinates, corner patches, and edge (midside -> two corners) topology
    nodeCoords = np.zeros((nGlobalNodes, 3))
    seen = np.zeros(nGlobalNodes, dtype=bool)
    cornerElements = defaultdict(list)  # global corner node -> element indices sharing it
    edgeCorners = {}  # global midside node -> (global cornerA, global cornerB)
    for e, conn in enumerate(connectivity):
        for a in range(N_NODES):
            g = conn[a]
            if not seen[g]:
                nodeCoords[g] = coordsAll[e, a]
                seen[g] = True
        for cornerLocal in _CORNER_LOCALS:
            cornerElements[conn[cornerLocal]].append(e)
        for midLocal, cornerA, cornerB in _MID_EDGES:
            edgeCorners[conn[midLocal]] = (conn[cornerA], conn[cornerB])

    # fit one polynomial per corner patch; the value at the corner is the constant term (the fit is
    # centred on the corner, so the basis reduces to [1, 0, ...] there)
    cornerFit = {}  # global corner -> (order, coeffs (nBasis, nComponents))
    recovered = np.zeros((nGlobalNodes, dim, 3))
    for corner, elements in cornerElements.items():
        elements = np.asarray(elements)
        sampleCoords = gaussCoords[elements].reshape(-1, 3) - nodeCoords[corner]
        sampleValues = gaussValues[elements].reshape(-1, nComponents)
        nSamples = sampleCoords.shape[0]
        order = 2 if nSamples >= 10 else (1 if nSamples >= 4 else 0)
        basis = _sprPolynomialBasis(sampleCoords, order)
        coeffs, _, _, _ = np.linalg.lstsq(basis, sampleValues, rcond=None)  # (nBasis, nComponents)
        cornerFit[corner] = (order, coeffs)
        recovered[corner] = coeffs[0].reshape(dim, 3)

    # midside nodes: average of the adjacent corner patches, each evaluated at the midside location
    hasRecovery = seen.copy()
    for midside, (cornerA, cornerB) in edgeCorners.items():
        values = []
        for corner in (cornerA, cornerB):
            fit = cornerFit.get(corner)
            if fit is not None:
                order, coeffs = fit
                basis = _sprPolynomialBasis((nodeCoords[midside] - nodeCoords[corner])[None, :], order)
                values.append((basis @ coeffs).ravel())
        if values:
            recovered[midside] = np.mean(values, axis=0).reshape(dim, 3)
        else:
            hasRecovery[midside] = False
    return recovered, hasRecovery


def _recoveryIndicators(coordsAll, valuesAll, connectivity, nGlobalNodes, recovery="averaging"):
    r"""Zienkiewicz-Zhu recovered-gradient error indicator, over all elements at once.

    The FE gradient at the 2x2x2 Gauss points and the final error norm are batched into single
    ``numpy`` calls; the recovery of the continuous gradient ``grad*`` is delegated to
    :func:`_recoverAveraging` (``recovery='averaging'``) or :func:`_recoverSpr` (``'spr'``).

    Parameters
    ----------
    coordsAll
        ``(nElem, 20, 3)`` node coordinates of each element, in C3D20 order.
    valuesAll
        ``(nElem, 20, dim)`` nodal values of the recovered field on each element.
    connectivity
        ``(nElem, 20)`` compact global node index of each element node, so nodes shared between
        elements accumulate into the same recovery slot.
    nGlobalNodes
        Number of distinct global nodes referenced by ``connectivity``.
    recovery
        ``'averaging'`` (nodal averaging) or ``'spr'`` (superconvergent patch recovery).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``indicators`` ``(nElem,)`` :math:`\eta_K = \lVert \nabla u^{*} - \nabla u^{h} \rVert_{L^2(K)}`
        and ``gradEnergy`` ``(nElem,)`` :math:`\lVert \nabla u^{h} \rVert_{L^2(K)}^2` -- the FE-gradient
        energy, used as the reference scale for the global relative error that gates marking (so a
        smooth, well-resolved field, whose indicators are pure floating-point noise, marks nothing).
    """
    dim = valuesAll.shape[2]

    # FE gradient at the 2x2x2 Gauss points (grad^h) and the element volumes
    jacobianGauss = np.einsum("eai,gaj->egij", coordsAll, _GP_DNDXI)  # (nElem, 8, 3, 3): dx_i/dxi_j
    detJGauss = np.linalg.det(jacobianGauss)  # (nElem, 8)
    paramGradGauss = np.einsum("ead,gaj->egdj", valuesAll, _GP_DNDXI)  # (nElem, 8, dim, 3): du_d/dxi_j
    gradH = np.einsum("egdj,egji->egdi", paramGradGauss, np.linalg.inv(jacobianGauss))  # (nElem, 8, dim, 3)
    volume = detJGauss.sum(axis=1)  # (nElem,)
    detJPos = np.where(detJGauss > 0.0, detJGauss, 0.0)  # (nElem, 8) drop inverted-element Gauss points

    # || grad^h ||^2_L2(K): the field's own gradient energy, the reference scale for the relative error
    gradEnergy = np.sum(detJPos * np.sum(gradH**2, axis=(2, 3)), axis=1)  # (nElem,)

    if recovery == "spr":
        gaussCoords = np.einsum("ga,ead->egd", _GP_N, coordsAll)  # (nElem, 8, 3) physical Gauss coords
        recovered, hasRecovery = _recoverSpr(coordsAll, connectivity, nGlobalNodes, dim, gaussCoords, gradH)
    else:
        recovered, hasRecovery = _recoverAveraging(coordsAll, valuesAll, connectivity, nGlobalNodes, dim, volume)

    # eta_K = || grad* - grad^h ||_{L2(K)}, integrated at the Gauss points
    gradStar = np.einsum("ga,eadi->egdi", _GP_N, recovered[connectivity])  # (nElem, 8, dim, 3)
    perGauss = np.sum((gradStar - gradH) ** 2, axis=(2, 3))  # (nElem, 8)
    errorSq = np.sum(detJPos * perGauss, axis=1)  # (nElem,)
    # an element touching a node that never received a recovered value (all its elements degenerate)
    # cannot be scored; leave its indicator at zero
    errorSq[~hasRecovery[connectivity].all(axis=1)] = 0.0
    return np.sqrt(errorSq), gradEnergy


def _growByNeighbors(seed, candidatePool, layers):
    """Dilate ``seed`` by ``layers`` rings of node-adjacent elements drawn from ``candidatePool``.

    Two elements are neighbours if they share at least one node (a node-adjacency, so it also picks up
    edge/corner touches, not only shared faces -- the coarser stencil is what a refinement halo wants).
    Returns a new ``set`` containing ``seed`` plus the grown ring; ``candidatePool`` bounds the growth
    so the halo can never leak onto non-refineable elements.
    """
    if layers <= 0:
        return set(seed)
    # node label -> elements of the pool touching it, built once
    elementsAtNode = defaultdict(list)
    for element in candidatePool:
        for node in element.nodes:
            elementsAtNode[node.label].append(element)
    grown = set(seed)
    frontier = set(seed)
    for _ in range(layers):
        nextFrontier = set()
        for element in frontier:
            for node in element.nodes:
                for neighbor in elementsAtNode[node.label]:
                    if neighbor not in grown:
                        grown.add(neighbor)
                        nextFrontier.add(neighbor)
        if not nextFrontier:
            break
        frontier = nextFrontier
    return grown


@dataclass(frozen=True)
class RecoveryErrorMarkerSchema(MarkerOptionsBase):
    """L2 options of a ``>>marker, type=recoveryError`` block."""

    nodeField: str | None = schemaField(
        description=(
            "Name of the nodal field whose recovered-gradient (Zienkiewicz-Zhu) error drives "
            "marking, e.g. 'nonlocal damage'."
        ),
        dtype=str,
        default=None,
        required=True,
    )
    markFraction: float = schemaField(
        description=(
            "Doerfler bulk-marking fraction theta in (0, 1]; refine the worst elements accumulating "
            "this fraction of the total squared error."
        ),
        dtype=float,
        default=0.5,
    )
    maxRefinedFraction: float = schemaField(
        description=(
            "Hard cap on the fraction of elements a single pass may mark (bounds the direct-solver "
            "factorization cost). Note that a non-zero 'halo' adds neighbor rings around these marks "
            "and can increase the final count beyond this fraction. Refinement may also be deferred "
            "across increments until 'minMarkedElements' accumulate at the model-modifier level."
        ),
        dtype=float,
        default=0.1,
    )
    fieldThreshold: float = schemaField(
        description=(
            "Absolute floor on the peak nodal |field| per element; an element is a refinement "
            "candidate only where the driving field is significant. This is what keeps the elastic "
            "phase (nonlocal field at machine zero) from being refined -- set it to a small fraction "
            "of the field value at damage initiation. 0 disables."
        ),
        dtype=float,
        default=1e-6,
    )
    relTol: float = schemaField(
        description=(
            "Global relative-error threshold applied among the significant elements; while "
            "||eta|| / ||grad|| stays below it the field is treated as well resolved and nothing is "
            "refined. 0 disables."
        ),
        dtype=float,
        default=1e-3,
    )
    halo: int = schemaField(
        description=(
            "Number of rings of node-adjacent elements to add around the marked set (a refinement "
            "buffer). Since the ZZ error peaks on the flanks of a localization band and dips over "
            "its core, halo=1 bridges the flank rows across the core; larger values also refine "
            "ahead of the front. Grows only within refineable elements. 0 disables."
        ),
        dtype=int,
        default=0,
    )
    recovery: str = schemaField(
        description=(
            "Gradient recovery method -- 'averaging' (nodal averaging) or 'spr' (superconvergent "
            "patch recovery, sharper for serendipity elements)."
        ),
        dtype=str,
        default="averaging",
    )
    entry: str = schemaField(
        description="Node-field value entry to read ('U', the current converged field, by default).",
        dtype=str,
        default="U",
    )


class RecoveryErrorMarker(MarkerBase):
    r"""Zienkiewicz-Zhu recovery-based error indicator on the *gradient* of a nodal field, with
    Doerfler bulk marking. Aimed at gradient-enhanced damage: the nonlocal driving field
    :math:`\bar\varepsilon` (governed by a screened-Poisson/Helmholtz equation) is smooth, but its
    gradient localizes in the process zone, and that is where the mesh must be resolved.

    The per-element indicator is the L2 norm of the difference between a recovered, continuous
    gradient field and the raw (discontinuous) FE gradient,

    .. math:: \eta_K = \lVert \nabla\bar\varepsilon^{*} - \nabla\bar\varepsilon^{h} \rVert_{L^2(K)},

    which is exactly the thermal-flux ZZ estimator applied to the nonlocal field (the constant
    length-scale factor :math:`\ell^2` cancels out of a pure ranking, so it is omitted). The
    recovered field :math:`\nabla\bar\varepsilon^{*}` is built by volume-weighted nodal averaging of
    the FE gradient sampled at the 2x2x2 (superconvergent) Gauss points -- the serendipity HEX20
    sweet spot. (An SPR patch-least-squares variant is a planned upgrade behind ``recovery``.)

    Selection is by Doerfler bulk marking on the squared indicators: the worst elements whose
    :math:`\sum \eta_K^2` first reaches ``markFraction`` of the total are marked -- but never more
    than ``maxRefinedFraction`` of all elements. Under a fixed single-level refinement this is the
    natural criterion: refinement depth is fixed, so marking reduces to ranking, and the hard cap
    keeps the direct-solver (PARDISO) factorization cost bounded.

    Doerfler is a purely *relative* criterion, so on its own it always marks a ``markFraction`` share
    of the ranking -- even in the elastic phase, before any damage, where the nonlocal field is
    *machine zero* (~1e-30). Crucially, a relative gate cannot fix this: the noise-level field's
    recovered-vs-FE gradient mismatch is O(1) *relative to itself*, indistinguishable from a genuinely
    localizing field. Marking is therefore gated by an **absolute** floor ``fieldThreshold`` on the
    peak nodal ``|field|`` per element -- an element is a candidate only where the driving field is
    significant (the active-damage field is 1e-5 and up, ~25 orders above the elastic noise, so the
    separation is unambiguous). Among those significant elements a secondary ZZ *global relative error*
    gate ``relTol`` skips regions the mesh already resolves well.

    Parameters
    ----------
    nodeFieldName
        Name of the nodal field whose recovered-gradient error drives marking (e.g.
        ``'nonlocal damage'``).
    markFraction
        Doerfler bulk fraction ``theta`` in ``(0, 1]``: mark the worst elements accumulating this
        fraction of the total squared error.
    maxRefinedFraction
        Hard cap on the fraction of eligible elements a single pass may mark.
    fieldThreshold
        Absolute floor on the peak nodal ``|field|`` per element: an element is a refinement candidate
        only where the driving field itself is significant. This is what keeps the elastic phase (where
        the nonlocal field is machine zero, ~1e-30) from being refined -- a purely relative criterion
        cannot, because the noise-level field's recovered-vs-FE gradient mismatch is O(1) relative to
        itself. Set it to a small fraction of the field value at damage initiation (for a strain-like
        driving field, damage onset is ``O(f_t / E)``, so a default of ``1e-6`` sits well below onset
        yet ~20 orders above the elastic noise). Set to ``0.0`` to disable and refine on gradient
        roughness alone.
    relTol
        Global relative-error threshold, applied among the significant elements only: while
        :math:`\lVert \eta \rVert / \lVert \nabla\bar\varepsilon^{h} \rVert < \text{relTol}` the field
        is treated as well resolved and no element is refined. Set to ``0.0`` to disable.
    halo
        Number of rings of node-adjacent elements to add around the Doerfler-marked set (a refinement
        buffer). The ZZ estimator responds to gradient error, which peaks on the *flanks* of a
        localization band and dips over its (locally smooth) core, so on its own it can refine two rows
        straddling a crack while leaving the core coarse. ``halo=1`` bridges those rows across the
        core; a larger halo also refines ahead of the propagating front. ``0`` (default) keeps the bare
        Doerfler set. The halo grows only within the refineable elements, so it never leaks elsewhere,
        and it is applied *after* the ``maxRefinedFraction`` cap, so the final count may exceed it.
    recovery
        Recovery method: ``'averaging'`` (volume-weighted nodal averaging, ZZ 1987) or ``'spr'``
        (superconvergent patch recovery, ZZ 1992 -- the sharper choice for serendipity elements).
    entry
        Node-field value entry to read (``'U'``, the current converged field, by default).
    """

    _RECOVERY_METHODS = ("averaging", "spr")

    schema = RecoveryErrorMarkerSchema

    @classmethod
    def fromOptions(cls, options):
        opts = buildSchemaFromOptions(cls.schema, options)
        return cls(
            opts.nodeField,
            markFraction=opts.markFraction,
            maxRefinedFraction=opts.maxRefinedFraction,
            fieldThreshold=opts.fieldThreshold,
            relTol=opts.relTol,
            halo=opts.halo,
            recovery=opts.recovery,
            entry=opts.entry,
            initialOnly=opts.initialOnly,
        )

    def __init__(
        self,
        nodeFieldName,
        markFraction=0.5,
        maxRefinedFraction=0.1,
        fieldThreshold=1e-6,
        relTol=1e-3,
        halo=0,
        recovery="averaging",
        entry="U",
        initialOnly=False,
    ):
        super().__init__(initialOnly)
        self.nodeFieldName = nodeFieldName
        self.markFraction = float(markFraction)
        self.maxRefinedFraction = float(maxRefinedFraction)
        self.fieldThreshold = float(fieldThreshold)
        self.relTol = float(relTol)
        self.halo = int(halo)
        if recovery not in self._RECOVERY_METHODS:
            raise ValueError(
                f"RecoveryErrorMarker: unknown recovery={recovery!r}; expected one of {self._RECOVERY_METHODS}."
            )
        self.recovery = recovery
        self.entry = entry

    def mark(self, model, refineElements, mesh):
        if model.nodeFields is None or self.nodeFieldName not in model.nodeFields:
            raise KeyError(
                f"hAdaptivity recoveryError marker references nodeField {self.nodeFieldName!r}, "
                "which is not defined on the model. Use the exact field name (e.g. 'nonlocal damage')."
            )
        nodeField = model.nodeFields[self.nodeFieldName]
        if self.entry not in nodeField:
            return set()
        values = nodeField[self.entry]
        indexOf = nodeField._indicesOfNodesInArray

        # gather, for every eligible element, its node coordinates and field values plus a compact
        # global node index (so shared nodes accumulate into the same recovery slot). Elements with a
        # node that does not carry the field are skipped -- they cannot contribute to recovery.
        elements = []
        coordsList = []
        valuesList = []
        connectivity = []
        nodeToIndex = {}
        for element in refineElements:
            if len(element.nodes) != N_NODES:
                continue
            rows = [indexOf.get(node, -1) for node in element.nodes]
            if -1 in rows:
                continue
            localConnectivity = []
            for node in element.nodes:
                globalIndex = nodeToIndex.get(node)
                if globalIndex is None:
                    globalIndex = len(nodeToIndex)
                    nodeToIndex[node] = globalIndex
                localConnectivity.append(globalIndex)
            elements.append(element)
            coordsList.append([node.coordinates for node in element.nodes])
            valuesList.append(values[rows])
            connectivity.append(localConnectivity)

        if not elements:
            return set()

        coordsAll = np.asarray(coordsList, dtype=float)  # (nElem, 20, 3)
        valuesAll = np.asarray(valuesList, dtype=float)  # (nElem, 20, dim)
        connectivity = np.asarray(connectivity, dtype=np.intp)  # (nElem, 20)
        indicators, gradEnergy = _recoveryIndicators(
            coordsAll, valuesAll, connectivity, len(nodeToIndex), self.recovery
        )

        # ---- absolute field-magnitude eligibility ----
        # An element is a refinement candidate only where the driving field itself is significant.
        # This is the decisive gate for gradient-enhanced damage: in the elastic phase the nonlocal
        # field is machine zero (~1e-30), yet its *shape* is floating-point noise whose recovered-vs-FE
        # gradient mismatch is O(1) relative to itself -- so a purely relative (scale-invariant)
        # criterion cannot tell it apart from a genuinely localizing field and would refine the elastic
        # structure everywhere. An absolute floor on the peak nodal |field| per element separates the
        # two cleanly (the active-damage field is 1e-5 and up, ~25 orders above the elastic noise) and
        # restricts marking to the process zone once damage is active. Elements below the floor are
        # scored zero and never reached by the ranking below.
        elementFieldMax = np.max(np.abs(valuesAll), axis=(1, 2))  # (nElem,) peak |field| per element
        indicators = np.where(elementFieldMax >= self.fieldThreshold, indicators, 0.0)

        # ---- global relative-error gate ----
        # Among the significant elements, skip refinement while the field is already well resolved: the
        # ZZ global relative error ||eta|| / ||grad|| must exceed relTol for any element to be marked.
        totalSquared = float(np.sum(indicators**2))
        if totalSquared <= 0.0:
            return set()
        referenceSquared = float(np.sum(np.where(elementFieldMax >= self.fieldThreshold, gradEnergy, 0.0)))
        if referenceSquared <= 0.0 or np.sqrt(totalSquared / referenceSquared) < self.relTol:
            return set()

        # ---- Doerfler bulk marking, capped by the DOF budget ----
        target = self.markFraction * totalSquared
        budget = max(1, int(np.ceil(self.maxRefinedFraction * len(elements))))
        order = np.argsort(indicators)[::-1]  # worst first

        marked = set()
        accumulated = 0.0
        for idx in order:
            if len(marked) >= budget or indicators[idx] <= 0.0:
                break
            marked.add(elements[idx])
            accumulated += indicators[idx] ** 2
            if accumulated >= target:
                break

        # optional refinement halo: bridge the flank-marked rows across the (locally smooth) band core
        # and buffer ahead of the front, growing only within the refineable elements
        if self.halo > 0 and marked:
            marked = _growByNeighbors(marked, refineElements, self.halo)
        return marked
