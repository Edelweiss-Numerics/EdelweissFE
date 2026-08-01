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

"""Refinement marking (WS-G): decide which elements to refine from a user expression evaluated on a
per-element quantity -- reusing EdelweissFE's math-expression stack
(:func:`edelweissfe.utils.math.createMathExpression`), the same mechanism as monitor/conditionalstop.
"""

from collections import defaultdict

import numpy as np

from edelweissfe.adaptivity.hex20shapefunctions import (
    EDGES,
    LOCAL_COORDS,
    N_NODES,
    hex20_shape,
    hex20_shape_grad,
)
from edelweissfe.utils.fieldoutput import ElementFieldOutput
from edelweissfe.utils.math import createMathExpression

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


class MarkerBase:
    def __init__(self, initialOnly=False):
        self.initialOnly = initialOnly

    def mark(self, model, refineElements, mesh):
        raise NotImplementedError()


class FieldOutputMarker(MarkerBase):
    """Marks elements by evaluating a boolean expression against an already-declared ``*fieldOutput``
    (a ``perElement`` one, covering every quadrature point of interest), instead of re-deriving a
    per-element scalar from the elements' raw quadrature-point results independently. This way
    refinement is driven by the exact same numbers the field output reports -- not a second,
    possibly divergent, reduction over the same underlying data (e.g. a marker silently reducing
    over all QPs while the corresponding output only reports QP 1).
    """

    def __init__(self, fieldOutputName, expression, initialOnly=False):
        super().__init__(initialOnly)
        self.fieldOutputName = fieldOutputName
        self._predicate = createMathExpression(expression)

    def mark(self, model, refineElements, mesh):
        fieldOutputController = model.fieldOutputController
        if fieldOutputController is None or self.fieldOutputName not in fieldOutputController.fieldOutputs:
            raise KeyError(
                f"hAdaptivity marker references fieldOutput {self.fieldOutputName!r}, which is not "
                "defined. Declare it under '*fieldOutput' (before the increment loop starts) so the "
                "marker can look it up by name."
            )
        fieldOutput = fieldOutputController.fieldOutputs[self.fieldOutputName]

        if not isinstance(fieldOutput, ElementFieldOutput):
            raise TypeError(
                f"hAdaptivity marker references fieldOutput {self.fieldOutputName!r}, which is a "
                f"{type(fieldOutput).__name__}; only a 'perElement' fieldOutput exposes one result "
                "row per element, as required for marking."
            )
        if fieldOutput.f is not None:
            raise ValueError(
                f"hAdaptivity marker references fieldOutput {self.fieldOutputName!r}, which defines "
                "'f(x)'; that reduces away the per-element axis, so it can no longer drive per-element "
                "marking. Declare a separate, unreduced fieldOutput (same result/quadraturePoint, no "
                "f(x)) for marking."
            )

        elements = list(fieldOutput.associatedSet)
        values = np.asarray(fieldOutput.getLastResult())

        if values.shape[0] != len(elements):
            raise ValueError(
                f"hAdaptivity marker: fieldOutput {self.fieldOutputName!r} reports {values.shape[0]} "
                f"result row(s) for {len(elements)} element(s) in its associated set; refusing to mark "
                "against a mismatched fieldOutput."
            )

        marked = set()
        for element, row in zip(elements, values):
            if bool(np.any(self._predicate(row))):
                marked.add(element)
        return marked


class ElementSetMarker(MarkerBase):
    def __init__(self, elSetName, initialOnly=False):
        super().__init__(initialOnly)
        self.elSetName = elSetName

    def mark(self, model, refineElements, mesh):
        if self.elSetName not in model.elementSets:
            return set()
        return set(model.elementSets[self.elSetName])


class NodeSetMarker(MarkerBase):
    def __init__(self, nSetName, initialOnly=False):
        super().__init__(initialOnly)
        self.nSetName = nSetName

    def mark(self, model, refineElements, mesh):
        if self.nSetName not in model.nodeSets:
            return set()
        ns_nodes = set(model.nodeSets[self.nSetName].nodes)
        marked = set()
        for el in refineElements:
            if any(n in ns_nodes for n in el.nodes):
                marked.add(el)
        return marked


class SurfaceMarker(MarkerBase):
    def __init__(self, surfaceName, initialOnly=False):
        super().__init__(initialOnly)
        self.surfaceName = surfaceName

    def mark(self, model, refineElements, mesh):
        if self.surfaceName not in model.surfaces:
            return set()
        marked = set()
        # model.surfaces[name] is a dict of faceID -> list of elements
        for elements in model.surfaces[self.surfaceName].values():
            for el in elements:
                marked.add(el)
        return marked


def _spr_polynomial_basis(localCoords, order):
    """Complete polynomial basis (3D) evaluated at ``localCoords`` (shape ``(n, 3)``, relative to a
    patch centre). ``order=1`` -> ``[1, x, y, z]`` (4 terms); ``order=2`` -> that plus
    ``[x^2, y^2, z^2, xy, yz, zx]`` (10 terms)."""
    x, y, z = localCoords[:, 0], localCoords[:, 1], localCoords[:, 2]
    ones = np.ones(len(localCoords))
    if order == 1:
        return np.stack([ones, x, y, z], axis=1)
    return np.stack([ones, x, y, z, x * x, y * y, z * z, x * y, y * z, z * x], axis=1)


def _recover_averaging(coordsAll, valuesAll, connectivity, nGlobalNodes, dim, volume):
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


def _recover_spr(coordsAll, connectivity, nGlobalNodes, dim, gaussCoords, gradGauss):
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
    cornerFit = {}  # global corner -> (order, coeffs (nBasis, nComponents)) or ("mean", value (nComponents,))
    recovered = np.zeros((nGlobalNodes, dim, 3))
    for corner, elements in cornerElements.items():
        elements = np.asarray(elements)
        sampleCoords = gaussCoords[elements].reshape(-1, 3) - nodeCoords[corner]
        sampleValues = gaussValues[elements].reshape(-1, nComponents)
        nSamples = sampleCoords.shape[0]
        order = 2 if nSamples >= 10 else (1 if nSamples >= 4 else 0)
        if order == 0:  # too few points even for a linear fit -> fall back to the plain mean
            meanValue = sampleValues.mean(axis=0)
            cornerFit[corner] = ("mean", meanValue)
            recovered[corner] = meanValue.reshape(dim, 3)
            continue
        basis = _spr_polynomial_basis(sampleCoords, order)
        coeffs, _, _, _ = np.linalg.lstsq(basis, sampleValues, rcond=None)  # (nBasis, nComponents)
        cornerFit[corner] = (order, coeffs)
        recovered[corner] = coeffs[0].reshape(dim, 3)

    # midside nodes: average of the adjacent corner patches, each evaluated at the midside location
    hasRecovery = seen.copy()
    for midside, (cornerA, cornerB) in edgeCorners.items():
        values = []
        for corner in (cornerA, cornerB):
            fit = cornerFit.get(corner)
            if fit is None:
                continue
            if fit[0] == "mean":
                values.append(fit[1])
            else:
                order, coeffs = fit
                basis = _spr_polynomial_basis((nodeCoords[midside] - nodeCoords[corner])[None, :], order)
                values.append((basis @ coeffs).ravel())
        if values:
            recovered[midside] = np.mean(values, axis=0).reshape(dim, 3)
        else:
            hasRecovery[midside] = False
    return recovered, hasRecovery


def _recovery_indicators(coordsAll, valuesAll, connectivity, nGlobalNodes, recovery="averaging"):
    r"""Zienkiewicz-Zhu recovered-gradient error indicator, over all elements at once.

    The FE gradient at the 2x2x2 Gauss points and the final error norm are batched into single
    ``numpy`` calls; the recovery of the continuous gradient ``grad*`` is delegated to
    :func:`_recover_averaging` (``recovery='averaging'``) or :func:`_recover_spr` (``'spr'``).

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
    np.ndarray
        ``(nElem,)`` indicator :math:`\eta_K = \lVert \nabla u^{*} - \nabla u^{h} \rVert_{L^2(K)}`.
    """
    dim = valuesAll.shape[2]

    # FE gradient at the 2x2x2 Gauss points (grad^h) and the element volumes
    jacobianGauss = np.einsum("eai,gaj->egij", coordsAll, _GP_DNDXI)  # (nElem, 8, 3, 3): dx_i/dxi_j
    detJGauss = np.linalg.det(jacobianGauss)  # (nElem, 8)
    paramGradGauss = np.einsum("ead,gaj->egdj", valuesAll, _GP_DNDXI)  # (nElem, 8, dim, 3): du_d/dxi_j
    gradH = np.einsum("egdj,egji->egdi", paramGradGauss, np.linalg.inv(jacobianGauss))  # (nElem, 8, dim, 3)
    volume = detJGauss.sum(axis=1)  # (nElem,)

    if recovery == "spr":
        gaussCoords = np.einsum("ga,ead->egd", _GP_N, coordsAll)  # (nElem, 8, 3) physical Gauss coords
        recovered, hasRecovery = _recover_spr(coordsAll, connectivity, nGlobalNodes, dim, gaussCoords, gradH)
    else:
        recovered, hasRecovery = _recover_averaging(coordsAll, valuesAll, connectivity, nGlobalNodes, dim, volume)

    # eta_K = || grad* - grad^h ||_{L2(K)}, integrated at the Gauss points
    gradStar = np.einsum("ga,eadi->egdi", _GP_N, recovered[connectivity])  # (nElem, 8, dim, 3)
    perGauss = np.sum((gradStar - gradH) ** 2, axis=(2, 3))  # (nElem, 8)
    errorSq = np.sum(np.where(detJGauss > 0.0, detJGauss, 0.0) * perGauss, axis=1)  # (nElem,)
    # an element touching a node that never received a recovered value (all its elements degenerate)
    # cannot be scored; leave its indicator at zero
    errorSq[~hasRecovery[connectivity].all(axis=1)] = 0.0
    return np.sqrt(errorSq)


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
    recovery
        Recovery method: ``'averaging'`` (volume-weighted nodal averaging, ZZ 1987) or ``'spr'``
        (superconvergent patch recovery, ZZ 1992 -- the sharper choice for serendipity elements).
    entry
        Node-field value entry to read (``'U'``, the current converged field, by default).
    """

    _RECOVERY_METHODS = ("averaging", "spr")

    def __init__(
        self,
        nodeFieldName,
        markFraction=0.5,
        maxRefinedFraction=0.1,
        recovery="averaging",
        entry="U",
        initialOnly=False,
    ):
        super().__init__(initialOnly)
        self.nodeFieldName = nodeFieldName
        self.markFraction = float(markFraction)
        self.maxRefinedFraction = float(maxRefinedFraction)
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
        indicators = _recovery_indicators(coordsAll, valuesAll, connectivity, len(nodeToIndex), self.recovery)

        # ---- Doerfler bulk marking, capped by the DOF budget ----
        totalSquared = float(np.sum(indicators**2))
        if totalSquared <= 0.0:
            return set()
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
        return marked
