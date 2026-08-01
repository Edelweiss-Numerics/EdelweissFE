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

import numpy as np

from edelweissfe.adaptivity.hex20shapefunctions import (
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
        Recovery method. Only ``'averaging'`` (volume-weighted nodal averaging) is implemented.
    entry
        Node-field value entry to read (``'U'``, the current converged field, by default).
    """

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
        if recovery != "averaging":
            raise NotImplementedError(
                f"RecoveryErrorMarker: recovery={recovery!r} is not implemented; only 'averaging' "
                "(volume-weighted nodal averaging) is available."
            )
        self.recovery = recovery
        self.entry = entry

    def _nodalValues(self, nodeField, element):
        """The (nNodes, dim) block of field values on ``element``'s nodes, or ``None`` if any node
        does not carry the field (so this element cannot contribute to recovery)."""
        values = nodeField[self.entry]
        indexOf = nodeField._indicesOfNodesInArray
        try:
            return np.array([values[indexOf[n]] for n in element.nodes])
        except KeyError:
            return None

    def mark(self, model, refineElements, mesh):
        if model.nodeFields is None or self.nodeFieldName not in model.nodeFields:
            raise KeyError(
                f"hAdaptivity recoveryError marker references nodeField {self.nodeFieldName!r}, "
                "which is not defined on the model. Use the exact field name (e.g. 'nonlocal damage')."
            )
        nodeField = model.nodeFields[self.nodeFieldName]
        if self.entry not in nodeField:
            return set()

        elements = [el for el in refineElements if len(el.nodes) == N_NODES]
        if not elements:
            return set()

        # ---- pass 1: FE gradient at the Gauss points + volume-weighted nodal recovery ----
        # Per element we cache the physical coordinates, the FE gradient at each Gauss point, and the
        # Gauss-point Jacobian determinants, so pass 2 needs no re-evaluation. Simultaneously we
        # accumulate, at every node, the volume-weighted FE gradient to form the recovered field.
        perElement = []  # list of (element, gradH (8, dim, 3), detJ (8,)) or None to skip
        recoveredSum = {}  # node -> summed (dim, 3) gradient, volume-weighted
        recoveredWeight = {}  # node -> summed volume weight
        for element in elements:
            nodalValues = self._nodalValues(nodeField, element)
            if nodalValues is None:
                perElement.append(None)
                continue
            coords = np.array([n.coordinates for n in element.nodes], dtype=float)  # (20, 3)

            gradH = np.empty((len(_GAUSS_POINTS), nodalValues.shape[1], 3))
            detJ = np.empty(len(_GAUSS_POINTS))
            for g in range(len(_GAUSS_POINTS)):
                jacobian = coords.T @ _GP_DNDXI[g]  # (3, 3), J[i, j] = dx_i/dxi_j
                detJ[g] = np.linalg.det(jacobian)
                dNdx = _GP_DNDXI[g] @ np.linalg.inv(jacobian)  # (20, 3)
                gradH[g] = nodalValues.T @ dNdx  # (dim, 3)
            volume = float(np.sum(detJ))
            perElement.append((element, gradH, detJ))
            if volume <= 0.0:
                continue

            # recovery: sample the FE gradient at each node's own parametric location and accumulate
            for a, node in enumerate(element.nodes):
                jacobian = coords.T @ _NODE_DNDXI[a]
                gradAtNode = nodalValues.T @ (_NODE_DNDXI[a] @ np.linalg.inv(jacobian))  # (dim, 3)
                if node in recoveredSum:
                    recoveredSum[node] += volume * gradAtNode
                    recoveredWeight[node] += volume
                else:
                    recoveredSum[node] = volume * gradAtNode
                    recoveredWeight[node] = volume

        recovered = {node: recoveredSum[node] / recoveredWeight[node] for node in recoveredSum}

        # ---- pass 2: per-element indicator eta_K = || grad* - grad^h ||_{L2(K)} ----
        indicators = np.zeros(len(elements))
        for i, record in enumerate(perElement):
            if record is None:
                continue
            element, gradH, detJ = record
            if any(n not in recovered for n in element.nodes):
                continue  # no recovered gradient available on a node (all its elements degenerate)
            recoveredNodal = np.array([recovered[n] for n in element.nodes])  # (20, dim, 3)
            errorSq = 0.0
            for g in range(len(_GAUSS_POINTS)):
                gradStar = np.tensordot(_GP_N[g], recoveredNodal, axes=(0, 0))  # (dim, 3)
                if detJ[g] > 0.0:
                    errorSq += detJ[g] * float(np.sum((gradStar - gradH[g]) ** 2))
            indicators[i] = np.sqrt(errorSq)

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
