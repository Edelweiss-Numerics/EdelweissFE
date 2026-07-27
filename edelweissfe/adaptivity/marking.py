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

from edelweissfe.utils.fieldoutput import ElementFieldOutput
from edelweissfe.utils.math import createMathExpression


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
