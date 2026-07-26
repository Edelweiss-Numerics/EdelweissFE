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
per-element quantity derived from the converged state -- reusing EdelweissFE's math-expression stack
(:func:`edelweissfe.utils.math.createMathExpression`), the same mechanism as monitor/conditionalstop.
"""

import numpy as np

from edelweissfe.utils.math import createMathExpression

_REDUCERS = {
    "max": np.max,
    "min": np.min,
    "mean": np.mean,
    "absmax": lambda a: np.max(np.abs(a)),
}


def elementScalar(element, result, reducer="absmax"):
    """Reduce a named quadrature-point result of an element to a single scalar over all its QPs."""
    red = _REDUCERS[reducer]
    values = [
        red(np.asarray(element.getResultArray(result, qp, getPersistentView=False)))
        for qp in range(element.getNumberOfQuadraturePoints())
    ]
    return float(red(np.asarray(values)))


def markElements(model, result, expression, reducer="absmax", elementLabels=None):
    """Return the set of element labels to refine.

    Parameters
    ----------
    model
        The FE model.
    result
        Name of the quadrature-point result to evaluate (e.g. ``"stress"``, ``"nonlocal damage"``).
    expression
        A boolean math expression in the symbol ``x`` (the reduced per-element scalar), e.g.
        ``"x > 0.1"``. Evaluated with :func:`createMathExpression`.
    reducer
        How to reduce the result over components and quadrature points: one of
        ``max``, ``min``, ``mean``, ``absmax`` (default).
    elementLabels
        Restrict marking to these element labels (default: all elements).

    Returns
    -------
    set
        Labels of the elements whose expression evaluates truthy.
    """
    if reducer not in _REDUCERS:  # fail loud on a mistyped reducer instead of silently marking nothing
        raise ValueError("Unknown reducer {!r}; expected one of {}.".format(reducer, sorted(_REDUCERS)))
    predicate = createMathExpression(expression)  # symbol "x"
    labels = model.elements.keys() if elementLabels is None else elementLabels
    marked = set()
    for label in labels:
        if label not in model.elements:  # e.g. a parent removed by a previous refinement
            continue
        element = model.elements[label]
        try:
            x = elementScalar(element, result, reducer)
        except (KeyError, ValueError):  # this element/material does not expose the marked result
            continue
        if bool(predicate(x)):
            marked.add(label)
    return marked
