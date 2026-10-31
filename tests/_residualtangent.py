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
"""Helpers for checking the rule every constraint follows: it adds a residual contribution ``P`` and the tangent
``K = dP/dU`` (see the solver documentation, "The residual and the tangent").

Because ``K`` must be the derivative of what is added to ``P``, comparing it with central differences of ``P`` checks
both at once: a wrong tangent and a ``P`` of the wrong sign each show up as a mismatch.
"""

import numpy as np


def localVector(constraint, nodalValue, scalarValues=()) -> np.ndarray:
    """A constraint's local degree-of-freedom vector, in the constraint's own layout.

    Parameters
    ----------
    constraint
        The constraint.
    nodalValue
        ``nodalValue(node, field)`` returns the values of ``field`` at ``node``.
    scalarValues
        The values of the constraint's own scalar variables (e.g. Lagrange multipliers), which come last.

    Returns
    -------
    np.ndarray
        The local vector of size ``constraint.nDof``.
    """

    parts = [
        np.atleast_1d(nodalValue(node, field))
        for node, fields in zip(constraint.nodes, constraint.fieldsOnNodes)
        for field in fields
    ]
    parts.append(np.asarray(scalarValues, dtype=float))
    U = np.concatenate(parts).astype(float)
    assert U.size == constraint.nDof, "the local vector does not match the constraint's layout"
    return U


def residualContributionAndTangent(
    constraint, U: np.ndarray, dU: np.ndarray, timeStep=None
) -> tuple[np.ndarray, np.ndarray]:
    """The residual contribution of a constraint and its tangent, the latter as one dense matrix.

    The tangent is scattered from the flat VIJ values with the constraint's own ``(I, J)`` pattern -- the one the
    global assembly uses -- so that the check covers the pattern together with the values.

    Parameters
    ----------
    constraint
        The constraint.
    U
        The local solution.
    dU
        The local solution increment.
    timeStep
        The time step handed to the constraint.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``P`` and the dense ``nDof x nDof`` tangent.
    """

    nDof = constraint.nDof
    size = constraint.getVIJContributionSize()
    I_ = np.zeros(size, dtype=int)
    J_ = np.zeros(size, dtype=int)
    constraint.initializeVIJContribution(np.arange(nDof), I_, J_, 0)

    flat = np.zeros(size)
    P = np.zeros(nDof)
    constraint.computeKernels(U, dU, P, constraint.shapeVIJContribution(flat), timeStep)

    K = np.zeros((nDof, nDof))
    np.add.at(K, (I_, J_), flat)
    return P, K


def centralDifferences(function, x: np.ndarray, h: float = 1e-7) -> np.ndarray:
    """The Jacobian of ``function`` at ``x`` by central differences.

    Parameters
    ----------
    function
        Maps a vector to a vector.
    x
        Where to differentiate.
    h
        The perturbation.

    Returns
    -------
    np.ndarray
        The matrix ``d function / d x``.
    """

    columns = []
    for i in range(x.size):
        perturbation = np.zeros(x.size)
        perturbation[i] = h
        columns.append((function(x + perturbation) - function(x - perturbation)) / (2.0 * h))
    return np.column_stack(columns)


def finiteDifferenceTangent(constraint, U: np.ndarray, dU: np.ndarray, timeStep=None, h: float = 1e-7) -> np.ndarray:
    """The derivative of a constraint's residual contribution by central differences.

    ``U`` and ``dU`` are perturbed together: within an increment the unknown is the increment, ``U = U_n + dU``, and a
    constraint with a history (friction) reads both.

    Parameters
    ----------
    constraint
        The constraint.
    U
        The local solution.
    dU
        The local solution increment.
    timeStep
        The time step handed to the constraint.
    h
        The perturbation.

    Returns
    -------
    np.ndarray
        The dense ``nDof x nDof`` matrix ``dP/dU``.
    """

    def residualContribution(perturbation):
        return residualContributionAndTangent(constraint, U + perturbation, dU + perturbation, timeStep)[0]

    return centralDifferences(residualContribution, np.zeros(constraint.nDof), h)
