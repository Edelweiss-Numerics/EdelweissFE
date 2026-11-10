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
both at once: a wrong tangent, or a ``P`` whose sign alone is wrong, shows up as a mismatch. A ``P`` and a ``K`` negated
together pass, so a test checks the direction of the forces the constraint exerts as well, which only the mechanics of
the constraint can say: that a penalty pulls its nodes back, that a contact pushes the bodies apart.

They are part of the package, so that the tests of a package that adds constraints of its own can check them the same
way as the tests of EdelweissFE check its constraints.
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

    Raises
    ------
    ValueError
        If the values do not fill the constraint's layout: their number is not ``constraint.nDof``.
    """

    parts = [
        np.atleast_1d(nodalValue(node, field))
        for node, fields in zip(constraint.nodes, constraint.fieldsOnNodes)
        for field in fields
    ]
    parts.append(np.asarray(scalarValues, dtype=float))
    U = np.concatenate(parts).astype(float)
    if U.size != constraint.nDof:
        raise ValueError(
            "the local vector has {:} entries, but the constraint has {:} degrees of freedom".format(
                U.size, constraint.nDof
            )
        )
    return U


def residualContributionAndTangent(
    constraint, U: np.ndarray, dU: np.ndarray, timeStep=None
) -> tuple[np.ndarray, np.ndarray]:
    """The residual contribution of a constraint and its tangent, the latter as one dense matrix.

    The tangent is scattered from the flat values with the constraint's own local entries -- the ones the global
    assembly uses -- so that the check covers the entries together with the values; every local entry must lie within
    the constraint's degrees of freedom.

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

    Raises
    ------
    ValueError
        If a local entry of the constraint's block lies outside its degrees of freedom.
    """

    nDof = constraint.nDof
    localRows, localColumns = constraint.localEntriesOfBlock()
    for localIndices in (localRows, localColumns):
        if not np.all((localIndices >= 0) & (localIndices < nDof)):
            raise ValueError(
                "a local entry of the constraint's block lies outside its {:} degrees of freedom".format(nDof)
            )

    flat = np.zeros(len(localRows))
    P = np.zeros(nDof)
    constraint.computeKernels(U, dU, P, constraint.shapeBlock(flat), timeStep)

    K = np.zeros((nDof, nDof))
    np.add.at(K, (localRows, localColumns), flat)
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
