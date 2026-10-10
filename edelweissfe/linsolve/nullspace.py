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
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation; either
#  version 2.1 of the License, or (at your option) any later version.
#
#  The full text of the license can be found in the file LICENSE.md at
#  the top level directory of EdelweissFE.
#  ---------------------------------------------------------------------
"""Rigid-body near-null-space bases for one vector field's diagonal block.

Solver-independent: both functions are pure geometry/linear algebra over a
:class:`~edelweissfe.linsolve.base.FieldBlock`, its diagonal-equilibration scaling, and (for the
richer basis) its node coordinates -- no solver-instance state. Any algebraic-multigrid-style
preconditioner needing an elasticity near-null-space can use them; today
:class:`~edelweissfe.linsolve.blockamg.blockamg.BlockAMGSolver` and its p-two-grid variant do, to
seed AMGCL's smoothed-aggregation coarsening.

:func:`rigidBodyNullspace` (translations *and* rotations) is preferred whenever node coordinates are
available: measured ~28-31% fewer isolated per-field outer iterations than :func:`translationNullspace`
alone on two real captured production systems, and unlike translations alone, robust to both thread
count and the Chebyshev ``power_iters`` setting. It costs one extra pass over the field's diagonal
scaling vector plus a handful of elementwise operations on the node coordinates -- O(nNodes),
negligible against the AMG hierarchy build it feeds.
"""

import numpy as np

from edelweissfe.linsolve.base import FieldBlock


def translationNullspace(block: FieldBlock, blockDinv: np.ndarray) -> np.ndarray:
    """The rigid-body translations of a vector field, transformed for the scaled operator.

    Translations are 1 on each of the ``dimension`` components (node-major). The near null-space of
    the scaled block :math:`D^{-1/2} A D^{-1/2}` is :math:`D^{1/2}` times that of :math:`A`, i.e.
    the raw translations divided by ``blockDinv``.
    """
    size = block.stop - block.start
    components = block.dimension
    B = np.zeros((size, components))
    rows = np.arange(size)
    B[rows, rows % components] = 1.0
    return B / blockDinv[:, None]


def rigidBodyNullspace(block: FieldBlock, coords: np.ndarray, blockDinv: np.ndarray) -> np.ndarray:
    """The full rigid-body near-null-space (translations *and* rotations) of a vector field,
    transformed for the scaled operator -- see :func:`translationNullspace` for the equilibration
    transform this mirrors exactly.

    3 translations + 3 rotations for a 3D field (6 modes); 2 translations + 1 in-plane rotation for
    a 2D field (3 modes) -- the standard elasticity near-null-space. Rotations are the classic
    infinitesimal rigid-rotation displacement fields about the field's coordinate centroid,
    node-major to match ``coords``' and the DOF vector's own layout: 3D rotations about x/y/z are
    ``(0,-z,y)``/``(z,0,-x)``/``(-y,x,0)``; the 2D in-plane rotation is ``(-y,x)``.

    Parameters
    ----------
    coords
        This field's node coordinates, node-major, shape ``(nNodes, >=dimension)`` (extra trailing
        columns, e.g. an out-of-plane coordinate on a 2D field, are ignored).
    """
    size = block.stop - block.start
    components = block.dimension
    nNodes = size // components
    coords = np.asarray(coords, dtype=float)[:, :components]
    if coords.shape[0] != nNodes:
        raise ValueError(
            "blockamg: node coordinates for field '{:}' have {:} rows, expected {:} to match its "
            "{:} dofs over {:} components".format(block.name, coords.shape[0], nNodes, size, components)
        )
    centered = coords - coords.mean(axis=0)

    if components == 3:
        numRotations = 3
    elif components == 2:
        numRotations = 1
    else:
        # No rigid-body rotation exists for any other nodal dimension -- translations are already
        # the full rigid-body space there.
        return translationNullspace(block, blockDinv)

    B = np.zeros((nNodes, components, components + numRotations))
    for c in range(components):
        B[:, c, c] = 1.0  # translations, identical construction to translationNullspace

    if components == 3:
        x, y, z = centered[:, 0], centered[:, 1], centered[:, 2]
        B[:, 1, 3], B[:, 2, 3] = -z, y  # rotation about x: (0, -z, y)
        B[:, 2, 4], B[:, 0, 4] = -x, z  # rotation about y: (z, 0, -x)
        B[:, 0, 5], B[:, 1, 5] = -y, x  # rotation about z: (-y, x, 0)
    else:  # components == 2
        x, y = centered[:, 0], centered[:, 1]
        B[:, 0, 2], B[:, 1, 2] = -y, x  # in-plane rotation: (-y, x)

    return B.reshape(size, components + numRotations) / blockDinv[:, None]
