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
"""Small linear systems with the field structure of a gradient-enhanced damage model, shared by the blockamg
tests: a 3D displacement field (3 DOFs per node) and a scalar nonlocal damage field, each a grid Laplacian."""

import numpy as np
import scipy.sparse as sp

from edelweissfe.linsolve.base import FieldBlock


def gridLaplacian(n: int) -> sp.csr_matrix:
    """The 7-point finite-difference Laplacian of an n x n x n node grid."""
    line = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n, n))
    identity = sp.identity(n)
    return (
        sp.kron(sp.kron(line, identity), identity)
        + sp.kron(sp.kron(identity, line), identity)
        + sp.kron(sp.kron(identity, identity), line)
    ).tocsr()


def displacementDamageSystem(n: int = 12, coupling: str = "none", rowScaling: bool = False):
    """A displacement / nonlocal damage system on an n^3 node grid, with a random right-hand side.

    Parameters
    ----------
    n
        Nodes per grid direction.
    coupling
        ``"none"`` (block diagonal), ``"one-way"`` (a weak random K_ud, K_du = 0 -- the structure of a
        gradient-enhanced damage model) or ``"symmetric"`` (K_du = K_ud^T).
    rowScaling
        Scale the displacement rows and columns over two orders of magnitude, so that the equilibration matters.

    Returns
    -------
    A (CSR), b, and the two field blocks.
    """
    laplacian = gridLaplacian(n)
    nodes = n**3
    K_uu = sp.kron(laplacian + 0.01 * sp.identity(nodes), sp.identity(3))
    if rowScaling:
        scaling = sp.diags(np.logspace(0, 2, 3 * nodes))
        K_uu = (scaling @ K_uu @ scaling).tocsr()
    K_dd = laplacian + sp.identity(nodes)
    if coupling == "none":
        A = sp.bmat([[K_uu, None], [None, K_dd]])
    else:
        K_ud = sp.random(3 * nodes, nodes, density=1e-3, random_state=1) * 1e-2
        K_du = K_ud.T if coupling == "symmetric" else None
        if coupling not in ("one-way", "symmetric"):
            raise ValueError("coupling must be 'none', 'one-way' or 'symmetric', got {!r}".format(coupling))
        A = sp.bmat([[K_uu, K_ud], [K_du, K_dd]])
    blocks = [FieldBlock("displacement", 0, 3 * nodes, 3), FieldBlock("nonlocal damage", 3 * nodes, 4 * nodes, 1)]
    b = np.random.default_rng(0).standard_normal(A.shape[0])
    return A.tocsr(), b, blocks
