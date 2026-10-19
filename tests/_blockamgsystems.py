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

import re

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from edelweissfe.linsolve.base import FieldBlock

#: A symmetric positive definite coupling of a node's three displacement components, for vector operators that
#: are not just three copies of a scalar one.
NODAL_COUPLING = np.array([[2.0, 0.3, 0.1], [0.3, 1.5, 0.2], [0.1, 0.2, 1.8]])


class DirichletStub:
    """What ``applyDirichletToStiffness`` reads from a boundary condition."""

    def __init__(self, dofs):
        self.constrainedDofIndices = np.asarray(dofs, dtype=np.int64)


def gridLaplacian(n: int, dimension: int = 3) -> sp.csr_matrix:
    """The finite-difference Laplacian of a node grid of n nodes per direction: 7-point in 3D, 5-point in 2D."""
    line = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n, n))
    identity = sp.identity(n)
    laplacian = sp.csr_matrix((n**dimension, n**dimension))
    for direction in range(dimension):
        term = line if direction == 0 else identity
        for other in range(1, dimension):
            term = sp.kron(term, line if other == direction else identity)
        laplacian = laplacian + term
    return laplacian.tocsr()


def variableCoefficientLaplacian(n: int, dimension: int = 3, decades: float = 2.0, seed: int = 0) -> sp.csr_matrix:
    """The Laplacian of a node grid of n nodes per direction whose edge stiffnesses are random over ``decades``
    orders of magnitude: sum over the edges e = (i, j) of k_e (e_i - e_j)(e_i - e_j)^T. Like a finite element
    stiffness of a heterogeneous material, its diagonal varies from node to node, while the constants stay its exact
    null-space (no boundary condition)."""
    line = sp.diags([-1.0, 1.0], [0, 1], shape=(n - 1, n))
    identity = sp.identity(n)
    rng = np.random.default_rng(seed)
    laplacian = sp.csr_matrix((n**dimension, n**dimension))
    for direction in range(dimension):
        edges = line if direction == 0 else identity
        for other in range(1, dimension):
            edges = sp.kron(edges, line if other == direction else identity)
        stiffnesses = sp.diags(10.0 ** rng.uniform(0.0, decades, edges.shape[0]))
        laplacian = laplacian + edges.T @ stiffnesses @ edges
    return laplacian.tocsr()


def displacementDamageSystem(n: int = 12, coupling: str = "none", rowScaling: bool = False):
    """A displacement / nonlocal damage system on an n^3 node grid, with a random right-hand side.

    Parameters
    ----------
    n
        Nodes per grid direction. AMGCL solves a block directly below 3000 scalar rows (1000 rows of 3x3
        blocks), so a test of the multi-level hierarchy needs n >= 15 for the damage field and n >= 10 for the
        displacement field.
    coupling
        ``"none"`` (block diagonal), ``"one-way"`` (a weak random K_ud and K_du = 0, as measured on a
        gradient-enhanced damage model, where K_du was about 1e-7 relative to K_dd) or ``"symmetric"``
        (K_du = K_ud^T). With ``"one-way"`` a forward block Gauss-Seidel sweep, displacement first, never
        uses the coupling: the damage field's residual has no displacement term. Tests of the sweep itself
        need ``"symmetric"``.
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


class ExactFieldSolve:
    """An exact solve with one field's diagonal block (a sparse LU factorization), in place of its AMG cycle: with
    these, a block Gauss-Seidel sweep is the exact block triangular solve that the AMG cycles approximate."""

    def __init__(self, block):
        self._factorization = splu(block.tocsc())

    def applyPreconditioner(self, residual):
        return self._factorization.solve(residual)


def hierarchyLevels(amg) -> int:
    """The number of levels of an AMGCL hierarchy, read from its report; one level is a direct solve."""
    return int(re.search(r"Number of levels:\s+(\d+)", amg.report()).group(1))
