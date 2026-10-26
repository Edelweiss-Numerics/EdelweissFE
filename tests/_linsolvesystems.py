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
"""What the linear-solver tests share: small linear systems -- with the field structure of a gradient-enhanced damage
model (a 3D displacement field of 3 DOFs per node and a scalar nonlocal damage field, each a grid Laplacian), and a bar
on an elastic foundation for the Schur complement -- the diagonal scaling, the GMRES iterations of a preconditioner,
and the test double of the Schur complement condensation."""

import re

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, gmres, splu

from edelweissfe.linsolve.base import FieldBlock
from edelweissfe.linsolve.composition.base import SetUpContext
from edelweissfe.linsolve.composition.direct import DirectSolver
from edelweissfe.linsolve.composition.schurcomplement import SchurComplementCondensation
from edelweissfe.linsolve.composition.systemmatrix import SystemMatrix

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


def equilibrated(A):
    """:math:`\\hat A = D A D` with :math:`D = \\mathrm{diag}(|a_{ii}|^{-1/2})`, as the diagonal scaling forms it (CSR,
    sorted columns), and the diagonal of :math:`D`."""
    scaling = 1.0 / np.sqrt(np.abs(A.diagonal()))
    As = (sp.diags(scaling) @ A @ sp.diags(scaling)).tocsr()
    As.sort_indices()
    return As, scaling


#: The iteration budget of :func:`gmresIterations`.
GMRES_CAP = 300


def gmresIterations(A, preconditioner) -> int:
    """The iterations of (left-preconditioned, unrestarted) GMRES to a preconditioned relative residual of 1e-8, or
    ``GMRES_CAP + 1`` if it does not reach a true relative residual of 1e-5 within ``GMRES_CAP``."""
    b = np.random.default_rng(7).standard_normal(A.shape[0])
    M = LinearOperator(A.shape, matvec=preconditioner.applyPreconditioner)
    history = []
    x, _ = gmres(A, b, M=M, rtol=1e-8, restart=GMRES_CAP, maxiter=1, callback=history.append, callback_type="pr_norm")
    converged = np.linalg.norm(b - A @ x) <= 1e-5 * np.linalg.norm(b)
    return len(history) if converged and len(history) <= GMRES_CAP else GMRES_CAP + 1


def barOnFoundation(n=14, barLength=None, springStiffness=1e3, barStiffness=1e4):
    """The host grid (n^3 dofs) followed by a bar of ``barLength`` dofs (default n^2). Like a bond, the spring of bar
    dof i ties it to the mean of the 8 corners of a host cell, N u_host: the energy k/2 (u_bar - N u_host)^2 gives the
    blocks k N^T N, -k N^T, k I. Returns A, a right-hand side b and the bar dofs."""
    host = (gridLaplacian(n) + 1e-3 * sp.identity(n**3)).tocsr()
    nBar = barLength or n * n
    bar = barStiffness * sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(nBar, nBar))
    # cell corners (x, y, z) .. (x + 1, y + 1, z + 1), the cells along a line through the grid
    cells = np.arange(nBar) % (n - 1)
    corners = [
        cells + dx + n * (cells + dy) + n * n * ((cells * 7) % (n - 1) + dz)
        for dx in (0, 1)
        for dy in (0, 1)
        for dz in (0, 1)
    ]
    N = sp.csr_matrix(
        (np.full(8 * nBar, 1.0 / 8.0), (np.tile(np.arange(nBar), 8), np.concatenate(corners))), shape=(nBar, n**3)
    )
    k = springStiffness
    A = sp.bmat([[host + k * (N.T @ N), -k * N.T], [-k * N, bar + k * sp.identity(nBar)]]).tocsr()
    A.sort_indices()
    barDofs = np.arange(n**3, n**3 + nBar)
    b = np.random.default_rng(1).standard_normal(A.shape[0])
    return A, b, barDofs


class FixedCondensation(SchurComplementCondensation):
    """A Schur complement condensation of given dofs, without a model and markers: the condensed dofs are set directly,
    and may be changed between updates (as the selection of a marker changes between increments). One dof per node:
    the windows along a chain count dofs.

    Parameters
    ----------
    dofs
        The condensed dofs.
    options
        See :class:`~edelweissfe.linsolve.composition.schurcomplement.SchurComplementCondensation`.
    """

    def __init__(self, dofs, **options):
        super().__init__([], **options)
        self.dofs = np.asarray(dofs)

    def _selectCondensedDofs(self):
        return self.dofs, self.dofs


def solveA_bbOf(condensation) -> DirectSolver:
    """:math:`A_{bb}^{-1}` of a condensation's last update: a SuperLU leaf set up on its :math:`A_{bb}`."""
    solveA_bb = DirectSolver()
    n = condensation.A_bb.shape[0]
    solveA_bb.setUp(SystemMatrix(condensation.A_bb), SetUpContext(fields=(FieldBlock("condensed", 0, n, 1),)))
    return solveA_bb


def condensingGivenDofs(schur, dofs, **options):
    """``schur`` (a :class:`~edelweissfe.linsolve.composition.schurcomplement.SchurComplement`), condensing the given
    dofs instead of those its markers select: a test without a model."""
    schur.condensation = FixedCondensation(dofs, **options)
    return schur
