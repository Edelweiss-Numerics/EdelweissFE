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
"""A small Hex20-like lattice and its Dirichlet-eliminated, equilibrated 3D vector operator, shared by the
p-two-grid tests: corner nodes on a cubic grid, one exclusive midside node on every grid edge, in a shuffled
node order, a vector-Laplacian-like 3x3-coupled operator with a mild non-symmetric part, fully clamped nodes on
one face and a single constrained component on another (partially constrained nodes)."""

import numpy as np
import scipy.sparse as sp
from _linsolvesystems import NODAL_COUPLING, equilibrated
from scipy.spatial import cKDTree


def lattice(nCornersPerAxis: int, seed: int = 0):
    """Corner and midside nodes of a cubic lattice, in a shuffled node order."""
    axis = np.arange(nCornersPerAxis, dtype=float)
    corners = np.array(np.meshgrid(axis, axis, axis, indexing="ij")).reshape(3, -1).T
    cornerIndex = {tuple(c): i for i, c in enumerate(corners)}
    midsides, endpoints = [], []
    for i, corner in enumerate(corners):
        for direction in range(3):
            neighbour = corner.copy()
            neighbour[direction] += 1.0
            j = cornerIndex.get(tuple(neighbour))
            if j is not None:
                midsides.append(0.5 * (corner + neighbour))
                endpoints.append((i, j))
    coords = np.vstack([corners, midsides])
    isCorner = np.r_[np.ones(len(corners), dtype=bool), np.zeros(len(midsides), dtype=bool)]
    edgeEndpoints = np.vstack([-np.ones((len(corners), 2), dtype=int), np.array(endpoints)])

    order = np.random.default_rng(seed).permutation(len(coords))
    newIndexOfOld = np.empty_like(order)
    newIndexOfOld[order] = np.arange(len(order))
    edgeEndpoints = edgeEndpoints[order]
    edgeEndpoints[~isCorner[order]] = newIndexOfOld[edgeEndpoints[~isCorner[order]]]
    return coords[order], isCorner[order], edgeEndpoints


def latticeSystem(coords, dirichletDofs, seed: int = 1, components: int = 3):
    """The equilibrated, Dirichlet-eliminated block of ``components`` dofs per node and its equilibration vector."""
    nNodes = len(coords)
    pairs = cKDTree(coords).query_pairs(1.0 + 1e-9, output_type="ndarray")
    W = sp.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(nNodes, nNodes))
    W = (W + W.T).tocsr()
    laplacian = sp.diags(np.asarray(W.sum(axis=1)).ravel() + 0.05) - W
    skew = np.random.default_rng(seed).uniform(-1.0, 1.0, (3, 3))[:components, :components]
    A = sp.kron(laplacian, NODAL_COUPLING[:components, :components]) + 0.02 * sp.kron(
        sp.triu(W, 1) - sp.tril(W, -1), skew
    )
    A = A.tolil()
    A[dirichletDofs, :] = 0.0
    A[:, dirichletDofs] = 0.0
    A[dirichletDofs, dirichletDofs] = 1.0
    A = A.tocsr()
    A.eliminate_zeros()
    return equilibrated(A)


def dirichletDofs(coords):
    """All components on the x = 0 face, only the z component on the rest of the y = 0 face."""
    clamped = np.nonzero(coords[:, 0] == 0.0)[0]
    sliding = np.nonzero((coords[:, 1] == 0.0) & (coords[:, 0] > 0.0))[0]
    return np.sort(np.r_[(3 * clamped[:, None] + np.arange(3)).ravel(), 3 * sliding + 2])


def fineChebyshevTree(npre: int = 1, npost: int = 1, degree: int = 5, powerIters: int = 300) -> dict:
    """The AMGCL tree of a p-two-grid's fine smoother: ``npre`` and ``npost`` Chebyshev sweeps of ``degree`` on
    [0.01, 1.1] times the spectral radius, estimated by ``powerIters`` power iterations. The defaults are the smoother
    the p-two-grid's recorded numbers were made with (BlockAMGSolver's p-two-grid defaults)."""
    return {
        "relax": {"type": "chebyshev", "degree": degree, "lower": 0.01, "higher": 1.1, "power_iters": powerIters},
        "npre": npre,
        "npost": npost,
    }
