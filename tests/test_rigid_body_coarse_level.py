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
"""The rigid-body coarse level (``TwoLevel(coarseSpace=RigidBodyModes())``, the default of the ``blockamg`` preset for
3D displacement fields, :mod:`edelweissfe.linsolve.composition.rigidbodycoarselevel`): the defining properties of its
construction -- the tentative prolongation reproduces the rigid-body modes, the coarse operator is the Galerkin product,
Dirichlet rows stay out of the coarse space, linearly dependent modes are dropped -- its control (with the translations
alone it is AMGCL's own first coarse level) and its purpose (fewer iterations on a bending-dominated structure). Its
place in the composed solvers is tested with them (test_linsolve_composition_*)."""

import os
import re
import subprocess
import sys

import numpy as np
import pytest
import scipy.sparse as sp
from _elasticbeam import slenderBeam
from _linsolvesystems import (
    GMRES_CAP,
    NODAL_COUPLING,
    equilibrated,
    gmresIterations,
    gridLaplacian,
    hierarchyLevels,
)

pytest.importorskip("edelweissfe.linsolve.amgcl.amgcl")

from edelweissfe.linsolve.amgcl.amgcl import PyAMGCLSolver  # noqa: E402
from edelweissfe.linsolve.base import FieldBlock  # noqa: E402
from edelweissfe.linsolve.composition.amg import defaultFieldTree  # noqa: E402
from edelweissfe.linsolve.composition.rigidbodycoarselevel import (  # noqa: E402
    RigidBodyCoarseLevelPreconditioner,
)
from edelweissfe.linsolve.nullspace import (  # noqa: E402
    rigidBodyNullspace,
    translationNullspace,
)


class _TranslationsOnly(RigidBodyCoarseLevelPreconditioner):
    """The control of the construction: the coarse level on the translations alone, M = B -- AMGCL's own coarse space
    on the block backend, so that it differs from the rigid-body coarse level in the rotations alone."""

    @staticmethod
    def nearNullspace(block, coordinates, scaling):
        return translationNullspace(block, scaling)


def _equilibratedBeam(dimension: int):
    """The equilibrated propped cantilever: ``\\hat A = D A D``, ``D``, the node coordinates and the constrained
    dofs."""
    K, coordinates, constrained = slenderBeam(dimension)
    As, scaling = equilibrated(K)
    return As, scaling, coordinates, constrained


def _preconditioner(
    dimension: int, backendPrecision: str = "double", kind=RigidBodyCoarseLevelPreconditioner, **options
) -> RigidBodyCoarseLevelPreconditioner:
    """The rigid-body coarse level (or its control, ``kind``) of the default tree of a field of this dimension, stored in
    backendPrecision."""
    return kind(defaultFieldTree(dimension), dimension, backendPrecision, **options)


def _relativeDifference(x, y) -> float:
    return np.linalg.norm(x - y) / np.linalg.norm(y)


def _rigidBodyModes(As, scaling, coordinates, dimension, aggregates):
    """The field's rigid-body modes about its centroid, divided by the equilibration, zero on the Dirichlet rows and on
    the nodes in no aggregate: what P_tent reproduces from the coarse near null-space."""
    modes = rigidBodyNullspace(FieldBlock("displacement", 0, As.shape[0], dimension), coordinates, scaling)
    modes[np.diff(As.indptr) == 1] = 0.0
    modes[np.repeat(aggregates < 0, dimension)] = 0.0
    return modes


def _assertGalerkinCoarseOperator(preconditioner, As) -> np.ndarray:
    """Assert that the coarse operator is the Galerkin product ``P^T A P``, and that the coarse dof of a dropped mode (a
    zero column of P) has the mean of the kept diagonal entries as its diagonal entry and nothing else; returns the
    dropped coarse dofs."""
    P, coarseOperator = preconditioner.prolongationMatrix(), preconditioner.coarseOperator().toarray()
    expected = (P.T @ As @ P).toarray()
    dropped = np.flatnonzero(abs(P).sum(axis=0).A1 == 0.0)
    kept = np.setdiff1d(np.arange(P.shape[1]), dropped)
    keptBlock = np.ix_(kept, kept)
    assert np.linalg.norm(coarseOperator[keptBlock] - expected[keptBlock]) <= 1e-12 * np.linalg.norm(expected)
    assert not coarseOperator[np.ix_(dropped, kept)].any() and not coarseOperator[np.ix_(kept, dropped)].any()
    padding = np.mean(np.diag(expected)[kept])
    assert np.allclose(coarseOperator[np.ix_(dropped, dropped)], padding * np.identity(len(dropped)), rtol=1e-12)
    return dropped


def _scaledGrid(n: int):
    """A grid of n^3 nodes with 3 coupled components each, rows scaled over two decades, equilibrated:
    ``\\hat A = D A D``, ``D`` and the node coordinates."""
    stiffness = sp.kron(gridLaplacian(n) + 0.01 * sp.identity(n**3), NODAL_COUPLING)
    rowScaling = sp.diags(10.0 ** np.random.default_rng(3).uniform(0.0, 2.0, 3 * n**3))
    As, scaling = equilibrated((rowScaling @ stiffness @ rowScaling).tocsr())
    axis = np.arange(n, dtype=float)
    # in the node order of gridLaplacian, the last direction fastest
    coordinates = np.array(np.meshgrid(axis, axis, axis, indexing="ij")).reshape(3, -1).T
    return As, scaling, coordinates


def _levelTable(report: str) -> list:
    """The (unknowns, nonzeros) of every level of the first AMGCL hierarchy in a report."""
    table = report[report.index("level     unknowns") :]
    return [(int(m.group(2)), int(m.group(3))) for m in re.finditer(r"^\s+(\d+)\s+(\d+)\s+(\d+)\s", table, re.M)]


def _collinearChain(direction, nNodes: int = 40):
    """A chain of nodes along ``direction``, every aggregate collinear: the rotation about its line is no displacement
    at all. ``\\hat A = D A D`` of a chain of nodal blocks, ``D`` and the node coordinates."""
    chain = sp.diags([-1.0, 2.01, -1.0], [-1, 0, 1], shape=(nNodes, nNodes))
    As, scaling = equilibrated(sp.kron(chain, NODAL_COUPLING).tocsr())
    unit = np.asarray(direction, dtype=float) / np.linalg.norm(direction)
    return As, scaling, np.outer(np.arange(nNodes, dtype=float), unit)


@pytest.mark.parametrize("dimension", [3, 2], ids=["3D, 6 modes", "2D, 3 modes"])
def test_the_tentative_prolongation_reproduces_the_rigid_body_modes(dimension):
    # P_tent B_c = B: the coarse near null-space are the coefficients of the field's rigid-body modes in the coarse
    # basis, so the hierarchy below inherits exactly the modes of the fine level. And the columns of P_tent are
    # orthonormal, the QR factors of every aggregate's modes.
    As, scaling, coordinates, _ = _equilibratedBeam(dimension)
    preconditioner = _preconditioner(dimension, prolongation="tentative")
    preconditioner.build(As, scaling, coordinates)
    modesPerAggregate = dimension * (dimension + 1) // 2
    assert preconditioner.nCoarse == modesPerAggregate * preconditioner.nAggregates > 0

    P, coarseNullspace = preconditioner.prolongationMatrix(), preconditioner.coarseNullspace()
    assert coarseNullspace.shape == (preconditioner.nCoarse, modesPerAggregate)
    modes = _rigidBodyModes(As, scaling, coordinates, dimension, preconditioner.aggregates)
    assert _relativeDifference(P @ coarseNullspace, modes) < 1e-12

    PtP = (P.T @ P).toarray()
    kept = np.diag(PtP) > 0.5
    assert np.abs(PtP - np.diag(kept.astype(float))).max() < 1e-12


@pytest.mark.parametrize("dimension", [3, 2], ids=["3D, 6 modes", "2D, 3 modes"])
@pytest.mark.parametrize("prolongation", ["smoothed", "tentative"])
def test_the_coarse_operator_is_the_galerkin_product(prolongation, dimension):
    As, scaling, coordinates, _ = _equilibratedBeam(dimension)
    preconditioner = _preconditioner(dimension, prolongation=prolongation)
    preconditioner.build(As, scaling, coordinates)
    _assertGalerkinCoarseOperator(preconditioner, As)
    if prolongation == "smoothed":
        # one damped Jacobi step widens every column of P_tent by a layer of nodes
        tentative = _preconditioner(dimension, prolongation="tentative")
        tentative.build(As, scaling, coordinates)
        assert preconditioner.prolongationMatrix().nnz > tentative.prolongationMatrix().nnz
        assert 0.0 < preconditioner.prolongationSmoothingWeight < 2.0
    else:
        assert preconditioner.prolongationSmoothingWeight == 0.0


def test_dirichlet_rows_stay_out_of_the_coarse_space_and_are_solved_exactly():
    # the beam is clamped (all components) and propped (one component): fully and partially constrained nodes
    As, scaling, coordinates, constrained = _equilibratedBeam(3)
    for backendPrecision in ("double", "float"):
        preconditioner = _preconditioner(3, backendPrecision)
        preconditioner.build(As, scaling, coordinates)
        P = preconditioner.prolongationMatrix()
        assert abs(P[constrained]).max() == 0.0
        r = np.random.default_rng(3).standard_normal(As.shape[0])
        x = preconditioner.applyPreconditioner(r)
        assert np.array_equal(x[constrained], r[constrained] / As.diagonal()[constrained])
    # a fully constrained node has no strong connection and is in no aggregate
    clampedNodes = np.flatnonzero(coordinates[:, 0] == 0.0)
    assert np.all(preconditioner.aggregates[clampedNodes] < 0)


@pytest.mark.parametrize("direction", [(1.0, 0.0, 0.0), (1.0, 2.0, 3.0)], ids=["along x", "oblique"])
def test_a_collinear_aggregate_drops_the_rotation_about_its_line(direction):
    # A chain of nodes: every aggregate is collinear, and the rotation about its line is no displacement at all. Along
    # an axis that rotation is one of the three modes, exactly zero; along an oblique line it is a combination of the
    # three, which the Gram-Schmidt sweeps reduce to rounding. Either is dropped, and the remaining modes still
    # reproduce all six rigid-body modes of the chain.
    As, scaling, coordinates = _collinearChain(direction)
    preconditioner = _preconditioner(3, prolongation="tentative")
    preconditioner.build(As, scaling, coordinates)

    aggregates = preconditioner.aggregates
    sizes = np.bincount(aggregates[aggregates >= 0])
    # one rotation of an aggregate of two or more nodes, all three of a single node
    assert preconditioner.droppedModes == np.sum(np.where(sizes > 1, 1, 3)) > 0
    P, coarseNullspace = preconditioner.prolongationMatrix(), preconditioner.coarseNullspace()
    modes = _rigidBodyModes(As, scaling, coordinates, 3, aggregates)
    assert _relativeDifference(P @ coarseNullspace, modes) < 1e-12
    # the dropped modes are decoupled coarse dofs: their diagonal entry is the padding, their right-hand side zero
    r = np.random.default_rng(4).standard_normal(As.shape[0])
    assert np.all(np.isfinite(preconditioner.applyPreconditioner(r)))


@pytest.mark.parametrize("prolongation", ["smoothed", "tentative"])
def test_the_coarse_dof_of_a_dropped_mode_is_padded(prolongation):
    # The Galerkin product of a chain, whose aggregates all drop a rotation: a dropped mode has a zero column of P, and
    # its coarse dof is padded with the mean of the kept diagonal entries, which keeps A_c regular for the hierarchy
    # below. The cycle is unaffected: P^T maps every residual to zero there, and P never prolongates it.
    As, scaling, coordinates = _collinearChain((1.0, 2.0, 3.0))
    preconditioner = _preconditioner(3, prolongation=prolongation)
    preconditioner.build(As, scaling, coordinates)
    dropped = _assertGalerkinCoarseOperator(preconditioner, As)
    assert len(dropped) == preconditioner.droppedModes > 0


def test_modes_whose_components_are_all_constrained_are_dropped():
    # A plane of nodes whose out-of-plane displacement is constrained, as on a symmetry plane: the translation normal
    # to the plane and the rotations about the two in-plane axes move constrained components only, so they are
    # dropped on every aggregate (and the in-plane rotation, too, on an aggregate of a single node). The in-plane modes
    # remain, reproduced exactly, and the coarse operator is the Galerkin product with the dropped modes padded.
    n = 16
    line = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n, n))
    plane = sp.kron(line, sp.identity(n)) + sp.kron(sp.identity(n), line) + 0.01 * sp.identity(n * n)
    K = sp.kron(plane, NODAL_COUPLING).tolil()
    outOfPlane = np.arange(2, 3 * n * n, 3)
    K[outOfPlane, :] = 0.0
    K[:, outOfPlane] = 0.0
    K[outOfPlane, outOfPlane] = 1.0
    K = K.tocsr()
    K.eliminate_zeros()
    As, scaling = equilibrated(K)
    axis = np.arange(n, dtype=float)
    # in the node order of the plane, the second direction fastest
    coordinates = np.column_stack([np.repeat(axis, n), np.tile(axis, n), np.zeros(n * n)])
    preconditioner = _preconditioner(3, prolongation="tentative")
    preconditioner.build(As, scaling, coordinates)

    aggregates = preconditioner.aggregates
    sizes = np.bincount(aggregates[aggregates >= 0])
    assert preconditioner.droppedModes == np.sum(np.where(sizes > 1, 3, 4)) > 0
    P, coarseNullspace = preconditioner.prolongationMatrix(), preconditioner.coarseNullspace()
    modes = _rigidBodyModes(As, scaling, coordinates, 3, aggregates)
    assert _relativeDifference(P @ coarseNullspace, modes) < 1e-12
    assert len(_assertGalerkinCoarseOperator(preconditioner, As)) == preconditioner.droppedModes


def test_the_translations_alone_are_amgcls_own_first_coarse_level():
    # The control of the construction: on the translations alone, the coarse space of every aggregate is spanned by its
    # translations D^-1 t, as in AMGCL's hierarchy on the block backend with blockTranslationNullspace -- the same
    # aggregates, the same tentative prolongation (per component, normalized), the same smoothing and the same
    # Galerkin product. With both coarse levels solved directly, the two preconditioners then differ only by the
    # spectral-radius estimate of the fine Chebyshev smoother (AMGCL's from per-thread random vectors, the two-level
    # cycle's from a deterministic one), which enough power iterations make negligible. Without Dirichlet rows:
    # AMGCL's translations are not zero there, those of the rigid-body coarse level are.
    As, scaling, coordinates = _scaledGrid(12)
    tree = defaultFieldTree(3)
    tree["relax"]["power_iters"] = 3000

    amgcl = PyAMGCLSolver(
        {
            "precond": tree,
            "backendBlockSize": 3,
            "backendPrecision": "double",
        }
    )
    amgcl.set_nullspace(translationNullspace(FieldBlock("displacement", 0, As.shape[0], 3), scaling))
    amgcl.build(As)
    assert hierarchyLevels(amgcl) == 2

    translations = _TranslationsOnly(tree, 3, "double")
    translations.build(As, scaling, coordinates)
    rigidBody = RigidBodyCoarseLevelPreconditioner(tree, 3, "double")
    rigidBody.build(As, scaling, coordinates)
    assert translations.coarseNullspace().shape[1] == 3 and translations.droppedModes == 0

    r = np.random.default_rng(5).standard_normal(As.shape[0])
    expected = amgcl.applyPreconditioner(r)
    print(
        "relative difference to AMGCL's first coarse level: translations alone {:.1e}, rigid-body modes {:.1e}".format(
            _relativeDifference(translations.applyPreconditioner(r), expected),
            _relativeDifference(rigidBody.applyPreconditioner(r), expected),
        )
    )
    assert _relativeDifference(translations.applyPreconditioner(r), expected) < 1e-6
    # and the test can tell: the rotations are another coarse space
    assert _relativeDifference(rigidBody.applyPreconditioner(r), expected) > 1e-3


@pytest.mark.parametrize(
    "kind, modes",
    [(RigidBodyCoarseLevelPreconditioner, 6), (_TranslationsOnly, 3)],
    ids=["rigid-body modes", "translations alone"],
)
def test_the_hierarchy_below_takes_the_modes_of_an_aggregate_as_one_point_on_every_level(kind, modes):
    # Below the rigid-body coarse level, AMGCL coarsens A_c further. It gives every aggregate of a level one coarse dof
    # per column of the near null-space and keeps aggr.block_size from level to level, so the M dofs of an aggregate are
    # one point on every level below if aggr.block_size is M, the number of modes: 6 with the rotations, 3 without.
    # Checked on three levels below against AMGCL's own hierarchy of the same coarse operator.
    As, scaling, coordinates = _scaledGrid(12)
    tree = defaultFieldTree(3) | {"coarse_enough": 20}
    preconditioner = kind(tree, 3, "double")
    preconditioner.build(As, scaling, coordinates)
    assert preconditioner.nCoarse == modes * preconditioner.nAggregates

    coarsening = tree["coarsening"] | {"aggr": {"eps_strong": 0.5 * tree["coarsening"]["aggr"]["eps_strong"]}}
    coarsening["aggr"]["block_size"] = modes
    reference = PyAMGCLSolver(
        {
            "precond": tree | {"coarsening": coarsening},
            "backendBlockSize": 1,
            "backendPrecision": "double",
        }
    )
    assert preconditioner.coarseNullspace().shape[1] == modes
    reference.set_nullspace(preconditioner.coarseNullspace())
    reference.build(preconditioner.coarseOperator())
    levels = _levelTable(reference.report())
    assert len(levels) >= 3
    assert _levelTable(preconditioner.report().split("AMGCL hierarchy below:")[1]) == levels
    r = np.random.default_rng(13).standard_normal(As.shape[0])
    assert np.all(np.isfinite(preconditioner.applyPreconditioner(r)))


#: Builds the rigid-body coarse level of the propped cantilever in a fresh process, whose OpenMP thread count is set
#: by OMP_NUM_THREADS at start-up: arguments the test directory, the output file and whether the prolongation smoothing
#: estimates its spectral radius.
_THREADS_SCRIPT = """
import sys
import numpy as np
sys.path.insert(0, sys.argv[1])
from _elasticbeam import slenderBeam
from _linsolvesystems import equilibrated, gmresIterations
from edelweissfe.linsolve.composition.amg import defaultFieldTree
from edelweissfe.linsolve.composition.rigidbodycoarselevel import RigidBodyCoarseLevelPreconditioner

K, coordinates, _ = slenderBeam(3)
As, scaling = equilibrated(K)
tree = defaultFieldTree(3)
tree["coarsening"]["estimate_spectral_radius"] = sys.argv[3] == "estimate"
preconditioner = RigidBodyCoarseLevelPreconditioner(tree, 3, "double")
preconditioner.build(As, scaling, coordinates)
iterations = gmresIterations(As, preconditioner)
Ac, P = preconditioner.coarseOperator(), preconditioner.prolongationMatrix()
np.savez(sys.argv[2], aggregates=preconditioner.aggregates, P=P.toarray(), Ac=Ac.toarray(), iterations=iterations)
"""


def _builtWithThreads(tmp_path, threads: int, estimate: bool) -> dict:
    """The aggregates, P, A_c and the GMRES iterations of the propped cantilever, built with this many threads."""
    output = tmp_path / "threads{:}_{:}.npz".format(threads, estimate)
    environment = os.environ | {"OMP_NUM_THREADS": str(threads)}
    testDirectory = os.path.dirname(os.path.abspath(__file__))
    arguments = [testDirectory, str(output), "estimate" if estimate else "fixed"]
    # run in tmp_path: "python -c" puts its working directory first on sys.path, and in the repository root that is
    # the source tree without the compiled extensions, which would shadow an installed package (as in CI)
    subprocess.run([sys.executable, "-c", _THREADS_SCRIPT, *arguments], env=environment, cwd=tmp_path, check=True)
    return dict(np.load(output))


def test_the_coarse_level_does_not_depend_on_the_thread_count(tmp_path):
    # Every stage of the build is threaded, and none may depend on how the rows are shared among the threads: with a
    # fixed damping weight of the prolongation smoothing, one thread and four build the same coarse level to rounding.
    # With the weight estimated (the default), AMGCL's power iteration starts from a random vector per thread, so the
    # weight, and with it P and A_c, differ slightly; the iterations hardly.
    serial, threaded = _builtWithThreads(tmp_path, 1, False), _builtWithThreads(tmp_path, 4, False)
    assert np.array_equal(serial["aggregates"], threaded["aggregates"])
    assert _relativeDifference(threaded["P"], serial["P"]) < 1e-14
    assert _relativeDifference(threaded["Ac"], serial["Ac"]) < 1e-13
    assert serial["iterations"] == threaded["iterations"]
    serial, threaded = _builtWithThreads(tmp_path, 1, True), _builtWithThreads(tmp_path, 4, True)
    assert np.array_equal(serial["aggregates"], threaded["aggregates"])
    assert abs(int(threaded["iterations"]) - int(serial["iterations"])) <= 1


@pytest.mark.parametrize("dimension", [3, 2], ids=["3D hexahedra", "2D quadrilaterals"])
def test_the_rotations_cut_the_iterations_of_a_bending_dominated_beam(dimension):
    # What the rigid-body coarse level is for: the smooth error of a slender beam is bending, which rotates every
    # aggregate. The translations alone cannot represent it, the rotations can.
    As, scaling, coordinates, _ = _equilibratedBeam(dimension)
    iterations = {}
    for kind in (_TranslationsOnly, RigidBodyCoarseLevelPreconditioner):
        preconditioner = _preconditioner(dimension, kind=kind)
        preconditioner.build(As, scaling, coordinates)
        iterations[kind] = gmresIterations(As, preconditioner)
    translations, rigidBody = iterations[_TranslationsOnly], iterations[RigidBodyCoarseLevelPreconditioner]
    print("GMRES iterations, translations alone / rigid-body modes:", translations, rigidBody)
    assert rigidBody <= GMRES_CAP and rigidBody < 0.5 * translations


def test_float_storage_rounds_only_the_stored_entries():
    As, scaling, coordinates, _ = _equilibratedBeam(3)
    double, single = _preconditioner(3, "double"), _preconditioner(3, "float")
    double.build(As, scaling, coordinates)
    single.build(As, scaling, coordinates)
    # the spectral radius comes from the double operator in both, and the setup is in double
    assert single.fineSpectralRadius == double.fineSpectralRadius
    r = np.random.default_rng(6).standard_normal(As.shape[0])
    assert _relativeDifference(single.applyPreconditioner(r), double.applyPreconditioner(r)) < 1e-5
    assert _relativeDifference(single.prolongationMatrix().data, double.prolongationMatrix().data) < 1e-7


def test_a_supplied_spectral_radius_is_rechecked_and_kept_on_the_same_operator():
    As, scaling, coordinates, _ = _equilibratedBeam(3)
    first = _preconditioner(3)
    first.build(As, scaling, coordinates)
    assert first.fineSpectralRadiusSource == "estimated"
    again = _preconditioner(3, fineSpectralRadius=first.fineSpectralRadius, fineDominantVector=first.fineDominantVector)
    again.build(As, scaling, coordinates)
    assert again.fineSpectralRadiusSource == "supplied" and again.fineSpectralRadius == first.fineSpectralRadius
    # the same smoother; the coarse level is rebuilt, and AMGCL's power iterations in it (the prolongation smoothing
    # weight, the coarse smoother's spectral radius) sum their per-thread parts in arrival order, which differs from
    # run to run in the last bits
    r = np.random.default_rng(8).standard_normal(As.shape[0])
    assert _relativeDifference(again.applyPreconditioner(r), first.applyPreconditioner(r)) < 1e-9


def test_without_a_strong_connection_there_is_no_coarse_level():
    # nodes that are not coupled at all: no aggregate, and the cycle is the smoother alone (as AMGCL ends its own
    # hierarchy at an empty level)
    nNodes = 50
    As, scaling = equilibrated(sp.kron(sp.identity(nNodes), NODAL_COUPLING).tocsr())
    preconditioner = _preconditioner(3)
    preconditioner.build(As, scaling, np.random.default_rng(9).uniform(size=(nNodes, 3)))
    assert preconditioner.nAggregates == 0 and preconditioner.nCoarse == 0
    assert "the smoother alone" in preconditioner.report()
    r = np.random.default_rng(10).standard_normal(As.shape[0])
    # one Chebyshev sweep of degree 16 on the nodal blocks reduces the residual
    assert _relativeDifference(As @ preconditioner.applyPreconditioner(r), r) < 0.5


def test_a_fully_constrained_field_is_the_identity():
    nNodes = 20
    preconditioner = _preconditioner(3)
    preconditioner.build(sp.identity(3 * nNodes, format="csr"), np.ones(3 * nNodes), np.zeros((nNodes, 3)))
    r = np.random.default_rng(11).standard_normal(3 * nNodes)
    assert np.array_equal(preconditioner.applyPreconditioner(r), r)
    assert "identity" in preconditioner.report()


def test_coordinates_of_other_nodes_and_a_near_null_space_it_cannot_take_are_refused():
    As, scaling, coordinates, _ = _equilibratedBeam(3)
    preconditioner = _preconditioner(3)
    with pytest.raises(ValueError, match="node coordinates .* have {:} rows".format(len(coordinates) - 1)):
        preconditioner.build(As, scaling, coordinates[:-1])
    with pytest.raises(RuntimeError, match="no preconditioner built"):
        preconditioner.applyPreconditioner(np.ones(As.shape[0]))
    # the C++ class takes the translations or all rigid-body modes, as many rows as the field has dofs
    native = preconditioner._preconditioner
    with pytest.raises(RuntimeError, match="near null-space of 3 columns .* or 6 .*, got 4"):
        native.build(As, np.ones((As.shape[0], 4)))
    with pytest.raises(RuntimeError, match="near null-space has {:} rows".format(As.shape[0] - 3)):
        native.build(As, np.ones((As.shape[0] - 3, 6)))


@pytest.mark.parametrize(
    "tree, blockSize, backendPrecision, options, reason",
    [
        ({"relax": {"type": "spai0"}}, 3, "double", {}, "relax.type must be chebyshev"),
        ({"relax": {"type": "chebyshev", "scale": True, "power_iters": 10}}, 3, "double", {}, "relax.scale"),
        ({"relax": {"type": "chebyshev"}}, 3, "double", {}, "relax.power_iters must be at least 1"),
        ({"coarsening": {"type": "ruge_stuben"}}, 3, "double", {}, "coarsening.type must be smoothed_aggregation"),
        ({}, 1, "double", {}, "backendBlockSize 2 or 3"),
        ({}, 2, "float", {}, "2x2 blocks are stored in double only"),
        ({}, 3, "double", {"prolongation": "galerkin"}, "prolongation must be smoothed or tentative"),
        ({}, 3, "double", {"coarseHierarchyPrecision": "half"}, "coarseHierarchyPrecision must be double or float"),
    ],
    ids=["relax", "scale", "power_iters", "coarsening", "scalar", "2D float", "prolongation", "precision"],
)
def test_settings_it_cannot_use_raise_before_any_build(tree, blockSize, backendPrecision, options, reason):
    fullTree = {key: value for key, value in defaultFieldTree(3).items() if key not in tree} | tree
    with pytest.raises(ValueError, match=reason):
        RigidBodyCoarseLevelPreconditioner(fullTree, blockSize, backendPrecision, **options)


def test_a_tree_without_a_smoother_is_told_that_amgcls_default_is_not_chebyshev():
    tree = {key: value for key, value in defaultFieldTree(3).items() if key != "relax"}
    with pytest.raises(ValueError, match=r"relax.type must be chebyshev, got spai0 \(AMGCL's default\)"):
        RigidBodyCoarseLevelPreconditioner(tree, 3, "double")
