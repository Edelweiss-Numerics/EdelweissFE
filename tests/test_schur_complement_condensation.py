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
"""The Schur complement condensation of :mod:`edelweissfe.linsolve.composition.schurcomplement` (the composed
:class:`~edelweissfe.linsolve.composition.schurcomplement.SchurComplement` is tested with the other components).

The test system is a bar on an elastic foundation: a 3D grid Laplacian (the host) and a chain of bar dofs, each tied
to a host dof by a stiff spring, the bar much stiffer along its axis than the host. The bar dofs are condensed.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp
from _linsolvesystems import (
    FixedCondensation,
    barOnFoundation,
    condensingGivenDofs,
    gridLaplacian,
    solveA_bbOf,
)

import edelweissfe.adaptivity.marking as marking
import edelweissfe.linsolve.composition.schurcomplement as schurcomplement
from edelweissfe.adaptivity.marking import (
    STABLE_FOR_THE_MODEL,
    ElementSetMarker,
    FieldOutputMarker,
    MarkerBase,
)
from edelweissfe.linsolve.base import FieldBlock
from edelweissfe.linsolve.composition import (
    DirectSolver,
    SchurComplement,
    SetUpContext,
    SystemMatrix,
)
from edelweissfe.linsolve.composition.amg import isSignificant
from edelweissfe.linsolve.composition.schurcomplement import (
    SchurComplementCondensation,
    _cutIntoWindows,
    _dropSmallEntries,
    _solveTransposed,
    _split,
    _splitCacheFor,
    blockLU,
)
from edelweissfe.linsolve.superlu import SuperLUBackend
from edelweissfe.numerics.p1topology import restrictP1Map
from edelweissfe.utils.fieldoutput import _FieldOutputBase


def _exactSchurComplement(A, barDofs):
    others = np.setdiff1d(np.arange(A.shape[0]), barDofs)
    A_kk, A_kb = A[others][:, others].toarray(), A[others][:, barDofs].toarray()
    A_bk, A_bb = A[barDofs][:, others].toarray(), A[barDofs][:, barDofs].toarray()
    return A_kk - A_kb @ np.linalg.solve(A_bb, A_bk)


@pytest.mark.parametrize("windowNodes", [10_000, 40])
def test_approximate_schur_complement_is_close_to_the_exact_one(windowNodes):
    """Without dropping, one window over the whole bar (an exact correction) gives the exact S, and short windows
    with halos give it up to the decay of A_bb^-1 along the bar."""
    A, _, barDofs = barOnFoundation(n=8)
    condensation = FixedCondensation(barDofs, dropTolerance=0.0, windowNodes=windowNodes, haloNodes=40)
    condensation.update(A)
    S = condensation.approximateSchurComplement(A).toarray()
    exact = _exactSchurComplement(A, barDofs)
    error = np.abs(S - exact).max() / np.abs(exact).max()
    assert error < (1e-12 if windowNodes > A.shape[0] else 1e-6)


def test_a_zone_that_is_no_chain_gets_its_exact_correction():
    """A cube of grid dofs inside a larger grid (a softening zone, say) is not banded: corrected exactly, with the
    factorization of A_bb, in several batches of right-hand sides."""
    n = 16
    A = (gridLaplacian(n) + 1e-2 * sp.identity(n**3)).tocsr()
    A.sort_indices()
    i, j, k = np.meshgrid(*(np.arange(4, 12),) * 3, indexing="ij")
    zone = np.sort((i + n * j + n * n * k).ravel())
    condensation = FixedCondensation(zone, dropTolerance=0.0, windowNodes=8, haloNodes=8)
    condensation.update(A)
    windows, zoneDofs = condensation._windowsAndZones()
    assert windows == [] and zoneDofs.size == zone.size
    S = condensation.approximateSchurComplement(A).toarray()
    exact = _exactSchurComplement(A, zone)
    assert np.abs(S - exact).max() / np.abs(exact).max() < 1e-12


def test_the_approximate_schur_complement_of_a_spd_system_has_a_positive_diagonal_and_is_nearly_symmetric():
    """What AMG on S~ relies on. For a symmetric positive definite A, the exact S is symmetric positive definite too;
    S~ keeps a positive diagonal. Restricted additive Schwarz keeps only the core rows of every window's inverse, so S~
    is symmetric to rounding only where a single window covers a whole bar (an exact correction); with windows along a
    bar, the symmetry defect is the decay of A_bb^-1 over a halo: small with halos of 40 nodes, larger with 10."""

    def symmetryDefect(S):
        return sp.linalg.norm(S - S.T) / sp.linalg.norm(S)

    A, _, barDofs = barOnFoundation(n=12)
    defects = {}
    for windowNodes, haloNodes in ((10_000, 0), (40, 40), (40, 10)):
        condensation = FixedCondensation(barDofs, dropTolerance=0.0, windowNodes=windowNodes, haloNodes=haloNodes)
        condensation.update(A)
        S = condensation.approximateSchurComplement(A)
        assert condensation.nonPositiveDiagonalEntries == 0 and S.diagonal().min() > 0.0
        defects[haloNodes] = symmetryDefect(S)
    assert defects[0] < 1e-14 and defects[40] < 1e-6 < defects[10]


def test_a_chain_is_cut_into_windows_whose_cores_partition_it():
    chain = np.arange(100, 110)
    windows = _cutIntoWindows(chain, core=4, halo=2)
    assert [list(w.dofs) for w in windows] == [list(range(100, 106)), list(range(102, 110)), list(range(106, 110))]
    assert np.array_equal(np.concatenate([w.dofs[w.core] for w in windows]), chain)
    # a chain no longer than one core with both halos is one window, an exact correction
    (window,) = _cutIntoWindows(chain, core=4, halo=3)
    assert np.array_equal(window.dofs, chain) and np.array_equal(window.core, np.arange(10))


def test_the_blocks_of_the_split_are_those_of_fancy_indexing(monkeypatch):
    """With blocks of a few entries for the parallel gathers, and int64 row pointers in ``A``."""
    monkeypatch.setattr(schurcomplement, "PARALLEL_BLOCK_ENTRIES", 17)
    A = sp.random(60, 60, density=0.2, format="csr", random_state=0) + sp.identity(60, format="csr")
    A.indptr, A.indices = A.indptr.astype(np.int64), A.indices.astype(np.int64)
    b = np.sort(np.random.default_rng(0).choice(60, 20, replace=False))
    k = np.setdiff1d(np.arange(60), b)
    blocks = _split(A, ("bb", "bk", "kb", "kk"), b, k, _splitCacheFor(A, None))
    for block, (rows, cols) in zip(blocks, ((b, b), (b, k), (k, b), (k, k))):
        expected = A[rows][:, cols].tocsr()
        expected.sort_indices()
        for name in ("data", "indices", "indptr"):
            assert np.array_equal(getattr(block, name), getattr(expected, name))


def test_the_drop_of_small_entries_is_the_simple_form(monkeypatch):
    """With row blocks of a few entries for the two passes, and stored zeros."""
    monkeypatch.setattr(schurcomplement, "PARALLEL_BLOCK_ENTRIES", 17)
    S = sp.random(60, 60, density=0.2, format="csr", random_state=1) + sp.identity(60, format="csr")
    S.data[::7] = 0.0
    sqrtAbsDiagonal = np.sqrt(np.abs(S.diagonal()))
    rowOfEntry = np.repeat(np.arange(60), np.diff(S.indptr))
    expected = S.copy()
    expected.data[~isSignificant(S.data, rowOfEntry, S.indices, sqrtAbsDiagonal, 0.3)] = 0.0
    expected.eliminate_zeros()
    dropped = _dropSmallEntries(S, 0.3)
    assert 0 < dropped.nnz < np.count_nonzero(S.data)
    for name in ("data", "indices", "indptr"):
        assert np.array_equal(getattr(dropped, name), getattr(expected, name))


def test_a_condensed_block_without_coupling_leaves_a_kk():
    """No window couples to the remaining dofs: no corrections, S~ is A_kk."""
    A = sp.block_diag([barOnFoundation(n=6)[0], 2.0 * sp.identity(5)], format="csr")
    b = np.arange(A.shape[0] - 5, A.shape[0])
    condensation = FixedCondensation(b, dropTolerance=0.0)
    condensation.update(A)
    S = condensation.approximateSchurComplement(A)
    assert abs(S - A[: b[0]][:, : b[0]]).max() == 0.0


class _Node:
    def __init__(self, label):
        self.label = label
        self.fields = {"displacement": label}


class _Element:
    def __init__(self, nodes):
        self.nodes = nodes


def test_condensed_dofs_are_those_of_the_interior_nodes_of_the_marked_elements():
    """A bar element (nodes 1, 2) and a bond element (nodes 2, 3) on a continuum element (nodes 3, 4): marking the
    bar and the bond condenses node 1 and 2; node 3 also belongs to the unmarked continuum element."""
    nodes = {label: _Node(label) for label in range(1, 5)}
    bar = _Element([nodes[1], nodes[2]])
    bond = _Element([nodes[2], nodes[3]])
    continuum = _Element([nodes[3], nodes[4]])
    model = SimpleNamespace(
        elements={1: bar, 2: bond, 3: continuum}, elementSets={"bars": [bar], "bonds": [bond], "all": []}
    )
    dofs = {label: np.array([2 * (label - 1), 2 * (label - 1) + 1]) for label in nodes}
    dofManager = SimpleNamespace(idcsOfFieldVariablesInDofVector=dofs)

    condensation = SchurComplementCondensation([ElementSetMarker("bars"), ElementSetMarker("bonds")])
    condensation.setModel(model, dofManager)
    assert np.array_equal(condensation._selectCondensedDofs()[0], [0, 1, 2, 3])

    onlyBars = SchurComplementCondensation([ElementSetMarker("bars")])
    onlyBars.setModel(model, dofManager)
    assert np.array_equal(onlyBars._selectCondensedDofs()[0], [0, 1])

    with pytest.raises(KeyError, match="element set 'rebars'"):
        SchurComplementCondensation([ElementSetMarker("rebars")]).setModel(model, dofManager)


def test_remaining_field_blocks_drop_the_condensed_nodes():
    condensation = FixedCondensation(np.array([3, 4, 5, 13]))
    condensation.condensedDofs = np.array([3, 4, 5, 13])
    blocks = [FieldBlock("displacement", 0, 12, 3), FieldBlock("damage", 12, 16, 1)]
    remaining = condensation.remainingFieldBlocks(blocks)
    assert remaining == [FieldBlock("displacement", 0, 9, 3), FieldBlock("damage", 9, 12, 1)]
    # a pure function of the condensed dofs and the field block, whatever was asked before
    assert np.array_equal(condensation.remainingNodes(blocks[1]), [True, False, True, True])
    assert np.array_equal(condensation.remainingNodes(blocks[0]), [True, False, True, True])


def test_restricted_p1_map_turns_a_midside_node_with_a_condensed_endpoint_into_a_corner():
    # corners 0, 1, 2; midside 3 on the edge (0, 1), midside 4 on the edge (1, 2); node 2 condensed
    isCorner = np.array([True, True, True, False, False])
    edgeEndpoints = np.array([[-1, -1], [-1, -1], [-1, -1], [0, 1], [1, 2]])
    remains = np.array([True, True, False, True, True])
    isCorner, edgeEndpoints = restrictP1Map(isCorner, edgeEndpoints, remains)
    assert np.array_equal(isCorner, [True, True, False, True])
    assert np.array_equal(edgeEndpoints, [[-1, -1], [-1, -1], [0, 1], [-1, -1]])


@pytest.mark.parametrize("bandwidth", [1, 3, 40])
def test_transposed_solve_of_banded_and_dense_matrices(bandwidth):
    n = 60
    rng = np.random.default_rng(bandwidth)
    A = sp.diags(
        [rng.standard_normal(n - abs(k)) for k in range(-bandwidth, bandwidth + 1)], range(-bandwidth, bandwidth + 1)
    )
    A = (A + 10.0 * bandwidth * sp.identity(n)).tocoo()
    B = rng.standard_normal((n, 4))
    assert np.allclose(_solveTransposed(A, B), np.linalg.solve(A.toarray().T, B))


def test_superlu_factorizes_once_for_several_right_hand_sides():
    A, b, _ = barOnFoundation(n=5)
    solver = SuperLUBackend()
    solver.factorize(A)
    B = np.column_stack([b, 2 * b])
    assert np.allclose(A @ solver.solveFactorized(B), B)


class _FailingFromTheSecondFactorization(DirectSolver):
    """SuperLU whose factorizations fail from the second one on."""

    def __init__(self):
        super().__init__("superlu")
        self.factorizations = 0

    def _setUp(self, A, context):
        self.factorizations += 1
        if self.factorizations > 1:
            raise np.linalg.LinAlgError("singular, for the test")
        super()._setUp(A, context)


def test_a_solve_whose_condensed_block_cannot_be_factorized_does_without_the_condensation():
    A, b, barDofs = barOnFoundation(n=8)
    solveA_bb = _FailingFromTheSecondFactorization()
    schur = condensingGivenDofs(SchurComplement(["bars"], preconditionS=DirectSolver(), solveA_bb=solveA_bb), barDofs)
    warnings = []
    context = SetUpContext(
        fields=(FieldBlock("u", 0, A.shape[0], 1),), log=lambda level, message: warnings.append(message)
    )
    schur.setUp(SystemMatrix(A), context)
    assert schur.condensing and schur.condensedDofs() is not None
    schur.setUp(SystemMatrix(A), context)
    assert not schur.condensing and schur.condensedDofs() is None and solveA_bb.factorizations == 2
    assert any("cannot be factorized" in message for message in warnings)
    # without the condensation, the child alone solves the system
    assert np.linalg.norm(A @ schur(b) - b) < 1e-10 * np.linalg.norm(b)


class _CountingMarker(MarkerBase):
    """Marks given elements, counts its calls, and reports a stamp the test sets."""

    def __init__(self, elements):
        super().__init__()
        self.elements, self.stamp, self.marks = set(elements), (), 0

    def selectionStamp(self, model):
        return self.stamp

    def mark(self, model, candidateElements, mesh):
        self.marks += 1
        return self.elements


def _chainModel(nElements=6):
    """A chain of two-node elements, one dof per node: the model and the dof manager setModel reads."""
    nodes = [_Node(i) for i in range(nElements + 1)]
    elements = {i: _Element([nodes[i], nodes[i + 1]]) for i in range(nElements)}
    dofManager = SimpleNamespace(idcsOfFieldVariablesInDofVector={i: i for i in range(nElements + 1)})
    return SimpleNamespace(elements=elements), dofManager


def test_the_condensed_dofs_are_selected_anew_only_when_a_marker_input_changes():
    # a selection is kept while every marker's stamp stays the same; a marker that cannot tell (None) marks on every
    # call, and a new model drops the selection
    model, dofManager = _chainModel()
    marker = _CountingMarker([model.elements[1], model.elements[2]])
    condensation = SchurComplementCondensation([marker])
    condensation.setModel(model, dofManager)
    first = condensation._selectCondensedDofs()
    assert np.array_equal(first[0], [2]) and condensation._selectCondensedDofs() is first and marker.marks == 1
    marker.stamp = (1,)
    condensation._selectCondensedDofs()
    assert marker.marks == 2
    marker.stamp = None
    condensation._selectCondensedDofs()
    condensation._selectCondensedDofs()
    assert marker.marks == 4
    marker.stamp = ()
    condensation.setModel(model, dofManager)
    condensation._selectCondensedDofs()
    assert marker.marks == 5


def test_a_field_output_marker_is_stamped_by_its_field_output_and_a_set_marker_by_the_model(monkeypatch):
    fieldOutput = _FieldOutputBase("damage", SimpleNamespace(time=0.0), None)
    monkeypatch.setattr(marking, "_perElementFieldOutput", lambda model, name: fieldOutput)
    marker = FieldOutputMarker("damage", threshold=0.1)
    stamp = marker.selectionStamp(None)
    fieldOutput._applyResultsPipleline(np.zeros(3))
    assert fieldOutput.resultVersion == 1 and marker.selectionStamp(None) != stamp
    assert ElementSetMarker("bars").selectionStamp(None) == STABLE_FOR_THE_MODEL
    assert MarkerBase().selectionStamp(None) is None


def test_superlu_reports_a_singular_matrix_as_a_linear_algebra_error():
    with pytest.raises(np.linalg.LinAlgError):
        SuperLUBackend().factorize(sp.csr_matrix(np.array([[1.0, 1.0], [1.0, 1.0]])))


def test_by_default_the_block_upper_triangular_factor_is_inverted():
    """With the exact S, the default applies the inverse of U = [A_bb A_bk; 0 S]."""
    A, b, barDofs = barOnFoundation(n=6)
    condensation = FixedCondensation(barDofs, dropTolerance=0.0, windowNodes=10_000)
    condensation.update(A)
    S = condensation.approximateSchurComplement(A).toarray()
    z = blockLU(b, condensation, lambda r_k: np.linalg.solve(S, r_k), solveA_bbOf(condensation), "U")
    U = sp.bmat([[condensation.A_bb, condensation.A_bk], [None, sp.csr_matrix(S)]])
    order = np.concatenate([barDofs, condensation.remainingDofs])
    assert np.allclose(U @ z[order], b[order], rtol=0.0, atol=1e-10 * np.abs(b).max())


def test_tutorial_why_the_condensation_works():
    """The method in three observations, on a bar on an elastic foundation (small enough for dense matrices)."""
    from scipy.sparse.linalg import LinearOperator, gmres

    A, b, barDofs = barOnFoundation(n=6, barLength=36)
    remaining = np.setdiff1d(np.arange(A.shape[0]), barDofs)

    # 1. The inverse of the bar block decays exponentially along the bar (Demko, Moss and Smith): a load on a bar
    #    on its bond is felt only nearby. This is why S = A_kk - A_kb A_bb^-1 A_bk may be cut into windows.
    inverseOfBar = np.linalg.inv(A[barDofs][:, barDofs].toarray())
    #    Here, the bar is ten times stiffer than its bond, and the factor per dof is about 0.73.
    alongTheBar = np.abs(inverseOfBar[0]) / abs(inverseOfBar[0, 0])
    assert np.all(np.diff(alongTheBar[:30]) < 0.0) and alongTheBar[30] < 1e-3

    # 2. With the exact S, the block LU factorization is the inverse of A: GMRES needs one iteration ...
    # 3. ... and the block upper triangular factor alone two (A U^-1 = L, unit block lower triangular, (L - I)^2 = 0)
    S = _exactSchurComplement(A, barDofs)
    for blockFactors, iterations in (("LU", 1), ("U", 2)):
        condensation = FixedCondensation(barDofs, dropTolerance=0.0, windowNodes=10_000)
        condensation.update(A)
        solveA_bb = solveA_bbOf(condensation)
        assert np.array_equal(condensation.remainingDofs, remaining)

        def preconditioner(r):
            return blockLU(
                np.asarray(r).ravel(), condensation, lambda r_k: np.linalg.solve(S, r_k), solveA_bb, blockFactors
            )

        counted = []
        x, info = gmres(
            A,
            b,
            M=LinearOperator(A.shape, matvec=preconditioner),
            rtol=1e-10,
            callback=lambda residual: counted.append(residual),
            callback_type="pr_norm",
        )
        assert info == 0 and len(counted) == iterations
