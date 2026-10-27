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
"""What is built again is forgotten first: a rebuild never holds the old and the new hierarchies, factorizations and
matrices at once, and what a component reuses on purpose survives. Each test looks at the moment a new state is
built and asks what of the old one is still held."""

import gc
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp

pytest.importorskip("edelweissfe.linsolve.amgcl.amgcl")

from _linsolvesystems import (  # noqa: E402
    barOnFoundation,
    condensingGivenDofs,
    displacementDamageSystem,
)

import edelweissfe.linsolve.composition.amg as amgModule  # noqa: E402
from edelweissfe.linsolve.base import FieldBlock  # noqa: E402
from edelweissfe.linsolve.composition import (  # noqa: E402
    AMG,
    LGMRES,
    ComposedLinearSolver,
    DirectSolver,
    FixedTolerance,
    Multiplicative,
    OnField,
    Preconditioner,
    SchurComplement,
)
from edelweissfe.linsolve.composition.schurcomplement import (  # noqa: E402
    SchurComplementCondensation,
)
from edelweissfe.linsolve.superlu import SuperLUBackend  # noqa: E402
from edelweissfe.numerics.csrgeneratorv2 import CSRGenerator  # noqa: E402


def _composed(root, fields):
    solver = ComposedLinearSolver(root, verbosity="silent")
    solver.setFieldStructure(fields)
    return solver


def _newPattern(A):
    """``A`` with one more stored entry, an explicit zero: a new sparsity pattern, for which the composed solver asks
    for a rebuild."""
    A = A.tocoo()
    n = A.shape[0]
    return sp.csr_matrix((np.append(A.data, 0.0), (np.append(A.row, 0), np.append(A.col, n - 1))), shape=A.shape)


def test_a_krylov_rebuild_forgets_every_hierarchy_before_it_builds_one(monkeypatch):
    A, b, fields = displacementDamageSystem(n=10, coupling="symmetric")
    amgs = [AMG(), AMG()]
    root = LGMRES(Multiplicative([OnField(f.name, amg) for f, amg in zip(fields, amgs)]), FixedTolerance(1e-8))
    solver = _composed(root, fields)
    solver(A, b)

    heldAtBuild = []
    original = amgModule.buildFieldHierarchy

    def build(*arguments):
        heldAtBuild.append(sum(amg.hierarchy is not None for amg in amgs))
        return original(*arguments)

    monkeypatch.setattr(amgModule, "buildFieldHierarchy", build)
    solver(_newPattern(A), b)
    assert root.lastRebuildReason == "pattern"
    # the first build of the rebuild finds no old hierarchy, the second only the one just built
    assert heldAtBuild == [0, 1]
    # the combination hands the rebuild on to its corrections as an update, with what changed
    assert [c.statistics.setUps for c in root.preconditioner.corrections] == [1, 1]

    # only new values: the hierarchies are reused, not forgotten
    heldAtBuild.clear()
    solver(1.0001 * _newPattern(A), b)
    assert root.lastRebuildReason is None and heldAtBuild == []
    assert all(amg.hierarchy is not None for amg in amgs)


def test_the_schur_complement_drops_its_old_complement_and_child_before_it_forms_the_next(monkeypatch):
    A, b, barDofs = barOnFoundation(n=8)
    fields = [FieldBlock("u", 0, A.shape[0], 1)]
    child = AMG()
    schur = condensingGivenDofs(SchurComplement(["bars"], preconditionS=child), barDofs)
    root = LGMRES(schur, FixedTolerance(1e-8))
    solver = _composed(root, fields)
    solver(A, b)
    assert schur.S_tilde is not None and child.hierarchy is not None

    heldAtFormation = []
    original = SchurComplementCondensation.approximateSchurComplement

    def approximateSchurComplement(condensation, matrix):
        heldAtFormation.append((schur.S_tilde, child.hierarchy))
        return original(condensation, matrix)

    monkeypatch.setattr(SchurComplementCondensation, "approximateSchurComplement", approximateSchurComplement)
    # other condensed dofs: S~ is formed anew, without a rebuild asked for by the Krylov solver
    schur.condensation.dofs = barDofs[1:]
    x = solver(1.0001 * A, b)
    assert heldAtFormation == [(None, None)]
    assert np.linalg.norm(1.0001 * (A @ x) - b) < 1e-7 * np.linalg.norm(b)


def test_a_direct_solve_releases_its_factorization_before_it_factorizes_again(monkeypatch):
    A, b, fields = displacementDamageSystem(n=6, coupling="symmetric")
    solver = _composed(DirectSolver("superlu"), fields)
    solver(A, b)

    heldAtFactorization = []
    original = SuperLUBackend.factorize

    def factorize(backend, matrix):
        heldAtFactorization.append(backend._factorization)
        original(backend, matrix)

    monkeypatch.setattr(SuperLUBackend, "factorize", factorize)
    x = solver(2.0 * A, b)
    assert heldAtFactorization == [None]
    assert np.allclose(2.0 * (A @ x), b)


def test_a_new_model_forgets_the_tree_at_once_and_keeps_what_is_reused_on_purpose():
    A, b, fields = displacementDamageSystem(n=10, coupling="symmetric")
    # the scalar field alone, as a model of one field
    damage = fields[1]
    A, b = A[damage.start : damage.stop, damage.start : damage.stop].tocsr(), b[damage.start : damage.stop]
    n = A.shape[0]
    amg = AMG()
    root = LGMRES(OnField(damage.name, amg), FixedTolerance(1e-8))
    solver = _composed(root, [FieldBlock(damage.name, 0, n, 1)])
    solver(A, b)
    scratch = root._krylov

    model = SimpleNamespace(nodeFields={damage.name: SimpleNamespace(dimension=1, nodes=[])})
    dofManager = SimpleNamespace(idcsOfFieldsInDofVector={damage.name: slice(0, n)})
    solver.setModel(model, dofManager)
    assert amg.hierarchy is None
    # the scratch vectors of the Krylov solve are sized for the system, not built for its matrix
    assert root._krylov is scratch
    x = solver(A, b)
    assert amg.hierarchy is not None and root.lastRebuildReason == "model"
    assert np.linalg.norm(A @ x - b) < 1e-7 * np.linalg.norm(b)


class _Singular(Preconditioner):
    """A preconditioner whose set-up meets a singular matrix."""

    def _setUp(self, A, context):
        raise np.linalg.LinAlgError("singular")


def test_a_failed_update_releases_the_matrix_of_the_solve():
    A, b, fields = displacementDamageSystem(n=6, coupling="symmetric")
    root = LGMRES(_Singular(), FixedTolerance(1e-8))
    x = _composed(root, fields)(A, b)
    assert np.all(np.isnan(x))
    # the increment is cut back: the tree must not hold this matrix while the next one is assembled
    assert root.A is None


def test_a_csr_generator_and_its_matrix_form_no_reference_cycle():
    """The matrix's pattern arrays keep the C++ core alive, not the generator: a replaced generator is freed by
    reference counting, and its matrix stays valid without it."""
    A = displacementDamageSystem(n=6, coupling="symmetric")[0].tocoo()
    vij = SimpleNamespace(I=A.row.astype(np.intc), J=A.col.astype(np.intc), nDof=A.shape[0])

    def generatorsAlive():
        return sum(isinstance(obj, CSRGenerator) for obj in gc.get_objects())

    gc.collect()
    gc.disable()
    try:
        before = generatorsAlive()
        generator = CSRGenerator(vij)
        matrix = generator.updateInPlace(A.data.copy())
        assert generatorsAlive() == before + 1
        del generator
        assert generatorsAlive() == before
        x = np.arange(A.shape[0], dtype=float)
        assert np.allclose(matrix @ x, A @ x)
    finally:
        gc.enable()
