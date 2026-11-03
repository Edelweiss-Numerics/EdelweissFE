#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit test for NIST's automatic ``pruneCondensedMatrixZeros`` decision
(:meth:`NIST._shouldPruneCondensedMatrixZeros`): unset defers to the active linear solver's own
:attr:`~edelweissfe.linsolve.base.LinearSolver.needsPrunedSystemMatrix`, an explicit True/False
always wins over that. Constructing a full NIST needs a job/journal/model; this calls the method on a
minimal stand-in that only provides what it reads (``options`` and ``linSolver``) instead.
"""

import pytest

from edelweissfe.solvers.nonlinearimplicitstatic import NIST


class _LinearSolverStub:
    """The one attribute ``_shouldPruneCondensedMatrixZeros`` reads off the active linear solver."""

    def __init__(self, needsPrunedSystemMatrix: bool):
        self.needsPrunedSystemMatrix = needsPrunedSystemMatrix


class _NISTStub:
    """The two attributes ``_shouldPruneCondensedMatrixZeros`` reads off ``self`` -- everything else a
    real NIST carries (model, journal, steps, ...) is irrelevant to that one decision."""

    def __init__(self, pruneCondensedMatrixZeros, needsPrunedSystemMatrix: bool):
        self.options = {"pruneCondensedMatrixZeros": pruneCondensedMatrixZeros}
        self.linSolver = _LinearSolverStub(needsPrunedSystemMatrix)


@pytest.mark.parametrize(
    "explicit, needsPrunedSystemMatrix, expected",
    [
        # Unset (None): automatic, from the active solver's own needsPrunedSystemMatrix.
        (None, True, True),  # e.g. PARDISO: its reordering is sensitive to the stored zeros -- prune.
        (None, False, False),  # e.g. blockamg: the same iterations with the zeros, pruning only costs -- don't.
        # An explicit value always wins, in both directions, regardless of the solver in force.
        (True, False, True),
        (False, True, False),
    ],
    ids=[
        "unset, solver needs a pruned matrix -> pruned",
        "unset, solver does not -> not pruned",
        "explicit True overrides a solver that does not need it",
        "explicit False overrides a solver that needs it",
    ],
)
def test_the_automatic_decision_table(explicit, needsPrunedSystemMatrix, expected):
    stub = _NISTStub(explicit, needsPrunedSystemMatrix)
    assert NIST._shouldPruneCondensedMatrixZeros(stub) is expected


def test_a_linear_solver_needs_a_pruned_matrix_if_it_factorizes_and_reuses_no_preconditioner():
    """Every linear solver is a composed one: a direct solver (one leaf) wants the stored zeros pruned out of the
    matrix, also one inside a Schur complement; a Krylov solver with an AMG preconditioner does not; and a tree
    with a Krylov solver that reuses its preconditioner across solves, at any depth, must not have the pattern pruned
    anew in every iteration, whatever else it holds."""
    from edelweissfe.linsolve.composition import (
        AMG,
        FGMRES,
        LGMRES,
        ComposedLinearSolver,
        DiagonalScaling,
        DirectSolver,
        FixedTolerance,
        Multiplicative,
        OnField,
        RebuildPolicy,
    )

    def reusing(preconditioner):
        return LGMRES(preconditioner, rebuild=RebuildPolicy(reusePreconditioner=True))

    cases = (
        (DirectSolver("superlu"), True),
        (LGMRES(AMG()), False),
        (DiagonalScaling(LGMRES(AMG())), False),
        (reusing(AMG()), False),
        (DiagonalScaling(reusing(AMG())), False),
        (FGMRES(reusing(AMG()), tolerance=FixedTolerance(1e-2)), False),
        # a direct solve inside a tree: factorized, so pruned -- unless the tree reuses its preconditioner
        (LGMRES(Multiplicative([OnField("u", DirectSolver("superlu"))])), True),
        (reusing(Multiplicative([OnField("u", DirectSolver("superlu"))])), False),
    )
    for root, expected in cases:
        stub = _NISTStub(None, ComposedLinearSolver(root).needsPrunedSystemMatrix)
        assert NIST._shouldPruneCondensedMatrixZeros(stub) is expected


@pytest.mark.parametrize("value, expected", [("false", False), ("False", False), ("true", True), ("1", True)])
def test_the_option_can_be_set_in_the_datalines_of_the_solver_keyword(value, expected):
    """The option has no default to take its type from (None: automatic), so its type is the schema's: a bool, parsed
    as one (``bool("false")`` would be truthy)."""
    stub = type("_Stub", (), {"schema": NIST.schema, "SolverSpecificOptions": NIST.SolverSpecificOptions})()
    stub.identification = "stub"
    stub.options = NIST.SolverSpecificOptions.copy()

    class _Journal:
        def message(self, *args, **kwargs):
            pass

    NIST._updateOptions(stub, {"pruneCondensedMatrixZeros": value}, _Journal(), strict=True)
    assert stub.options["pruneCondensedMatrixZeros"] is expected
