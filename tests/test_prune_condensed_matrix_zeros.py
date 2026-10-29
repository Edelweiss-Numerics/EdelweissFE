#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit test for NIST's automatic ``pruneCondensedMatrixZeros`` decision
(:meth:`NIST._shouldPruneCondensedMatrixZeros`): unset defers to the active linear solver's own
:attr:`~edelweissfe.linsolve.base.LinearSolver.needsStableSparsityPattern`, an explicit True/False
always wins over that. Constructing a full NIST needs a job/journal/model; this calls the method on a
minimal stand-in that only provides what it reads (``options`` and ``linSolver``) instead.
"""

import pytest

from edelweissfe.solvers.nonlinearimplicitstatic import NIST


class _LinearSolverStub:
    """The one attribute ``_shouldPruneCondensedMatrixZeros`` reads off the active linear solver."""

    def __init__(self, needsStableSparsityPattern: bool):
        self.needsStableSparsityPattern = needsStableSparsityPattern


class _NISTStub:
    """The two attributes ``_shouldPruneCondensedMatrixZeros`` reads off ``self`` -- everything else a
    real NIST carries (model, journal, steps, ...) is irrelevant to that one decision."""

    def __init__(self, pruneCondensedMatrixZeros, needsStableSparsityPattern: bool):
        self.options = {"pruneCondensedMatrixZeros": pruneCondensedMatrixZeros}
        self.linSolver = _LinearSolverStub(needsStableSparsityPattern)


@pytest.mark.parametrize(
    "explicit, needsStableSparsityPattern, expected",
    [
        # Unset (None): automatic, from the active solver's own needsStableSparsityPattern.
        (None, False, True),  # e.g. PARDISO: no reuse to protect, pruning is free -- prune.
        (None, True, False),  # e.g. blockamg reusing its preconditioner: pruning would churn the pattern -- don't.
        # An explicit value always wins, in both directions, regardless of the solver in force.
        (True, True, True),
        (False, False, False),
    ],
    ids=[
        "unset, solver tolerates pruning -> pruned",
        "unset, solver needs a stable pattern -> not pruned",
        "explicit True overrides a solver that needs a stable pattern",
        "explicit False overrides a solver that tolerates pruning",
    ],
)
def test_the_automatic_decision_table(explicit, needsStableSparsityPattern, expected):
    stub = _NISTStub(explicit, needsStableSparsityPattern)
    assert NIST._shouldPruneCondensedMatrixZeros(stub) is expected


def test_a_linear_solver_needs_a_stable_pattern_if_a_krylov_solver_of_its_tree_reuses_its_preconditioner():
    """Every linear solver is a composed one: a direct solver (one leaf, factorized anew on every solve) and a Krylov
    solver that builds its preconditioner anew for every solve (the default) tolerate pruning; a Krylov solver that
    reuses its preconditioner across solves, at any depth of the tree, does not."""
    from edelweissfe.linsolve.composition import (
        AMG,
        FGMRES,
        LGMRES,
        ComposedLinearSolver,
        DiagonalScaling,
        DirectSolver,
        FixedTolerance,
        RebuildPolicy,
    )

    def reusing():
        return LGMRES(AMG(), rebuild=RebuildPolicy(reusePreconditioner=True))

    cases = (
        (DirectSolver("superlu"), True),
        (LGMRES(AMG()), True),
        (reusing(), False),
        (DiagonalScaling(reusing()), False),
        (FGMRES(LGMRES(AMG(), tolerance=FixedTolerance(1e-2), rebuild=RebuildPolicy(reusePreconditioner=True))), False),
    )
    for root, expected in cases:
        stub = _NISTStub(None, ComposedLinearSolver(root).needsStableSparsityPattern)
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
