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
        (None, True, False),  # e.g. blockamg: pruning would churn the pattern every iteration -- don't.
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


def test_matrixdumpsolver_forwards_the_decision_to_its_delegate():
    """MatrixDumpSolver wraps another linear solver for offline dumping; the automatic decision must see
    through the wrapper to the delegate's own needsStableSparsityPattern, not the wrapper's."""
    from edelweissfe.linsolve.matrixdump.matrixdump import MatrixDumpSolver

    for delegateNeedsStablePattern in (True, False):
        delegate = _LinearSolverStub(delegateNeedsStablePattern)
        wrapper = MatrixDumpSolver.__new__(MatrixDumpSolver)
        wrapper._delegate = delegate
        assert wrapper.needsStableSparsityPattern is delegateNeedsStablePattern

        stub = _NISTStub(None, wrapper.needsStableSparsityPattern)
        assert NIST._shouldPruneCondensedMatrixZeros(stub) is (not delegateNeedsStablePattern)
