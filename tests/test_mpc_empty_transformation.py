#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A model whose multi-point constraints constrain no DOF gets no condensation at all.

An adaptivity registers its hanging-node constraint up front, so before the first refinement the
model has a multi-point constraint without a single record. Condensing with the identity would cost a
``T^T K T`` product per Newton iteration and, through the zero pruning after it, a sparsity pattern
that changes every iteration -- so :meth:`buildMPCTransformation` must return None.
"""

import edelweissfe.utils.inputfileparser  # noqa: F401  bootstrap input language
from edelweissfe.models.femodel import FEModel
from edelweissfe.numerics.mpctransformation import MultiPointConstraintTransformation
from edelweissfe.solvers.base.nonlinearsolverbase import NonlinearSolverBase


class _FakeConstraint:
    mayYieldSlaveToEarlierClaim = True

    def __init__(self, records):
        self._records = records

    def getMultiPointConstraints(self, dofManager):
        return list(self._records)


class _DofManager:
    nDof = 10


class _Host:
    """The minimum a solver must provide for :meth:`buildMPCTransformation`."""

    identification = "TestSolver"
    supportsMPC = True

    def __init__(self):
        self.theDofManager = _DofManager()
        self.options = {}
        self.journal = self

    def message(self, text, identification, level=1):
        pass

    def _collectMultiPointConstraintRecords(self, model):
        return NonlinearSolverBase._collectMultiPointConstraintRecords(self, model)

    def build(self, model):
        return NonlinearSolverBase.buildMPCTransformation(self, model)


def _model(constraint):
    model = FEModel(2)
    model.multiPointConstraints["hangingNodes"] = constraint
    return model


def test_a_constraint_without_records_gives_no_transformation():
    assert _Host().build(_model(_FakeConstraint([]))) is None


def test_a_constraint_with_records_still_gives_a_transformation():
    transformation = _Host().build(_model(_FakeConstraint([(7, [(1, 0.5), (2, 0.5)])])))
    assert isinstance(transformation, MultiPointConstraintTransformation)
    assert transformation.nEliminatedDof == 1
