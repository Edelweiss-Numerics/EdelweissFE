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
"""The constraint forces of an explicit solver
(:meth:`~edelweissfe.solvers.base.nonlinearsolverbase.NonlinearSolverBase.assembleConstraintForces`): the force buffer
and the DOF indices of a constraint are made once per discrete system and reused in every increment, and a plan made
for indices the constraint no longer has is never used. The tangent that a constraint's default explicit evaluation
throws away is sized once per discrete system as well."""

from types import SimpleNamespace

import numpy as np

from edelweissfe.constraints.base.constraintbase import ConstraintBase
from edelweissfe.numerics.dofvector import DofVector
from edelweissfe.solvers.base.nonlinearsolverbase import NonlinearSolverBase


class _UnitForceConstraint(ConstraintBase):
    """A constraint that adds 1 at each of its two degrees of freedom, and a tangent; evaluated in an explicit
    analysis by the default of :meth:`computeKernelsExplicit`."""

    def __init__(self):
        super().__init__("spring", None)

    @property
    def nodes(self):
        return []

    @property
    def fields(self):
        return []

    @property
    def nDof(self):
        return 2

    def computeKernels(self, U_np, dU, P, K, timeStep):
        P += 1.0
        K += 1.0


class _ListedUnitForceConstraint(_UnitForceConstraint):
    """The same constraint with a block of its own layout, its diagonal; it counts how often it builds the list."""

    def __init__(self):
        super().__init__()
        self.listings = 0

    def localEntriesOfBlock(self):
        self.listings += 1
        return np.arange(2, dtype=np.intc), np.arange(2, dtype=np.intc)


class _Locations(SimpleNamespace):
    """The locations the constraint forces read: ``locations[constraint]``, from a mapping."""

    def __getitem__(self, entity):
        return self.ofEntity[entity]


class _Host:
    """The minimum a solver must provide for :meth:`assembleConstraintForces`: a discrete system with locations."""

    _constraintForcePlans = None
    system = None

    def assemble(self, constraints, P):
        return NonlinearSolverBase.assembleConstraintForces(self, constraints, P, P, P, None)


def _forces(host, constraints, ofEntity):
    """The forces of the constraints in the discrete system whose locations are ``ofEntity``."""

    locations = _Locations(ofEntity=ofEntity)
    host.system = SimpleNamespace(locations=locations)
    P = DofVector(4, locations)
    return np.asarray(host.assemble(constraints, P))


def test_the_plan_follows_a_constraint_that_is_located_anew():
    """A constraint located anew has indices of its own: the plan made for the old indices must not be used for the
    new ones."""
    host = _Host()
    constraint = _UnitForceConstraint()
    constraints = [constraint]

    np.testing.assert_array_equal(_forces(host, constraints, {constraint: np.array([0, 1])}), [1, 1, 0, 0])
    np.testing.assert_array_equal(_forces(host, constraints, {constraint: np.array([3, 1])}), [0, 1, 0, 1])


def test_the_plan_is_reused_while_the_indices_stay():
    host = _Host()
    constraint = _UnitForceConstraint()
    constraints = [constraint]
    ofEntity = {constraint: np.array([0, 1])}

    _forces(host, constraints, ofEntity)
    plan = host._constraintForcePlans[constraint]
    _forces(host, constraints, ofEntity)
    assert host._constraintForcePlans[constraint] is plan


def test_the_discarded_tangent_is_sized_once_per_discrete_system():
    """A constraint that lists the entries of its block builds the list to size the tangent its explicit evaluation
    throws away: once per discrete system, not in every increment."""
    host = _Host()
    constraint = _ListedUnitForceConstraint()
    constraints = [constraint]
    ofEntity = {constraint: np.array([0, 1])}

    for _ in range(3):
        np.testing.assert_array_equal(_forces(host, constraints, ofEntity), [1, 1, 0, 0])
    assert constraint.listings == 1

    ofEntity = {constraint: np.array([2, 3])}
    for _ in range(3):
        np.testing.assert_array_equal(_forces(host, constraints, ofEntity), [0, 0, 1, 1])
    assert constraint.listings == 2
