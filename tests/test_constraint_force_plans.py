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
and the DOF indices of a constraint are made once per equation system and reused in every increment, and a plan made
for indices the constraint no longer has is never used."""

import numpy as np

from edelweissfe.numerics.dofvector import DofVector
from edelweissfe.solvers.base.nonlinearsolverbase import NonlinearSolverBase


class _UnitForceConstraint:
    """A constraint that adds 1 at each of its two degrees of freedom."""

    active = True
    nDof = 2

    def computeKernelsExplicit(self, U_np, dU, P, timeStep):
        P += 1.0


class _Host:
    """The minimum a solver must provide for :meth:`assembleConstraintForces`."""

    _constraintForcePlans = None

    def assemble(self, constraints, P):
        return NonlinearSolverBase.assembleConstraintForces(self, constraints, P, P, P, None)


def _forces(host, constraints, entitiesInDofVector):
    P = DofVector(4, entitiesInDofVector)
    return np.asarray(host.assemble(constraints, P))


def test_the_plan_follows_a_constraint_located_anew_in_the_same_mapping():
    """The meshfree explicit solver re-locates its constraints in place (the DofManager updates its entity mapping):
    the plan made for the old indices must not be used for the new ones."""
    host = _Host()
    constraint = _UnitForceConstraint()
    constraints = {"spring": constraint}
    entitiesInDofVector = {constraint: np.array([0, 1])}

    np.testing.assert_array_equal(_forces(host, constraints, entitiesInDofVector), [1, 1, 0, 0])
    np.testing.assert_array_equal(_forces(host, constraints, entitiesInDofVector), [1, 1, 0, 0])

    entitiesInDofVector[constraint] = np.array([2, 3])
    np.testing.assert_array_equal(_forces(host, constraints, entitiesInDofVector), [0, 0, 1, 1])


def test_the_plan_follows_a_rebuilt_equation_system():
    host = _Host()
    constraint = _UnitForceConstraint()
    constraints = {"spring": constraint}

    np.testing.assert_array_equal(_forces(host, constraints, {constraint: np.array([0, 1])}), [1, 1, 0, 0])
    np.testing.assert_array_equal(_forces(host, constraints, {constraint: np.array([3, 1])}), [0, 1, 0, 1])


def test_the_plan_is_reused_while_the_indices_stay():
    host = _Host()
    constraint = _UnitForceConstraint()
    constraints = {"spring": constraint}
    entitiesInDofVector = {constraint: np.array([0, 1])}

    _forces(host, constraints, entitiesInDofVector)
    plan = host._constraintForcePlans["spring"]
    _forces(host, constraints, entitiesInDofVector)
    assert host._constraintForcePlans["spring"] is plan
