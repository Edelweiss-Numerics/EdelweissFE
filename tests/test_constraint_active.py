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
"""A constraint that is not active (``ConstraintBase.active``) is not part of the equation system: it gets no degrees
of freedom, the solver does not evaluate it, and its Lagrange multipliers leave the equation system with it. It is
active while it is switched on from outside (``switchedOn``, set through ``active``) and within reach of what it
constrains (``withinReach``, which the constraint finds itself).

A Lagrange multiplier constraint is the sharp case: if its multipliers stayed in the equation system without the
constraint that gives them an equation, the system matrix would be singular. Switched off by a ``modelupdate``, the
constraint below lets the loaded corner of a plate move freely; switched on in the next step, it pulls the top edge
back to one common displacement. The explicit solver ``NED``, which refuses a constraint with multipliers, does not
refuse one that is switched off.
"""

import numpy as np
import pytest

from edelweissfe.constraints.base.constraintbase import ConstraintBase
from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.solvers.nonlinearexplicitdynamic import NED
from edelweissfe.utils.inputfileparser import parseInputFile

DECK = """
*material, name=LinearElastic, id=linearelastic, provider=edelweiss
100.0, 0.3

*job, name=job, domain=2d
*solver, solver=NIST, name=theSolver

*modelGenerator, generator=planeRectQuad, name=gen
x0=0, y0=0, l=1.0, h=1.0, nX=2, nY=2, elType=CPE4, elProvider=edelweiss

*section, name=section1, thickness=1.0, material=linearelastic, type=plane
gen_all

*constraint, type=equalValueLagrangian, name=topEqual
nSet=gen_top
field=displacement
component=1

*step, solver=theSolver
maxInc=1e0, minInc=1e0, maxNumInc=1, maxIter=5, stepLength=1
>>modelupdate, update='model.constraints["topEqual"].active=False'
>>dirichlet, name=bottom, nSet=gen_bottom, field=displacement, 1=0, 2=0
>>nodeforces, name=pull, nSet=gen_rightTop, field=displacement, components='0, 1'
{secondStep}
"""

SECOND_STEP = """
*step, solver=theSolver
maxInc=1e0, minInc=1e0, maxNumInc=1, maxIter=5, stepLength=1
>>modelupdate, update='model.constraints["topEqual"].active=True'
"""


def _run(tmp_path, secondStep: str):
    path = tmp_path / "deck.inp"
    path.write_text(DECK.format(secondStep=secondStep))
    model, _ = finiteElementSimulation(parseInputFile(str(path)), verbose=False, suppressPlots=True)
    return model


def _verticalDisplacementOfTopEdge(model) -> np.ndarray:
    return model.nodeFields["displacement"].subset(model.nodeSets["gen_top"])["U"][:, 1]


def test_switched_off_lagrange_constraint_leaves_no_singular_rows(tmp_path):
    model = _run(tmp_path, secondStep="")

    assert not model.constraints["topEqual"].active
    assert model.activeConstraints() == []

    uTop = _verticalDisplacementOfTopEdge(model)
    assert np.all(np.isfinite(uTop))
    # without the constraint, the loaded corner runs ahead of the rest of the top edge
    assert np.ptp(uTop) > 1e-3

    # the multipliers were not part of the equation system, so they kept their initial value
    assert all(model.scalarVariables[name].value == 0.0 for name in model.scalarVariables)


def test_switched_on_again_the_constraint_takes_part_again(tmp_path):
    model = _run(tmp_path, secondStep=SECOND_STEP)

    assert model.activeConstraints() == [model.constraints["topEqual"]]

    uTop = _verticalDisplacementOfTopEdge(model)
    assert np.ptp(uTop) < 1e-10 * np.abs(uTop).max()
    assert any(model.scalarVariables[name].value != 0.0 for name in model.scalarVariables)


def test_the_explicit_solver_refuses_a_lagrange_constraint_only_while_it_is_switched_on(tmp_path):
    """NED refuses a constraint with scalar variables: they carry no inertia, so it has no equation of motion for
    them. A constraint that is switched off takes no part in the equation system, nor do its multipliers."""

    model = _run(tmp_path, secondStep="")
    explicitSolver = NED({}, Journal(verbose=False))

    explicitSolver.validateModelCapabilities(model)

    model.constraints["topEqual"].active = True
    with pytest.raises(NotImplementedError, match="topEqual"):
        explicitSolver.validateModelCapabilities(model)


class _ConstraintWithAPartner(ConstraintBase):
    """A constraint that finds itself within reach of its partner or not, as the test places the partner
    (``partnerInReach``), and reports a change of :attr:`active` as a change of its connectivity."""

    def __init__(self):
        super().__init__("withAPartner", None)
        self.name = "withAPartner"
        self.partnerInReach = True

    @property
    def nodes(self):
        return []

    @property
    def fieldsOnNodes(self):
        return []

    @property
    def nDof(self):
        return 0

    def updateConnectivity(self, model):
        wasActive = self.active
        self.withinReach = self.partnerInReach
        return self.active != wasActive

    def computeKernels(self, U_np, dU, P, K, timeStep):
        pass


def _modelWithAPartnerConstraint() -> tuple[FEModel, _ConstraintWithAPartner]:
    model = FEModel(2)
    constraint = _ConstraintWithAPartner()
    model.constraints[constraint.name] = constraint
    return model, constraint


def test_a_constraint_out_of_reach_is_not_active():
    model, constraint = _modelWithAPartnerConstraint()
    assert constraint.active and model.activeConstraints() == [constraint]

    constraint.partnerInReach = False
    assert constraint.updateConnectivity(model), "switching itself off is a change of the connectivity"
    assert constraint.switchedOn and not constraint.active
    assert model.activeConstraints() == []

    constraint.partnerInReach = True
    assert constraint.updateConnectivity(model), "switching itself on is a change of the connectivity"
    assert constraint.active and model.activeConstraints() == [constraint]


def test_a_constraint_switched_off_from_outside_stays_off_whatever_it_finds_in_reach():
    model, constraint = _modelWithAPartnerConstraint()

    constraint.active = False
    constraint.updateConnectivity(model)
    assert constraint.withinReach and not constraint.switchedOn and not constraint.active
    assert model.activeConstraints() == []

    constraint.active = True
    assert constraint.switchedOn and constraint.active

    constraint.partnerInReach = False
    constraint.updateConnectivity(model)
    assert constraint.switchedOn and not constraint.active, "switched on again, it follows the reach of its partner"
