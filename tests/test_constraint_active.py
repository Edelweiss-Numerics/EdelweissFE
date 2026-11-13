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
"""A constraint that is not active (``ConstraintBase.active``) is not part of the discrete system: it gets no degrees
of freedom, the solver does not evaluate it, and its Lagrange multipliers leave the discrete system with it. It is
active while it is switched on from outside (``switchedOn``, set through ``active``) and within reach of what it
constrains (``withinReach``, which the constraint finds itself).

A Lagrange multiplier constraint is the sharp case: if its multipliers stayed in the discrete system without the
constraint that gives them an equation, the system matrix would be singular. Switched off by a ``modelupdate``, the
constraint below lets the loaded corner of a plate move freely; switched on in the next step, it pulls the top edge
back to one common displacement. The explicit solver ``NED``, which refuses a constraint with multipliers, does not
refuse one that is switched off, nor does ``NEST`` refuse a contact switched off from outside. A constraint switched
off from outside rests: the model does not locate it anew.
"""

import numpy as np
import pytest

from edelweissfe.config import registry
from edelweissfe.constraints.base.constraintbase import ConstraintBase
from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.journal.journal import Journal
from edelweissfe.models.change import Change
from edelweissfe.models.femodel import FEModel
from edelweissfe.solvers.nonlinearexplicitdynamic import NED
from edelweissfe.solvers.nonlinearexplicitstatic import NEST
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


def _runDeck(tmp_path, deck: str):
    path = tmp_path / "deck.inp"
    path.write_text(deck)
    model, _ = finiteElementSimulation(parseInputFile(str(path)), verbose=False, suppressPlots=True)
    return model


def _run(tmp_path, secondStep: str):
    return _runDeck(tmp_path, DECK.format(secondStep=secondStep))


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

    # the multipliers were not part of the discrete system, so they kept their initial value
    assert all(model.scalarVariables[name].value == 0.0 for name in model.scalarVariables)


def test_switched_on_again_the_constraint_takes_part_again(tmp_path):
    model = _run(tmp_path, secondStep=SECOND_STEP)

    assert model.activeConstraints() == [model.constraints["topEqual"]]

    uTop = _verticalDisplacementOfTopEdge(model)
    assert np.ptp(uTop) < 1e-10 * np.abs(uTop).max()
    assert any(model.scalarVariables[name].value != 0.0 for name in model.scalarVariables)


def test_the_explicit_solver_refuses_a_lagrange_constraint_only_while_it_is_switched_on(tmp_path):
    """NED refuses a constraint with scalar variables: they carry no inertia, so it has no equation of motion for
    them. A constraint that is switched off takes no part in the discrete system, nor do its multipliers."""

    model = _run(tmp_path, secondStep="")
    explicitSolver = NED({}, Journal(verbose=False))

    explicitSolver.validateModelCapabilities(model)

    model.constraints["topEqual"].active = True
    with pytest.raises(NotImplementedError, match="topEqual"):
        explicitSolver.validateModelCapabilities(model)

    # switched on, but out of reach at the start of the step: it may come within reach during the step
    model.constraints["topEqual"].withinReach = False
    with pytest.raises(NotImplementedError, match="topEqual"):
        explicitSolver.validateModelCapabilities(model)


class _ConstraintWithAPartner(ConstraintBase):
    """A constraint that finds itself within reach of its partner or not, as the test places the partner
    (``partnerInReach``), and couples no nodes. It counts how often it is located (``timesLocated``)."""

    def __init__(self):
        super().__init__("withAPartner", None)
        self.name = "withAPartner"
        self.partnerInReach = True
        self.timesLocated = 0

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
        self.timesLocated += 1
        self.withinReach = self.partnerInReach
        return False

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
    constraint.updateConnectivity(model)
    assert constraint.switchedOn and not constraint.active
    assert model.activeConstraints() == []

    constraint.partnerInReach = True
    constraint.updateConnectivity(model)
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


def test_the_model_locates_a_constraint_out_of_reach_but_not_one_switched_off_from_outside():
    model, constraint = _modelWithAPartnerConstraint()

    constraint.active = False
    assert model.updateConnectivity() == Change.NONE
    assert constraint.timesLocated == 0, "switched off from outside, the constraint rests"

    constraint.active = True
    constraint.partnerInReach = False
    assert model.updateConnectivity() == Change.LOCATIONS, "the model sees the constraint leave reach"
    assert constraint.timesLocated == 1 and not constraint.active

    constraint.partnerInReach = True
    assert (
        model.updateConnectivity() == Change.LOCATIONS
    ), "out of reach, it is located, so that it finds itself within reach again"
    assert constraint.timesLocated == 2 and constraint.active


def test_the_system_holds_exactly_the_constraints_its_locations_hold():
    """The solver evaluates the constraints of the discrete system: they must be those that have rows in it."""
    model, constraint = _modelWithAPartnerConstraint()

    system = model.buildDiscreteSystem()
    assert system.constraints == [constraint]
    assert system.locations[constraint] is not None
    with pytest.raises(AttributeError):
        system.constraints = []

    constraint.active = False
    system = model.buildDiscreteSystem()
    assert system.constraints == [] and constraint not in system.locations


def test_the_system_follows_its_locations_when_a_constraint_leaves_reach_and_returns():
    model, constraint = _modelWithAPartnerConstraint()
    system = model.buildDiscreteSystem()

    constraint.partnerInReach = False
    assert model.updateConnectivity() == Change.LOCATIONS
    outOfReach = system.withNewLocations()
    assert outOfReach.constraints == [] and constraint not in outOfReach.locations
    assert outOfReach.numbering is system.numbering and system.constraints == [constraint], "a new system"

    constraint.partnerInReach = True
    assert model.updateConnectivity() == Change.LOCATIONS
    back = outOfReach.withNewLocations()
    assert back.constraints == [constraint]
    assert back.locations[constraint] is not None


class _ReachProbe(ConstraintBase):
    """A constraint on one node that leaves reach once it has been evaluated three times, and counts its explicit
    evaluations (``nEvaluations``, and ``nEvaluationsAtLeaving``). It stands in for a contact: it searches for its
    connectivity."""

    connectivityChangesWithMotion = True

    def __init__(self, name, model, **kwargs):
        super().__init__(name, model)
        self.name = name
        self._node = next(iter(model.nodeSets["gen_top"]))
        self.nEvaluations = 0
        self.nEvaluationsAtLeaving = None

    @property
    def nodes(self):
        return [self._node]

    @property
    def fieldsOnNodes(self):
        return [["displacement"]]

    @property
    def nDof(self):
        return 2

    def updateConnectivity(self, model):
        self.withinReach = self.nEvaluations < 3
        if not self.withinReach and self.nEvaluationsAtLeaving is None:
            self.nEvaluationsAtLeaving = self.nEvaluations
        return False

    def computeKernels(self, U_np, dU, P, K, timeStep):
        pass

    def computeKernelsExplicit(self, U_np, dU, P, timeStep):
        self.nEvaluations += 1


registry.register("constraint", "reachprobe", _ReachProbe)

REACH_PROBE_DECK = """
*material, name=LinearElastic, id=linearelastic, provider=edelweiss
1.0e4, 0.0, 1.0

*section, name=section1, material=linearelastic, type=plane, thickness=1
all

*job, name=job, domain=2d

*modelGenerator, generator=planeRectQuad, name=gen
l=1, h=1, nX=1, nY=1, elType=CPE4, elProvider=edelweiss

*solver, solver=NED, name=theSolver
courant-number=1.0
output-frequency=1
contact-update-frequency=1

*constraint, name=probe, type=reachprobe

*step, type=adaptiveForExplicitSimulations, solver=theSolver
maxInc=1, minInc=1e-12, maxNumInc=1000, maxIter=25, stepLength=0.02
>>dirichlet, name=base, nSet=gen_bottom, field=displacement, 1=0.0, 2=0.0
"""


def test_the_explicit_dynamic_solver_evaluates_the_constraints_of_its_system(tmp_path, monkeypatch):
    """A constraint that leaves reach is located out of the system (the solver makes new locations of the discrete
    system, without the constraint): from then on the system holds no constraint, and the solver evaluates none."""

    monkeypatch.setattr(NED, "getCriticalTimeStepForExplicitDynamics", lambda self, model, U: 2.0e-3)

    model = _runDeck(tmp_path, REACH_PROBE_DECK)

    probe = model.constraints["probe"]
    assert not probe.active
    assert probe.nEvaluationsAtLeaving == 3, "left reach after three evaluations"
    assert probe.nEvaluations == probe.nEvaluationsAtLeaving, "evaluated while in reach, never once it has left"


#: Two plane-strain blocks, the upper one 0.05 above the lower one, and a frictional contact with augmented Lagrange
#: multipliers between them: the model of ``testfiles/edelweiss-only/NodeToDeformableSurfaceContact2D``.
CONTACT_MODEL = """
*material, name=LinearElastic, id=linearelastic, provider=edelweiss
1.8e4, 0.3

*job, name=job, domain=2d
*solver, solver=NIST, name=theSolver

*modelGenerator, generator=planeRectQuad, name=lower
x0=0, y0=0, l=2.0, h=1.0, nX=3, nY=2, elType=CPE8, elProvider=edelweiss

*modelGenerator, generator=planeRectQuad, name=upper
x0=0.7, y0=1.05, l=0.6, h=1.0, nX=2, nY=2, elType=CPE8, elProvider=edelweiss

*modelGenerator, generator=surfaceElementGenerator, name=slaveFacets
surface=upper_bottom
name=slaveSurf
triangulation=midside

*modelGenerator, generator=surfaceElementGenerator, name=masterFacets
surface=lower_top
name=masterSurf
triangulation=midside

*section, name=section1, thickness=1.0, material=linearelastic, type=plane
lower_all
upper_all

*constraint, name=contact, type=nodeToDeformableSurfacePenalty
slaveSurface=slaveSurf_facets, masterSurface=masterSurf_facets, penalty=3e6, type=quadratic,
searchDistance=2.0, sliding=small, mu=0.3, tangentPenalty=1e6, augmentedLagrange=True
"""

#: Press the upper block onto the lower one, then drag it sideways, so that the contact holds normal multipliers and a
#: frictional history: the steps of ``testfiles/edelweiss-only/NodeToDeformableSurfaceContact2D``.
PRESS_AND_DRAG = """
*step, solver=theSolver
maxInc=0.34, minInc=1e-6, maxNumInc=100, maxIter=25, stepLength=1
>>options, name=theSolver, extrapolation=off
>>dirichlet, name=fixLower, nSet=lower_bottom, field=displacement, 1=0.0, 2=0.0
>>dirichlet, name=holdUpper, nSet=upper_top, field=displacement, 1=0.0, 2=-0.15

*step, solver=theSolver
maxInc=0.1, minInc=1e-6, maxNumInc=100, maxIter=25, stepLength=1
>>options, name=theSolver, extrapolation=off
>>dirichlet, name=fixLower, nSet=lower_bottom, field=displacement, 1=0.0, 2=0.0
>>dirichlet, name=holdUpper, nSet=upper_top, field=displacement, 1=0.4, 2=0.0
"""

#: A step of three increments with the contact switched off, the bottom of the lower block fixed and the top of the
#: upper block held.
SWITCHED_OFF = """
*step, solver=theSolver
maxInc=0.34, minInc=1e-6, maxNumInc=100, maxIter=25, stepLength=1
>>modelupdate, update='model.constraints["contact"].active=False'
>>dirichlet, name=fixLower, nSet=lower_bottom, field=displacement, 1=0.0, 2=0.0
>>dirichlet, name=holdUpper, nSet=upper_top, field=displacement, 1=0.0, 2=0.0
"""


def test_the_explicit_static_solver_refuses_a_contact_only_while_it_is_switched_on(tmp_path):
    """NEST never updates the connectivity, so it refuses a contact, which searches for what it touches in every
    increment. A contact switched off from outside rests, and takes no part in the step."""

    model = _runDeck(tmp_path, CONTACT_MODEL + SWITCHED_OFF)
    explicitSolver = NEST({}, Journal(verbose=False))

    explicitSolver.validateModelCapabilities(model)

    model.constraints["contact"].active = True
    with pytest.raises(Exception, match="changes with the motion"):
        explicitSolver.validateModelCapabilities(model)

    # switched on, but out of reach: it would have to be located to find itself within reach again
    model.constraints["contact"].withinReach = False
    with pytest.raises(Exception, match="changes with the motion"):
        explicitSolver.validateModelCapabilities(model)


#: A block, and a rigid floor pushing back the bottom nodes: fixed nodes, a gap that is stateless per stage.
RIGID_SURFACE_MODEL = """
*material, name=LinearElastic, id=linearelastic, provider=edelweiss
100.0, 0.3

*job, name=job, domain=2d
*solver, solver=NIST, name=theSolver

*modelGenerator, generator=planeRectQuad, name=gen
x0=0, y0=0, l=1.0, h=1.0, nX=2, nY=2, elType=CPE4, elProvider=edelweiss

*section, name=section1, thickness=1.0, material=linearelastic, type=plane
gen_all

*constraint, type=nodeToRigidSurfacePenalty, name=floor
nSet=gen_bottom
field=displacement
component=1
penalty=20.0
value=-0.1
direction=-1.0

*step, solver=theSolver
maxInc=1e0, minInc=1e0, maxNumInc=1, maxIter=5, stepLength=1
>>dirichlet, name=top, nSet=gen_top, field=displacement, 1=0.0, 2=0.0
"""


def test_the_explicit_static_solver_accepts_a_constraint_with_fixed_nodes(tmp_path):
    """A rigid surface the nodes of a set may not penetrate couples the same nodes throughout, and is only refreshed
    after a change of the mesh: NEST has nothing to update for it, even while it is switched on."""

    model = _runDeck(tmp_path, RIGID_SURFACE_MODEL)

    NEST({}, Journal(verbose=False)).validateModelCapabilities(model)


def test_a_contact_switched_off_for_a_step_keeps_its_multipliers_and_its_frictional_history(tmp_path):
    """The blocks are pressed together and dragged, then held for a step with the contact switched off. The contact,
    which is not part of that step, accepts no state: its augmented Lagrange multipliers would otherwise grow by the
    penalty force of its last gap in every increment, and act across the gap once it is switched on again."""

    (tmp_path / "pressedAndDragged").mkdir()
    (tmp_path / "switchedOff").mkdir()
    pressedAndDragged = _runDeck(tmp_path / "pressedAndDragged", CONTACT_MODEL + PRESS_AND_DRAG)
    switchedOff = _runDeck(tmp_path / "switchedOff", CONTACT_MODEL + PRESS_AND_DRAG + SWITCHED_OFF)
    assert pressedAndDragged.time == pytest.approx(2.0) and switchedOff.time == pytest.approx(3.0)

    history = pressedAndDragged.constraints["contact"].getRestartData()
    assert np.any(history["lambdaN"] < 0.0), "the check is vacuous without multipliers"
    assert np.any(history["tangentialForceConverged"] != 0.0), "the check is vacuous without a frictional history"

    historyAfterTheSwitchedOffStep = switchedOff.constraints["contact"].getRestartData()
    assert not switchedOff.constraints["contact"].active
    for name in ("lambdaN", "tangentialForceConverged"):
        np.testing.assert_allclose(historyAfterTheSwitchedOffStep[name], history[name], rtol=1e-12, atol=0.0)
