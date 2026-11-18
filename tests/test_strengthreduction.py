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
"""Strength reduction with the arc length solver: under an active arc length controller, the
``changeMaterialProperty`` actions of a step follow the load parameter instead of the step time.

A von Mises cube under a constant uniaxial pressure ``p`` is brought to yielding by reducing its
yield stress to ``fy0 / (1 + lambda)``. With linear hardening ``H``, equilibrium requires
``p = fy0 / (1 + lambda) + H * kappa``, which the load parameter and the equivalent plastic strain
found by the solver have to satisfy at any stage of plastic flow.

The parts a strength reduction relies on are tested on their own as well: a relative change of a material
property, the weight of the load parameter in the indirect displacement control, and the yield check of the
von Mises material for a strain increment of zero.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.materials.vonmises.vonmises import VonMisesMaterial
from edelweissfe.solvers.nonlinearimplicitstaticparallelarclength import NISTPArcLength
from edelweissfe.stepactions.base.indirectdisplacementcontrolbase import (
    IndirectDisplacementControlBase,
)
from edelweissfe.stepactions.changematerialproperty import (
    StepAction as ChangeMaterialProperty,
)
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.inputfileparser import parseInputFile

FY0 = 550.0
PRESSURE = 400.0

DECK = """
*material, name=VonMises, id=myMaterial, provider=edelweiss
210000, 0.3, {fy0}, {hardening}, 0, 0

*job, name=job, domain=3d
*solver, solver={solver}, name=theSolver

*modelGenerator, generator=boxGen, name=gen
nX=1
nY=1
nZ=1
lX=10
lY=10
lZ=10
elType=C3D8
elProvider=edelweiss

*section, name=section1, material=myMaterial, type=solid
all

*fieldOutput
>>perElement, elSet=all, result=kappa, name=kappa, quadraturePoint=0:8

*step, solver=theSolver
maxInc=5e-1, minInc=1e-6, maxNumInc=100, maxIter=25, stepLength=1
>>dirichlet, name=x0u, nSet=gen_left, field=displacement, 1=0.
>>dirichlet, name=y0u, nSet=gen_bottom, field=displacement, 2=0.
>>dirichlet, name=z0u, nSet=gen_back, field=displacement, 3=0.
>>nodeforces, name=top, nSet=gen_top, field=displacement, 2={nodeForce}

*step, solver=theSolver
maxInc=5e-2, minInc=1e-6, maxNumInc=1000, maxIter=25, stepLength=1
{controller}
{change}
"""

# the settlement of a top node plus cLambda times the load parameter grows by L over the step
CONTROLLER = """>>options, name=theSolver, extrapolation=off, arcLengthController=settlement
>>indirectcontrol, name=settlement, dof1='list(model.nodeSets["gen_top"])[0].fields["displacement"]', cVector1='0, -1, 0', dof2='list(model.nodeSets["gen_bottom"])[0].fields["displacement"]', cVector2='0, 0, 0', L=0.1, absolute=False, cLambda=0.1"""

ABSOLUTE = f">>changeMaterialProperty, name=reduce, material=myMaterial, index=2, f(t)='{FY0}/(1+t)'"
RELATIVE = ">>changeMaterialProperty, name=reduce, material=myMaterial, index=2, relative=True, f(t)='1/(1+t)'"


def _run(tmp_path, hardening: float, change: str, solver: str = "NISTPArcLength", controller: str = CONTROLLER):
    path = tmp_path / "job.inp"
    path.write_text(
        DECK.format(
            fy0=FY0,
            hardening=hardening,
            # the pressure on the top face of 10 x 10, lumped to its four nodes
            nodeForce=-PRESSURE * 100.0 / 4.0,
            solver=solver,
            controller=controller,
            change=change,
        )
    )

    return finiteElementSimulation(parseInputFile(str(path)), verbose=False, suppressPlots=True)


def _kappa(model) -> np.ndarray:
    return np.hstack(
        [
            np.copy(el.getResultArray("kappa", qp, getPersistentView=False)).flatten()
            for el in model.elements.values()
            for qp in range(8)
        ]
    )


@pytest.mark.parametrize("hardening", [1000.0, 0.0])
def test_load_parameter_satisfies_equilibrium_at_yield(tmp_path, hardening):
    model, _ = _run(tmp_path, hardening, ABSOLUTE)

    Lambda = model.additionalParameters["arc length parameter"]
    kappa = _kappa(model)

    assert kappa.min() > 1e-3, "the cube did not yield"
    assert np.allclose(kappa, kappa[0], rtol=1e-8), "the cube is not in a homogeneous state"
    assert FY0 / (1.0 + Lambda) + hardening * kappa[0] == pytest.approx(PRESSURE, rel=1e-8)


def test_relative_change_equals_absolute_change(tmp_path):
    absolute, _ = _run(tmp_path, 1000.0, ABSOLUTE)
    relative, _ = _run(tmp_path, 1000.0, RELATIVE)

    # equal up to the round-off of the two property functions, carried through the Newton iterations
    assert relative.additionalParameters["arc length parameter"] == pytest.approx(
        absolute.additionalParameters["arc length parameter"], rel=1e-9
    )
    assert np.allclose(_kappa(relative), _kappa(absolute), rtol=1e-9, atol=0.0)


def test_relative_change_in_time_keeps_the_values_of_the_elements(tmp_path):
    # without an arc length controller the property follows the step time: reduced by 20 % at t = 1
    model, _ = _run(
        tmp_path,
        1000.0,
        ">>changeMaterialProperty, name=reduce, material=myMaterial, index=2, relative=True, f(t)='1-0.2*t'",
        solver="NIST",
        controller="",
    )

    assert _kappa(model).max() == 0.0, "a yield stress of 440 must not be reached by a pressure of 400"
    for el in model.elements.values():
        assert el.material.materialProperties[2] == pytest.approx(0.8 * FY0, rel=1e-12)


def test_line_search_evaluates_its_trial_states_at_their_load_parameter(tmp_path, monkeypatch):
    # a line search from the second iteration on: every trial state sets the properties at its own load parameter
    numberOfTrialStates = 0
    setTrialState = NISTPArcLength.setTrialState

    def countingSetTrialState(self, dU, ddU_0, stepLength):
        nonlocal numberOfTrialStates
        numberOfTrialStates += 1
        return setTrialState(self, dU, ddU_0, stepLength)

    monkeypatch.setattr(NISTPArcLength, "setTrialState", countingSetTrialState)

    lineSearch = "extrapolation=off, lineSearch=True, lineSearchAfterIterations=0, lineSearchEveryIterations=1"
    model, _ = _run(tmp_path, 1000.0, ABSOLUTE, controller=CONTROLLER.replace("extrapolation=off", lineSearch))

    Lambda = model.additionalParameters["arc length parameter"]
    kappa = _kappa(model)

    assert numberOfTrialStates > 0, "no line search took place"
    assert kappa.min() > 1e-3, "the cube did not yield"
    assert FY0 / (1.0 + Lambda) + 1000.0 * kappa[0] == pytest.approx(PRESSURE, rel=1e-8)


def test_loads_and_properties_follow_the_load_parameter_together(tmp_path):
    # the load grows by 25 % per unit of the load parameter while the yield stress is reduced: the reference load
    # and the derivative of the internal forces both enter the second right-hand side
    increase = 0.25
    # a re-declared node force is the increment of its step
    raisedLoad = "\n>>nodeforces, name=top, 2={:}".format(-increase * PRESSURE * 100.0 / 4.0)
    model, _ = _run(tmp_path, 1000.0, ABSOLUTE + raisedLoad)

    Lambda = model.additionalParameters["arc length parameter"]
    kappa = _kappa(model)

    assert kappa.min() > 1e-3, "the cube did not yield"
    assert FY0 / (1.0 + Lambda) + 1000.0 * kappa[0] == pytest.approx((1.0 + increase * Lambda) * PRESSURE, rel=1e-8)
    # yielding starts earlier than under the constant load, where it starts at FY0 / PRESSURE - 1
    assert 0.0 < Lambda < FY0 / PRESSURE - 1.0 + 0.1


def test_load_alone_follows_the_load_parameter(tmp_path):
    # no property is driven: the elastic cube settles by the load increment times the load parameter, and the
    # controller's equation, settlement + cLambda * lambda = L, gives the load parameter
    increment = 0.25 * PRESSURE
    model, _ = _run(tmp_path, 1000.0, ">>nodeforces, name=top, 2={:}".format(-increment * 100.0 / 4.0))

    Lambda = model.additionalParameters["arc length parameter"]
    settlementPerLoadParameter = increment * 10.0 / 210000.0

    assert _kappa(model).max() == 0.0
    assert Lambda == pytest.approx(0.1 / (settlementPerLoadParameter + 0.1), rel=1e-10)


def test_without_controller_the_properties_follow_the_step_time(tmp_path):
    # the arc length solver without a controller reduces the yield stress in time, as NIST does
    change = ">>changeMaterialProperty, name=reduce, material=myMaterial, index=2, f(t)='550/(1+0.5*t)'"
    arcLength, _ = _run(tmp_path, 1000.0, change, controller=">>options, name=theSolver, extrapolation=off")
    nist, _ = _run(tmp_path, 1000.0, change, solver="NIST", controller=">>options, name=theSolver, extrapolation=off")

    assert _kappa(nist).min() > 0.0, "a yield stress of 367 must be reached by a pressure of 400"
    assert np.allclose(_kappa(arcLength), _kappa(nist), rtol=1e-12, atol=0.0)


class _Element:
    """An element as the Marmot elements hold their properties, in a vector of their own."""

    def __init__(self, properties):
        self._materialProperties = np.array(properties, dtype=float)

    def updateMaterialProperty(self, index, value):
        self._materialProperties[index] = value


class _ElementWithMaterialInstance:
    """An element as the edelweiss elements hold their properties, in their material instance."""

    def __init__(self, properties):
        self.material = SimpleNamespace(materialProperties=np.array(properties, dtype=float))

    def updateMaterialProperty(self, index, value):
        self.material.materialProperties[index] = value


def _relativeChange(elements, f_t=lambda t: 1.0 / (1.0 + t)):
    """A relative change of property 1 of a material with the given elements, and what it logs."""
    material = {"name": "myMaterial", "properties": np.array([1.0, 10.0])}
    model = SimpleNamespace(
        sections={
            "mine": SimpleNamespace(material=material, elSets=[elements]),
            "other": SimpleNamespace(material={"name": "otherMaterial"}, elSets=[[_Element([1.0, 99.0])]]),
        }
    )
    messages = []
    journal = SimpleNamespace(message=lambda message, *args, **kwargs: messages.append(message))
    action = ChangeMaterialProperty("reduce", material, 1, f_t, model, journal, relative=True)
    return action, model, material, messages


def test_relative_change_scales_the_value_of_every_element():
    marmotLike, edelweissLike = _Element([1.0, 10.0]), _ElementWithMaterialInstance([1.0, 30.0])
    action, model, material, _ = _relativeChange([marmotLike, edelweissLike])

    action.applyAtTime(model, 1.0)
    assert marmotLike._materialProperties[1] == pytest.approx(5.0)
    assert edelweissLike.material.materialProperties[1] == pytest.approx(15.0)
    # the model's record of the material follows, and other properties and other materials are left alone
    assert material["properties"][1] == pytest.approx(5.0)
    assert marmotLike._materialProperties[0] == 1.0
    assert model.sections["other"].elSets[0][0]._materialProperties[1] == 99.0

    # the factor refers to the values at the start of the step, not to the current ones
    action.applyAtTime(model, 3.0)
    assert marmotLike._materialProperties[1] == pytest.approx(2.5)
    assert edelweissLike.material.materialProperties[1] == pytest.approx(7.5)
    action.applyAtTime(model, 0.0)
    assert marmotLike._materialProperties[1] == pytest.approx(10.0)


def test_relative_change_at_increment_start_logs_the_factor():
    element = _Element([1.0, 10.0])
    action, model, _, messages = _relativeChange([element])

    action.applyAtIncrementStart(model, TimeStep(1, 0.5, 0.5, 0.5, 1.0, 1.0))

    assert element._materialProperties[1] == pytest.approx(5.0)
    assert "by the factor 0.5" in messages[-1]


def test_relative_change_refers_to_the_start_of_the_step_it_is_declared_in():
    element = _Element([1.0, 10.0])
    action, model, _, _ = _relativeChange([element])

    action.applyAtTime(model, 1.0)
    action.applyAtStepEnd(model)

    # switched off at the end of its step
    action.applyAtTime(model, 3.0)
    action.applyAtIncrementStart(model, TimeStep(1, 1.0, 1.0, 1.0, 3.0, 3.0))
    assert element._materialProperties[1] == pytest.approx(5.0)

    # declared again in a later step: relative to the value reached, with the function kept
    action.updateStepAction()
    action.applyAtTime(model, 1.0)
    assert element._materialProperties[1] == pytest.approx(2.5)


def test_relative_change_refuses_a_changed_set_of_elements():
    elements = [_Element([1.0, 10.0])]
    action, model, _, _ = _relativeChange(elements)
    action.applyAtTime(model, 1.0)

    elements.append(_Element([1.0, 10.0]))
    with pytest.raises(NotImplementedError, match="change of the elements"):
        action.applyAtTime(model, 2.0)


class _Controller(IndirectDisplacementControlBase):
    """An indirect displacement control of the degrees of freedom 0 and 2."""

    cVector = np.array([1.0, -2.0])

    def controlledDofIndices(self, dofManager):
        return np.array([0, 2])

    def controlledValuesOfModel(self):
        return np.array([0.3, 0.1])

    def finishIncrement(self, U, dU, dLambda, timeStep, dofManager):
        pass


@pytest.mark.parametrize("cLambda", [0.0, 0.5])
def test_the_weight_of_the_load_parameter_enters_the_controllers_equation(cLambda):
    controller = _Controller("control", absolute=False, cLambda=cLambda)
    controller.declareTarget(2.0)

    rng = np.random.default_rng(1)
    dU, ddU_0, ddU_f = rng.standard_normal((3, 4))
    dLambda = 0.7
    timeStep = TimeStep(1, 0.1, 0.4, 0.1, 0.4, 0.4)

    ddLambda = controller.computeDDLambda(dU, ddU_0, ddU_f, timeStep, None, dLambda)

    idcs = controller.controlledDofIndices(None)
    controlled = controller.cVector.dot((dU + ddU_0 + ddLambda * ddU_f)[idcs]) + cLambda * (dLambda + ddLambda)
    assert controlled == pytest.approx(0.1 * 2.0, rel=1e-12)


def test_the_weight_keeps_the_equation_regular_without_response_of_the_displacements():
    # no point yields: the reference solve moves nothing, and the load parameter advances by L / cLambda over the step
    controller = _Controller("control", absolute=False, cLambda=0.5)
    controller.declareTarget(2.0)
    zero = np.zeros(4)

    ddLambda = controller.computeDDLambda(zero, zero, zero, TimeStep(1, 0.1, 0.1, 0.1, 0.1, 0.1), None, 0.0)

    assert ddLambda == pytest.approx(0.1 * 2.0 / 0.5, rel=1e-14)


def _vonMises(yieldStress):
    material = VonMisesMaterial(np.array([210000.0, 0.3, yieldStress, 1000.0, 0.0, 0.0]))
    kappa = np.zeros(1)
    material.assignCurrentStateVars(kappa)
    return material, kappa


def _misesStress(stress):
    deviator = stress - np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]) * stress[:3].mean()
    return np.sqrt(1.5 * (deviator[:3] @ deviator[:3] + 2.0 * deviator[3:] @ deviator[3:]))


def test_von_mises_at_rest_stays_elastic_inside_the_yield_surface():
    material, kappa = _vonMises(FY0)
    stress = np.array([0.0, -PRESSURE, 0.0, 0.0, 0.0, 0.0])
    tangent = np.zeros((6, 6))

    material.computeStress(stress, tangent, np.zeros(6), np.zeros(2), 1.0)

    assert np.array_equal(stress, [0.0, -PRESSURE, 0.0, 0.0, 0.0, 0.0])
    assert kappa[0] == 0.0
    assert np.array_equal(tangent, material.elasticityMatrix())


def test_von_mises_at_rest_yields_if_the_yield_surface_has_shrunk_onto_its_stress():
    # the yield stress reduced below the stress the material carries, without any strain increment
    material, kappa = _vonMises(350.0)
    stress = np.array([0.0, -PRESSURE, 0.0, 0.0, 0.0, 0.0])
    tangent = np.zeros((6, 6))

    material.computeStress(stress, tangent, np.zeros(6), np.zeros(2), 1.0)

    assert kappa[0] > 0.0
    assert _misesStress(stress) == pytest.approx(350.0 + 1000.0 * kappa[0], rel=1e-10)
    assert not np.allclose(tangent, material.elasticityMatrix())
