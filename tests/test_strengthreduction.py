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
"""

import numpy as np
import pytest

from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
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
