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
"""Verification of the Newmark-beta implicit dynamic solver (``NID``) against closed forms.

The model is ``testfiles/marmot/NID``: one Marmot ``C3D8`` with ``LinearElastic`` (``nu = 0``) as
a uniaxial bar, fixed at ``x = 0``, every lateral displacement suppressed, and a suddenly applied
end load. By symmetry the four loaded nodes move together, so the finite element model IS a
single-degree-of-freedom oscillator whose stiffness is ``K = E A / L`` and whose CONSISTENT mass is
the sum of the consistent mass matrix over the free x-dofs, ``rho * int (x/L)^2 dV = rho A L / 3``.
With ``E = 4 pi^2`` and ``rho = 3``: ``M = 1``, ``K = 4 pi^2``, ``omega = 2 pi``, period ``T = 1``.

Three independent checks, each of which fails on a distinct class of defect:

* **The exact discrete Newmark recurrence** of that SDOF system, replicated here in a dozen lines,
  must be reproduced to round-off. A wrong coefficient anywhere in the residual, the effective
  tangent, the velocity update or the initial acceleration shows up here at O(1), not O(dt^2).
* **The continuous closed form** ``u(t) = (F/K) (1 - cos omega t)`` must be approached at
  **second order**: halving ``dt`` must quarter the error. This is what a wrong initial
  acceleration (a step load entering the trapezoidal rule as though it were zero at ``t_0``)
  breaks -- it leaves an O(dt) error in the momentum, and the ratio drops to 2.
* **The discrete energy** ``1/2 M v^2 + 1/2 K u^2 - F u`` is conserved EXACTLY by the
  average-acceleration rule on a linear system, so it must stay at its initial value (zero) to
  round-off over both periods. Any drift is a bug, never a property of the scheme.

The velocity and acceleration are read through ordinary ``*fieldOutput`` blocks (``result=V``,
``result=A``), which is also what pins that those entries are published as node-field entries.
"""

import numpy as np
import pytest

from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.utils.inputfileparser import parseInputFile

E = 4.0 * np.pi**2
RHO = 3.0
MASS = RHO / 3.0  # consistent mass of the uniform tip mode: rho * A * L / 3
STIFFNESS = E  # E * A / L
OMEGA = np.sqrt(STIFFNESS / MASS)  # = 2 pi, so the period is 1
U_STATIC = 0.01
FORCE = STIFFNESS * U_STATIC
STEP_LENGTH = 2.0

BETA = 0.25
GAMMA = 0.5


def _deck(dt: float, maxNumInc: int = 100000, extra: str = "", stepLength: float = STEP_LENGTH) -> str:
    return f"""
*material, name=LinearElastic, id=bar
{E!r}, 0.0, {RHO!r}

*section, name=section1, material=bar, type=solid
gen_all

*job, name=nidjob, domain=3d
*solver, solver=NID, name=theSolver
newmarkBeta={BETA}
newmarkGamma={GAMMA}

*modelGenerator, generator=boxGen, name=gen
nX=1
nY=1
nZ=1
lX=1
lY=1
lZ=1
elType=C3D8

*fieldOutput
>>perNode, nSet=gen_right, field=displacement, result=U, name=tipU, f(x)='np.mean(x[:,0])', saveHistory=True
>>perNode, nSet=gen_right, field=displacement, result=V, name=tipV, f(x)='np.mean(x[:,0])', saveHistory=True
>>perNode, nSet=gen_right, field=displacement, result=A, name=tipA, f(x)='np.mean(x[:,0])', saveHistory=True
{extra}
*step, solver=theSolver
stepLength={stepLength!r}, startInc={dt / stepLength!r}, maxInc={dt / stepLength!r}, minInc=1e-6, maxNumInc={maxNumInc}, maxIter=25
>>options, name=theSolver, extrapolation=off
>>dirichlet, name=fixedEnd, nSet=gen_left, field=displacement, 1=0.0
>>dirichlet, name=lateral, nSet=all, field=displacement, 2=0.0, 3=0.0
>>nodeforces, name=endLoad, nSet=gen_right, field=displacement, 1={FORCE / 4.0!r}, f(t)='1'
"""


def _run(tmp_path, name: str, deck: str):
    path = tmp_path / f"{name}.inp"
    path.write_text(deck)
    inputfile = parseInputFile(str(path))
    model, fieldOutputController = finiteElementSimulation(inputfile, verbose=False, suppressPlots=True)
    return model, fieldOutputController


def _tipHistories(fieldOutputController) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Times and tip histories of u, v, a, without the sample taken at job initialisation."""

    outputs = fieldOutputController.fieldOutputs
    t = np.asarray(outputs["tipU"].getTimeHistory(), dtype=float)
    u = np.asarray(outputs["tipU"].getResultHistory(), dtype=float).ravel()
    v = np.asarray(outputs["tipV"].getResultHistory(), dtype=float).ravel()
    a = np.asarray(outputs["tipA"].getResultHistory(), dtype=float).ravel()

    # The controller samples once at job initialisation (t = 0, everything zero) and then once per
    # converged increment. The time stepper may close the step with a round-off remainder (400
    # increments of 0.0025 leave 2e-14), which the solver skips with the state kept; drop that
    # duplicate sample so the histories are the uniform-dt sequence the closed forms describe.
    steps = np.diff(t)
    regular = np.r_[t[0] > 0.0, steps > 0.5 * np.median(steps)]
    return t[regular], u[regular], v[regular], a[regular]


def _exactContinuous(t: np.ndarray) -> np.ndarray:
    return U_STATIC * (1.0 - np.cos(OMEGA * t))


def _exactDiscreteNewmark(dt: float, nIncrements: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The Newmark-beta solution of ``M a + K u = F`` from rest, with the initial acceleration from
    equilibrium at ``t_0`` -- ``a_0 = F / M`` for a step load -- written in the same
    increment/predictor form the solver uses."""

    u, v, a = 0.0, 0.0, FORCE / MASS
    us, vs, accs = [], [], []
    for _ in range(nIncrements):
        aPredictor = -v / (BETA * dt) - (0.5 / BETA - 1.0) * a
        KEff = STIFFNESS + MASS / (BETA * dt * dt)
        du = (FORCE - STIFFNESS * u - MASS * aPredictor) / KEff
        aNew = aPredictor + du / (BETA * dt * dt)
        vNew = v + dt * ((1.0 - GAMMA) * a + GAMMA * aNew)
        u, v, a = u + du, vNew, aNew
        us.append(u)
        vs.append(v)
        accs.append(a)
    return np.array(us), np.array(vs), np.array(accs)


def test_matches_exact_discrete_newmark_recurrence(tmp_path):
    dt = 0.02
    _, fieldOutputController = _run(tmp_path, "nid_discrete", _deck(dt))
    t, u, v, a = _tipHistories(fieldOutputController)

    nIncrements = int(round(STEP_LENGTH / dt))
    assert len(t) == nIncrements
    np.testing.assert_allclose(t, dt * np.arange(1, nIncrements + 1), rtol=0.0, atol=1e-12)

    uRef, vRef, aRef = _exactDiscreteNewmark(dt, nIncrements)

    # round-off: the Newton loop solves a LINEAR system, so its solve is exact; a wrong coefficient
    # anywhere would show at O(1e-3) relative, the size of one increment's change
    np.testing.assert_allclose(u, uRef, rtol=0.0, atol=1e-10 * U_STATIC)
    np.testing.assert_allclose(v, vRef, rtol=0.0, atol=1e-10 * U_STATIC * OMEGA)
    np.testing.assert_allclose(a, aRef, rtol=0.0, atol=1e-10 * U_STATIC * OMEGA**2)


def test_discrete_energy_is_conserved_exactly(tmp_path):
    dt = 0.02
    _, fieldOutputController = _run(tmp_path, "nid_energy", _deck(dt))
    _, u, v, _ = _tipHistories(fieldOutputController)

    # kinetic + strain - work of the constant load; zero at t = 0 (at rest, unloaded)
    energy = 0.5 * MASS * v**2 + 0.5 * STIFFNESS * u**2 - FORCE * u

    energyScale = FORCE * U_STATIC
    assert np.max(np.abs(energy)) < 1e-11 * energyScale


def test_converges_at_second_order_to_the_closed_form(tmp_path):
    timeIncrements = [0.04, 0.02, 0.01]
    errors = []
    for dt in timeIncrements:
        _, fieldOutputController = _run(tmp_path, f"nid_dt_{dt}", _deck(dt))
        t, u, _, _ = _tipHistories(fieldOutputController)
        errors.append(float(np.max(np.abs(u - _exactContinuous(t)))))

    # The trapezoidal rule has no amplitude error, only a period elongation of (omega dt)^2 / 12
    # per period, which over the two periods at dt = 0.04 accumulates to a phase lag of
    # 4 pi (omega dt)^2 / 12 = 0.066 rad, i.e. an error of about 6.6e-2 of the amplitude. Pin the
    # magnitude too, from both sides: a comparison accidentally made against something other than
    # the continuous closed form would either be far too good or far too bad at a "convergent" rate.
    assert 3e-2 * U_STATIC < errors[0] < 1e-1 * U_STATIC, errors

    ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
    for ratio in ratios:
        assert 3.5 < ratio < 4.5, "expected quadratic error reduction, got ratios {:}".format(ratios)


def test_period_is_reproduced(tmp_path):
    dt = 0.01
    _, fieldOutputController = _run(tmp_path, "nid_period", _deck(dt))
    t, u, v, _ = _tipHistories(fieldOutputController)

    # after one period the oscillator is back at rest at the origin, up to the O(dt^2) phase error
    atOnePeriod = np.argmin(np.abs(t - 1.0))
    assert abs(t[atOnePeriod] - 1.0) < 1e-12
    assert abs(u[atOnePeriod]) < 1e-3 * U_STATIC
    assert abs(v[atOnePeriod]) < 1e-2 * U_STATIC * OMEGA
    # and at half a period it is at the far turning point, 2 u_static
    atHalfPeriod = np.argmin(np.abs(t - 0.5))
    assert abs(u[atHalfPeriod] - 2.0 * U_STATIC) < 1e-3 * U_STATIC


def test_the_published_force_is_the_internal_inertial_and_damping_force(tmp_path):
    """``P = f_int + M a + C v`` (documentation page *Node field results*). At the loaded end, which no support holds,
    it is the end load at every increment, while the internal force alone, ``K u``, swings between 0 and twice the load.
    At the fixed end it is the reaction, the inertia of the consistent mass there included: the internal forces of the
    bar sum to zero, so the two ends add up to its inertial force, the row sums of the mass (``rho V / 8`` per node)
    times the acceleration of the four loaded nodes. The zero-length increment of a second step integrates nothing
    and keeps the force the first step ended with."""

    dt = 0.02
    sums = (
        ">>perNode, nSet=gen_right, field=displacement, result=P, name=tipP, f(x)='np.sum(x[:,0])', saveHistory=True\n"
        ">>perNode, nSet=gen_left, field=displacement, result=P, name=fixedP, f(x)='np.sum(x[:,0])', saveHistory=True\n"
    )
    secondStep = f"""
*step, solver=theSolver
stepLength={2 * dt!r}, startInc=0.5, maxInc=0.5, minInc=1e-6, maxNumInc=10, maxIter=25
"""
    _, fieldOutputController = _run(tmp_path, "nid_force", _deck(dt, extra=sums) + secondStep)
    outputs = fieldOutputController.fieldOutputs
    t = np.asarray(outputs["tipP"].getTimeHistory(), dtype=float)
    tipP = np.asarray(outputs["tipP"].getResultHistory(), dtype=float).ravel()
    fixedP = np.asarray(outputs["fixedP"].getResultHistory(), dtype=float).ravel()
    a = np.asarray(outputs["tipA"].getResultHistory(), dtype=float).ravel()

    # every sample after the start, the skipped increments included: they keep the state, and with it the force
    moving = t > 0.0
    assert np.count_nonzero(t > STEP_LENGTH + 1e-9) == 2, "the second step has two increments of motion"
    np.testing.assert_allclose(tipP[moving], FORCE, rtol=1e-10)
    np.testing.assert_allclose(tipP[moving] + fixedP[moving], 4 * (RHO / 8.0) * a[moving], rtol=0.0, atol=1e-10 * FORCE)

    # the last increment of the first step, the zero-length increment of the second (and a skipped round-off
    # remainder of the first, if the stepper leaves one)
    atTheSwitch = np.flatnonzero(np.isclose(t, STEP_LENGTH, rtol=0.0, atol=1e-9))
    assert len(atTheSwitch) >= 2
    assert fixedP[atTheSwitch[0]] != 0.0
    np.testing.assert_array_equal(tipP[atTheSwitch], tipP[atTheSwitch[0]])
    np.testing.assert_array_equal(fixedP[atTheSwitch], fixedP[atTheSwitch[0]])


def test_a_quasi_static_step_leaves_the_model_at_rest(tmp_path):
    """A quasi-static step between two dynamic ones (documentation page *Node field results*). The first ``NID`` step
    ends a quarter period after the load was applied, at the static deflection with the largest velocity. ``NIST``
    then holds the load: its solution is the static one, and the state it accepts is at rest, ``V = A = 0`` on every
    node. The second ``NID`` step starts from there -- at rest in equilibrium, its acceleration from equilibrium zero
    -- and stays at the static deflection. Had the quasi-static step left the velocity of the first dynamic step in
    the model, the bar would swing about the static deflection with an amplitude of about that deflection."""

    dt = 0.025
    quarterPeriod = 0.25
    endOfQuasiStaticStep = quarterPeriod + 1.0
    extra = (
        ">>perNode, nSet=all, field=displacement, result=V, name=maxV, f(x)='np.max(np.abs(x))', saveHistory=True\n"
        ">>perNode, nSet=all, field=displacement, result=A, name=maxA, f(x)='np.max(np.abs(x))', saveHistory=True\n"
        "\n*solver, solver=NIST, name=quasiStatic\n"
    )
    deck = (
        _deck(dt, extra=extra, stepLength=quarterPeriod)
        + f"""
*step, solver=quasiStatic
stepLength=1.0, startInc=0.5, maxInc=0.5, minInc=1e-6, maxNumInc=10, maxIter=25

*step, solver=theSolver
stepLength={quarterPeriod!r}, startInc={dt / quarterPeriod!r}, maxInc={dt / quarterPeriod!r}, minInc=1e-6, maxNumInc=100, maxIter=25
"""
    )
    _, fieldOutputController = _run(tmp_path, "nid_after_nist_after_nid", deck)
    outputs = fieldOutputController.fieldOutputs
    t = np.asarray(outputs["tipU"].getTimeHistory(), dtype=float)
    u, v, maxV, maxA = (
        np.asarray(outputs[name].getResultHistory(), dtype=float).ravel() for name in ("tipU", "tipV", "maxV", "maxA")
    )

    # the last increment of the first dynamic step ends moving: the defect would show
    endOfFirstStep = np.flatnonzero(np.isclose(t, quarterPeriod, rtol=0.0, atol=1e-9))[0]
    assert abs(v[endOfFirstStep]) > 0.5 * U_STATIC * OMEGA

    # both increments of the quasi-static step (the zero-length increment of NID that follows, at the same time, keeps
    # this state; NID computes the acceleration from equilibrium at the start of its first increment of motion)
    quasiStatic = np.flatnonzero((t > quarterPeriod + 1e-9) & (t < endOfQuasiStaticStep - 1e-9))
    quasiStatic = np.r_[quasiStatic, np.flatnonzero(np.isclose(t, endOfQuasiStaticStep, rtol=0.0, atol=1e-9))[0]]
    assert len(quasiStatic) == 2
    np.testing.assert_allclose(u[quasiStatic], U_STATIC, rtol=1e-8)
    np.testing.assert_array_equal(maxV[quasiStatic], 0.0)
    np.testing.assert_array_equal(maxA[quasiStatic], 0.0)

    secondDynamic = t > endOfQuasiStaticStep + 1e-9
    assert np.count_nonzero(secondDynamic) >= int(round(quarterPeriod / dt))
    np.testing.assert_allclose(u[secondDynamic], U_STATIC, rtol=1e-8)
    np.testing.assert_allclose(v[secondDynamic], 0.0, rtol=0.0, atol=1e-8 * U_STATIC * OMEGA)


@pytest.mark.parametrize("beta", [0.0, -0.25])
def test_refuses_non_positive_beta(tmp_path, beta):
    deck = _deck(0.02).replace(f"newmarkBeta={BETA}", f"newmarkBeta={beta}")
    with pytest.raises(ValueError, match="newmarkBeta"):
        _run(tmp_path, "nid_bad_beta", deck)
