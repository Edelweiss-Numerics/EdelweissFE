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
#  ---------------------------------------------------------------------
"""Live h-adaptivity under the Newmark-beta implicit dynamic solver (``NID``).

Four things have to hold for a refinement mid-step to be more than "it did not crash", and there
is one test for each:

* **The kinematics ride the warm start.** ``V`` and ``A`` are node-field entries, so the nodes a
  refinement creates get them by the same isoparametric interpolation that already carries ``U``
  (``hadaptivity.WARM_STARTED_NODE_FIELD_ENTRIES``). Both decks below prescribe a motion whose
  velocity field is a **polynomial of degree at most one in space**, which a 20-node hexahedron
  reproduces *exactly* -- so the value on every new node is known in closed form and is asserted
  against it, rather than merely being "not zero".
* **Conservation.** Total mass is an identity of the assembly and is asserted by the solver
  itself. Linear momentum and kinetic energy are exact for a velocity field of degree at most one
  (partition of unity for the first, plus an exactly-integrated quadratic integrand on an affine
  mesh for the second), which is what makes this a bug detector rather than a plausibility check
  -- the same argument, and the same role in the test ladder, as
  ``testfiles/marmot/NEDLiveAMR`` plays for the explicit solver.
* **The acceleration is re-equilibrated.** An interpolated acceleration is not in equilibrium with
  the mass that was just reassembled on the refined mesh, so the increment after a topology change
  re-solves it from equilibrium. That has to fire exactly on those increments and nowhere else.
* **Nothing else changed.** The highest-risk part of this feature is not the adaptive path at all:
  it is that the re-arm might fire on an ordinary multi-step deck that never refines anything --
  a step boundary rebuilds the ``DofManager`` too. That is pinned by running a two-step deck twice,
  once as shipped and once with the re-arm suppressed, and requiring the two to agree bit for bit.
"""

import numpy as np
import pytest

from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.models.femodel import FEModel
from edelweissfe.solvers.nonlinearimplicitdynamic import NonlinearImplicitDynamic
from edelweissfe.utils.inputfileparser import parseInputFile

#: The unpatched methods, captured once at import -- a test that runs the solver more than once must
#: wrap THESE rather than whatever is currently on the class, or the wrappers chain. Same reason as
#: in test_explicit_operator_reuse.py.
_TRUE_ENSURE = NonlinearImplicitDynamic._updateNewmarkSystem
_TRUE_REPORT = NonlinearImplicitDynamic.reportTopologyChangeConservation
_TRUE_INITIAL_ACCELERATION = NonlinearImplicitDynamic._computeInitialAcceleration

E = 20000.0
RHO = 2.0e-3
DT = 0.1
X_MOTION = 0.02


def _deck(
    *,
    modifier: str = "",
    analyticalField: bool = False,
    steps: str | None = None,
    nX: int = 1,
    lX: float = 2.0,
    maxNumInc: int = 5,
) -> str:
    """A box of ``C3D20R`` with every node's x-motion prescribed, so the displacement field -- and
    with it the velocity and the acceleration -- is a polynomial of degree at most one in space.

    Without ``analyticalField`` the prescribed value is the same on every node: a rigid translation,
    the uniform-velocity case the explicit solver's own live-adaptivity check uses. With it, the
    value is scaled by the node's ``x``: a uniform extension, whose velocity field is linear rather
    than constant -- a strictly harder case for the interpolation, and still exact.
    """

    field = (
        """
*analyticalField, name=linearInX, type=scalarExpression
"f(x,y,z)"="x"
"""
        if analyticalField
        else ""
    )
    dirichletField = ", analyticalField=linearInX" if analyticalField else ""

    steps = (
        steps
        or f"""
*step, solver=theSolver
stepLength=1.0, startInc={DT}, maxInc={DT}, minInc=1e-6, maxNumInc={maxNumInc}, maxIter=25
>>options, name=theSolver, extrapolation=off
>>dirichlet, name=motion, nSet=all, field=displacement, 1={X_MOTION}, 2=0.0, 3=0.0{dirichletField}
"""
    )

    return f"""
*material, name=LinearElastic, id=bar
{E!r}, 0.0, {RHO!r}

*section, name=section1, material=bar, type=solid
gen_all

*job, name=nidAmrJob, domain=3d
*solver, solver=NID, name=theSolver
newmarkBeta=0.25
newmarkGamma=0.5

*modelGenerator, generator=boxGen, name=gen
nX={nX}
nY=1
nZ=1
lX={lX!r}
lY=1.0
lZ=1.0
elType=C3D20R
{modifier}{field}
*fieldOutput
>>perNode, elSet=gen_all, field=displacement, result=U, name=displacement
{steps}"""


#: Refine everything, twice. The first pass lands on the first REAL increment -- the zero-length one
#: the stepper yields first consumes the modifier's initial-call latch, so a live marker fires right
#: after it -- which is still the first build of the Newmark system and therefore not a re-arm. The
#: second pass, one increment later, is the genuine mid-flight event this file is about: by then the
#: velocity is non-zero, so the conservation statements have something to say.
_REFINE_EVERYTHING_TWICE = """
*modelModifier, type=hAdaptivity, name=amr
>>marker, type=elementSet, elSet=gen_all, initialOnly=False
refineElSet=gen_all
maxLevel=2
"""

#: Refine only what touches the fixed end, twice. Partial refinement, so the 2:1 interface carries
#: hanging-node multi-point constraints -- the path the uniform refinement above never reaches.
_REFINE_THE_FIXED_END_TWICE = """
*modelModifier, type=hAdaptivity, name=amr
>>marker, type=nodeSet, nSet=gen_left, initialOnly=False
refineElSet=gen_all
maxLevel=2
"""


#: A genuine dynamic state to refine into: only the two end faces are prescribed, so the interior
#: carries a real, non-uniform acceleration -- unlike the rigidly translated decks above, whose
#: every degree of freedom is prescribed and whose re-equilibrated acceleration is therefore zero.
_CANTILEVER_STEP = f"""
*step, solver=theSolver
stepLength=1.0, startInc={DT}, maxInc={DT}, minInc=1e-6, maxNumInc=5, maxIter=25
>>options, name=theSolver, extrapolation=off
>>dirichlet, name=fixedEnd, nSet=gen_left, field=displacement, 1=0.0, 2=0.0, 3=0.0
>>dirichlet, name=pull, nSet=gen_right, field=displacement, 1={X_MOTION}
"""


class _Recorder:
    """What the solver did, recorded from its own methods.

    Attributes
    ----------
    conservation
        One ``(before, after)`` pair per reported topology change.
    initialAccelerations
        The step time at which each equilibrium solve for the acceleration was started.
    builds
        One entry per (re)built Newmark system: its DofManager and the interpolated velocity and
        acceleration it was built with, sampled BEFORE any re-equilibration, as
        ``(dofManager, [(node, component, V, A)])``. The manager is kept, not merely identified:
        CPython reuses the address of a collected object, so ``id()`` of a discarded manager can
        equal ``id()`` of its successor.
    """

    def __init__(self):
        self.conservation = []
        self.initialAccelerations = []
        self.builds = []


def _instrument(monkeypatch, suppressRearm: bool = False) -> _Recorder:
    """Wrap the three solver methods this file makes assertions about.

    ``suppressRearm`` restores ``_initialAccelerationPending`` to whatever it was on entry to
    ``_updateNewmarkSystem``, which is exactly and only the behaviour this change added -- the
    disarming happens in ``solveIncrement``, not here. So a run with it set is the shipped solver
    minus this feature, and nothing else.
    """

    recorder = _Recorder()

    def recordingEnsure(self, model, stepActions):
        previousSystem = self._newmarkSystem
        armedOnEntry = self._initialAccelerationPending

        system = _TRUE_ENSURE(self, model, stepActions)

        if suppressRearm:
            self._initialAccelerationPending = armedOnEntry

        if system is not previousSystem:
            samples = []
            V = np.asarray(system.V)
            A = np.asarray(system.A)
            for fieldName in system.dynamicFields:
                indices = system.dofManager.idcsOfFieldsInDofVector[fieldName]
                dimension = model.nodeFields[fieldName].dimension
                for offset, index in enumerate(range(indices.start, indices.stop)):
                    samples.append(
                        (
                            system.dofManager.getNodeForIndexInDofVector(index),
                            offset % dimension,
                            V[index],
                            A[index],
                        )
                    )
            recorder.builds.append((system.dofManager, samples))

        return system

    def recordingReport(self, before, after):
        recorder.conservation.append((before, after))
        return _TRUE_REPORT(self, before, after)

    def recordingInitialAcceleration(self, system, U_n, stepActions, model, timeStep):
        recorder.initialAccelerations.append(float(timeStep.stepTime - timeStep.timeIncrement))
        return _TRUE_INITIAL_ACCELERATION(self, system, U_n, stepActions, model, timeStep)

    monkeypatch.setattr(NonlinearImplicitDynamic, "_updateNewmarkSystem", recordingEnsure)
    monkeypatch.setattr(NonlinearImplicitDynamic, "reportTopologyChangeConservation", recordingReport)
    monkeypatch.setattr(NonlinearImplicitDynamic, "_computeInitialAcceleration", recordingInitialAcceleration)

    return recorder


def _run(tmp_path, name: str, deck: str):
    path = tmp_path / f"{name}.inp"
    path.write_text(deck)
    return finiteElementSimulation(parseInputFile(str(path)), verbose=False, suppressPlots=True)


def _finalState(model):
    field = model.nodeFields["displacement"]
    return np.asarray(field["U"]), np.asarray(field["V"]), np.asarray(field["A"])


def _hangingSlaveNodes(model) -> int:
    return sum(len(constraint.claimedSlaveNodes()) for constraint in model.multiPointConstraints.values())


def test_the_mesh_really_refines_twice_and_the_run_converges(tmp_path, monkeypatch):
    """The premise every other test in this file rests on. A deck that quietly never refined --
    a retired marker, a maxLevel already reached -- would make all of them pass on an unexercised
    code path."""

    recorder = _instrument(monkeypatch)
    model, _ = _run(tmp_path, "amr_runs", _deck(modifier=_REFINE_EVERYTHING_TWICE))

    # one element -> 8 -> 64, and the parents are gone
    assert len(model.elements) == 64
    assert len(model.topologyHistory) == 2
    # uniform refinement, so there is no 2:1 interface and nothing hangs -- which is what makes
    # test_a_hanging_node_interface_refines_and_converges a different case rather than a rerun
    assert _hangingSlaveNodes(model) == 0

    # Only the SECOND refinement is a re-arm: the first lands on the increment that also builds the
    # Newmark system for the first time, which is a step start, not a topology event.
    assert len(recorder.conservation) == 1
    assert len(recorder.builds) == 2, "expected one build per refinement, the first being the step's own"


def test_a_uniform_velocity_field_is_carried_onto_new_nodes_exactly(tmp_path, monkeypatch):
    """The explicit solver's own reference case, ported: the whole body translates rigidly, so the
    velocity field is spatially uniform and every new node must receive exactly that value.

    Uniform is the one configuration in which mass, momentum AND kinetic energy are all exact, so
    any deviation here is a defect in the transfer, not discretisation error -- there is none to
    hide behind."""

    recorder = _instrument(monkeypatch)
    _run(tmp_path, "amr_uniform", _deck(modifier=_REFINE_EVERYTHING_TWICE))

    _, samples = recorder.builds[-1]
    xVelocities = np.array([V for _, component, V, _ in samples if component == 0])
    assert np.max(np.abs(xVelocities)) > 1e-3, "the body is not moving; this test would pass on a model at rest"
    np.testing.assert_allclose(xVelocities, xVelocities[0], rtol=0.0, atol=1e-14 * np.max(np.abs(xVelocities)))

    # nothing transverse was invented
    transverse = np.array([V for _, component, V, _ in samples if component != 0])
    assert np.max(np.abs(transverse)) < 1e-14 * np.max(np.abs(xVelocities))

    _assertConservedExactly(recorder)


def test_a_linear_velocity_field_is_carried_onto_new_nodes_exactly(tmp_path, monkeypatch):
    """One step harder than the uniform case, and the one that actually tests the isoparametric
    map rather than only the partition of unity: a uniform extension, whose velocity is linear in
    ``x``. A 20-node hexahedron reproduces a linear field exactly at every point of its own
    reference cube, so every new node's value is ``c * x`` for the same ``c`` a retained node has,
    to round-off -- an interpolation that used the wrong parametric coordinates, the wrong parent,
    or the wrong shape functions cannot satisfy this, whereas the uniform case above would forgive
    all three."""

    recorder = _instrument(monkeypatch)
    _run(tmp_path, "amr_linear", _deck(modifier=_REFINE_EVERYTHING_TWICE, analyticalField=True))

    _, samples = recorder.builds[-1]
    xSamples = [(node.coordinates[0], V, A) for node, component, V, A in samples if component == 0]
    coordinates = np.array([x for x, _, _ in xSamples])
    velocities = np.array([V for _, V, _ in xSamples])
    accelerations = np.array([A for _, _, A in xSamples])

    # the rate is read off the node furthest from the origin, where it is best conditioned
    reference = int(np.argmax(np.abs(coordinates)))
    assert abs(coordinates[reference]) > 0.0
    velocityRate = velocities[reference] / coordinates[reference]
    accelerationRate = accelerations[reference] / coordinates[reference]

    assert abs(velocityRate) > 1e-3, "the body is not moving; this test would pass on a model at rest"
    assert abs(accelerationRate) > 1e-3, "the acceleration is zero; the A transfer is not being exercised"

    scale = np.max(np.abs(velocities))
    np.testing.assert_allclose(velocities, velocityRate * coordinates, rtol=0.0, atol=1e-12 * scale)
    accelerationScale = np.max(np.abs(accelerations))
    np.testing.assert_allclose(accelerations, accelerationRate * coordinates, rtol=0.0, atol=1e-12 * accelerationScale)

    _assertConservedExactly(recorder)


def _assertConservedExactly(recorder: _Recorder):
    """Mass, linear momentum and kinetic energy, for a velocity field of degree at most one.

    All three are exact here, for three different reasons: the total mass because the children
    tile their parent at the same density; the momentum because the shape functions are a partition
    of unity, so the interpolated field is literally the same function of space before and after;
    the kinetic energy because that same identity makes its integrand the same degree-two
    polynomial, which the element quadrature integrates exactly on an affine mesh.
    """

    assert recorder.conservation, "no topology change was reported, so nothing was checked"
    for before, after in recorder.conservation:
        for fieldName, massBefore in before.massByField.items():
            np.testing.assert_allclose(after.massByField[fieldName], massBefore, rtol=1e-12, atol=0.0)

        momentumScale = float(np.max(np.abs(before.momentum)))
        assert momentumScale > 0.0, "the momentum is zero before the event; conservation is vacuous"
        np.testing.assert_allclose(after.momentum, before.momentum, rtol=0.0, atol=1e-11 * momentumScale)

        assert before.kineticEnergy > 0.0, "the kinetic energy is zero before the event; conservation is vacuous"
        np.testing.assert_allclose(after.kineticEnergy, before.kineticEnergy, rtol=1e-11, atol=0.0)


def test_the_acceleration_is_re_equilibrated_exactly_on_the_increments_that_refined(tmp_path, monkeypatch):
    """The equilibrium solve must fire on the step start and on every increment that followed a
    topology change -- and on no other. Firing everywhere would make it a different integrator;
    firing nowhere would leave an interpolated acceleration inconsistent with the mass that was
    just reassembled."""

    recorder = _instrument(monkeypatch)
    _run(tmp_path, "amr_rearm", _deck(modifier=_REFINE_EVERYTHING_TWICE))

    # The step's own cold start, then one per refinement. The first refinement coincides with the
    # cold start (see _REFINE_EVERYTHING_TWICE), so the two collapse into one solve at t = 0.
    assert recorder.initialAccelerations == [0.0, DT], recorder.initialAccelerations


def test_the_re_equilibrated_acceleration_replaces_the_interpolated_one(tmp_path, monkeypatch):
    """Not merely "the solve ran": the acceleration it produces must actually differ from the
    interpolated one it replaces, on a deck whose acceleration is not identically zero. Otherwise
    the whole re-arm is an expensive no-op that no assertion would notice."""

    solves = []

    def capture(self, system, U_n, stepActions, model, timeStep):
        interpolated = np.array(system.A)
        result = _TRUE_INITIAL_ACCELERATION(self, system, U_n, stepActions, model, timeStep)
        solves.append((float(timeStep.stepTime - timeStep.timeIncrement), interpolated, np.array(system.A)))
        return result

    monkeypatch.setattr(NonlinearImplicitDynamic, "_computeInitialAcceleration", capture)

    _run(
        tmp_path,
        "amr_rearm_effect",
        _deck(modifier=_REFINE_THE_FIXED_END_TWICE, nX=2, lX=4.0, steps=_CANTILEVER_STEP),
    )

    startTime, interpolated, committed = solves[-1]
    assert startTime > 0.0, "the last solve was the step's cold start, not a post-refinement one"
    assert np.max(np.abs(interpolated)) > 0.0, "the interpolated acceleration was zero; there was nothing to replace"
    assert np.max(np.abs(committed - interpolated)) > 1e-6 * np.max(np.abs(committed))


def test_a_hanging_node_interface_refines_and_converges(tmp_path, monkeypatch):
    """Partial refinement, so the 2:1 interface carries hanging-node multi-point constraints and
    the Newmark residual, tangent and equilibrium solve all go through the MPC condensation. The
    uniform refinements above never create one."""

    recorder = _instrument(monkeypatch)
    model, _ = _run(
        tmp_path,
        "amr_hanging",
        _deck(modifier=_REFINE_THE_FIXED_END_TWICE, nX=2, lX=4.0, steps=_CANTILEVER_STEP),
    )

    # The constraint OBJECT exists after any refinement, holding no records when the refinement was
    # uniform, so its mere presence proves nothing; count the nodes it actually claims.
    assert _hangingSlaveNodes(model) > 0, "no node is actually constrained; this is not the 2:1 case"
    assert len(model.topologyHistory) == 2
    assert recorder.conservation, "the second refinement did not report; it may not have happened"

    # the mass is asserted by the solver itself; here only that the run reached the end intact
    U, V, A = _finalState(model)
    assert np.all(np.isfinite(U)) and np.all(np.isfinite(V)) and np.all(np.isfinite(A))
    assert model.time > 0.0


# --------------------------------------------------------------------------------------------
# The regression that matters most: an ordinary multi-step deck that never refines anything.
# --------------------------------------------------------------------------------------------

_TWO_STEPS = f"""
*step, solver=theSolver
stepLength=1.0, startInc={DT}, maxInc={DT}, minInc=1e-6, maxNumInc=3, maxIter=25
>>options, name=theSolver, extrapolation=off
>>dirichlet, name=motion, nSet=gen_left, field=displacement, 1=0.0, 2=0.0, 3=0.0
>>dirichlet, name=pull, nSet=gen_right, field=displacement, 1={X_MOTION}

*step, solver=theSolver
stepLength=1.0, startInc={DT}, maxInc={DT}, minInc=1e-6, maxNumInc=3, maxIter=25
>>options, name=theSolver, extrapolation=off
>>dirichlet, name=motion, nSet=gen_left, field=displacement, 1=0.0, 2=0.0, 3=0.0
>>dirichlet, name=pull, nSet=gen_right, field=displacement, 1={X_MOTION}, 2=0.0
"""


def test_a_two_step_deck_without_adaptivity_is_bit_identical_to_before_the_change(tmp_path, monkeypatch):
    """**The highest-risk item of this feature.**

    A step boundary rebuilds the ``DofManager`` -- unconditionally, since ``solveStep`` sets it to
    ``None`` first -- so the identity check that detects a refinement fires there too, on every
    existing multi-step deck. Two gates stop it from re-arming anything there: the Newmark system
    is also dropped at a step start, so the "not the first build" condition is false; and the
    topology history has not grown. This asserts, empirically rather than by reading the gates,
    that a two-step non-adaptive run is untouched: the same deck is run twice, once as shipped and
    once with the re-arm neutralised, and the two final states must agree to the last bit.

    The second step deliberately prescribes a different set of degrees of freedom from the first,
    so the two steps really do present different equation systems rather than accidentally the
    same one.
    """

    shipped = _instrument(monkeypatch, suppressRearm=False)
    shippedModel, _ = _run(tmp_path, "two_step_shipped", _deck(steps=_TWO_STEPS, nX=2, lX=4.0))
    shippedState = _finalState(shippedModel)
    shippedCalls = list(shipped.initialAccelerations)
    shippedDofManagers = [dofManager for dofManager, _ in shipped.builds]
    monkeypatch.undo()

    asBefore = _instrument(monkeypatch, suppressRearm=True)
    asBeforeModel, _ = _run(tmp_path, "two_step_as_before", _deck(steps=_TWO_STEPS, nX=2, lX=4.0))
    asBeforeState = _finalState(asBeforeModel)
    monkeypatch.undo()

    # The deck has to actually cross a step boundary with a rebuilt DofManager, or this proves
    # nothing at all.
    assert len(shippedDofManagers) == 2, shippedDofManagers
    assert (
        shippedDofManagers[0] is not shippedDofManagers[1]
    ), "the two steps shared a DofManager, so this deck is not the case at risk"

    # No topology change, so nothing may be reported and the acceleration may be solved exactly
    # once per step -- at each step's own start, which is what solveStep already decided.
    assert not shipped.conservation
    assert shippedCalls == [0.0, 0.0], shippedCalls
    assert not asBefore.conservation
    assert asBefore.initialAccelerations == shippedCalls

    for shippedArray, asBeforeArray, name in zip(shippedState, asBeforeState, "UVA"):
        np.testing.assert_array_equal(shippedArray, asBeforeArray, err_msg=f"'{name}' changed on a non-adaptive deck")


def test_suppressing_the_rearm_does_change_an_adaptive_run(tmp_path, monkeypatch):
    """The control for the test above. ``suppressRearm`` has to be a real difference detector --
    otherwise the bit-identical result there would only prove the patch does nothing, on any deck.
    """

    _instrument(monkeypatch, suppressRearm=False)
    shippedModel, _ = _run(
        tmp_path,
        "amr_shipped",
        _deck(modifier=_REFINE_THE_FIXED_END_TWICE, nX=2, lX=4.0, steps=_CANTILEVER_STEP),
    )
    shippedU, shippedV, shippedA = _finalState(shippedModel)
    monkeypatch.undo()

    _instrument(monkeypatch, suppressRearm=True)
    asBeforeModel, _ = _run(
        tmp_path,
        "amr_as_before",
        _deck(modifier=_REFINE_THE_FIXED_END_TWICE, nX=2, lX=4.0, steps=_CANTILEVER_STEP),
    )
    asBeforeU, asBeforeV, asBeforeA = _finalState(asBeforeModel)
    monkeypatch.undo()

    assert not np.array_equal(shippedA, asBeforeA), "suppressing the re-arm changed nothing, so it detects nothing"


def test_a_rebuild_without_a_topology_change_does_not_re_equilibrate(tmp_path, monkeypatch):
    """A constraint reporting that its connectivity changed -- a contact candidate list, most of
    all -- rebuilds the equation system without moving a single node. There is then no interpolated
    acceleration to replace and no conservation statement to make, and restarting the acceleration
    of the whole model there would be both wrong and, repeated per increment, expensive.

    Driven by making the mesh-dependent sweep report a change on one increment, which is exactly
    the signal such a constraint raises, without needing a contact deck to raise it.
    """

    recorder = _instrument(monkeypatch)

    trueRefresh = FEModel.refreshMeshDependents
    calls = {"n": 0}

    def refreshAndClaimAChangeOnce(self, *args, **kwargs):
        refreshed = trueRefresh(self, *args, **kwargs)
        calls["n"] += 1
        # The third call is the second REAL increment: the stepper yields a zero-length increment
        # first, and this sweep runs on it too.
        return refreshed or calls["n"] == 3

    monkeypatch.setattr(FEModel, "refreshMeshDependents", refreshAndClaimAChangeOnce)

    _run(tmp_path, "connectivity_only", _deck(nX=2, lX=4.0))

    assert calls["n"] > 3, "the forced connectivity change never happened"
    assert len(recorder.builds) == 2, "the equation system was not rebuilt mid-step; nothing was exercised"
    assert not recorder.conservation, "a rebuild that moved no node reported a topology change"
    assert recorder.initialAccelerations == [0.0], recorder.initialAccelerations


@pytest.mark.parametrize("modifier", [_REFINE_EVERYTHING_TWICE, _REFINE_THE_FIXED_END_TWICE])
def test_keeping_the_interpolated_acceleration_is_selectable(tmp_path, monkeypatch, modifier):
    """``computeInitialAcceleration=False`` continues from the carried-over acceleration at a step
    start; it must mean the same thing after a refinement, rather than being quietly overridden by
    a feature added later."""

    recorder = _instrument(monkeypatch)
    deck = _deck(modifier=modifier, nX=2, lX=4.0).replace(
        "newmarkGamma=0.5", "newmarkGamma=0.5\ncomputeInitialAcceleration=False"
    )
    _run(tmp_path, "amr_no_rearm", deck)

    assert recorder.conservation, "the refinement did not happen, so the option was never reached"
    assert recorder.initialAccelerations == []
