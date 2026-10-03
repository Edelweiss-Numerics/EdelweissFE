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
"""Restart must be exact from EVERY checkpoint, with live AMR, for every solver.

The rule a resumed run has to obey: from the checkpointed increment on, it performs exactly the
operations the uninterrupted run performed -- the same topology decisions at the same increments,
the same contact searches, the same state. Anything else (a step-start action repeated on resume, a
piece of state not checkpointed, a schedule counted from the resume instead of from the step start)
shows up as a resumed run that ends somewhere else.

Such defects depend on WHERE the run is resumed: on or off the topology-check cycle, before or after
a refinement, right after a contact change, in the first or a later step. A test that resumes from
one hand-picked checkpoint finds them by luck. So this one runs each scenario once without
interruption, writing a checkpoint after every increment, and then resumes from every one of those
checkpoints -- including checkpoint 0, written at the start of the step (one more per step). Each resume is its
own test case (``<solver>-<scenario>-resume<k>``) and must end bitwise identical to the uninterrupted
run: time, element numbering and connectivity, every node field entry, every scalar variable, every
element's quadrature-point state, every constraint's restart data and the times of the topology
history. No tolerances, no per-case exceptions.

In addition, ``-chain<k>-<j>`` cases resume from checkpoint k, let that resumed run write checkpoints
of its own, resume from its j-th one and compare again: a checkpoint written by a resumed run must
be as good as one written by an uninterrupted run.

Every scenario asserts, in its uninterrupted run, that it actually exercises its feature (AMR fired
repeatedly, multipliers exist and are nonzero, the time step grew and was cut back, ...); a scenario
that silently stopped doing so would test nothing.

The models are tiny (a few C3D20 elements, 10--20 increments, a second or so per run) and use only
the pure-Python element and material library, so this runs everywhere the test suite runs. No
resume subset is taken: every checkpoint of every scenario is resumed from. To cover a new feature,
add a row to ``_SCENARIOS``.
"""

from functools import cache
from pathlib import Path

import h5py
import numpy as np
import pytest

import edelweissfe.outputmanagers.restart as restartOutputManager
from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.timesteppers.adaptivetimestepper import AdaptiveTimeStepper
from edelweissfe.timesteppers.simpletimestepper import SimpleTimeStepper
from edelweissfe.utils.exceptions import RestartError
from edelweissfe.utils.inputfileparser import parseInputFile

#: The explicit solver offers the model modifiers a topology update only every this many increments,
#: so most checkpoints lie off that cycle -- which is where resuming went wrong before.
TOPOLOGY_CHECK_FREQUENCY = 5

_MATERIAL = """
*material, name=LinearElastic, id=linearelastic
2.0e4, 0.25, 1.0e-3
"""

_BEAM_GEOMETRY = """
*modelGenerator, generator=boxGen, name=beam
nX      =6
nY      =2
nZ      =1
lX      =6
lY      =1
lZ      =1
elType  =C3D20

*section, name=beamSection, material=linearelastic, type=solid
beam_all
"""

_AMR = """
*fieldOutput
>>perElement, elSet=beam_all, result=stress, quadraturePoint=0:27, name=stressForAMR, f(x)='np.abs(x)'

*modelModifier, type=hAdaptivity, name=amr
>>marker, type=fieldOutput, fieldOutput=stressForAMR, operator='>', threshold={threshold}
refineElSet=beam_all
maxLevel={maxLevel}
{amrOptions}
"""

_BEAM = _BEAM_GEOMETRY + _AMR

#: A rigid plate the beam's right end is pressed onto while the plate slides sideways.
_RIGID_PLATE = """
*modelGenerator, generator=surfaceElementGenerator, name=genBeamBottom
surface = beam_bottom
name    = beamBottom

*modelGenerator, generator=discreteRigidBodyGenerator, name=plate, executeAfterManualGeneration=True
filename={stl}
rpCoordinate='5.0, -0.5, 0.5'
mass=1.0
inertia="1.0, 1.0, 1.0"

*constraint, name=contactPlate, type=surfaceToDiscreteRigidBodyPenalty
slaveSurface=beamBottom_facets, rigidBody=plate, penalty=1e4, type=linear, searchDistance=0.5
"""

#: The x-displacements of the beam's right face are kept equal by Lagrange multipliers (scalar
#: variables, extrapolated by NIST's linear predictor like every other unknown).
_EQUAL_VALUE = """
*constraint, type=equalValueLagrangian, name=rightFaceEqualX
nSet=beam_right
field=displacement
component=0
"""

#: A second, shorter beam on top of the first one, bonded to it by a tie (non-matching meshes).
_TIED_BEAM = """
*modelGenerator, generator=boxGen, name=top
nX      =3
nY      =1
nZ      =1
z0      =1
lX      =6
lY      =1
lZ      =0.5
elType  =C3D20

*section, name=topSection, material=linearelastic, type=solid
top_all

*modelGenerator, generator=surfaceElementGenerator, name=genTieSlave
surface = top_back
name    = tieSlave
triangulation = midside

*modelGenerator, generator=surfaceElementGenerator, name=genTieMaster
surface = beam_front
name    = tieMaster
triangulation = midside

*constraint, name=tieTop, type=tie
slaveSurface=tieSlave_facets, masterSurface=tieMaster_facets
"""

#: The beam in a hardening von Mises material, bent far into the plastic range.
_PLASTIC_BEAM = """
*material, name=VonMises, id=vonmises
2.0e4, 0.25, 15.0, 100.0, 10.0, 100.0, 1.0e-3
""" + _BEAM_GEOMETRY.replace(
    "material=linearelastic", "material=vonmises"
)

#: A penalty constraint (no scalar variables) on a node set that live AMR refines: the top face's
#: z-displacements are kept equal. The node set grows with every refinement of a top-face element.
_EQUAL_VALUE_PENALTY = """
*constraint, type=equalValuePenalty, name=topFaceEqualZ
nSet=beam_top
field=displacement
component=2
penalty=1e5
"""

#: Node count of ``beam_top`` before any refinement (6 x 1 C3D20 faces).
_BEAM_TOP_NODES = 33

#: A small block (``block``) pressed onto and dragged across a deformable base (named ``beam`` so
#: that every scenario has ``beam_right`` for the exported field output), frictional node-to-surface
#: penalty contact. The block initially touches the base.
_DEFORMABLE_CONTACT = """
*modelGenerator, generator=boxGen, name=beam
nX      =2
nY      =2
nZ      =1
lX      =2
lY      =2
lZ      =1
elType  =C3D20

*modelGenerator, generator=boxGen, name=block
nX      =1
nY      =1
nZ      =1
x0      =0.3
y0      =0.7
z0      =1.0
lX      =0.6
lY      =0.6
lZ      =0.5
elType  =C3D20

*modelGenerator, generator=surfaceElementGenerator, name=genSlave
surface = block_back
name    = slaveSurf
triangulation = midside

*modelGenerator, generator=surfaceElementGenerator, name=genMaster
surface = beam_front
name    = masterSurf
triangulation = midside

*section, name=contactSection, material=linearelastic, type=solid
beam_all
block_all

*constraint, name=contact, type=nodeToDeformableSurfacePenalty
slaveSurface=slaveSurf_facets, masterSurface=masterSurf_facets, penalty={contactPenalty}, type=quadratic,
searchDistance=2.0, sliding=small, mu=0.3, tangentPenalty={tangentPenalty}
"""

_JOB = "\n*job, name=exhaustiveRestart, domain=3d\n"

_SOLVERS = {
    "implicit": "*solver, solver=NIST, name=theSolver\n{solverOptions}\n",
    "NID": "*solver, solver=NID, name=theSolver\nnewmarkBeta=0.25\nnewmarkGamma=0.5\n{solverOptions}\n",
    "explicit": """*solver, solver=NED, name=theSolver
courant-number=0.1
output-frequency=1
contact-update-frequency={contactUpdateFrequency}
topology-check-frequency={topologyCheckFrequency}
{solverOptions}
""",
}

_STEP = {
    "implicit": """
*step, solver=theSolver
startInc={startInc}, maxInc={maxInc}, minInc=1e-6, maxNumInc=1000, maxIter={maxIter}, stepLength=1
""",
    "explicit": """
*step, type=adaptiveForExplicitSimulations, solver=theSolver
maxInc=1, minInc=1e-14, maxNumInc={maxNumInc}, maxIter=25, stepLength=1.0
""",
}
_STEP["NID"] = _STEP["implicit"]

#: The prescribed displacement at the end of a step. The explicit run covers a far shorter time
#: span within its increments, so it is loaded correspondingly faster.
_AMPLITUDE = {"implicit": 0.05, "NID": 0.05, "explicit": 50.0}

_LOADS_BEND = """
>>dirichlet, name=clamp, nSet=beam_left, field=displacement, 1=0, 2=0, 3=0
>>dirichlet, name=bend, nSet=beam_right, field=displacement, 2=-{amplitude}
"""

#: The second step of the two-step analysis bends the beam back up, past its initial position.
_LOADS_BEND_BACK = """
>>dirichlet, name=bend, nSet=beam_right, field=displacement, 2={amplitude}
"""

#: Press the block into the base and drag it sideways at the same time.
_LOADS_PRESS_AND_DRAG = """
>>dirichlet, name=fixBase, nSet=beam_back, field=displacement, 1=0, 2=0, 3=0
>>dirichlet, name=drag, nSet=block_front, field=displacement, 1={amplitude}, 2=0, 3=-{amplitude}
"""

#: The step-1 loads of the modelupdate scenario: the bend, and the penalty constraint switched off by
#: a modelupdate at the start of step 1. Step 2 must still run without it.
_LOADS_BEND_AND_DEACTIVATE = (
    _LOADS_BEND
    + """>>modelupdate, update='model.constraints["topFaceEqualZ"].active=False'
"""
)

_LOADS_SLIDE = """
>>dirichlet, name=clamp, nSet=beam_left, field=displacement, 1=0, 2=0, 3=0
>>dirichlet, name=slidePlate, nSet=plate_rp, field=displacement, 1=0.0, 2={amplitude}, 3={amplitude}
>>dirichlet, name=fixPlateRotation, nSet=plate_rp, field=rotation, 1=0, 2=0, 3=0
"""


def _refinements(model) -> int:
    return len([record for record in model.topology.history if record.modifier == "amr"])


def _refinesRepeatedly(model, run) -> str:
    return "" if _refinements(model) >= 2 else "AMR must fire repeatedly, or there is nothing to test"


def _hasNonzeroMultipliers(model, run) -> str:
    values = [variable.value for variable in model.scalarVariables.values()]
    return "" if values and any(v != 0.0 for v in values) else "expected nonzero Lagrange multipliers"


def _hasTie(model, run) -> str:
    return "" if "tieTop" in model.multiPointConstraints else "the tie constraint is missing"


def _adaptiveGrowthAndCutback(model, run) -> str:
    if not any(increment > run["startInc"] for increment in run["increments"]):
        return "the time step never grew beyond startInc"
    if run["nCutbacks"] < 1:
        return "no cutback happened"
    return ""


def _timeIncrementLowered(model, run) -> str:
    increments = [dt for dt in run["timeIncrements"] if dt > 0.0]
    if len(set(increments)) < 2:
        return "the explicit time increment never changed during the run"
    if any(later > earlier for earlier, later in zip(increments, increments[1:])):
        return "the explicit time increment grew"
    return ""


def _isPlastic(model, run) -> str:
    alphas = [np.max(model.elements[n].getStateVars()) for n in model.elements]
    return "" if max(alphas) > 0.0 else "no element state is nonzero: the beam never yielded"


def _penaltyNodeSetRefined(model, run) -> str:
    n = len(model.nodeSets["beam_top"])
    return "" if n > _BEAM_TOP_NODES else "beam_top was never refined ({:} nodes)".format(n)


def _frictionalContact(model, run) -> str:
    data = model.constraints["contact"].getRestartData()
    if not np.any(np.asarray(data["tangentialForceConverged"]) != 0.0):
        return "no frictional contact history was built up"
    return ""


def _all(*checks):
    return lambda model, run: "; ".join(message for message in (check(model, run) for check in checks) if message)


#: One row per scenario: the solvers, the model blocks, the steps (their load blocks), and what the
#: uninterrupted run must show to prove it exercises the feature. Remaining keys parametrize the deck.
_SCENARIOS = {
    "amr": dict(
        solvers=["implicit", "explicit"],
        blocks=[_BEAM],
        threshold={"implicit": 6.0, "explicit": 20.0},
        steps=[_LOADS_BEND],
        check=_refinesRepeatedly,
    ),
    "amrRigidContact": dict(
        solvers=["implicit", "explicit", "NID"],
        blocks=[_BEAM, _RIGID_PLATE],
        threshold={"implicit": 6.0, "explicit": 40.0, "NID": 6.0},
        steps=[_LOADS_SLIDE],
        check=_refinesRepeatedly,
    ),
    # (a) Marks below minMarkedElements are held pending across increments -- state to checkpoint.
    "amrMinMarked": dict(
        solvers=["implicit", "explicit"],
        blocks=[_BEAM],
        threshold={"implicit": 6.0, "explicit": 10.0},
        amrOptions="minMarkedElements=3",
        steps=[_LOADS_BEND],
        check=_refinesRepeatedly,
    ),
    # (b) The increment after a refinement starts from a zero predictor instead of extrapolating.
    "amrNoExtrapolationAfterChange": dict(
        solvers=["implicit"],
        blocks=[_BEAM],
        threshold={"implicit": 6.0},
        solverOptions="extrapolateAfterModelChange=False",
        steps=[_LOADS_BEND],
        check=_refinesRepeatedly,
    ),
    # (c) Lagrange multipliers (scalar variables) under NIST's default linear extrapolation.
    "lagrangeEqualValue": dict(
        solvers=["implicit"],
        blocks=[_BEAM_GEOMETRY, _EQUAL_VALUE],
        steps=[_LOADS_BEND],
        check=_hasNonzeroMultipliers,
    ),
    # (d) Contact searched only every 3rd increment: most checkpoints lie off the contact cycle.
    "amrRigidContactSearchEvery3": dict(
        solvers=["explicit"],
        blocks=[_BEAM, _RIGID_PLATE],
        threshold={"explicit": 40.0},
        contactUpdateFrequency=3,
        steps=[_LOADS_SLIDE],
        check=_refinesRepeatedly,
    ),
    # (g) Adaptive time stepping: the increment grows (startInc < maxInc), and a tight iteration
    # limit forces cutbacks once the beam yields (maxIter=6 would converge every increment).
    "plasticAdaptiveDt": dict(
        solvers=["implicit"],
        blocks=[_PLASTIC_BEAM],
        startInc=0.05,
        maxInc=0.2,
        maxIter=4,
        amplitude=1.0,
        steps=[_LOADS_BEND],
        check=_adaptiveGrowthAndCutback,
    ),
    # (h) A surface tie between two non-matching meshes, the master side refined by live AMR.
    "amrTie": dict(
        solvers=["implicit"],
        blocks=[_BEAM, _TIED_BEAM],
        threshold={"implicit": 6.0},
        steps=[_LOADS_BEND],
        check=_all(_refinesRepeatedly, _hasTie),
    ),
    # (i) Newmark implicit dynamics, with live AMR.
    "amrDynamic": dict(
        solvers=["NID"],
        blocks=[_BEAM],
        threshold={"NID": 6.0},
        steps=[_LOADS_BEND],
        check=_refinesRepeatedly,
    ),
    # (j) Two steps; resumes land in the first and in the second one.
    "amrTwoSteps": dict(
        solvers=["implicit"],
        blocks=[_BEAM],
        threshold={"implicit": 6.0},
        steps=[_LOADS_BEND, _LOADS_BEND_BACK],
        check=_refinesRepeatedly,
    ),
    # (k) Explicit dynamics with a von Mises material and live AMR: the stable time increment depends
    # on the state (refinement lowers it, mid-run, lower-only).
    "amrPlasticExplicit": dict(
        solvers=["explicit"],
        blocks=[_PLASTIC_BEAM, _AMR],
        threshold={"explicit": 20.0},
        steps=[_LOADS_BEND],
        check=_all(_refinesRepeatedly, _isPlastic, _timeIncrementLowered),
    ),
    # (l) A penalty constraint on a node set that live AMR refines.
    "amrEqualValuePenalty": dict(
        solvers=["implicit"],
        blocks=[_BEAM, _EQUAL_VALUE_PENALTY],
        threshold={"implicit": 6.0},
        steps=[_LOADS_BEND],
        check=_all(_refinesRepeatedly, _penaltyNodeSetRefined),
    ),
    # (m) Checkpoints only every 3rd increment: the restart writer's own counter is state.
    "amrWriteEvery3": dict(
        solvers=["implicit", "explicit"],
        blocks=[_BEAM],
        threshold={"implicit": 6.0, "explicit": 20.0},
        writeInterval=3,
        nCheckpoints={"implicit": 3, "explicit": 7},
        steps=[_LOADS_BEND],
        check=_refinesRepeatedly,
    ),
    # (n) Frictional node-to-surface contact between two deformable bodies.
    "deformableFrictionalContact": dict(
        solvers=["implicit", "explicit"],
        blocks=[_DEFORMABLE_CONTACT],
        amplitude={"implicit": 0.05, "explicit": 50.0},
        steps=[_LOADS_PRESS_AND_DRAG],
        check=_frictionalContact,
    ),
    # (o) A modelupdate executed at the start of step 1 changes the model by an arbitrary expression
    # (here: switches a penalty constraint off). No checkpoint records its effect, so every resume
    # after it -- in step 1 or step 2 -- is refused with a RestartError instead of silently running
    # with the constraint on again.
    "modelUpdateInStep1": dict(
        solvers=["implicit"],
        blocks=[_BEAM_GEOMETRY, _EQUAL_VALUE_PENALTY],
        steps=[_LOADS_BEND_AND_DEACTIVATE, _LOADS_BEND_BACK],
        resumeIsRefused=True,
        check=lambda model, run: "" if not model.constraints["topFaceEqualZ"].active else "modelupdate did not run",
    ),
}

#: Fixed increments per step: implicit runs are costlier per increment; explicit runs need more
#: increments for the load (and AMR) to develop, and to put most checkpoints off the topology cycle.
_N_INCREMENTS = {"implicit": 10, "NID": 10, "explicit": 20}

#: How often an element may be split. The explicit runs need a second level to refine more than once
#: within their short time span; the implicit ones refine repeatedly with one level, which keeps
#: their meshes -- and so the test's runtime -- small.
_MAX_LEVEL = {"implicit": 1, "NID": 1, "explicit": 2}

#: Adaptive runs have no a-priori increment count: this is the count calibrated for the adaptive
#: scenario (25 increments with cutbacks, plus the step-start checkpoint). The test asserts it, so a
#: run that changes its incrementation cannot silently leave checkpoints untested.
_ADAPTIVE_CHECKPOINTS = 26

#: Chained resumes: from checkpoint k, then from the j-th checkpoint the resumed run wrote itself.
_CHAINS = [(0, 3), (3, 2)]
_CHAINED_SCENARIOS = ("amr", "amrTwoSteps")

_PLATE_STL = "solid plate\n{:}\nendsolid plate\n"


def _writePlateStl(path: Path):
    """A 2 x 0.2 x 2 cuboid under the beam's right end, its top face touching the beam's bottom."""

    corners = np.array([[x, y, z] for x in (4.0, 6.0) for y in (-0.2, 0.0) for z in (-0.5, 1.5)])
    center = corners.mean(axis=0)
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    facets = []
    for a, b, c, d in faces:
        for triangle in ((a, b, c), (a, c, d)):
            p0, p1, p2 = corners[list(triangle)]
            normal = np.cross(p1 - p0, p2 - p0)
            if normal @ ((p0 + p1 + p2) / 3 - center) < 0.0:
                p1, p2, normal = p2, p1, -normal
            normal /= np.linalg.norm(normal)
            facets.append(
                " facet normal {:} {:} {:}\n  outer loop\n".format(*normal)
                + "".join("   vertex {:} {:} {:}\n".format(*p) for p in (p0, p1, p2))
                + "  endloop\n endfacet"
            )
    path.write_text(_PLATE_STL.format("\n".join(facets)))


def _deck(directory: Path, scenario: str, solver: str, extra: str = "") -> str:
    spec = _SCENARIOS[scenario]
    stl = directory / "plate.stl"
    _writePlateStl(stl)
    blocks = "".join(
        block.format(
            threshold=spec.get("threshold", {}).get(solver),
            maxLevel=_MAX_LEVEL[solver],
            stl=stl,
            amrOptions=spec.get("amrOptions", ""),
            contactPenalty={"explicit": 5e4}.get(solver, 3e7),
            tangentPenalty={"explicit": 5e3}.get(solver, 1e6),
        )
        for block in spec["blocks"]
    )
    solverBlock = _SOLVERS[solver].format(
        solverOptions=spec.get("solverOptions", ""),
        contactUpdateFrequency=spec.get("contactUpdateFrequency", 1),
        topologyCheckFrequency=TOPOLOGY_CHECK_FREQUENCY,
    )
    dt = 1.0 / _N_INCREMENTS[solver]
    step = _STEP[solver].format(
        startInc=spec.get("startInc", dt),
        maxInc=spec.get("maxInc", dt),
        maxIter=spec.get("maxIter", 25),
        maxNumInc=_N_INCREMENTS[solver],
    )
    amplitude = spec.get("amplitude", _AMPLITUDE[solver])
    if isinstance(amplitude, dict):
        amplitude = amplitude[solver]
    steps = "".join(step + loads.format(amplitude=amplitude) for loads in spec["steps"])
    return _MATERIAL + blocks + _export(directory) + extra + _JOB + solverBlock + steps


def _export(directory: Path) -> str:
    """A field output exported to a CSV file, one row per output increment."""

    return """
*fieldOutput
>>perNode, nSet=beam_right, field=displacement, result=U, name=exportU, export={:}, f(x)='np.mean(x, axis=0)', saveHistory=True
""".format(
        directory / "exportU"
    )


def _csvRows(directory: Path) -> int:
    path = directory / "exportU.csv"
    return len(path.read_text().splitlines()) if path.exists() else 0


def _run(path: Path, text: str):
    path.write_text(text)
    model, _ = finiteElementSimulation(parseInputFile(str(path)), verbose=False, suppressPlots=True)
    return model


def _state(model) -> dict:
    """Everything a resumed run must reproduce, as arrays to compare bitwise."""

    history = model.topology.history
    state = {
        "time": np.array([model.time]),
        "element numbers": np.array(sorted(model.elements)),
        "connectivity": np.array(
            [node.label for number in sorted(model.elements) for node in model.elements[number].nodes]
        ),
        "topology history times": np.array([record.time for record in history]),
        "topology history modifiers": np.array([str(record.modifier) for record in history]),
    }
    for name, field in model.nodeFields.items():
        for entry in field._values:
            state["{:}/{:}".format(name, entry)] = np.array(field[entry])
    for name in sorted(model.scalarVariables):
        state["scalar variable {:}".format(name)] = np.array([model.scalarVariables[name].value])
    for number in sorted(model.elements):
        try:
            state["element {:} state".format(number)] = np.array(model.elements[number].getStateVars())
        except NotImplementedError:
            pass
    for name in sorted(model.constraints):
        data = model.constraints[name].getRestartData()
        if data is None:
            continue
        for key in sorted(data):
            state["constraint {:}/{:}".format(name, key)] = np.asarray(data[key])
    return state


def _checkpoints(directory: Path) -> list:
    return sorted(directory.glob("ckpt_*.h5"), key=lambda p: int(p.stem.split("_")[-1]))


def _checkpointKey(checkpoint: Path) -> tuple:
    """Where a checkpoint lies in the analysis: (step number, model time)."""

    with h5py.File(checkpoint, "r") as f:
        return int(f.attrs["stepNumber"]), float(f.attrs["time"])


def _writer(directory: Path, writeInterval: int = 1) -> str:
    return "\n*output, type=restart, name=restart\nwriteInterval={:}, baseName={:}, numberOfFilesToKeep=1000\n".format(
        writeInterval, directory / "ckpt"
    )


def _reader(checkpoint: Path) -> str:
    return "\n*restart, readFrom={:}\n".format(checkpoint)


@cache
def _uninterrupted(scenario: str, solver: str, directory: Path):
    """The reference run, writing a checkpoint after every increment; returns its final state, its
    checkpoints, and why it does not exercise its feature ('' if it does)."""

    directory.mkdir(parents=True, exist_ok=True)
    # Instrument the time stepper to see increment sizes and cutbacks (the feature check only).
    increments, timeIncrements, nCutbacks = [], [], [0]
    generate, discard = AdaptiveTimeStepper.generateTimeStep, AdaptiveTimeStepper.discardAndChangeIncrement
    generateSimple = SimpleTimeStepper.generateTimeStep
    write = restartOutputManager.writeCheckpoint
    #: The exported CSV rows at the moment each checkpoint was written, by checkpoint key.
    rowsAtCheckpoint = {}

    def recordingGenerate(self, *args, **kwargs):
        for timeStep in generate(self, *args, **kwargs):
            increments.append(timeStep.stepProgressIncrement)
            timeIncrements.append(timeStep.timeIncrement)
            yield timeStep

    def recordingGenerateSimple(self, *args, **kwargs):
        for timeStep in generateSimple(self, *args, **kwargs):
            timeIncrements.append(timeStep.timeIncrement)
            yield timeStep

    def countingDiscard(self, scaleFactor):
        nCutbacks[0] += 1
        return discard(self, scaleFactor)

    def recordingWrite(fileName, model, step, outputManagers):
        write(fileName, model, step, outputManagers)
        rowsAtCheckpoint[_checkpointKey(Path(fileName))] = _csvRows(directory)

    AdaptiveTimeStepper.generateTimeStep = recordingGenerate
    AdaptiveTimeStepper.discardAndChangeIncrement = countingDiscard
    SimpleTimeStepper.generateTimeStep = recordingGenerateSimple
    restartOutputManager.writeCheckpoint = recordingWrite
    spec = _SCENARIOS[scenario]
    try:
        deck = _deck(directory, scenario, solver, _writer(directory, spec.get("writeInterval", 1)))
        model = _run(directory / "uninterrupted.inp", deck)
    finally:
        AdaptiveTimeStepper.generateTimeStep, AdaptiveTimeStepper.discardAndChangeIncrement = generate, discard
        SimpleTimeStepper.generateTimeStep = generateSimple
        restartOutputManager.writeCheckpoint = write
    checkpoints = _checkpoints(directory)
    run = dict(
        increments=increments,
        timeIncrements=timeIncrements,
        nCutbacks=nCutbacks[0],
        startInc=spec.get("startInc", 1.0),
    )
    output = dict(
        checkpointKeys=[_checkpointKey(c) for c in checkpoints],
        rowsAtCheckpoint=rowsAtCheckpoint,
        rows=_csvRows(directory),
    )
    return _state(model), checkpoints, output, spec["check"](model, run)


def _isAdaptive(scenario: str) -> bool:
    return "startInc" in _SCENARIOS[scenario]


def _nCheckpoints(scenario: str, solver: str) -> int:
    if "nCheckpoints" in _SCENARIOS[scenario]:
        return _SCENARIOS[scenario]["nCheckpoints"][solver]
    if _isAdaptive(scenario):
        return _ADAPTIVE_CHECKPOINTS
    return (_N_INCREMENTS[solver] + 1) * len(_SCENARIOS[scenario]["steps"])


def _cases():
    for scenario, spec in _SCENARIOS.items():
        for solver in spec["solvers"]:
            for k in range(_nCheckpoints(scenario, solver)):
                yield pytest.param(scenario, solver, (k,), id="{:}-{:}-resume{:02d}".format(solver, scenario, k))
            if scenario in _CHAINED_SCENARIOS:
                for k, j in _CHAINS:
                    caseId = "{:}-{:}-chain{:02d}-{:02d}".format(solver, scenario, k, j)
                    yield pytest.param(scenario, solver, (k, j), id=caseId)


@pytest.fixture(scope="module")
def workDirectory(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("exhaustiveRestart")


@pytest.mark.parametrize("scenario, solver, chain", list(_cases()))
def test_resume_from_every_checkpoint_is_exact(workDirectory, scenario, solver, chain):
    spec = _SCENARIOS[scenario]
    reference, checkpoints, output, featureMissing = _uninterrupted(scenario, solver, workDirectory / scenario / solver)
    assert not featureMissing, featureMissing
    assert len(checkpoints) == _nCheckpoints(scenario, solver), "expected {:} checkpoints, got {:}".format(
        _nCheckpoints(scenario, solver), len(checkpoints)
    )

    def writer(directory):
        return _writer(directory, spec.get("writeInterval", 1))

    resumeDirectory = workDirectory / scenario / solver / "resume{:}".format("-".join(map(str, chain)))
    resumeDirectory.mkdir()
    checkpoint = checkpoints[chain[0]]
    for j in chain[1:]:
        deck = _deck(resumeDirectory, scenario, solver, _reader(checkpoint) + writer(resumeDirectory))
        _run(resumeDirectory / "resumed.inp", deck)
        written = _checkpoints(resumeDirectory)
        assert j < len(written), "the resumed run wrote only {:} checkpoints".format(len(written))
        checkpoint = written[j]
        resumeDirectory = resumeDirectory / "chained"
        resumeDirectory.mkdir()

    resumeKey = _checkpointKey(checkpoint)
    deck = _deck(resumeDirectory, scenario, solver, _reader(checkpoint) + writer(resumeDirectory))
    if spec.get("resumeIsRefused"):
        with pytest.raises(RestartError):
            _run(resumeDirectory / "resumed.inp", deck)
        return
    resumed = _state(_run(resumeDirectory / "resumed.inp", deck))

    differing = [key for key in reference if key not in resumed or not np.array_equal(resumed[key], reference[key])]
    differing += [key for key in resumed if key not in reference]

    # The resumed run writes the checkpoints the uninterrupted run wrote after the resume point --
    # the same number, at the same (step, time) -- and gains as many exported CSV rows.
    expectedKeys = [key for key in output["checkpointKeys"] if key > resumeKey]
    writtenKeys = [_checkpointKey(c) for c in _checkpoints(resumeDirectory)]
    if writtenKeys != expectedKeys:
        differing.append("checkpoints written {:} != expected {:}".format(writtenKeys, expectedKeys))
    expectedRows = output["rows"] - output["rowsAtCheckpoint"][resumeKey]
    if _csvRows(resumeDirectory) != expectedRows:
        differing.append("CSV rows gained {:} != expected {:}".format(_csvRows(resumeDirectory), expectedRows))

    assert not differing, "resuming from {:} ends elsewhere: {:}".format(checkpoint, differing[:8])


if __name__ == "__main__":
    # Calibration aid: one uninterrupted run per scenario and solver, reporting the feature check.
    import sys
    import tempfile

    for scenario, spec in _SCENARIOS.items():
        if len(sys.argv) > 1 and scenario not in sys.argv[1:]:
            continue
        for solver in spec["solvers"]:
            _, checkpoints, _, missing = _uninterrupted(scenario, solver, Path(tempfile.mkdtemp()))
            print(scenario, solver, "checkpoints:", len(checkpoints), "feature:", missing or "ok", file=sys.stderr)
