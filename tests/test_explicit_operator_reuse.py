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

"""Keeping the lumped operators across a contact-connectivity rebuild must not change the answer.

The unit tests in ``test_dofmanager_refresh.py`` pin the refreshed bookkeeping against a freshly
constructed DofManager, but they never run a solver, so on their own they would pass even if the
solver reused an operator it had no business reusing. This drives the explicit solver over a real
contact problem whose connectivity changes mid-run, twice:

* once as shipped, where a connectivity change refreshes the constraints and keeps the operators,
* once with the reuse refused, where every such change rebuilds everything from scratch,

and requires the two to agree to the last bit. Refusing the reuse is exactly the fallback the guard
takes when the model does not match, so the same pair also covers that path.

Driven single-threaded on purpose: the comparison is bit-for-bit, and the parallel element loop
sums the chunk energies in completion order.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.solvers.nonlinearexplicitdynamic import NED
from edelweissfe.utils.inputfileparser import parseInputFile

#: The unpatched methods, captured once at import. A test that runs the solver twice must wrap
#: THESE rather than whatever is currently on the class -- wrapping the previous run's wrapper
#: would chain them, and the second run would then also record into the first run's lists.
_TRUE_REUSABLE = NED._operatorsReusable
_TRUE_BUILD = NED.buildEquationSystem

#: One way each to make the recorded model description disagree with the live model.
_PERTURBATIONS = {
    "elementKeys": lambda cache: replace(cache, elementKeys=frozenset(list(cache.elementKeys)[:-1])),
    "multiPointConstraintKeys": lambda cache: replace(cache, multiPointConstraintKeys=frozenset({"noSuchTie"})),
    "nNodes": lambda cache: replace(cache, nNodes=cache.nNodes + 1),
    "nScalarVariables": lambda cache: replace(cache, nScalarVariables=cache.nScalarVariables + 1),
}

#: Two blocks pressed into each other, with a penalty contact between them whose connectivity is
#: searched again every few increments -- which is what asks for the rebuild this test is about.
#: A coarsened, shortened relative of testfiles/edelweiss-only/NEDContact, small enough to run in
#: seconds and still cross a connectivity change.
_DECK = """
*material, name=LinearElastic, id=linearelastic, provider=edelweiss
1.8e4,    0.3,   1.0

*job, name=operatorReuseJob, domain=3d

*solver, solver=NED, name=theSolver
courant-number=0.8
output-frequency=1000
contact-update-frequency=5

*modelGenerator, generator=boxGen, name=lower
nX=2
nY=2
nZ=2
x0=0
y0=0
z0=0
lX=2
lY=2
lZ=1
elProvider=edelweiss
elType=C3D8

*modelGenerator, generator=boxGen, name=upper
nX=2
nY=2
nZ=2
x0=0.3
y0=1.1
z0=1.05
lX=0.6
lY=0.6
lZ=1
elProvider=edelweiss
elType=C3D8

*modelGenerator, generator=surfaceElementGenerator, name=gen1
surface = upper_back
name    = slaveSurf

*modelGenerator, generator=surfaceElementGenerator, name=gen2
surface = lower_front
name    = masterSurf

*section, name=section1, material=linearelastic, type=solid
lower_all
upper_all

*constraint, name=contact, type=nodeToDeformableSurfacePenalty
slaveSurface=slaveSurf_facets, masterSurface=masterSurf_facets, penalty=5e4, type=linear, searchDistance=2.0

*fieldOutput
>>perNode, elSet=lower_all, field=displacement, result=U, name=dispLower
>>perNode, elSet=upper_all, field=displacement, result=U, name=dispUpper

*step, type=adaptiveForExplicitSimulations, solver=theSolver
maxInc=1, minInc=1e-12, maxNumInc=400, maxIter=25, stepLength=0.2
>>dirichlet, name=fixLower, nSet=lower_back, field=displacement, 1=0.0, 2=0.0, 3=0.0
>>dirichlet, name=pushDown, nSet=upper_front, field=displacement, 3=-0.2
>>dirichlet, name=pinUpperXY, nSet=upper_bottomLeftBack, field=displacement, 1=0.0, 2=0.0
>>dirichlet, name=pinUpperRotZ, nSet=upper_bottomRightBack, field=displacement, 2=0.0
"""


def _deckFile(tmp_path: Path) -> str:
    """The deck on disk, which is what the input file parser reads."""

    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "operatorReuse.inp"
    path.write_text(_DECK)
    return str(path)


def _run(tmp_path, monkeypatch, refuseReuse: bool):
    """Run the deck, counting how often the solver took each rebuild path."""

    # single-threaded, so the run is deterministic and the comparison may be exact
    monkeypatch.setenv("OMP_NUM_THREADS", "1")

    taken = {"reused": 0, "rebuilt": 0}
    operators = []

    trueReusable = _TRUE_REUSABLE
    trueBuild = _TRUE_BUILD

    def countingReusable(self, model, stepActions):
        reusable = False if refuseReuse else trueReusable(self, model, stepActions)
        taken["reused" if reusable else "rebuilt"] += 1
        return reusable

    def recordingBuild(self, model, step, previous=None):
        system = trueBuild(self, model, step, previous)
        # every operator the reuse path restores instead of assembling, at every rebuild
        operators.append(
            (
                np.array(self._lumpedMass),
                np.array(self._rawLumpedMass),
                np.array(self._dampingRate),
                np.array(system.Minv),
            )
        )
        return system

    monkeypatch.setattr(NED, "_operatorsReusable", countingReusable)
    monkeypatch.setattr(NED, "buildEquationSystem", recordingBuild)

    model, _ = finiteElementSimulation(parseInputFile(_deckFile(tmp_path)), verbose=False, suppressPlots=True)

    # put the class back, so a second run in the same test starts from the shipped methods
    monkeypatch.undo()

    return model, taken, operators


def _finalState(model):
    """The solution and the internal force over the displacement field.

    Both runs build the same model from the same deck, so the field's node ordering is the same
    in both and the arrays may be compared as they are.
    """

    displacement = model.nodeFields["displacement"]
    return np.asarray(displacement["U"]), np.asarray(displacement["P"])


def test_reusing_the_operators_gives_exactly_the_rebuilt_answer(tmp_path, monkeypatch):
    reusedModel, reusedPaths, _ = _run(tmp_path / "reused", refuseReuse=False, monkeypatch=monkeypatch)
    rebuiltModel, rebuiltPaths, _ = _run(tmp_path / "rebuilt", refuseReuse=True, monkeypatch=monkeypatch)

    # The test is worthless unless the run really did cross a connectivity change and really did
    # take the reuse path there -- otherwise both runs are the same code and would agree trivially.
    assert reusedPaths["reused"] > 0, (
        "the deck never reused the operators, so this comparison proves nothing; "
        "the contact connectivity has to change while the run is going"
    )
    assert rebuiltPaths["reused"] == 0
    assert rebuiltPaths["rebuilt"] == reusedPaths["reused"] + reusedPaths["rebuilt"]

    reusedU, reusedP = _finalState(reusedModel)
    rebuiltU, rebuiltP = _finalState(rebuiltModel)

    np.testing.assert_array_equal(reusedU, rebuiltU)
    np.testing.assert_array_equal(reusedP, rebuiltP)


@pytest.mark.parametrize(
    "perturbedField",
    ["elementKeys", "multiPointConstraintKeys", "nNodes", "nScalarVariables"],
)
def test_every_recorded_part_of_the_model_description_can_refuse_the_reuse(perturbedField, tmp_path, monkeypatch):
    """Each field of the recorded model description must be able to refuse the reuse on its own.

    A field that cannot refuse anything is dead weight that reads as protection -- which is what
    a recorded DOF count would have been here, since a constraint refresh cannot move it. Nothing
    in the shipped solvers can reach these mismatches, because a topology change never passes a
    previous system; that is exactly why they are worth pinning rather than trusting.
    """

    monkeypatch.setenv("OMP_NUM_THREADS", "1")

    refusals = []
    trueReusable = _TRUE_REUSABLE

    def perturbTheRecordedModel(self, model, stepActions):
        reusable = trueReusable(self, model, stepActions)
        if reusable and self._reusableOperators is not None:
            intact = self._reusableOperators
            self._reusableOperators = _PERTURBATIONS[perturbedField](intact)
            refusals.append(trueReusable(self, model, stepActions))
            self._reusableOperators = intact
        return reusable

    monkeypatch.setattr(NED, "_operatorsReusable", perturbTheRecordedModel)

    finiteElementSimulation(parseInputFile(_deckFile(tmp_path)), verbose=False, suppressPlots=True)

    assert refusals, "the deck never got as far as a reusable rebuild"
    assert not any(refusals), "a perturbed {:} still allowed the reuse".format(perturbedField)


def test_every_restored_operator_equals_the_one_a_full_assembly_would_have_produced(tmp_path, monkeypatch):
    """Compare the operators themselves, not only the solution they lead to.

    The deck this runs is elastic and undamped, so its damping rate is zero throughout and its
    lumped mass only reaches the energy diagnostics -- meaning the end-to-end comparison above,
    which looks at the solution, cannot tell whether those two were restored correctly. Reading
    them off at every rebuild closes that gap, and keeps doing so for a deck that does damp.
    """

    _, reusedPaths, reusedOperators = _run(tmp_path / "reused", refuseReuse=False, monkeypatch=monkeypatch)
    _, _, rebuiltOperators = _run(tmp_path / "rebuilt", refuseReuse=True, monkeypatch=monkeypatch)

    assert reusedPaths["reused"] > 0
    assert len(reusedOperators) == len(rebuiltOperators)

    names = ("lumped mass", "raw lumped mass", "damping rate", "inverse lumped mass")
    for rebuildIndex, (reused, rebuilt) in enumerate(zip(reusedOperators, rebuiltOperators)):
        for name, fromReuse, fromAssembly in zip(names, reused, rebuilt):
            np.testing.assert_array_equal(
                fromReuse,
                fromAssembly,
                err_msg="the restored {:} differs from the assembled one at rebuild {:}".format(name, rebuildIndex),
            )
