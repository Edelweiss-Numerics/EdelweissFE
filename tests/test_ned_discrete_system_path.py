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

"""``NED`` on the one build path of the discrete system (``NonlinearSolverBase.updateDiscreteSystem``).

What is sized by the numbering -- the lumped operators the increment divides by, the multi-point-constraint fold, the
classification of the fields by time derivative, the stable time increment -- is built once for a numbering
(``onNewNumbering``), and a contact search that couples other nodes only makes new locations on the same numbering
(``onNewLocations``): the operators are not touched, and the vectors keep their values. The tests drive the explicit
solver over real problems whose connectivity changes mid-run, and over a refinement of the mesh, and pin

* how often each hook is called, that what a new numbering builds is the same object afterwards, and that the vectors
  of the increments keep their values and are wrapped with the locations of the discrete system,
* that a tie and a contact together keep the fold through new locations, and the slaves follow their masters,
* that a change of the mesh builds a new numbering whose net force starts at zero, and whose stable time increment is
  that of the refined mesh.

Driven single-threaded where the hooks are recorded, to keep the order of the calls deterministic.
"""

import numpy as np
import pytest

from edelweissfe.config.configurator import loadConfiguration
from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.journal.journal import Journal
from edelweissfe.models.change import Change
from edelweissfe.solvers.base.nonlinearsolverbase import NonlinearSolverBase
from edelweissfe.solvers.nonlinearexplicitdynamic import NED
from edelweissfe.utils.inputfileparser import parseInputFile

#: Two blocks pressed into each other, with a penalty contact between them whose connectivity is
#: searched again every few increments -- which is what makes new locations, which this test is about.
#: A coarsened, shortened relative of testfiles/edelweiss-only/NEDContact, small enough to run in
#: seconds and still cross a connectivity change.
_DECK = """
*material, name=LinearElastic, id=linearelastic, provider=edelweiss
1.8e4,    0.3,   1.0

*job, name=nedDiscreteSystemPathJob, domain=3d

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


#: A tie and a contact together: a lower block, a middle block tied to it on a mismatched mesh, and a top block pushed
#: down and dragged along the top of the middle one, which pairs its nodes with other facets as it slides.
_TIE_AND_CONTACT_DECK = """
*material, name=LinearElastic, id=linearelastic, provider=edelweiss
100, 0.0, 1.0

*job, name=nedTieAndContactJob, domain=2d
*solver, solver=NED, name=theSolver
output-frequency=1000
contact-update-frequency=2

*modelGenerator, generator=planeRectQuad, name=lower
nX=3
nY=2
x0=0
y0=0
l=2
h=1
elType=CPE4
elProvider=edelweiss

*modelGenerator, generator=planeRectQuad, name=middle
nX=4
nY=2
x0=0
y0=1.02
l=2
h=1
elType=CPE4
elProvider=edelweiss

*modelGenerator, generator=planeRectQuad, name=top
nX=2
nY=2
x0=0.1
y0=2.07
l=0.8
h=0.6
elType=CPE4
elProvider=edelweiss

*modelGenerator, generator=surfaceElementGenerator, name=tieSlaveFacets
surface=middle_bottom
name=tieSlave

*modelGenerator, generator=surfaceElementGenerator, name=tieMasterFacets
surface=lower_top
name=tieMaster

*modelGenerator, generator=surfaceElementGenerator, name=contactSlaveFacets
surface=top_bottom
name=contactSlave

*modelGenerator, generator=surfaceElementGenerator, name=contactMasterFacets
surface=middle_top
name=contactMaster

*section, name=section1, thickness=1.0, material=linearelastic, type=plane
lower_all
middle_all
top_all

*constraint, name=tie1, type=tie
slaveSurface=tieSlave_facets, masterSurface=tieMaster_facets

*constraint, name=contact, type=nodeToDeformableSurfacePenalty
slaveSurface=contactSlave_facets, masterSurface=contactMaster_facets, penalty=100, type=linear, searchDistance=1.0

*step, type=adaptiveForExplicitSimulations, solver=theSolver
maxInc=1, minInc=1e-12, maxNumInc=100000, maxIter=25, stepLength=1.0
>>dirichlet, name=fixBottom, nSet=lower_bottom, field=displacement, 1=0, 2=0
>>dirichlet, name=dragTop, nSet=top_top, field=displacement, 1=1.0, 2=-0.05
"""

#: A bar that is refined once, live, while it is translated rigidly: ``testfiles/marmot/NEDLiveAMR``.
_REFINEMENT_DECK = """
*material, name=LinearElastic, id=linearelastic
20000.0, 0.2, 2.0e-9

*section, name=section1, material=linearelastic, type=solid
gen_all

*job, name=nedRefinementJob, domain=3d

*solver, solver=NED, name=theSolver
courant-number=0.1
output-frequency=20
topology-check-frequency=20

*modelGenerator, generator=boxGen, name=gen
nX=2
nY=1
nZ=1
lX=20
lY=10
lZ=10
elType=C3D20R

*modelModifier, type=hAdaptivity, name=amrLive
>>marker, type=elementSet, elSet=gen_all, initialOnly=False
refineElSet=gen_all
maxLevel=1

*fieldOutput
>>perNode, elSet=gen_all, field=displacement, result=U, name=displacement

*step, type=adaptiveForExplicitSimulations, solver=theSolver
maxInc=1, minInc=1e-14, maxNumInc=100, maxIter=25, stepLength=1e-3
>>dirichlet, name=rigidTranslation, nSet=all, field=displacement, 1=0.05, 2=0, 3=0
"""


def _deckFile(tmp_path, deck: str) -> str:
    """The deck on disk, which is what the input file parser reads."""

    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "nedDiscreteSystemPath.inp"
    path.write_text(deck)
    return str(path)


def _run(tmp_path, monkeypatch, deck: str, solver: str = "NED"):
    """Run a deck; what each hook of the solver is called with, and what it finds, is recorded."""

    # single-threaded, so the order of the calls is deterministic
    monkeypatch.setenv("OMP_NUM_THREADS", "1")

    record = {
        "solver": None,
        "onNewNumbering": 0,
        "onNewLocations": 0,
        "assembled": 0,
        "operatorsAfterNewNumbering": None,
        "operatorsKept": True,
        "vectorsKept": True,
        "vectorsOnTheLocations": True,
        "mpcKept": True,
        "criticalTimeSteps": [],
        "nDofs": [],
        "netForceBeforeAndAfterAMeshChange": [],
    }

    onNewNumbering = NED.onNewNumbering
    onNewLocations = NED.onNewLocations
    assembleLumpedOperators = NED._assembleLumpedOperators
    updateDiscreteSystem = NonlinearSolverBase.updateDiscreteSystem

    def operatorsOf(solver):
        return (solver._Minv, solver._lumpedMass, solver._rawLumpedMass, solver._dampingRate, solver._inertia)

    def recordingOnNewNumbering(self):
        record["solver"] = self
        record["onNewNumbering"] += 1
        onNewNumbering(self)
        record["operatorsAfterNewNumbering"] = operatorsOf(self)
        record["mpcTransformationAfterNewNumbering"] = self.mpcTransformation
        record["criticalTimeSteps"].append(self._criticalTimeStep)
        record["nDofs"].append(self.system.numbering.nDof)

    def recordingOnNewLocations(self):
        record["onNewLocations"] += 1
        stateBefore = [np.array(vector) for vector in (self._U, self._V, self._A, self._netForce)]

        onNewLocations(self)

        record["operatorsKept"] &= all(
            new is old for new, old in zip(operatorsOf(self), record["operatorsAfterNewNumbering"])
        )
        record["mpcKept"] &= self.mpcTransformation is record["mpcTransformationAfterNewNumbering"]
        stateAfter = [np.array(vector) for vector in (self._U, self._V, self._A, self._netForce)]
        record["vectorsKept"] &= all(np.array_equal(before, after) for before, after in zip(stateBefore, stateAfter))
        record["vectorsOnTheLocations"] &= all(
            vector.locations is self.system.locations
            for vector in (self._U, self._dU, self._V, self._A, self._netForce, self._internalForce)
        )

    def countingAssembleLumpedOperators(self):
        record["assembled"] += 1
        assembleLumpedOperators(self)

    def recordingUpdateDiscreteSystem(self, model, change):
        netForceBefore = None if self._netForce is None else np.array(self._netForce)
        updateDiscreteSystem(self, model, change)
        if change == Change.MESH:
            record["netForceBeforeAndAfterAMeshChange"].append((netForceBefore, np.array(self._netForce)))

    monkeypatch.setattr(NED, "onNewNumbering", recordingOnNewNumbering)
    monkeypatch.setattr(NED, "onNewLocations", recordingOnNewLocations)
    monkeypatch.setattr(NED, "_assembleLumpedOperators", countingAssembleLumpedOperators)
    monkeypatch.setattr(NED, "updateDiscreteSystem", recordingUpdateDiscreteSystem)

    deck = deck.replace("solver=NED, name=theSolver", "solver={:}, name=theSolver".format(solver))
    model, _ = finiteElementSimulation(parseInputFile(_deckFile(tmp_path, deck)), verbose=False, suppressPlots=True)

    return model, record


@pytest.mark.parametrize("solver", ["NED", "NEDParallel"])
def test_new_locations_keep_what_the_numbering_built_and_the_state_the_increments_continue_from(
    solver, tmp_path, monkeypatch
):
    _, record = _run(tmp_path, monkeypatch, _DECK, solver)

    # The test is worthless unless the run really did cross a connectivity change.
    assert record["onNewLocations"] > 1, (
        "the deck never made new locations, so this test proves nothing; "
        "the contact connectivity has to change while the run is going"
    )
    assert record["onNewNumbering"] == 1, "a contact search is no change of the numbering"
    assert record["assembled"] == 1, "the lumped operators are built once for the numbering, not for the locations"
    assert record["operatorsKept"], "new locations replaced an operator"
    assert record[
        "vectorsKept"
    ], "new locations changed the values of the solution, velocity, acceleration or net force"
    assert record["vectorsOnTheLocations"], "a vector of the increments is not wrapped with the locations of the system"


def test_a_tie_and_a_contact_keep_the_fold_through_new_locations_and_the_slaves_follow_their_masters(
    tmp_path, monkeypatch
):
    _, record = _run(tmp_path, monkeypatch, _TIE_AND_CONTACT_DECK)
    solver = record["solver"]
    transformation = solver.mpcTransformation

    assert record["onNewLocations"] > 1, "the contact never paired anew, so this test proves nothing"
    assert record["onNewNumbering"] == 1 and record["assembled"] == 1
    assert transformation is not None, "the tie has no condensation"
    assert record["mpcKept"], "new locations replaced the condensation of the tie"
    assert record["operatorsKept"] and record["vectorsKept"]

    slaves = transformation.slaveDofIndices
    assert slaves.size
    # the fold: the slaves carry no mass of their own, and the mass of the system is conserved
    assert np.all(solver._lumpedMass[slaves] == 0.0) and np.all(solver._Minv[slaves] == 0.0)
    assert np.sum(solver._lumpedMass) == pytest.approx(np.sum(solver._rawLumpedMass), rel=1e-12)
    # the slaves follow their masters in the velocity and the solution, and carry no net force of their own
    for vector in (solver._V, solver._U):
        slaved = vector.copy()
        transformation.applySlaveKinematics(slaved)
        np.testing.assert_allclose(slaved, vector, rtol=0.0, atol=1e-12 * max(1.0, np.max(np.abs(vector))))
    assert np.all(solver._netForce[slaves] == 0.0)
    assert np.any(solver._V), "nothing moves in the deck"


def test_a_refinement_of_the_mesh_builds_a_new_numbering_with_a_zero_net_force_and_the_stable_increment_of_the_mesh(
    tmp_path, monkeypatch
):
    _, record = _run(tmp_path, monkeypatch, _REFINEMENT_DECK)

    assert record["onNewNumbering"] == 2, "the first discrete system of the step and the one after the refinement"
    assert record["nDofs"][1] > record["nDofs"][0]
    assert record["assembled"] == 2, "the operators are assembled anew for the refined mesh"
    assert record["criticalTimeSteps"][1] < record["criticalTimeSteps"][0], "a refinement lowers the stable increment"

    ((netForceBefore, netForceAfter),) = record["netForceBeforeAndAfterAMeshChange"]
    assert np.any(netForceBefore), "the bar is accelerated: there is a net force to be dropped"
    assert not np.any(netForceAfter)


def test_forgetting_the_discrete_system_forgets_everything_the_numbering_built():
    """What a new numbering builds is sized by it: none of it is left standing, as an array of the old numbering, until
    the next build."""

    solver = NED(loadConfiguration({}), Journal(verbose=False))
    solver.system = object()
    solver._Minv = solver._lumpedMass = solver._rawLumpedMass = np.ones(3)
    solver._dampingRate = solver._halfDampingRate = solver._secondOrderMask = np.ones(3)
    solver._inertia = solver._damping = np.ones(3)
    solver._criticalTimeStep = 1.0
    solver.firstOrderFields, solver.secondOrderFields = ["pressure"], ["displacement"]
    solver.linearMomentumFields, solver.nonMechanicalSecondOrderFields = ["displacement"], ["pressure"]
    solver.ids_1st = solver.ids_2nd = solver.ids_mechanicalEnergy = np.arange(3)
    solver.mpcTransformation = object()

    solver.forgetDiscreteSystem()

    assert solver.system is None and solver.mpcTransformation is None
    assert all(
        operator is None
        for operator in (
            solver._Minv,
            solver._lumpedMass,
            solver._rawLumpedMass,
            solver._dampingRate,
            solver._halfDampingRate,
            solver._secondOrderMask,
            solver._inertia,
            solver._damping,
            solver._criticalTimeStep,
            solver.ids_1st,
            solver.ids_2nd,
            solver.ids_mechanicalEnergy,
        )
    )
    assert not (
        solver.firstOrderFields
        or solver.secondOrderFields
        or solver.linearMomentumFields
        or solver.nonMechanicalSecondOrderFields
    )
