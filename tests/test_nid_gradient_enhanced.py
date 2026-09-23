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
"""``NID`` on a gradient-enhanced element (``GC3D8``/``AT2PHASEFIELD``): the one carrying a SECOND
field (``nonlocal damage``) that is not integrated in time, which ``test_nid_newmark.py``'s plain
``C3D8``/``LinearElastic`` bar cannot exercise -- there is only one field there.

On this element/material combination ``NID`` must integrate the mechanical field in time while
leaving the nonlocal field quasi-static, as :mod:`edelweissfe.solvers.nonlinearimplicitdynamic`'s
module docstring states.

Uses ``AT2PHASEFIELD`` because it is part of every public Marmot build, and it declares the optional
``density``/``nonlocalViscosity``/``microInertia`` material properties (idx 4/5/6) through
``MarmotMaterialGeneralGradientEnhancedHypoElastic``, the base class every gradient-enhanced
hypoelastic material shares -- so the multi-field bookkeeping it exercises is that of all of them.

``AT2PHASEFIELD`` requires ``density`` and ``nonlocalViscosity`` (idx 4/5) under ``NID``, which a
quasi-static deck never needs; a deck omitting them fails loudly, with Marmot naming the missing
property.
"""

import numpy as np

from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.solvers.nonlinearimplicitdynamic import NonlinearImplicitDynamic
from edelweissfe.utils.inputfileparser import parseInputFile

# A small enough tip force that the elastic strain energy stays far below what would drive the
# phase field away from (near-)zero -- the point of this test is the multi-field bookkeeping, not
# AT2PhaseField's fracture mechanics, which are already covered elsewhere (testfiles/marmot/AT2PhaseField).
E = 30000.0
NU = 0.2
GC = 1.0
L = 0.5
DENSITY = 4.8e-6
NONLOCAL_VISCOSITY = 3e-4
FORCE = 1.0
DT = 0.01

DECK = f"""
*material, name=AT2PHASEFIELD, id=at2
**E    nu   Gc   l    density        eta
{E!r}, {NU!r}, {GC!r}, {L!r}, {DENSITY!r}, {NONLOCAL_VISCOSITY!r}

*section, name=section1, material=at2, type=solid
gen_all

*job, name=nidat2job, domain=3d
*solver, solver=NID, name=theSolver
newmarkBeta=0.25
newmarkGamma=0.5

*modelGenerator, generator=boxGen, name=gen
nX=1
nY=1
nZ=1
lX=1
lY=1
lZ=1
elType=GC3D8

*fieldOutput
>>perNode, nSet=gen_right, field=displacement, result=U, name=tipU, f(x)='np.mean(x[:,0])', saveHistory=True
>>perNode, nSet=gen_right, field=displacement, result=V, name=tipV, f(x)='np.mean(x[:,0])', saveHistory=True

*step, solver=theSolver
stepLength=1.0, startInc={DT!r}, maxInc={DT!r}, minInc=1e-6, maxNumInc=5, maxIter=25
>>options, name=theSolver, extrapolation=off
>>dirichlet, name=fixedEnd, nSet=gen_left, field=displacement, 1=0.0
>>dirichlet, name=lateral, nSet=all, field=displacement, 2=0.0, 3=0.0
>>nodeforces, name=endLoad, nSet=gen_right, field=displacement, 1={FORCE / 4.0!r}, f(t)='1'
"""


def test_displacement_is_dynamic_nonlocal_damage_stays_quasistatic(tmp_path):
    path = tmp_path / "nid_gc3d8_at2.inp"
    path.write_text(DECK)
    inputfile = parseInputFile(str(path))
    model, fieldOutputController = finiteElementSimulation(inputfile, verbose=False, suppressPlots=True)

    displacementField = model.nodeFields["displacement"]
    nonlocalField = model.nodeFields["nonlocal damage"]

    # the mechanical field is time-integrated: it has V/A entries, and the tip actually moved
    assert "V" in displacementField and "A" in displacementField
    assert np.max(np.abs(displacementField["V"])) > 0.0

    # the nonlocal field carries no Newmark state at all -- carriesLinearMomentum is False for it,
    # so the input-file driver never pre-creates V/A there in the first place
    assert "V" not in nonlocalField and "A" not in nonlocalField

    # near-elastic regime: unlike GCDP, AT2PhaseField's phase field has no yield-like threshold, so
    # it is never bit-exactly zero once any strain energy exists -- but the applied force is far too
    # small to drive it away from that, which is what makes the mechanical response close to the
    # equivalent LinearElastic bar's
    phaseField = np.asarray(nonlocalField["U"])
    assert np.all(np.isfinite(phaseField))
    assert np.max(np.abs(phaseField)) < 1e-3

    tipU = np.asarray(fieldOutputController.fieldOutputs["tipU"].getResultHistory())
    assert np.all(np.isfinite(tipU))
    assert tipU[-1] > 0.0


def test_the_discarded_nonlocal_inertia_is_warned_about_once(tmp_path, monkeypatch):
    """The element reports a damping on the nonlocal field (its ``nonlocalViscosity``), which this
    solver discards because it keeps that field quasi-static. That must not happen silently, and
    the displacement field, which keeps its inertia, must not be named."""

    trueWarn = NonlinearImplicitDynamic._warnAboutDiscardedInertia
    warned = []

    def recordingWarn(self, Mvij, Cvij, couplesDynamicOnly):
        before = set(self._fieldsWarnedAboutDiscardedInertia)
        trueWarn(self, Mvij, Cvij, couplesDynamicOnly)
        warned.extend(self._fieldsWarnedAboutDiscardedInertia - before)

    monkeypatch.setattr(NonlinearImplicitDynamic, "_warnAboutDiscardedInertia", recordingWarn)

    path = tmp_path / "nid_gc3d8_at2_warn.inp"
    path.write_text(DECK)
    finiteElementSimulation(parseInputFile(str(path)), verbose=False, suppressPlots=True)

    assert warned == ["nonlocal damage"], warned
