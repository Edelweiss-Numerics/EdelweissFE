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
"""``NID`` on a gradient-enhanced element (``GC3D8``/``GCDP``): the one carrying a SECOND field
(``nonlocal damage``) that is not integrated in time, which ``test_nid_newmark.py``'s plain
``C3D8``/``LinearElastic`` bar cannot exercise -- there is only one field there.

Promotes an ad hoc manual check from PR #154's own description into a real regression test: on
this exact element/material combination, ``NID`` must integrate the mechanical field in time while
leaving the nonlocal field quasi-static, exactly as
:mod:`edelweissfe.solvers.nonlinearimplicitdynamic`'s module docstring says it does -- Marmot's
``GeneralGradientEnhancedDisplacementFiniteElement::computeConsistentInertia`` reports a nonzero
micro-inertia on the nonlocal block whenever the material declares one (GCDP does, via its
``nonlocalViscosity``/``microInertia`` properties), so this is the one element type where "the
non-mechanical block is zeroed" is not a vacuous claim.

GCDP requires ``density`` and ``nonlocalViscosity`` as material properties for ``NID`` (idx 19/20)
that a purely static deck never needs -- omitting them is exactly the mistake made once already
this session, and it fails loudly (Marmot raises naming the missing index) rather than silently.
"""

import numpy as np

from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.utils.inputfileparser import parseInputFile

# Same E, nu=0.2, and a small enough tip force to stay well inside GCDP's elastic branch (fcy=9.3
# is the compressive yield stress; the applied stress here is two orders below it) -- the point of
# this test is the multi-field bookkeeping, not GCDP's damage mechanics, which are already covered
# elsewhere (testfiles/marmot/GCDP, GC3D8, ...).
E = 30000.0
NU = 0.2
DENSITY = 4.8e-6
NONLOCAL_VISCOSITY = 3e-4
FORCE = 1.0
DT = 0.01

DECK = f"""
*material, name=GCDP, id=gcdp
**E    nu   fcy  fcu  fbu  ftu  Df    Ah
{E!r}, {NU!r}, 9.3, 28, 33, 2.2, 0.85, 0.08,
**Bh    Ch   Dh       As  epsF    lDamage  m    maxDmg
0.003, 2.0, 0.000001, 1,  0.0130, 2,       1.0, 0.99,
**drvdvMthd dTThrshHold viscosity  density        nonlocalViscosity
0,          1e-12,      0,         {DENSITY!r},   {NONLOCAL_VISCOSITY!r}

*section, name=section1, material=gcdp, type=solid
gen_all

*job, name=nidgcdpjob, domain=3d
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
>>perElement, name=omega, elSet=all, result=omega, quadraturePoint=0, saveHistory=True

*step, solver=theSolver
stepLength=1.0, startInc={DT!r}, maxInc={DT!r}, minInc=1e-6, maxNumInc=5, maxIter=25
>>options, name=theSolver, extrapolation=off
>>dirichlet, name=fixedEnd, nSet=gen_left, field=displacement, 1=0.0
>>dirichlet, name=lateral, nSet=all, field=displacement, 2=0.0, 3=0.0
>>nodeforces, name=endLoad, nSet=gen_right, field=displacement, 1={FORCE / 4.0!r}, f(t)='1'
"""


def test_displacement_is_dynamic_nonlocal_damage_stays_quasistatic(tmp_path):
    path = tmp_path / "nid_gc3d8_gcdp.inp"
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

    # elastic regime: damage never triggers, which is what makes the mechanical response close to
    # the equivalent LinearElastic bar's -- not asserted here bit-for-bit (GCDP is not a linear
    # material even below yield), only that nothing failed and nothing's damaged
    omega = np.asarray(fieldOutputController.fieldOutputs["omega"].getResultHistory())
    assert np.all(omega == 0.0)

    tipU = np.asarray(fieldOutputController.fieldOutputs["tipU"].getResultHistory())
    assert np.all(np.isfinite(tipU))
    assert tipU[-1] > 0.0
