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
"""A step that reaches its maximum number of increments (``maxNumInc``) before its end fails, unless it is under arc
length control.

Without arc length control, the loads follow the progress of the time stepper, and a load keeps its value at the end of
the step for the next one. A step that stopped short of its end would hand the full load to the next step, whose first
increment would jump there.
"""

from pathlib import Path

import pytest

from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.utils.exceptions import StepFailed
from edelweissfe.utils.inputfileparser import parseInputFile

_MODEL = (
    Path(__file__).resolve().parent.parent / "testfiles" / "edelweiss-only" / "ArcLengthPenaltyConstraint" / "model.inp"
)

#: Two steps of a node force; the first (step 0) can reach only 20 % of its load in two increments of 10 %.
_STEP_SHORT_OF_INCREMENTS = """
*material, name=linearelastic, id=le, provider=edelweiss
1000.0, 0.3
*section, name=section1, thickness=1.0, material=le, type=plane
all
*job, name=job, domain=2d
*solver, solver=NIST, name=theSolver
*fieldOutput
>>perNode, elSet=all, field=displacement, result=U, name=displacement
*modelGenerator, generator=planeRectQuad, name=gen
x0=0, l=1
y0=0, h=1
elType=CPE4
elProvider=edelweiss
nX=1
nY=1
*step, solver=theSolver
startInc=0.1, maxInc=0.1, minInc=1e-3, maxNumInc=2, maxIter=25, stepLength=1
>>dirichlet, name=bottom, nSet=gen_bottom, field=displacement, 1=0, 2=0
>>nodeforces, name=top, nSet=gen_top, field=displacement, 2=1000
*step, solver=theSolver
startInc=0.1, maxInc=0.1, minInc=1e-3, maxNumInc=100, maxIter=25, stepLength=1
"""

#: An arc length step of two increments, which ends by maxNumInc long before the tip has moved down by its target.
_ARC_LENGTH_STEP_ENDING_BY_MAXNUMINC = """
*include, input={model}

*solver, solver=NISTPArcLength, name=theSolver

*step, solver=theSolver
startInc=0.1, maxInc=0.1, minInc=1e-6, maxNumInc=2, maxIter=25, stepLength=1
>>options, name=theSolver, arcLengthController=tipDisplacement
>>indirectcontrol, name=tipDisplacement, dof1='model.nodeSets["gen_rightTop"][0].fields["displacement"]', cVector1='0, 1', dof2='model.nodeSets["gen_leftBottom"][0].fields["displacement"]', cVector2='0, 0', L=-0.125
>>dirichlet, name=clamp, nSet=gen_left, field=displacement, 1=0.0, 2=0.0
>>nodeforces, name=tipLoad, nSet=gen_rightTop, field=displacement, 2=-1.0
"""


def solve(tmp_path, deck: str):
    path = tmp_path / "test.inp"
    path.write_text(deck)
    return finiteElementSimulation(parseInputFile(str(path)), verbose=False, suppressPlots=True)


def test_a_step_that_reaches_maxnuminc_before_its_end_fails(tmp_path):
    with pytest.raises(StepFailed, match=r"step 0: maxNumInc reached at step progress 0\.2\b"):
        solve(tmp_path, _STEP_SHORT_OF_INCREMENTS)


def test_an_arc_length_step_may_end_by_maxnuminc(tmp_path):
    model, _ = solve(tmp_path, _ARC_LENGTH_STEP_ENDING_BY_MAXNUMINC.format(model=_MODEL))

    assert 0.0 < model.additionalParameters["arc length parameter"]
