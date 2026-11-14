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
"""``NISTPArcLength`` with a constraint, against an independent reference.

The model is ``testfiles/edelweiss-only/ArcLengthPenaltyConstraint``: a cantilever under its self-weight (the dead
load), bent down onto a rigid floor (a quadratic penalty constraint) by a tip load (the reference load), whose load
factor is controlled such that the tip moves down by 0.25. Every part of the model is conservative -- linear elastic
elements, an elastic penalty, loads that do not follow the deformation -- so the state at the end of the step depends
only on the tip displacement reached, not on the path that led there.

``NIST`` therefore reaches the same state with the tip displacement prescribed instead of controlled. There the tip
load is replaced by the reaction at the prescribed degree of freedom, which gives the load factor. This checks how
``NISTPArcLength`` assembles the constraint, the dead load and the reference load, each with its sign (see the solver
documentation, "The residual and the tangent"):

- a constraint entering with the sign of an external load would push the cantilever into the floor, and change the
  displacements;
- a reference load of the wrong sign would leave the displacements as they are, as the controller reaches the same
  tip displacement with the opposite load factor -- which the comparison of the load factor catches.
"""

from pathlib import Path

import numpy as np

from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.utils.inputfileparser import parseInputFile

_DECK_DIRECTORY = Path(__file__).resolve().parent.parent / "testfiles" / "edelweiss-only" / "ArcLengthPenaltyConstraint"

#: The same model, solved by NIST with the tip displacement prescribed; the step actions besides the control are the
#: ones of ``test.inp``.
_REFERENCE = """
*include, input={model}

*solver, solver=NIST, name=theSolver

*step, solver=theSolver
startInc=0.1, maxInc=0.1, minInc=1e-6, maxNumInc=100, maxIter=25, stepLength=1
>>dirichlet, name=clamp, nSet=gen_left, field=displacement, 1=0.0, 2=0.0
>>dirichlet, name=tipDisplacement, nSet=gen_rightTop, field=displacement, 2=-0.25
>>bodyforce, name=selfWeight, elSet=gen_all, forceVector='0.0, -0.005', f(t)='1'
"""


def _solve(inputFile: Path):
    model, _ = finiteElementSimulation(parseInputFile(str(inputFile)), verbose=False, suppressPlots=True)
    return model


def test_arc_length_with_a_penalty_constraint_reaches_the_displacement_controlled_state(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    arcLengthModel = _solve(_DECK_DIRECTORY / "test.inp")

    referenceFile = tmp_path / "reference.inp"
    referenceFile.write_text(_REFERENCE.format(model=_DECK_DIRECTORY / "model.inp"))
    referenceModel = _solve(referenceFile)

    displacement = referenceModel.nodeFields["displacement"]
    tip = displacement.indexOfNode(referenceModel.nodeSets["gen_rightTop"][0])
    np.testing.assert_allclose(displacement["U"][tip, 1], -0.25, rtol=1e-12, err_msg="the end of the step")
    assert np.any(displacement["U"][:, 1] < -0.1), "the cantilever must have reached the floor"

    # the same state: the displacements, and the elements' internal forces (the nodal result P)
    U = arcLengthModel.nodeFields["displacement"]["U"]
    np.testing.assert_allclose(U, displacement["U"], rtol=0.0, atol=1e-8)
    np.testing.assert_allclose(arcLengthModel.nodeFields["displacement"]["P"], displacement["P"], rtol=0.0, atol=1e-8)

    # At the tip, the internal force balances the external loads, f_int = -1 * loadFactor + selfWeightOnTheTip: the
    # tip load is -1 times the load factor, and the corner node of a square element of edge 0.5 carries a quarter of
    # the element's weight.
    selfWeightOnTheTip = -0.005 * 0.5 * 0.5 / 4.0
    internalForceAtTheTip = displacement["P"][tip, 1]
    expectedLoadFactor = selfWeightOnTheTip - internalForceAtTheTip
    loadFactor = arcLengthModel.solvers["theSolver"].Lambda
    np.testing.assert_allclose(loadFactor, expectedLoadFactor, rtol=1e-7)
