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
"""A body force on the element set ``all`` in a model with contact facets.

The surface element generator puts the contact facets into the element set ``all``. A facet carries geometry only: it has
no block in the system matrix, so the discrete system leaves it out and it has no location. A load evaluates the elements
of its set that are located, so the facets are skipped, and the answer is that of the model without them -- for the
implicit and the explicit solver alike.
"""

import numpy as np
import pytest

from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.utils.inputfileparser import parseInputFile

_FACETS = """
*modelGenerator, generator=surfaceElementGenerator, name=gen
surface=lower_front
name=facetSurf
"""

_STEPS = {
    "NIST": """
*step, solver=theSolver
maxInc=1, startInc=1, minInc=1e-6, maxNumInc=10, maxIter=25, stepLength=1
""",
    "NED": """
*step, type=adaptiveForExplicitSimulations, solver=theSolver
maxInc=1, minInc=1e-12, maxNumInc=1000, maxIter=25, stepLength=0.01
""",
}

_DECK = """
*material, name=LinearElastic, id=linearelastic, provider=edelweiss
1.8e4, 0.3, 1.0

*job, name=bodyloadjob, domain=3d
*solver, solver={solver}, name=theSolver

*modelGenerator, generator=boxGen, name=lower
nX=2
nY=2
nZ=2
lX=1
lY=1
lZ=1
elProvider=edelweiss
elType=C3D8
{facets}
*section, name=section1, material=linearelastic, type=solid
lower_all
{step}
>>dirichlet, name=fixBottom, nSet=lower_bottom, field=displacement, 1=0.0, 2=0.0, 3=0.0
>>bodyforce, name=weight, elSet=all, forceVector='0.0, 0.0, -1.0', f(t)='1'
"""


def _run(tmp_path, monkeypatch, solver: str, withFacets: bool):
    """The model after the step, and the solution and the force of the displacement field."""

    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    deck = _DECK.format(solver=solver, facets=_FACETS if withFacets else "", step=_STEPS[solver])
    path = tmp_path / "bodyload.inp"
    path.write_text(deck)
    model, _ = finiteElementSimulation(parseInputFile(str(path)), verbose=False, suppressPlots=True)
    field = model.nodeFields["displacement"]

    return model, np.array(field["U"]), np.array(field["P"])


@pytest.mark.parametrize("solver", ["NIST", "NED"])
def test_a_body_force_on_all_skips_the_facets_in_it_and_gives_the_answer_without_them(tmp_path, monkeypatch, solver):
    model, U, P = _run(tmp_path / "facets", monkeypatch, solver, withFacets=True)
    _, UWithout, PWithout = _run(tmp_path / "without", monkeypatch, solver, withFacets=False)

    assert any(not element.hasKernels for element in model.elementSets["all"]), "the facets are not in the set all"
    assert np.any(U), "the weight moves the block"
    np.testing.assert_array_equal(U, UWithout)
    np.testing.assert_array_equal(P, PWithout)
