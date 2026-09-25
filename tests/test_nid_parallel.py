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
"""The parallel Newmark-beta implicit dynamic solver (``NIDParallel``).

``NIDParallel`` is ``NID`` with ``NISTParallel``'s thread-parallel element loop mixed in, and adds no
code of its own. What can go wrong is therefore the composition, not the dynamics:

* **The method resolution order.** Every ``super()`` call of ``NID`` must reach ``NISTParallel``, so
  that the Newton loop and the initial-acceleration solve both evaluate the elements through the
  thread pool, and ``NID``'s own overrides must still win over ``NIST``'s.
* **The result.** On a live-refining deck with hanging-node constraints -- the path through every
  part of ``NID``: the Newmark residual and tangent, the MPC condensation, the re-equilibrated
  acceleration after a topology change -- the parallel run must reproduce the serial one. Only up
  to round-off: the thread pool accumulates the element contributions in a different order.
"""

import numpy as np
from test_nid_amr import (
    _CANTILEVER_STEP,
    _REFINE_THE_FIXED_END_TWICE,
    _deck,
    _finalState,
    _hangingSlaveNodes,
    _run,
)

from edelweissfe.config.solvers import getSolverByName
from edelweissfe.solvers.nonlinearimplicitdynamic import NonlinearImplicitDynamic
from edelweissfe.solvers.nonlinearimplicitdynamicparallel import NIDParallel
from edelweissfe.solvers.nonlinearimplicitstatic import NIST
from edelweissfe.solvers.nonlinearimplicitstaticparallel import NISTParallel


def test_the_parallel_element_loop_is_reached_through_nid():
    assert getSolverByName("NIDParallel") is NIDParallel
    assert NIDParallel.__mro__[:4] == (NIDParallel, NonlinearImplicitDynamic, NISTParallel, NIST)

    # the element loop is NISTParallel's, everything NID overrides is still NID's
    assert NIDParallel.computeElements is NISTParallel.computeElements
    assert NIDParallel.solveIncrement is NonlinearImplicitDynamic.solveIncrement
    assert NIDParallel.solveStep is NonlinearImplicitDynamic.solveStep
    assert NIDParallel.schema is NonlinearImplicitDynamic.schema


def test_parallel_reproduces_serial_under_live_refinement(tmp_path):
    deck = _deck(modifier=_REFINE_THE_FIXED_END_TWICE, nX=2, lX=4.0, steps=_CANTILEVER_STEP)

    serialModel, _ = _run(tmp_path, "serial", deck)
    parallelModel, _ = _run(tmp_path, "parallel", deck.replace("solver=NID,", "solver=NIDParallel,"))

    assert _hangingSlaveNodes(parallelModel) > 0, "no hanging node; the MPC path was not exercised"
    assert len(parallelModel.topologyHistory) == len(serialModel.topologyHistory) == 2
    assert parallelModel.time == serialModel.time

    for serial, parallel, name in zip(_finalState(serialModel), _finalState(parallelModel), "UVA"):
        assert serial.shape == parallel.shape, name
        scale = float(np.max(np.abs(serial)))
        assert scale > 0.0, name
        np.testing.assert_allclose(parallel, serial, rtol=0.0, atol=1e-9 * scale, err_msg=name)
