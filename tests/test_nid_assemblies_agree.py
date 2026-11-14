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
"""``NID`` sums its system matrix either way (``systemMatrixAssembly=vij|direct``) with the same result: the mass is
assembled through the same assembly as the stiffness, and the effective tangent
``K_eff = K + M / (beta dT^2) + gamma C / (beta dT)`` is added during the assembly on both paths (see
:class:`~edelweissfe.numerics.systemmatrixassembly.SystemMatrixAssembly`). The two paths sum in different orders, so
``U``, ``V`` and ``A`` agree to round-off, not bit for bit:

* on the live-refining deck with hanging-node constraints (the MPC condensation on the effective tangent, the mass
  reassembled after every refinement), serially and with ``NIDParallel`` on several workers, whose blocks the direct
  path adds atomically;
* on the contact deck, whose rebuilds copy the mass onto the widened sparsity pattern instead of reassembling it.

The other ``NID`` tests run on both paths each (``conftest.py``: ``systemMatrixAssemblyOfNID``); this one compares the
two.
"""

import numpy as np
import pytest
from test_nid_amr import (
    _CANTILEVER_STEP,
    _REFINE_THE_FIXED_END_TWICE,
    _deck,
    _finalState,
    _hangingSlaveNodes,
)
from test_nid_contact import _DECK as _CONTACT_DECK

from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.utils.inputfileparser import parseInputFile


def _withAssembly(deck: str, solver: str, assembly: str) -> str:
    return deck.replace(
        "*solver, solver=NID, name=theSolver",
        "*solver, solver={:}, name=theSolver\nsystemMatrixAssembly={:}".format(solver, assembly),
    )


def _run(tmp_path, name: str, deck: str):
    path = tmp_path / f"{name}.inp"
    path.write_text(deck)
    model, _ = finiteElementSimulation(parseInputFile(str(path)), verbose=False, suppressPlots=True)
    return model


def _assertAgreeToRoundOff(model, reference):
    assert model.time == reference.time > 0.0
    for state, referenceState, name in zip(_finalState(model), _finalState(reference), "UVA"):
        assert state.shape == referenceState.shape, name
        scale = float(np.max(np.abs(referenceState)))
        assert scale > 0.0, name
        np.testing.assert_allclose(state, referenceState, rtol=0.0, atol=1e-9 * scale, err_msg=name)


@pytest.mark.parametrize("solver, numThreads", [("NID", 1), ("NIDParallel", 3)])
def test_both_assemblies_agree_under_live_refinement(tmp_path, workerThreads, solver, numThreads):
    workerThreads(numThreads)
    deck = _deck(modifier=_REFINE_THE_FIXED_END_TWICE, nX=2, lX=4.0, steps=_CANTILEVER_STEP)

    vij = _run(tmp_path, "vij", _withAssembly(deck, "NID", "vij"))
    direct = _run(tmp_path, "direct", _withAssembly(deck, solver, "direct"))

    assert _hangingSlaveNodes(direct) > 0, "no hanging node; the MPC path was not exercised"
    assert len(direct.topology.history) == len(vij.topology.history) == 2
    _assertAgreeToRoundOff(direct, vij)


def test_both_assemblies_agree_under_contact(tmp_path):
    vij = _run(tmp_path, "vij", _withAssembly(_CONTACT_DECK, "NID", "vij"))
    direct = _run(tmp_path, "direct", _withAssembly(_CONTACT_DECK, "NID", "direct"))

    _assertAgreeToRoundOff(direct, vij)
