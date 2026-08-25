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
"""Regression test for BlockAMGSolver._forcingTolerance's zero-residual guard.

Found while adding a restart+blockamg integration test (tests/test_restart_integration.py): a
linear-elastic-like problem (VonMises before yielding) can converge with an exactly-zero residual
on one Newton iteration, and the very next call's ``ratio = residualNorm / self._lastResidualNorm``
then divides by that stored zero -- a ZeroDivisionError unrelated to restart itself, just first
surfaced by that test's simple mesh. Constructs a bare instance via __new__, bypassing
BlockAMGSolver.__init__ (which needs a live model/AMG setup), since _forcingTolerance only reads
plain scalar attributes.
"""

from edelweissfe.linsolve.blockamg.blockamg import BlockAMGSolver


def _bareSolver(lastResidualNorm, lastEta=1.0, etaMin=1e-4, etaMax=0.9, ewGamma=0.9, ewAlpha=2.0):
    solver = BlockAMGSolver.__new__(BlockAMGSolver)
    solver._lastResidualNorm = lastResidualNorm
    solver._lastEta = lastEta
    solver._etaMin = etaMin
    solver._etaMax = etaMax
    solver._ewGamma = ewGamma
    solver._ewAlpha = ewAlpha
    return solver


def test_forcing_tolerance_falls_back_to_etamax_with_no_history():
    solver = _bareSolver(lastResidualNorm=None)
    assert solver._forcingTolerance(residualNorm=1.0, newIncrement=False) == solver._etaMax


def test_forcing_tolerance_falls_back_to_etamax_on_new_increment():
    solver = _bareSolver(lastResidualNorm=1e-6)
    assert solver._forcingTolerance(residualNorm=1.0, newIncrement=True) == solver._etaMax


def test_forcing_tolerance_does_not_divide_by_a_zero_previous_residual():
    solver = _bareSolver(lastResidualNorm=0.0)
    assert solver._forcingTolerance(residualNorm=0.0, newIncrement=False) == solver._etaMax


def test_forcing_tolerance_computes_a_ratio_with_real_history():
    solver = _bareSolver(lastResidualNorm=1.0, lastEta=0.05, etaMin=1e-4, etaMax=0.9, ewGamma=0.9, ewAlpha=2.0)
    eta = solver._forcingTolerance(residualNorm=0.1, newIncrement=False)
    # ratio=0.1 -> eta = 0.9 * 0.1**2 = 0.009, safeguard = 0.9 * 0.05**2 = 0.00225 (not > 0.1, ignored),
    # clamped to [1e-4, 0.9] -- 0.009 is already inside that range, so the clamp is a no-op here.
    assert abs(eta - 0.009) < 1e-9
