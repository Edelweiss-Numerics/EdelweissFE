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
"""The log of an implicit solver with a composed linear solver reads as one table: what the linear solver reports while
it sets up (its tree) stands above the table header, in a frame, and nothing it says about a solve interrupts the rows.
"""

import pytest

from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.utils.inputfileparser import parseInputFile

pytest.importorskip("edelweissfe.linsolve.amgcl.amgcl")

SCRIPT = """
from edelweissfe.linsolve.composition import AMG, LGMRES, DiagonalScaling, FixedTolerance, Multiplicative, OnField


def fieldCorrections(context):
    return [OnField(field.name, AMG()) for field in context.fields]


linearSolver = LGMRES(DiagonalScaling(Multiplicative(fieldCorrections)), tolerance=FixedTolerance(1e-10))
"""

DECK = """
*material, name=linearelastic, id=linearelastic, provider=edelweiss
210000.0, 0.15

*section, name=section1, thickness=40.0, material=linearelastic, type=plane
all

*job, name=cps4job, domain=2d
*linearSolver, name=fieldSplit, type=custom
script=solver.py
verbosity=debug

*solver, solver=NIST, name=theSolver
linsolver=fieldSplit

*modelGenerator, generator=planeRectQuad, name=gen
x0=0, l=10000
y0=0, h=100
elType=CPE4
elProvider=edelweiss
nX=60
nY=4

*step, solver=theSolver
maxInc=5e-1, minInc=5e-1, maxNumInc=10, maxIter=25, stepLength=1
>>dirichlet, name=left, nSet=gen_left, field=displacement, 1=0, 2=0
>>nodeforces, name=load, nSet=gen_rightTop, components='0,-100', field=displacement
"""


@pytest.fixture
def log(tmp_path, monkeypatch, capsys) -> list:
    (tmp_path / "solver.py").write_text(SCRIPT)
    (tmp_path / "test.inp").write_text(DECK)
    monkeypatch.chdir(tmp_path)
    finiteElementSimulation(parseInputFile(str(tmp_path / "test.inp")), verbose=True, suppressPlots=True)
    return capsys.readouterr().out.splitlines()


def _index(lines: list, text: str) -> int:
    return next(i for i, line in enumerate(lines) if text in line)


def test_the_tree_of_the_solver_stands_in_a_frame_above_the_first_table_header(log):
    assert _index(log, "composed linear solver") < _index(log, "||R||∞")


def test_a_solve_is_not_narrated_between_the_rows_of_the_table(log):
    assert not any("outer iterations" in line for line in log)


def test_the_linear_solver_signs_with_a_name_that_fits_the_column(log):
    senders = {line.split()[-1] for line in log if "rebuilt(first)" in line}
    assert senders == {"ComposedSolver"}
