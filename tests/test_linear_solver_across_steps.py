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
"""A nonlinear solver keeps one linear solver for all the steps that name it.

A new linear solver per step would leave the previous one, with everything it built (its matrices, hierarchies and
factorizations), to Python's cyclic garbage collector whenever its tree refers back to itself -- which a function in
the solver script does, since the function's globals hold the script's tree. The garbage collector does not see the
size of that memory, so the steps' matrices would pile up. The tests run with the garbage collector off, so that only
what reference counting frees is freed.
"""

import gc

import pytest

import edelweissfe.solvers.nonlinearimplicitstatic as nonlinearimplicitstatic
from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.linsolve.composition import AMG
from edelweissfe.numerics.csrgeneratorv2 import CSRGenerator
from edelweissfe.utils.inputfileparser import parseInputFile

pytest.importorskip("edelweissfe.linsolve.amgcl.amgcl")

# the corrections made by a function: the pattern of the documented field-split scripts, whose function keeps the
# script's globals, and with them the script's tree, alive
SCRIPT = """
from edelweissfe.linsolve.composition import AMG, LGMRES, DiagonalScaling, FixedTolerance, Multiplicative, OnField


def fieldCorrections(context):
    return [OnField(field.name, AMG()) for field in context.fields]


linearSolver = LGMRES(DiagonalScaling(Multiplicative(fieldCorrections)), tolerance=FixedTolerance(1e-10))
"""

# the first step declares the boundary conditions, the later ones raise the load
FIRST_STEP_ACTIONS = """>>dirichlet, name=left, nSet=gen_left, field=displacement, 1=0, 2=0
>>nodeforces, name=load, nSet=gen_rightTop, components='0,{load}', field=displacement"""

STEP = """
*step, solver=theSolver
maxInc=1e0, minInc=1e0, maxNumInc=10, maxIter=25, stepLength=1
{actions}
{options}
"""


def _step(i: int, options: str) -> str:
    load = -100.0 * (i + 1)
    actions = FIRST_STEP_ACTIONS if i == 0 else ">>nodeforces, name=load, components='0,{load}'"
    return STEP.format(actions=actions.format(load=load), options=options)


DECK = """
*material, name=linearelastic, id=linearelastic, provider=edelweiss
210000.0, 0.15

*section, name=section1, thickness=40.0, material=linearelastic, type=plane
all

*job, name=cps4job, domain=2d
*linearSolver, name=fieldSplit, type=custom
script=solver.py

*solver, solver=NIST, name=theSolver
linsolver=fieldSplit

*modelGenerator, generator=planeRectQuad, name=gen
x0=0, l=10000
y0=0, h=100
elType=CPE4
elProvider=edelweiss
nX=60
nY=4
"""


def _simulate(tmp_path, monkeypatch, steps: list) -> list:
    """Run the deck with the given steps (their options lines) and the garbage collector off; the linear solvers the
    nonlinear solver created, and the AMG components this simulation left alive. The AMG components that other
    tests left alive in the process are not this simulation's, so they are not counted."""
    (tmp_path / "solver.py").write_text(SCRIPT)
    deck = tmp_path / "test.inp"
    deck.write_text(DECK + "".join(_step(i, options) for i, options in enumerate(steps)))
    monkeypatch.chdir(tmp_path)

    created = []

    def createLinearSolver(*args):
        created.append(original(*args))
        return created[-1]

    original = nonlinearimplicitstatic.createLinearSolver
    monkeypatch.setattr(nonlinearimplicitstatic, "createLinearSolver", createLinearSolver)

    gc.collect()
    gc.disable()
    try:
        # held, so that none of them is freed and its id taken by an AMG of this simulation
        aliveBefore = [obj for obj in gc.get_objects() if isinstance(obj, AMG)]
        idsAliveBefore = {id(amg) for amg in aliveBefore}
        finiteElementSimulation(parseInputFile(str(deck)), verbose=False, suppressPlots=True)
        aliveAMGs = sum(isinstance(obj, AMG) and id(obj) not in idsAliveBefore for obj in gc.get_objects())
    finally:
        gc.enable()
    return created, aliveAMGs


def test_the_steps_that_name_a_linear_solver_share_it_and_leave_no_tree_behind(tmp_path, monkeypatch):
    created, aliveAMGs = _simulate(tmp_path, monkeypatch, ["", "", ""])
    # one field, displacement: the one tree has one AMG
    assert aliveAMGs == 1
    assert len(created) == 1


def test_a_new_step_drops_the_last_equation_system_before_it_builds_the_next(tmp_path, monkeypatch):
    generatorsAtBuild = []
    original = nonlinearimplicitstatic.DofManager

    def dofManager(*args):
        generatorsAtBuild.append(sum(isinstance(obj, CSRGenerator) for obj in gc.get_objects()))
        return original(*args)

    monkeypatch.setattr(nonlinearimplicitstatic, "DofManager", dofManager)
    _simulate(tmp_path, monkeypatch, ["", ""])
    assert len(generatorsAtBuild) == 2 and generatorsAtBuild[1] == generatorsAtBuild[0]


def test_a_step_that_names_another_linear_solver_gets_a_new_one(tmp_path, monkeypatch):
    # an option set in a step holds until a later step sets it again
    steps = ["", ">>options, name=theSolver, linsolver=superlu", ">>options, name=theSolver, linsolver=fieldSplit"]
    created, _ = _simulate(tmp_path, monkeypatch, steps)
    assert len(created) == 3
