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
"""The views of a composed solver, and the solver script of an input file."""

import textwrap

import numpy as np
import pytest
from _linsolvesystems import displacementDamageSystem

pytest.importorskip("edelweissfe.linsolve.amgcl.amgcl")

from edelweissfe.config.linsolve import getLinSolverByName  # noqa: E402
from edelweissfe.linsolve.composition import views  # noqa: E402
from edelweissfe.linsolve.composition import (  # noqa: E402
    AMG,
    LGMRES,
    ComposedLinearSolver,
    DiagonalScaling,
    DirectSolver,
    FixedTolerance,
    Multiplicative,
    OnField,
    SchurComplement,
    blockAMG,
    loadSolverScript,
)


def _solvedTree():
    A, b, fields = displacementDamageSystem(n=10, coupling="symmetric")
    root = LGMRES(
        DiagonalScaling(Multiplicative([OnField("displacement", AMG()), OnField("nonlocal damage", AMG())])),
        tolerance=FixedTolerance(1e-8),
    )
    solver = ComposedLinearSolver(root, verbosity="silent")
    solver.setFieldStructure(fields)
    solver(A, b)
    return root


def test_the_structure_view_shows_one_line_per_component_with_the_roles_of_the_children():
    root = _solvedTree()
    lines = str(root).splitlines()
    assert lines[0].startswith("LGMRES(tolerance=FixedTolerance(1e-08)")
    assert lines[1].startswith("└─ DiagonalScaling")
    assert lines[2].startswith("   └─ Multiplicative")
    assert lines[3].startswith("      ├─ M1: OnField(field=displacement)")
    assert lines[4].startswith("      │  └─ AMG(")
    assert lines[5].startswith("      └─ M2: OnField(field=nonlocal damage)")


def test_the_structure_of_the_block_amg_preset_before_it_has_seen_a_matrix():
    """Its corrections are made per field at the first set-up; before, the view says so."""
    lines = views.structure(blockAMG()).splitlines()
    assert len(lines) == 3 and "ForcingTolerance" in lines[0] and "Multiplicative" in lines[2]


def test_the_formula_puts_each_child_in_the_place_of_its_role():
    root = _solvedTree()
    assert views.formula(root) == "LGMRES(A, D · (Rᵀ AMG(nonlocal damage)⁻¹ R ∘ Rᵀ AMG(displacement)⁻¹ R) · D)⁻¹"
    assert views.formula(root, latex=True).startswith("\\mathrm{LGMRES}\\left(A,\\, D \\, \\left(R^T")


def test_the_schur_complement_formula_and_steps():
    schur = SchurComplement(["a marker"], preconditionS=AMG(), solveA_bb=DirectSolver(), blockFactors="LU")
    assert views.formula(schur) == "BlockLU[A⁻¹, AMG(A)⁻¹]"
    assert [step.child for step in schur.describe().steps] == ["solveA_bb", "preconditionS", "solveA_bb"]
    assert schur.describe().parameters["S̃"] == "A_kk − A_kb A_bb⁻¹ A_bk, local"
    assert SchurComplement(["a marker"], preconditionS=AMG(), approximation="none").describe().parameters["S̃"] == "A_kk"


def test_the_run_statistics_annotate_every_component():
    text = views.runStatistics(_solvedTree())
    assert text.count("applied") == 7
    assert "rows 3000" in text and "rows 1000" in text


def test_the_data_flow_is_a_graph_of_nested_clusters():
    dot = views.dataFlow(_solvedTree())
    assert dot.startswith("digraph composedSolver {")
    assert dot.count("subgraph") == 7
    assert '"r" -> ' in dot and '-> "z"' in dot


def test_the_html_page_holds_every_view(tmp_path):
    path = tmp_path / "solver.html"
    views.writeHtml(_solvedTree(), str(path))
    page = path.read_text(encoding="utf-8")
    for heading in ("Structure", "Formula", "Run statistics", "One application"):
        assert "<h3>" + heading + "</h3>" in page


def test_the_matrix_view_draws_the_given_matrix(tmp_path):
    pytest.importorskip("matplotlib")
    A, _, _ = displacementDamageSystem(n=6)
    figure = views.spy(_solvedTree(), A=A)
    assert len(figure.axes) == 1


def test_a_solver_script_is_the_custom_linear_solver(tmp_path):
    script = tmp_path / "solver.py"
    script.write_text(
        textwrap.dedent(
            """
            from edelweissfe.linsolve.composition import AMG, LGMRES, DiagonalScaling, FixedTolerance, Multiplicative, OnField

            blockGaussSeidel = Multiplicative([OnField("displacement", AMG()), OnField("nonlocal damage", AMG())])
            linearSolver = LGMRES(DiagonalScaling(blockGaussSeidel), tolerance=FixedTolerance(1e-9))
            """
        )
    )
    A, b, fields = displacementDamageSystem(n=10, coupling="symmetric")
    for solver in (
        ComposedLinearSolver(loadSolverScript(str(script))),
        getLinSolverByName("custom", {"script": str(script), "verbosity": "silent"}),
    ):
        solver.setFieldStructure(fields)
        x = solver(A, b)
        assert solver.root.tolerance.eta == 1e-9
        assert np.linalg.norm(b - A @ x) <= 1e-9 * np.linalg.norm(b)


def test_a_script_must_bind_a_composed_solver(tmp_path):
    script = tmp_path / "solver.py"
    script.write_text("linearSolver = 42\n")
    with pytest.raises(TypeError, match="not a composed solver"):
        loadSolverScript(str(script))
    script.write_text("solver = 42\n")
    with pytest.raises(ValueError, match="defines no 'linearSolver'"):
        loadSolverScript(str(script))


def test_a_script_holds_only_the_math(tmp_path):
    script = tmp_path / "solver.py"
    script.write_text(
        "from edelweissfe.linsolve.composition import ComposedLinearSolver, blockAMG\n"
        "linearSolver = ComposedLinearSolver(blockAMG())\n"
    )
    with pytest.raises(TypeError, match="holds only the math"):
        loadSolverScript(str(script))
