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
"""The driver of a composed solver, :class:`ComposedLinearSolver`: what blockamg did around its solve -- the warnings
and the journal, the validation of the settings, and the hot reload of the solver script during a run."""

import textwrap

import numpy as np
import pytest
from _linsolvesystems import displacementDamageSystem

pytest.importorskip("edelweissfe.linsolve.amgcl.amgcl")

from edelweissfe.linsolve.base import FieldBlock  # noqa: E402
from edelweissfe.linsolve.composition import (  # noqa: E402
    AMG,
    LGMRES,
    ComposedLinearSolver,
    FixedTolerance,
    ForcingTolerance,
    GapCompensation,
    OnField,
    P1Interpolation,
    RebuildPolicy,
    RigidBodyModes,
    SetUpContext,
    SolveDumps,
    SystemMatrix,
    TwoLevel,
    blockAMG,
)


class _Journal:
    """Collects the messages of a solver."""

    def __init__(self):
        self.messages = []
        self.frames = []

    def message(self, text, identification, level=None):
        self.messages.append(text)

    def framedMessage(self, text, identification, title, level=None):
        self.frames.append(title)
        self.messages.append(text)

    def take(self) -> str:
        """All messages since the last call, as one text."""
        text = "\n".join(self.messages)
        self.messages = []
        return text


def _driven(root, fields, **options):
    solver = ComposedLinearSolver(root, **options)
    solver.setFieldStructure(fields)
    journal = _Journal()
    solver.setJournal(journal)
    return solver, journal


# ----- warnings and the journal (verbosity, warnOuterIterationsThreshold) -----


def test_a_slow_solve_is_warned_about():
    A, b, fields = displacementDamageSystem(n=8, coupling="symmetric")
    solver, journal = _driven(blockAMG(tolerance=FixedTolerance(1e-8)), fields, warnOuterIterations=1)
    solver(A, b)
    assert "⚠ solve #1 needed" in journal.take()


def test_a_solve_stopped_at_the_cap_is_warned_about_and_rebuilds_the_preconditioner():
    A, b, fields = displacementDamageSystem(n=8, coupling="symmetric")
    solver, journal = _driven(blockAMG(tolerance=FixedTolerance(1e-12), maxIterations=2), fields)
    solver(A, b)
    assert "⚠ solve #1 did not converge" in journal.take()
    assert not solver.lastSolveSummary.residualMet
    solver(A, b)
    assert solver.root.lastRebuildReason == "stale"


def test_the_debug_column_says_why_the_preconditioner_was_rebuilt():
    A, b, fields = displacementDamageSystem(n=8, coupling="symmetric")
    solver, journal = _driven(blockAMG(), fields, verbosity="debug")
    solver(A, b)
    assert solver.reportsSolveSummary
    assert solver.lastSolveSummary.detailLines[0] == "rebuilt(first)"
    solver(A, 0.5 * b)
    assert solver.lastSolveSummary.detailLines[0] == "reused"


def test_the_tree_is_shown_in_a_frame_once_and_a_solve_is_not_narrated():
    A, b, fields = displacementDamageSystem(n=8, coupling="symmetric")
    solver, journal = _driven(blockAMG(), fields, verbosity="info")
    solver(A, b)
    solver(A, 0.5 * b)
    assert journal.frames == ["composed linear solver"]
    assert "solve #" not in journal.take()


def test_the_continuations_on_the_scaled_system_are_logged_at_debug():
    A, b, fields = displacementDamageSystem(n=8, coupling="symmetric", rowScaling=True)
    solver, journal = _driven(blockAMG(krylovOnScaledSystem=True, tolerance=FixedTolerance(1e-6)), fields)
    solver.verbosity = "debug"
    solver(A, b)
    if solver.lastSolveSummary.retries:
        assert "continuation 1/2" in journal.take()
    assert np.linalg.norm(b - A @ solver(A, b)) <= 1e-6 * np.linalg.norm(b) * (1 + 1e-9)


# ----- the settings are checked at construction, as blockamg checked its keys -----


@pytest.mark.parametrize(
    "construct",
    [
        lambda: FixedTolerance(1.0),
        lambda: ForcingTolerance(minimum=1e-2, maximum=1e-3),
        lambda: ForcingTolerance(gamma=1.5),
        lambda: ForcingTolerance(alpha=1.0),
        lambda: RebuildPolicy(stalenessFactor=0.5),
        lambda: GapCompensation(safetyFactor=0.0),
        lambda: GapCompensation(maxFactor=0.5),
        lambda: LGMRES(AMG(), restart=0),
        lambda: LGMRES(AMG(), continuations=-1),
        lambda: AMG(dropTolerance=-1.0),
        lambda: AMG(backendPrecision="half"),
        lambda: AMG(backendBlockSize=2, backendPrecision="float"),
        lambda: TwoLevel(RigidBodyModes(), coarseHierarchyPrecision="half"),
        lambda: SolveDumps("unused", maxDumps=-1),
        lambda: ComposedLinearSolver(blockAMG(), verbosity="loud"),
        lambda: ComposedLinearSolver(blockAMG(), residualGrowthFactor=1.0),
    ],
)
def test_a_setting_out_of_range_is_refused_at_construction(construct):
    with pytest.raises(ValueError):
        construct()


@pytest.mark.parametrize(
    "component, message",
    [
        (AMG(backendBlockSize=2), "does not match the field's dimension 3"),
        (AMG(tree={"coarsening": {"type": "ruge_stuben"}}), "blockTranslationNullspace=False"),
        (TwoLevel(RigidBodyModes(), backendBlockSize=1), "is on the scalar backend"),
        (TwoLevel(RigidBodyModes(), tree={"relax": {"type": "spai0"}}), "relax.type must be chebyshev"),
        (TwoLevel(P1Interpolation(), backendBlockSize=3), "needs its P1 map"),
    ],
)
def test_a_field_tree_that_cannot_be_built_fails_at_the_set_up(component, message):
    """The checks the components make when they meet the field."""
    A, b, fields = displacementDamageSystem(n=4, coupling="symmetric")
    nNodes = fields[0].stop // 3
    coordinates = np.random.default_rng(0).random((nNodes, 3))
    context = SetUpContext(fields=tuple(fields), coordinatesOf=lambda name: coordinates)
    with pytest.raises(ValueError, match=message):
        OnField("displacement", component).setUp(SystemMatrix(A), context)


def test_node_wise_fine_blocks_of_the_p1_interpolation_need_a_field_of_as_many_components():
    A, _, _ = displacementDamageSystem(n=4, coupling="symmetric")
    n = A.shape[0] // 2 * 2
    context = SetUpContext(fields=(FieldBlock("displacement", 0, n, 2),))
    with pytest.raises(ValueError, match="backendBlockSize 3 does not match the field's dimension 2"):
        TwoLevel(P1Interpolation(), backendBlockSize=3).setUp(SystemMatrix(A[:n, :n]), context)


# ----- type=custom, hotReload=true: the solver script is run again -----


def _script(path, body: str) -> None:
    path.write_text(
        textwrap.dedent(
            """
            from edelweissfe.linsolve.composition import *

            """
        )
        + textwrap.dedent(body)
    )


def _scriptSolver(tmp_path, eta: float, **settings):
    script = tmp_path / "solver.py"
    _script(script, "linearSolver = blockAMG(tolerance=FixedTolerance({:g}))\n".format(eta))
    A, b, fields = displacementDamageSystem(n=8, coupling="symmetric")
    solver = ComposedLinearSolver.fromScript(str(script), hotReload=True, **settings)
    solver.setFieldStructure(fields)
    journal = _Journal()
    solver.setJournal(journal)
    return solver, journal, script, A, b


def test_a_changed_script_replaces_the_tree_and_says_so(tmp_path):
    solver, journal, script, A, b = _scriptSolver(tmp_path, 1e-6)
    solver(A, b)
    assert "hot reload" not in journal.take()

    _script(script, "linearSolver = blockAMG(tolerance=FixedTolerance(1e-9))\n")
    x = solver(A, b)
    text = journal.take()
    assert "hot reload of {:} (sha256 ".format(script) in text and "the tree is replaced" in text
    assert solver.root.tolerance.eta == 1e-9
    assert solver.root.lastRebuildReason == "reload"
    assert np.linalg.norm(b - A @ x) <= 1e-9 * np.linalg.norm(b) * (1 + 1e-9)
    # unchanged since: no reload, the new preconditioner is reused
    solver(A, b)
    assert "hot reload" not in journal.take() and solver.root.lastRebuildReason is None


def test_a_script_that_does_not_compile_is_retried_on_the_next_solve(tmp_path):
    solver, journal, script, A, b = _scriptSolver(tmp_path, 1e-6)
    solver(A, b)
    script.write_text("linearSolver = blockAMG(\n")
    solver(A, b)
    assert "does not compile" in journal.take() and solver.root.tolerance.eta == 1e-6
    solver(A, b)
    assert "does not compile" in journal.take()
    _script(script, "linearSolver = blockAMG(tolerance=FixedTolerance(1e-7))\n")
    solver(A, b)
    assert solver.root.tolerance.eta == 1e-7


def test_a_script_that_fails_is_reported_once_and_the_solver_kept(tmp_path):
    solver, journal, script, A, b = _scriptSolver(tmp_path, 1e-6)
    solver(A, b)
    _script(script, "raise RuntimeError('a typo')\n")
    solver(A, b)
    text = journal.take()
    assert "REJECTED, RuntimeError: a typo" in text and solver.root.tolerance.eta == 1e-6
    solver(A, b)
    assert "hot reload" not in journal.take()


def test_a_new_tree_that_fails_to_set_up_is_rolled_back(tmp_path):
    solver, journal, script, A, b = _scriptSolver(tmp_path, 1e-6)
    solver(A, b)
    standing = solver.root
    _script(script, 'linearSolver = LGMRES(Multiplicative([OnField("no such field", AMG())]))\n')
    x = solver(A, b)
    assert "ROLLED BACK" in journal.take()
    # the tree in force forgot what it built when it was replaced: it is built again
    assert solver.root is standing and solver.root.lastRebuildReason == "reload"
    assert np.linalg.norm(b - A @ x) <= 1e-6 * np.linalg.norm(b) * (1 + 1e-9)


def test_the_dump_numbering_continues_across_a_reload(tmp_path):
    solver, journal, script, A, b = _scriptSolver(tmp_path, 1e-6, dumps=SolveDumps(str(tmp_path / "dumps"), "every"))
    solver(A, b)
    _script(script, "linearSolver = blockAMG(tolerance=FixedTolerance(1e-7))\n")
    solver(A, b)
    files = sorted(path.name for path in (tmp_path / "dumps").glob("A_*.npz"))
    stem = "{:02d}".format(solver._instanceOrdinal)
    assert files == ["A_{:}_00001.npz".format(stem), "A_{:}_00002.npz".format(stem)]
