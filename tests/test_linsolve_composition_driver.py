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
from types import SimpleNamespace

import numpy as np
import pytest
from _linsolvesystems import displacementDamageSystem

pytest.importorskip("edelweissfe.linsolve.amgcl.amgcl")

from edelweissfe.linsolve.base import FieldBlock  # noqa: E402
from edelweissfe.linsolve.composition import (  # noqa: E402
    AMG,
    LGMRES,
    ComposedLinearSolver,
    CornerNodeInterpolation,
    DiagonalScaling,
    FixedTolerance,
    ForcingTolerance,
    GapCompensation,
    OnField,
    RebuildPolicy,
    RigidBodyModesPerAggregate,
    SetUpContext,
    SolveDumps,
    SystemMatrix,
    TwoLevel,
    blockAMG,
)
from edelweissfe.linsolve.composition.amg import (  # noqa: E402
    defaultCornerNodeTwoLevelTree,
    defaultFieldTree,
)
from edelweissfe.linsolve.composition.presets import (  # noqa: E402
    blockAMGFieldPreconditioner,
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


def test_the_default_rebuild_policy_builds_every_preconditioner_anew():
    assert RebuildPolicy().reuse is False


@pytest.mark.parametrize(
    "isCorner, options, expected",
    [
        ([True, False, True], {}, "corner-node"),
        ([True, True, True], {}, "rigid-body"),
        (None, {}, "rigid-body"),
        ([True, False, True], {"cornerNodeInterpolation": False}, "rigid-body"),
        ([True, False, True], {"cornerNodeInterpolation": False, "rigidBodyModesPerAggregate": False}, "AMG"),
    ],
    ids=["quadratic", "linear", "no topology", "corner nodes off", "both off"],
)
def test_the_blockamg_preset_gives_a_3d_displacement_field_the_fastest_measured_cycle(isCorner, options, expected):
    """A quadratic field (with midside nodes) gets the two-level cycle on its corner nodes, a linear one that on its
    rigid-body modes; each can be switched off."""
    nNodes = 3
    midsideCorners = None if isCorner is None else (np.array(isCorner), np.array([[-1, -1], [0, 2], [-1, -1]]))
    context = SetUpContext(
        fields=(FieldBlock("displacement", 0, 3 * nNodes, 3),),
        coordinatesOf=lambda name: np.zeros((nNodes, 3)),
        midsideCornersOf=lambda name: midsideCorners,
    )
    preconditioner = blockAMGFieldPreconditioner(context.fields[0], context, **options)
    if expected == "AMG":
        assert isinstance(preconditioner, AMG)
    else:
        assert isinstance(preconditioner, TwoLevel) and expected in preconditioner.coarseSpace.name


@pytest.mark.parametrize("rigidBodyModesPerAggregate", [True, False], ids=["rigid-body modes", "AMG"])
def test_the_blockamg_preset_gives_the_corner_node_coarse_level_its_options(rigidBodyModesPerAggregate):
    """The preset passes the coarse solve of the corner-node interpolation explicitly: under a diagonal scaling of its
    own, the case of a linear 3D displacement field, with the preset's options; the rigid-body coarse level with the
    smoother of the corner-node two-level cycle, AMG() with its default tree."""
    context = SetUpContext(
        fields=(FieldBlock("displacement", 0, 9, 3),),
        coordinatesOf=lambda name: np.zeros((3, 3)),
        midsideCornersOf=lambda name: (np.array([True, False, True]), np.array([[-1, -1], [0, 2], [-1, -1]])),
    )
    filtering = dict(dropTolerance=1e-3, lumpDroppedEntries=True)
    preconditioner = blockAMGFieldPreconditioner(
        context.fields[0], context, rigidBodyModesPerAggregate=rigidBodyModesPerAggregate, **filtering
    )
    assert isinstance(preconditioner.coarseSpace, CornerNodeInterpolation)
    coarse = preconditioner.coarseSpace.coarseSolve
    assert isinstance(coarse, DiagonalScaling)
    if rigidBodyModesPerAggregate:
        assert isinstance(coarse.inner, TwoLevel) and isinstance(coarse.inner.coarseSpace, RigidBodyModesPerAggregate)
        assert coarse.inner.tree == defaultCornerNodeTwoLevelTree()
    else:
        assert isinstance(coarse.inner, AMG) and coarse.inner.tree is None
    for level in (preconditioner, coarse.inner):
        assert (level.dropTolerance, level.lumpDroppedEntries) == (1e-3, True)


def test_a_field_on_elements_of_an_unknown_topology_gets_the_rigid_body_cycle():
    """A quadratic element the corner-node interpolation does not know (a Tet10, 10 nodes in 3D): the solver finds no
    midside corners for its field, says so once in the log, and the preset gives the rigid-body cycle (it raised before
    #246's review)."""

    class Node:
        """A node as the topology pass reads it (a dict key, so hashable)."""

        fields = {"displacement": True}
        coordinates = np.zeros(3)

    nodes = [Node() for _ in range(10)]
    tet10 = SimpleNamespace(nodes=nodes, fields=[["displacement"]] * 10, nSpatialDimensions=3)
    model = SimpleNamespace(nodeFields={"displacement": SimpleNamespace(nodes=nodes, dimension=3)}, elements={1: tet10})
    solver, journal = _driven(blockAMG(), [], verbosity="info")
    solver.setModel(model, SimpleNamespace(idcsOfFieldsInDofVector={"displacement": range(0, 30)}))
    assert solver.midsideCornersOf("displacement") is None
    assert "no midside corners for the field 'displacement'" in journal.take()
    field = FieldBlock("displacement", 0, 30, 3)
    context = SetUpContext(
        fields=(field,), coordinatesOf=solver.coordinatesOf, midsideCornersOf=solver.midsideCornersOf
    )
    preconditioner = blockAMGFieldPreconditioner(field, context)
    assert isinstance(preconditioner, TwoLevel) and isinstance(preconditioner.coarseSpace, RigidBodyModesPerAggregate)


def test_the_corner_node_smoother_is_the_vector_default_with_degree_8():
    """One place of the vector-field defaults: the corner-node tree is the 3D default but for its Chebyshev smoother."""
    vector, cornerNode = defaultFieldTree(3), defaultCornerNodeTwoLevelTree()
    assert vector["relax"]["degree"] == 16 and cornerNode["relax"]["degree"] == 8
    assert {key: value for key, value in cornerNode.items() if key != "relax"} == {
        key: value for key, value in vector.items() if key != "relax"
    }


def test_the_debug_column_says_why_the_preconditioner_was_rebuilt():
    A, b, fields = displacementDamageSystem(n=8, coupling="symmetric")
    solver, journal = _driven(blockAMG(rebuild=RebuildPolicy(reuse=True)), fields, verbosity="debug")
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
        lambda: RigidBodyModesPerAggregate(coarseHierarchyPrecision="half"),
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
        (TwoLevel(RigidBodyModesPerAggregate(), backendBlockSize=1), "is on the scalar backend"),
        (TwoLevel(RigidBodyModesPerAggregate(), tree={"relax": {"type": "spai0"}}), "relax.type must be chebyshev"),
        (TwoLevel(CornerNodeInterpolation(), backendBlockSize=3), "needs its midside corners"),
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
        TwoLevel(CornerNodeInterpolation(), backendBlockSize=3).setUp(SystemMatrix(A[:n, :n]), context)


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
    _script(
        script,
        "linearSolver = blockAMG(tolerance=FixedTolerance({:g}), rebuild=RebuildPolicy(reuse=True))\n".format(eta),
    )
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

    _script(script, "linearSolver = blockAMG(tolerance=FixedTolerance(1e-9), rebuild=RebuildPolicy(reuse=True))\n")
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
