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
"""
P0.1 safety net (see PLAN_INPUT_SYSTEM.md): a model built and solved entirely via real Python
constructors -- no ``.inp`` file, no ``parseInputFile``, no dependency on the ``InputLanguage``
singleton having been populated.

The first test stops short of driving the model through the production
``StepManager``/``StepAction``/``NIST`` solver stack, and drives the single element to equilibrium
itself instead: it assembles the element's tangent stiffness and internal force vector and solves
the reduced (Dirichlet-eliminated) linear system with plain numpy, which is mathematically exactly
what the production solver does for a single linear-elastic Newton iteration, just without the
``Step``/``StepAction`` wrapping.

It did so because of a gap that P3(c) has now started closing. Originally, constructing a
``StepAction`` required a fully-populated *parser-shaped* ``action`` dict -- every optional key
(``"components"``, ``"analyticalField"``, ``"f(t)"``) had to be present even as ``None`` -- so
hand-assembling one here would have meant writing a second, hidden input-file parser. The second
test below is the successor that gap was blocking: it builds a real ``dirichlet.StepAction``
through its typed constructor, with no dict and no parser anywhere, and checks the boundary
condition it produces. The remaining 12 step actions still take the dict (they keep working through
the default hooks on ``StepActionBase``), so a full ``StepManager``-driven programmatic cycle waits
on the rest of P3(c).
"""

import numpy as np
import pytest

from edelweissfe.config.elementlibrary import getElementClass
from edelweissfe.config.materiallibrary import getMaterialClass
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.points.node import Node
from edelweissfe.sections.plane import Section
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.stepactions.dirichlet import StepAction as DirichletStepAction
from edelweissfe.timesteppers.timestep import TimeStep


def test_single_cpe4_patch_test_pure_python_no_parser():
    """Build a single CPE4 (plane-strain quad) patch test end to end using only Node, Section,
    a material class, and FEModel -- then solve it directly to equilibrium and check the result
    against exact, mesh-independent physical invariants (Newton's third law / global force
    balance, and the mirror symmetry of the geometry and loading), not a hard-coded number.
    """
    E = 1000.0
    nu = 0.3
    thickness = 1.0

    # unit square, standard CCW node order matching the Quad4 shape functions
    # (see edelweissfe/elements/displacementelement/_elementcomputationmatrices.py: N[0] is 1 at
    # (xi,eta)=(-1,-1), N[1] at (1,-1), N[2] at (1,1), N[3] at (-1,1))
    n1 = Node(1, np.array([0.0, 0.0]))  # bottom left
    n2 = Node(2, np.array([1.0, 0.0]))  # bottom right
    n3 = Node(3, np.array([1.0, 1.0]))  # top right
    n4 = Node(4, np.array([0.0, 1.0]))  # top left

    ElementClass = getElementClass("CPE4", "edelweiss")
    element = ElementClass("CPE4", 1)
    element.setNodes([n1, n2, n3, n4])

    material = getMaterialClass("linearelastic", "edelweiss")(np.array([E, nu]))

    model = FEModel(2)
    for n in (n1, n2, n3, n4):
        model.nodes[n.label] = n
    model.elements[element.elNumber] = element
    model.elementSets["all"] = ElementSet("all", [element])
    model.nodeSets["all"] = NodeSet("all", [n1, n2, n3, n4])
    model.nodeSets["bottom"] = NodeSet("bottom", [n1, n2])
    model.nodeSets["top"] = NodeSet("top", [n3, n4])
    model.materials["linearelastic"] = material

    section = Section(
        "section1",
        model,
        thickness,
        material,
        [model.elementSets["all"]],
        materialParameterFromFieldDefs=[],
        writeMaterialPropertiesToFileDefs=[],
    )
    model.sections["section1"] = section
    section.assignSectionPropertiesToElement(element)

    journal = Journal(verbose=False)
    model.prepareYourself(journal)

    assert "displacement" in model.nodeFields
    assert len(model.nodeFields["displacement"].nodes) == 4

    # --- Drive the element to equilibrium directly (see module docstring for why) ---
    # dof layout for this element: [n1x, n1y, n2x, n2y, n3x, n3y, n4x, n4y]
    fixedDofs = np.array([0, 1, 2, 3])  # n1, n2 (bottom) clamped
    freeDofs = np.array([4, 5, 6, 7])  # n3, n4 (top) free

    FyPerNode = 0.5
    externalForce = np.zeros(8)
    externalForce[5] = FyPerNode  # n3 (top right)
    externalForce[7] = FyPerNode  # n4 (top left)

    zeroU = np.zeros(8)
    K = np.zeros((8, 8))
    P = np.zeros(8)
    element.computeKernels(K, P, zeroU, zeroU, 0.0, 1.0)  # linear elastic -> tangent is exact everywhere

    K_ff = K[np.ix_(freeDofs, freeDofs)]
    U = np.zeros(8)
    U[freeDofs] = np.linalg.solve(K_ff, externalForce[freeDofs])

    Kfinal = np.zeros((8, 8))
    Pfinal = np.zeros(8)
    element.computeKernels(Kfinal, Pfinal, U, U, 0.0, 1.0)

    # the solve must reproduce the applied load exactly at the free dofs (this is what "solving a
    # step" means for a linear-elastic single Newton iteration)
    np.testing.assert_allclose(Pfinal[freeDofs], externalForce[freeDofs], atol=1e-10)

    # Newton's third law: reactions at the clamped dofs must balance the applied load exactly,
    # independent of any specific numeric target
    reactionFy = Pfinal[fixedDofs[1::2]].sum()
    np.testing.assert_allclose(reactionFy, -2 * FyPerNode, atol=1e-10)
    reactionFx = Pfinal[fixedDofs[0::2]].sum()
    np.testing.assert_allclose(reactionFx, 0.0, atol=1e-10)

    # mirror symmetry of geometry and loading about x=0.5 -> mirrored response
    ux_n3, uy_n3 = U[4], U[5]
    ux_n4, uy_n4 = U[6], U[7]
    np.testing.assert_allclose(ux_n4, -ux_n3, atol=1e-10)
    np.testing.assert_allclose(uy_n4, uy_n3, atol=1e-10)

    # a tensile load must elongate the bar
    assert uy_n3 > 0


def test_dirichlet_step_action_built_from_python_without_a_parser_dict():
    """A real ``dirichlet.StepAction`` constructed through its typed L1 constructor.

    This is the P0.1 gap closing: no ``.inp`` file, no ``parseInputFile``, and -- the part that was
    impossible before P3(c) -- no hand-assembled parser-shaped ``action`` dict either. The node set
    is a ``NodeSet``, the prescribed values are a ``dict``, and the amplitude is an ordinary Python
    callable, so nothing about this construction path knows that an input file exists.
    """
    n1 = Node(1, np.array([0.0, 0.0]))
    n2 = Node(2, np.array([1.0, 0.0]))

    model = FEModel(2)
    for n in (n1, n2):
        model.nodes[n.label] = n
    nSet = NodeSet("bottom", [n1, n2])
    model.nodeSets["bottom"] = nSet

    journal = Journal(verbose=False)

    # prescribe only the y component (index 1), leaving x free, and ramp it quadratically
    bc = DirichletStepAction(
        "bottom",
        nSet,
        "displacement",
        {1: 0.5},
        model,
        journal,
        f_t=lambda t: t**2,
    )

    assert bc.components == [1]
    assert bc.field == "displacement"
    assert bc.fieldSize == 2

    # one prescribed component per node in the set
    assert bc.delta.shape == (2, 1)

    # The increment handed to the solver is delta * (f_t(progress) - f_t(progress - increment)),
    # so over the whole step it must sum to the prescribed value regardless of the amplitude: the
    # amplitude shapes the path, not the destination.
    total = 0.0
    nIncrements = 4
    for i in range(nIncrements):
        progress = (i + 1) / nIncrements
        timeStep = TimeStep(i, 1.0 / nIncrements, progress, 1.0 / nIncrements, progress, progress)
        total += bc.getDelta(timeStep)[0, 0]
    np.testing.assert_allclose(total, 0.5, atol=1e-12)

    # a nonlinear amplitude must actually be nonlinear in the increments, otherwise the callable
    # was silently ignored and the assertion above would pass for the wrong reason
    firstHalf = bc.getDelta(TimeStep(0, 0.5, 0.5, 0.5, 0.5, 0.5))[0, 0]
    assert firstHalf < 0.5 * 0.5

    # updating on the same set is typed too, and re-prescribing must replace, not accumulate
    bc.updateStepAction({0: 1.0, 1: 2.0})
    assert bc.components == [0, 1]
    np.testing.assert_allclose(bc.delta[0], [1.0, 2.0])

    # a component the field does not have must be rejected, rather than silently ignored the way
    # an unknown key in a definition dict would have been
    with pytest.raises(ValueError, match="do not exist on field"):
        bc.updateStepAction({7: 1.0})
