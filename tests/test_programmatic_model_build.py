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

This test intentionally stops short of driving the model through the production
``StepManager``/``StepAction``/``NIST`` solver stack: constructing a ``StepAction`` (e.g.
``edelweissfe.stepactions.dirichlet.StepAction``) currently requires a fully-populated,
parser-shaped ``action`` dict (every optional key, e.g. ``"components"``, ``"analyticalField"``,
``"f(t)")``, must already be present, even if ``None``) -- see
``edelweissfe/stepactions/dirichlet.py:105-168``. Hand-assembling that dict here would amount to
writing a second, hidden input-file parser, which is explicitly out of scope. So instead, this
test drives the single element directly to equilibrium: it assembles the element's tangent
stiffness and internal force vector itself and solves the reduced (Dirichlet-eliminated) linear
system with plain numpy -- which is mathematically exactly what the production solver does for a
single, linear-elastic Newton iteration, just without the ``Step``/``StepAction`` wrapping.

The precise gap this documents for P1/P2: ``StepAction`` subclasses need a real L1 constructor
(explicit typed kwargs mirroring ``createFieldOutputFromInputFile``'s pattern in
``edelweissfe/helpers/inputfilehelpers.py:72``), with the current dict-consuming ``__init__``
demoted to a thin adapter called from the L4 (input-file) layer.
"""

import numpy as np

from edelweissfe.config.elementlibrary import getElementClass
from edelweissfe.config.materiallibrary import getMaterialClass
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.points.node import Node
from edelweissfe.sections.plane import Section
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.sets.nodeset import NodeSet


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
