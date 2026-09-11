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
A model built and solved entirely via real Python constructors -- no ``.inp`` file, no
``parseInputFile``, no dependency on the ``InputLanguage`` singleton having been populated.

The first test stops short of driving the model through the production
``StepManager``/``StepAction``/``NIST`` solver stack, and drives the single element to equilibrium
itself instead: it assembles the element's tangent stiffness and internal force vector and solves
the reduced (Dirichlet-eliminated) linear system with plain numpy, which is mathematically exactly
what the production solver does for a single linear-elastic Newton iteration, just without the
``Step``/``StepAction`` wrapping.

Originally, constructing a ``StepAction`` required a fully-populated *parser-shaped* ``action``
dict -- every optional key (``"components"``, ``"analyticalField"``, ``"f(t)"``) had to be present
even as ``None`` -- so hand-assembling one here would have meant writing a second, hidden
input-file parser. The second test below builds a real ``dirichlet.StepAction`` through its typed
constructor, with no dict and no parser anywhere, and checks the boundary condition it produces,
but it stops at ``getPrescribedIncrement`` and never reaches a solver.

With all 13 step actions now built the same way, the third test closes the remaining gap: it
drives a real ``AdaptiveStep`` through the production ``NIST`` solver, with typed step actions and
a real ``FieldOutputController``, mirroring the lifecycle of
``edelweissfe/drivers/inputfiledrivensimulation.py`` -- but built from Python objects rather than
from a parsed input file.
"""

import numpy as np
import pytest

from edelweissfe.config.configurator import loadConfiguration
from edelweissfe.config.elementlibrary import getElementClass
from edelweissfe.config.materiallibrary import getMaterialClass
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.points.node import Node
from edelweissfe.sections.plane import PlaneSectionSchema, Section
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.solvers.nonlinearimplicitstatic import NIST
from edelweissfe.stepactions.dirichlet import StepAction as DirichletStepAction
from edelweissfe.stepactions.nodeforces import StepAction as NodeForcesStepAction
from edelweissfe.steps.adaptivestep import AdaptiveStep
from edelweissfe.steps.stepmanager import StepActionCollection
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.fieldoutput import FieldOutputController


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
        material,
        [model.elementSets["all"]],
        configuration=PlaneSectionSchema(thickness=thickness),
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
    """A real ``dirichlet.StepAction`` constructed through its typed constructor.

    This closes that gap: no ``.inp`` file, no ``parseInputFile``, and no hand-assembled
    parser-shaped ``action`` dict either. The node set is a ``NodeSet``, the prescribed values are
    a ``dict``, and the amplitude is an ordinary Python callable, so nothing about this
    construction path knows that an input file exists.
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
        total += bc.getPrescribedIncrement(timeStep)[0, 0]
    np.testing.assert_allclose(total, 0.5, atol=1e-12)

    # a nonlinear amplitude must actually be nonlinear in the increments, otherwise the callable
    # was silently ignored and the assertion above would pass for the wrong reason
    firstHalf = bc.getPrescribedIncrement(TimeStep(0, 0.5, 0.5, 0.5, 0.5, 0.5))[0, 0]
    assert firstHalf < 0.5 * 0.5

    # updating on the same set is typed too, and re-prescribing must replace, not accumulate
    bc.updateStepAction({0: 1.0, 1: 2.0})
    assert bc.components == [0, 1]
    np.testing.assert_allclose(bc.delta[0], [1.0, 2.0])

    # a component the field does not have must be rejected, rather than silently ignored the way
    # an unknown key in a definition dict would have been
    with pytest.raises(ValueError, match="do not exist on field"):
        bc.updateStepAction({7: 1.0})


def _buildPatchModel(youngsModulus: float, poissonsRatio: float, thickness: float) -> FEModel:
    """Build a 2x2 patch of CPE4 elements on the unit-spaced square [0,2] x [0,2].

    Nodes are labelled row-wise from the bottom left (1..9), so that ``bottom`` = 1,2,3,
    ``middle`` = 4,5,6 and ``top`` = 7,8,9, and ``left`` = 1,4,7 / ``right`` = 3,6,9 are ordered by
    increasing y -- i.e. the i-th node of ``left`` is the mirror image of the i-th node of ``right``
    about the axis x = 1, which is what makes the symmetry assertion below a per-row comparison.

    Parameters
    ----------
    youngsModulus
        Young's modulus of the linear elastic material.
    poissonsRatio
        Poisson's ratio of the linear elastic material.
    thickness
        The out-of-plane thickness of the section.

    Returns
    -------
    FEModel
        The model, with nodes, elements, sets, material and section assigned -- but not yet
        prepared: the lifecycle calls are the business of the test itself.
    """

    nodes = {}
    label = 1
    for j in range(3):
        for i in range(3):
            nodes[label] = Node(label, np.array([float(i), float(j)]))
            label += 1

    ElementClass = getElementClass("CPE4", "edelweiss")
    elements = {}
    # CCW node order per element, matching the Quad4 shape functions (see the first test)
    for elNumber, connectivity in enumerate([(1, 2, 5, 4), (2, 3, 6, 5), (4, 5, 8, 7), (5, 6, 9, 8)], start=1):
        element = ElementClass("CPE4", elNumber)
        element.setNodes([nodes[label] for label in connectivity])
        elements[elNumber] = element

    model = FEModel(2)
    model.nodes.update(nodes)
    model.elements.update(elements)

    model.elementSets["all"] = ElementSet("all", list(elements.values()))
    model.nodeSets["all"] = NodeSet("all", list(nodes.values()))
    model.nodeSets["bottom"] = NodeSet("bottom", [nodes[1], nodes[2], nodes[3]])
    model.nodeSets["middle"] = NodeSet("middle", [nodes[4], nodes[5], nodes[6]])
    model.nodeSets["top"] = NodeSet("top", [nodes[7], nodes[8], nodes[9]])
    model.nodeSets["left"] = NodeSet("left", [nodes[1], nodes[4], nodes[7]])
    model.nodeSets["right"] = NodeSet("right", [nodes[3], nodes[6], nodes[9]])

    material = getMaterialClass("linearelastic", "edelweiss")(np.array([youngsModulus, poissonsRatio]))
    model.materials["linearelastic"] = material

    section = Section(
        "section1",
        model,
        material,
        [model.elementSets["all"]],
        configuration=PlaneSectionSchema(thickness=thickness),
    )
    model.sections["section1"] = section
    for element in elements.values():
        section.assignSectionPropertiesToElement(element)

    return model


def test_full_step_and_solver_cycle_driven_programmatically():
    """A complete ``Step``/``StepAction``/solver cycle driven from Python objects only.

    This is the point where the programmatic path reaches the same place the ``.inp`` front-end
    does: a 2x2 CPE4 patch is built, prepared through the very lifecycle calls
    ``drivers/inputfiledrivensimulation.py`` makes (``prepareYourself``, ``advanceToTime``,
    ``loadConfiguration``, the ``"U"``/``"P"`` field value entries, ``_linkFieldVariableObjects``,
    ``FieldOutputController.initializeJob``), and solved by a real ``NIST`` instance inside a real
    ``AdaptiveStep`` via ``step.solve()`` -- which is exactly what the driver calls. No ``.inp``
    file, no ``parseInputFile``, no ``InputLanguage`` lookup, and no parser-shaped definition dict
    anywhere: every step action is built through its typed constructor, and the step actions are
    handed over in the same ``StepActionCollection`` the ``StepManager`` would fill.

    This relies on constructing *any* ``StepAction`` without a fully-populated parser-shaped
    ``action`` dict (see the module docstring); a step cannot be solved without one -- ``NIST``
    unconditionally reads ``stepActions["dirichlet"]``, ``["nodeforces"]`` and friends.

    The assertions are physical invariants of the setup rather than expected numbers:

    1. The prescribed Dirichlet value is reached *and* traversed along the prescribed amplitude:
       the recorded history of the top edge's vertical displacement must equal
       ``uyPrescribed * t**2`` at every recorded time, for the quadratic ``f(t)`` handed to the
       boundary condition. This constrains both the destination (the amplitude cancels at t=1) and
       the path, so a step action whose amplitude never reached the solver fails it.
    2. Global equilibrium (Newton's third law): the element internal forces are self-equilibrated,
       so the reactions collected on the constrained edges must balance the total externally applied
       nodal load. This holds only if the ``nodeforces`` action actually reached the assembly *and*
       the Newton iteration converged.
    3. Mirror symmetry about x = 1, which the geometry, the boundary conditions and the load all
       respect: the horizontal displacements of the left and right edge must be opposite and the
       vertical ones equal.
    """

    youngsModulus = 1000.0
    poissonsRatio = 0.3
    thickness = 1.0
    uyPrescribed = 0.02
    FyPerNode = 25.0
    nIncrements = 4

    journal = Journal(verbose=False)

    model = _buildPatchModel(youngsModulus, poissonsRatio, thickness)

    # --- the model lifecycle of the production driver, in the production order ---
    model.prepareYourself(journal)
    model.advanceToTime(0.0)

    jobInfo = loadConfiguration(dict())

    for nodeField in model.nodeFields.values():
        nodeField.createFieldValueEntry("U")
        nodeField.createFieldValueEntry("P")

    model._linkFieldVariableObjects(model.nodeSets["all"])

    # --- field outputs, built through the controller's typed methods ---
    fieldOutputController = FieldOutputController(model, journal)
    displacement = model.nodeFields["displacement"]
    fieldOutputController.addPerNodeFieldOutput(
        "topU", displacement.subset(model.nodeSets["top"]), "U", saveHistory=True
    )
    fieldOutputController.addPerNodeFieldOutput("bottomP", displacement.subset(model.nodeSets["bottom"]), "P")
    fieldOutputController.addPerNodeFieldOutput("topP", displacement.subset(model.nodeSets["top"]), "P")
    fieldOutputController.addPerNodeFieldOutput("leftU", displacement.subset(model.nodeSets["left"]), "U")
    fieldOutputController.addPerNodeFieldOutput("rightU", displacement.subset(model.nodeSets["right"]), "U")
    model.fieldOutputController = fieldOutputController
    fieldOutputController.initializeJob()

    # --- step actions, all through their typed constructors, in the collection the
    # StepManager would hand to the step (keyed by step action module name) ---
    stepActions = StepActionCollection()
    stepActions["dirichlet"]["clamp"] = DirichletStepAction(
        "clamp", model.nodeSets["bottom"], "displacement", {0: 0.0, 1: 0.0}, model, journal
    )
    stepActions["dirichlet"]["stretch"] = DirichletStepAction(
        "stretch",
        model.nodeSets["top"],
        "displacement",
        {1: uyPrescribed},
        model,
        journal,
        f_t=lambda t: t**2,
    )
    stepActions["nodeforces"]["pull"] = NodeForcesStepAction(
        "pull", model.nodeSets["middle"], "displacement", np.array([0.0, FyPerNode]), model, journal
    )

    # --- the production solver and step, then solve ---
    solver = NIST(jobInfo, journal)

    step = AdaptiveStep(
        0,
        model,
        fieldOutputController,
        journal,
        jobInfo,
        solver,
        [],
        stepActions,
        stepLength=1.0,
        startInc=1.0 / nIncrements,
        maxInc=1.0 / nIncrements,
    )
    step.solve()
    fieldOutputController.finalizeJob()

    # ------------------------------------------------------------------
    # 1. the prescribed value is reached, along the prescribed amplitude
    # ------------------------------------------------------------------
    # The step starts at time 0 and is one time unit long, so the recorded total time equals the
    # step progress, and the accumulated Dirichlet increments must sum to uyPrescribed * f(t).
    topU = fieldOutputController.fieldOutputs["topU"]
    recordedTime = topU.getTimeHistory()
    uyHistory = topU.getResultHistory()[:, :, 1]

    assert recordedTime[-1] == 1.0
    assert len(np.unique(recordedTime)) == nIncrements + 1  # every increment converged, none lost

    np.testing.assert_allclose(
        uyHistory,
        uyPrescribed * np.tile((recordedTime**2)[:, None], (1, uyHistory.shape[1])),
        atol=1e-12,
        err_msg="the top edge did not follow the prescribed quadratic amplitude to the prescribed value",
    )

    # -------------------------------------------------
    # 2. global equilibrium: reactions balance the load
    # -------------------------------------------------
    # The internal force vector of a solid element is self-equilibrated (a rigid body translation is
    # in its kernel), hence the assembled P sums to zero over all nodes. At convergence P equals the
    # external load on every free DOF, so the reactions on the constrained DOFs -- the clamped bottom
    # edge and the vertically prescribed top edge -- must equal minus the total applied nodal load.
    reactionsBottom = fieldOutputController.fieldOutputs["bottomP"].getLastResult()
    reactionsTop = fieldOutputController.fieldOutputs["topP"].getLastResult()

    appliedFy = FyPerNode * len(model.nodeSets["middle"])
    reactionFy = reactionsBottom[:, 1].sum() + reactionsTop[:, 1].sum()
    np.testing.assert_allclose(
        reactionFy,
        -appliedFy,
        rtol=1e-8,
        err_msg="the vertical reactions do not balance the applied nodal forces",
    )

    # nothing is loaded horizontally, so the horizontal reactions must cancel
    reactionFx = reactionsBottom[:, 0].sum() + reactionsTop[:, 0].sum()
    np.testing.assert_allclose(reactionFx, 0.0, atol=1e-8 * appliedFy)

    # ---------------------------------------
    # 3. mirror symmetry of the response
    # ---------------------------------------
    leftU = fieldOutputController.fieldOutputs["leftU"].getLastResult()
    rightU = fieldOutputController.fieldOutputs["rightU"].getLastResult()

    np.testing.assert_allclose(leftU[:, 0], -rightU[:, 0], atol=1e-12)
    np.testing.assert_allclose(leftU[:, 1], rightU[:, 1], atol=1e-12)

    # the upward pull on the middle row lifts it beyond the prescribed top displacement, so the
    # solution is not merely the homogeneous stretch: the load genuinely deforms the patch
    assert leftU[1, 1] > uyPrescribed
