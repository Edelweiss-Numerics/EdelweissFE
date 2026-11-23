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


"""The compiled Marmot element wrapper: the element number reaches Marmot, Marmot's exceptions reach Python, and a
buffer of the wrong length is refused before Marmot reads or writes behind its end.

Skipped where Marmot is not built.
"""

import numpy as np
import pytest

elementModule = pytest.importorskip("edelweissfe.elements.marmotelement.element")


def _elementOrSkip(elementType: str, elNumber: int):
    try:
        return elementModule.MarmotElementWrapper(elementType, elNumber)
    except NotImplementedError:
        pytest.skip(f"Marmot has no {elementType} element")


def test_a_misspelled_result_raises_instead_of_aborting():
    """Marmot throws for an unknown state name; undeclared as `except +`, that reached std::terminate."""
    element = _elementOrSkip("CPE4", 42)
    element.setMaterial("LINEARELASTIC", np.array([30000.0, 0.2]))

    with pytest.raises(ValueError, match="nonsense"):
        element.getResultArray("nonsense", 0)


def test_the_element_number_reaches_marmot():
    """__cinit__ runs before __init__: it passed the still unset self._elNumber, so every Marmot element was 0.

    The embedded bond element names itself in its errors (Marmot >= the bond-slip elements)."""
    element = _elementOrSkip("EB2D2Q4", 42)

    with pytest.raises(ValueError, match="EmbeddedBondElement 42:"):
        element.setProperties(np.array([-1.0, -1.0, 1.0]))


def _unitSquareCPE4(density: float = 2.0):
    """A plane-strain CPE4 on the unit square, linear elastic, with a density."""
    from edelweissfe.points.node import Node

    element = _elementOrSkip("CPE4", 1)
    coordinates = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    element.setNodes([Node(i + 1, np.array(c)) for i, c in enumerate(coordinates)])
    element.setProperties(np.array([1.0]))
    element.setMaterial("LINEARELASTIC", np.array([30000.0, 0.2, density]))
    element.initializeElement()
    return element


def _elementCalls(element):
    """Every buffer-taking method of the element, as (method, buffer name, call taking the buffers by name)."""
    n = element.nDof
    nDim = element.nSpatialDimensions

    def buffers(**lengths):
        return {name: np.zeros(length) for name, length in lengths.items()}

    return [
        (
            "computeKernels",
            buffers(Ke=n * n, Pe=n, U=n, dU=n),
            lambda b: element.computeKernels(b["Ke"], b["Pe"], b["U"], b["dU"], 1.0, 1.0),
        ),
        (
            "computeKernelsExplicit",
            buffers(Pe=n, U=n, dU=n),
            lambda b: element.computeKernelsExplicit(b["Pe"], b["U"], b["dU"], 1.0, 1.0),
        ),
        (
            "computeDistributedLoad",
            buffers(P=n, K=n * n, load=1, U=n),
            lambda b: element.computeDistributedLoad("pressure", b["P"], b["K"], 1, b["load"], b["U"], 1.0, 1.0),
        ),
        (
            "computeBodyForce",
            buffers(P=n, K=n * n, load=nDim, U=n),
            lambda b: element.computeBodyForce(b["P"], b["K"], b["load"], b["U"], 1.0, 1.0),
        ),
        ("computeLumpedInertia", buffers(M=n), lambda b: element.computeLumpedInertia(b["M"])),
        ("computeLumpedDamping", buffers(C=n), lambda b: element.computeLumpedDamping(b["C"])),
        ("computeConsistentInertia", buffers(M=n * n), lambda b: element.computeConsistentInertia(b["M"])),
        (
            "computeCriticalTimeStepForExplicitDynamics",
            buffers(Q=n),
            lambda b: element.computeCriticalTimeStepForExplicitDynamics(b["Q"]),
        ),
    ]


def _oneEntryShortCases():
    """(method index, buffer name) of every buffer of every method of :func:`_elementCalls`."""
    from types import SimpleNamespace

    stub = SimpleNamespace(nDof=8, nSpatialDimensions=2)
    return [
        pytest.param(i, name, id=f"{method}-{name}")
        for i, (method, buffers, _) in enumerate(_elementCalls(stub))
        for name in buffers
    ]


@pytest.mark.parametrize("callIndex, bufferName", _oneEntryShortCases())
def test_a_buffer_one_entry_too_short_is_refused_and_not_written(callIndex, bufferName):
    """Marmot writes nDof (or nDof * nDof) entries behind a bare pointer; one entry less is refused beforehand."""
    element = _unitSquareCPE4()
    method, buffers, call = _elementCalls(element)[callIndex]
    buffers[bufferName] = np.zeros(buffers[bufferName].shape[0] - 1)

    with pytest.raises(ValueError, match=f"{method}: {bufferName} has"):
        call(buffers)

    assert not buffers[bufferName].any()


def test_buffers_of_the_required_length_give_the_element_operators():
    """With the required lengths, the operators are those of the unit square: the masses sum to rho * A per
    direction, the stiffness is symmetric and leaves a rigid translation free, and the pressure on one face and
    the body force sum to their loads."""
    density, pressure = 2.0, 3.0
    element = _unitSquareCPE4(density)
    n = element.nDof

    M = np.zeros(n)
    element.computeLumpedInertia(M)
    assert M.sum() == pytest.approx(2 * density)

    MConsistent = np.zeros(n * n)
    element.computeConsistentInertia(MConsistent)
    assert MConsistent.sum() == pytest.approx(2 * density)

    Ke, Pe = np.zeros(n * n), np.zeros(n)
    element.computeKernels(Ke, Pe, np.zeros(n), np.zeros(n), 1.0, 1.0)
    K = Ke.reshape(n, n)
    assert np.allclose(K, K.T)
    assert np.allclose(K @ np.tile([1.0, 0.0], n // 2), 0.0, atol=1e-9 * np.abs(K).max())

    P, KLoad = np.zeros(n), np.zeros(n * n)
    element.computeDistributedLoad("pressure", P, KLoad, 1, np.array([pressure]), np.zeros(n), 1.0, 1.0)
    assert np.linalg.norm(P.reshape(-1, 2).sum(axis=0)) == pytest.approx(pressure)

    P = np.zeros(n)
    element.computeBodyForce(P, np.zeros(n * n), np.array([0.0, -10.0]), np.zeros(n), 1.0, 1.0)
    assert P.reshape(-1, 2).sum(axis=0) == pytest.approx([0.0, 10.0])


def test_the_nodes_must_be_those_of_the_element():
    """Marmot reads nNodes * nSpatialDimensions coordinates."""
    from edelweissfe.points.node import Node

    element = _elementOrSkip("CPE4", 1)
    with pytest.raises(ValueError, match="setNodes: nodes has 3 entries, but 4 are required"):
        element.setNodes([Node(i + 1, np.array([float(i), 0.0])) for i in range(3)])
    with pytest.raises(ValueError, match="setNodes: node coordinates has 12 entries, but 8 are required"):
        element.setNodes([Node(i + 1, np.array([float(i), 0.0, 0.0])) for i in range(4)])


def test_the_marmot_materials_refuse_buffers_of_the_wrong_length():
    """The point-wise Marmot material reads and writes 6 Voigt components and its own number of state variables."""
    materialModule = pytest.importorskip("edelweissfe.materials.marmot.marmothypoelastic")
    material = materialModule.MarmotHypoElasticMaterial("LinearElastic", np.array([30000.0, 0.2]))
    nStateVars = material.getNumberOfRequiredStateVars()

    with pytest.raises(ValueError, match="assignCurrentStateVars: currentStateVars has"):
        material.assignCurrentStateVars(np.zeros(nStateVars + 1))
    material.assignCurrentStateVars(np.zeros(nStateVars))

    for compute in (material.computeStress, material.computePlaneStress, material.computeUniaxialStress):
        with pytest.raises(ValueError, match=f"{compute.__name__}: stress has 5 entries, but 6 are required"):
            compute(np.zeros(5), np.zeros((6, 6)), np.zeros(6), 1.0, 1.0)
        with pytest.raises(ValueError, match=f"{compute.__name__}: dStrain has 5 entries, but 6 are required"):
            compute(np.zeros(6), np.zeros((6, 6)), np.zeros(5), 1.0, 1.0)

    stress, dStress_dStrain = np.zeros(6), np.zeros((6, 6))
    material.computeStress(stress, dStress_dStrain, np.array([1e-3, 0, 0, 0, 0, 0]), 1.0, 1.0)
    assert stress[0] == pytest.approx(dStress_dStrain[0, 0] * 1e-3)


def test_the_gradient_enhanced_marmot_material_refuses_containers_of_the_wrong_shape():
    """The gradient-enhanced material writes 6 stress components and n x n tangents of the nonlocal variables."""
    materialModule = pytest.importorskip("edelweissfe.materials.marmot.marmotgradientenhancedhypoelastic")
    from edelweissfe.materials.base.basegradientenhancedhypoelasticmaterial import (
        GradientEnhancedIncrement,
        GradientEnhancedResponse,
        GradientEnhancedTangents,
    )
    from edelweissfe.utils.exceptions import ModuleNotBuilt

    try:
        material = materialModule.MarmotGradientEnhancedHypoElasticMaterial(
            "AT2PhaseField", np.array([20000.0, 0.2, 0.1, 5.0])
        )
    except ModuleNotBuilt:
        pytest.skip("Marmot has no AT2PhaseField material")

    n = material.nNonlocalVariables
    material.assignCurrentStateVars(np.zeros(material.getNumberOfRequiredStateVars()))

    def containers():
        return (
            GradientEnhancedResponse.createZero(n),
            GradientEnhancedTangents.createZero(n),
            GradientEnhancedIncrement.createZero(n),
        )

    response, tangents, increment = containers()
    response.stress = np.zeros(5)
    with pytest.raises(ValueError, match="computeStress: response.stress has 5 entries, but 6 are required"):
        material.computeStress(response, tangents, increment, 1.0, 1.0)

    response, tangents, increment = containers()
    tangents.dStress_dK = np.zeros((6, n + 1))
    with pytest.raises(ValueError, match="computePlaneStress: tangents.dStress_dK columns has"):
        material.computePlaneStress(response, tangents, increment, 1.0, 1.0)

    response, tangents, increment = containers()
    increment.K = np.zeros(n + 1)
    with pytest.raises(ValueError, match="computeStress: increment.K has"):
        material.computeStress(response, tangents, increment, 1.0, 1.0)

    response, tangents, increment = containers()
    increment.dStrain[0] = 1e-5
    material.computeStress(response, tangents, increment, 1.0, 1.0)
    assert response.stress[0] == pytest.approx(tangents.dStress_dStrain[0, 0] * 1e-5)
