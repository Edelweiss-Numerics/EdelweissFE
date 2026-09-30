"""Integrating element states in place, as a Marmot element does in explicit dynamics.

An explicit kernel evaluation makes the accepted state and the state the MarmotElement writes into
one buffer, so that acceptance has nothing to copy. An implicit evaluation, whose increment may be
rejected, splits them into two independent buffers with the same values again.
"""

import numpy as np
import pytest

marmotelement = pytest.importorskip("edelweissfe.elements.marmotelement.element")

from edelweissfe.points.node import Node  # noqa: E402

UNIT_CUBE = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ]
)


def makeElement():
    """A linear elastic C3D8 on the unit cube, with distinct state values."""

    element = marmotelement.MarmotElementWrapper("C3D8", 1)
    element.setNodes([Node(i + 1, coordinates) for i, coordinates in enumerate(UNIT_CUBE)])
    element.setMaterial("LINEARELASTIC", np.array([30000.0, 0.2]))
    element.initializeElement()
    element.setStateVars(np.arange(element.getStateVars().shape[0], dtype=float))
    return element


def evaluateExplicit(element):
    nDof = element.nDof
    element.computeKernelsExplicit(np.zeros(nDof), np.zeros(nDof), np.zeros(nDof), 0.0, 1.0)


def evaluateImplicit(element):
    nDof = element.nDof
    element.computeKernels(np.zeros(nDof * nDof), np.zeros(nDof), np.zeros(nDof), np.zeros(nDof), 0.0, 1.0)


def test_explicit_evaluation_integrates_into_the_accepted_state():
    element = makeElement()
    evaluateExplicit(element)

    # One buffer: whatever the element integrates is accepted without acceptLastState.
    element._stateVarsTemp[0] = -1.0
    assert element.getStateVars()[0] == -1.0

    element.acceptLastState()
    assert element.getStateVars()[0] == -1.0


def test_implicit_evaluation_splits_the_buffers_again_with_the_same_values():
    element = makeElement()
    evaluateExplicit(element)
    accepted = element.getStateVars()

    evaluateImplicit(element)
    np.testing.assert_array_equal(element.getStateVars(), accepted)

    # Two buffers again: a trial state is accepted only by acceptLastState.
    element._stateVarsTemp[0] = -1.0
    assert element.getStateVars()[0] != -1.0

    element.acceptLastState()
    assert element.getStateVars()[0] == -1.0
