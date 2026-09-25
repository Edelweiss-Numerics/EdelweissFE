"""Integrating element states in place, as the explicit dynamic solver does.

An element that integrates in place writes its new state directly into the accepted state, so it
has no trial state that could be restored. These tests check what that means for a Marmot element:
the accepted state and the state the MarmotElement writes into are one buffer while in place, and
two independent buffers with the same values again afterwards.
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


def makeElement(withMaterial: bool = True):
    """A linear elastic C3D8 on the unit cube."""

    element = marmotelement.MarmotElementWrapper("C3D8", 1)
    element.setNodes([Node(i + 1, coordinates) for i, coordinates in enumerate(UNIT_CUBE)])
    if withMaterial:
        element.setMaterial("LINEARELASTIC", np.array([30000.0, 0.2]))
    element.initializeElement()
    return element


def test_an_element_without_material_declines():
    element = makeElement(withMaterial=False)

    assert not element.requestStateIntegrationInPlace(True)
    assert not element.integratesStateInPlace


def test_in_place_the_accepted_state_is_the_state_the_element_writes():
    element = makeElement()
    values = np.arange(element.getStateVars().shape[0], dtype=float)
    element.setStateVars(values)

    assert element.requestStateIntegrationInPlace(True)
    assert element.integratesStateInPlace
    np.testing.assert_array_equal(element.getStateVars(), values)

    # One buffer: whatever the element integrates is accepted without acceptLastState.
    element._stateVarsTemp[0] = -1.0
    assert element.getStateVars()[0] == -1.0

    element.acceptLastState()
    assert element.getStateVars()[0] == -1.0


def test_switching_off_splits_the_buffers_again_with_the_same_values():
    element = makeElement()
    values = np.arange(element.getStateVars().shape[0], dtype=float)
    element.setStateVars(values)
    element.requestStateIntegrationInPlace(True)

    assert not element.requestStateIntegrationInPlace(False)
    assert not element.integratesStateInPlace
    np.testing.assert_array_equal(element.getStateVars(), values)

    # Two buffers again: a trial state is accepted only by acceptLastState.
    element._stateVarsTemp[0] = -1.0
    assert element.getStateVars()[0] == 0.0

    element.acceptLastState()
    assert element.getStateVars()[0] == -1.0
