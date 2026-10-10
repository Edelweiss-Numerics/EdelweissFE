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
"""``changeMaterialProperty`` changes the properties of a material during an analysis, but must keep
the state of every element using it.

It used to re-assign the section to every element at every increment, which handed each element a
new material along with new, zeroed state variables: the stress, plastic strain, ... of every
element was wiped at every increment, and field outputs kept viewing the orphaned arrays. These
tests drive a plastic material while mildly hardening it, for each element family
``changeMaterialProperty`` can reach and for both Marmot and pure-Python materials, and check

* that a no-op change (the property set to its current value) is bitwise identical to no change,
* that a mild change only mildly changes the state, which therefore survived,
* that the field outputs keep viewing the state the elements actually carry, and
* that :meth:`updateMaterialProperty` keeps the state arrays themselves, not only their values.
"""

import numpy as np
import pytest

from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.utils.inputfileparser import parseInputFile

CHANGE = ">>changeMaterialProperty, name=chProp, material=myMaterial, index={index}, f(t)='{value}'"

BOX_DECK = """
*material, name={material}, id=myMaterial{materialProvider}
{properties}

*job, name=job, domain=3d
*solver, solver=NIST, name=theSolver

*modelGenerator, generator=boxGen, name=gen
nX=1
nY=1
nZ=1
lX=10
lY=10
lZ=10
elType={elType}
{elementProvider}

*section, name=section1, material=myMaterial, type=solid
all

*fieldOutput
>>perElement, elSet=all, result=kappa, name=kappa, quadraturePoint=0:8

*step, solver=theSolver
maxInc=1e0, minInc=1e0, maxNumInc=100, maxIter=25, stepLength=1
>>dirichlet, name=x0u, nSet=gen_left, field=displacement, 1=0.
>>dirichlet, name=y0u, nSet=gen_bottom, field=displacement, 2=0.
>>dirichlet, name=z0u, nSet=gen_back, field=displacement, 3=0.

*step, solver=theSolver
maxInc=1e-1, minInc=1e-6, maxNumInc=1000, maxIter=25, stepLength=1
>>dirichlet, name=top, nSet=gen_top, field=displacement, 2=0.1
{change}
"""

SINGLE_QP_DECK = """
*material, name={material}, id=myMaterial{materialProvider}
{properties}

*section, name=section1, material=myMaterial, type=solid
all

*node
1, 0, 0, 0

*element, provider=MarmotSingleQpElement, type=MarmotMaterialHypoElastic
1, 1

*job, name=job, domain=3d
*solver, solver=NIST, name=theSolver

*fieldOutput
>>perElement, elSet=all, result=kappa, name=kappa, quadraturePoint=0

*step, solver=theSolver
maxInc=1e0, minInc=1e0, maxNumInc=100, maxIter=25, stepLength=1
>>dirichlet, name=prescribed_strain, nSet=all, field=strain symmetric, components='[ 0, x, x, 0, 0, 0 ]',
>>nodeforces, name=prescribed_stress, nSet=all, field=strain symmetric, components='[ x, 0, 0, x, x, x ]'

*step, solver=theSolver
maxInc=1e-1, minInc=1e-6, maxNumInc=100, maxIter=25, stepLength=1
>>dirichlet, name=prescribed_strain, components='[ 0.01, x, x, 0, 0, 0 ]'
{change}
"""

# von Mises: E, nu, yield stress, linear hardening modulus (changed), nonlinear hardening magnitude
# and exponent
VONMISES = dict(material="VonMises", properties="210000, 0.3, 550, 1000, 200, 1400", index=3, value=1000.0)
# finite strain plasticity: formulation, mu, K, yield stress (changed), hardening parameters
NEOHOOKEPLASTIC = dict(
    material="NeoHookePlastic", properties="1, 91304.34783, 100000., 260, 70, 320, 9", index=3, value=260.0
)

FAMILIES = {
    "marmot element, Marmot material": dict(
        VONMISES, deck=BOX_DECK, elType="C3D8", materialProvider="", elementProvider="", marmot=True
    ),
    "edelweiss element, edelweiss material": dict(
        VONMISES,
        deck=BOX_DECK,
        elType="C3D8",
        materialProvider=", provider=edelweiss",
        elementProvider="elProvider=edelweiss",
        marmot=False,
    ),
    "edelweiss total Lagrangian element, edelweiss material": dict(
        NEOHOOKEPLASTIC,
        deck=BOX_DECK,
        elType="C3D8TL",
        materialProvider=", provider=edelweiss",
        elementProvider="elProvider=edelweiss",
        marmot=False,
    ),
    "single qp element, Marmot material": dict(
        VONMISES, deck=SINGLE_QP_DECK, elType="", materialProvider="", elementProvider="", marmot=True
    ),
    "single qp element, edelweiss material": dict(
        VONMISES,
        deck=SINGLE_QP_DECK,
        elType="",
        materialProvider=", provider=edelweiss",
        elementProvider="",
        marmot=False,
    ),
}

NO_CHANGE, NOOP_CHANGE, MILD_CHANGE = "none", "noop", "mild"


def _run(tmp_path, family: str, change: str):
    spec = FAMILIES[family]

    if spec["marmot"]:
        pytest.importorskip("edelweissfe.elements.marmotelement.element")
        pytest.importorskip("edelweissfe.materials.marmot.marmothypoelastic")

    changeDefinition = {
        NO_CHANGE: "",
        NOOP_CHANGE: CHANGE.format(index=spec["index"], value=spec["value"]),
        # a 20 % stiffer hardening by the end of the step
        MILD_CHANGE: CHANGE.format(index=spec["index"], value=f"{spec['value']} * (1 + 0.2 * t)"),
    }[change]

    path = tmp_path / f"{change}.inp"
    path.write_text(
        spec["deck"].format(
            material=spec["material"],
            materialProvider=spec["materialProvider"],
            properties=spec["properties"],
            elType=spec["elType"],
            elementProvider=spec["elementProvider"],
            change=changeDefinition,
        )
    )

    return finiteElementSimulation(parseInputFile(str(path)), verbose=False, suppressPlots=True)


def _solution(model) -> np.ndarray:
    return np.hstack([f["U"].flatten() for f in model.nodeFields.values()])


def _stateVars(model) -> np.ndarray:
    return np.hstack([np.asarray(el.getStateVars()).flatten() for el in model.elements.values()])


def _kappa(model) -> np.ndarray:
    return np.hstack(
        [
            np.copy(el.getResultArray("kappa", qp, getPersistentView=False)).flatten()
            for el in model.elements.values()
            for qp in range(_nQuadraturePoints(el))
        ]
    )


def _nQuadraturePoints(el) -> int:
    return 1 if el.elType == "MarmotMaterialHypoElastic" else 8


@pytest.mark.parametrize("family", FAMILIES)
def test_noop_change_is_bitwise_identical_to_no_change(tmp_path, family):
    reference, _ = _run(tmp_path, family, NO_CHANGE)
    noop, _ = _run(tmp_path, family, NOOP_CHANGE)

    assert np.array_equal(_solution(noop), _solution(reference))
    assert np.array_equal(_stateVars(noop), _stateVars(reference))


@pytest.mark.parametrize("family", FAMILIES)
def test_mild_change_keeps_the_state(tmp_path, family):
    reference, _ = _run(tmp_path, family, NO_CHANGE)
    changed, fieldOutputController = _run(tmp_path, family, MILD_CHANGE)

    kappaReference = _kappa(reference)
    kappaChanged = _kappa(changed)

    # plastic loading throughout the step's ten increments: a wiped state would leave at most the
    # plastic strain of the very last increment, about a tenth of the accumulated one
    assert np.all(kappaReference > 1e-3)
    # a stiffer hardening yields slightly less plastic strain, accumulated over the whole step
    assert np.all(kappaChanged < kappaReference)
    np.testing.assert_allclose(kappaChanged, kappaReference, rtol=5e-2)

    # the field output views the state the elements actually carry, not orphaned arrays
    np.testing.assert_array_equal(
        np.asarray(fieldOutputController.fieldOutputs["kappa"].getLastResult()).flatten(), kappaChanged
    )


@pytest.mark.parametrize("family", FAMILIES)
def test_update_material_property_keeps_the_state_arrays(tmp_path, family):
    model, _ = _run(tmp_path, family, NO_CHANGE)

    for el in model.elements.values():
        stateVars = np.copy(el.getStateVars())
        persistentKappa = el.getResultArray("kappa", 0, getPersistentView=True)
        kappa = np.copy(persistentKappa)

        el.updateMaterialProperty(FAMILIES[family]["index"], 2 * FAMILIES[family]["value"])

        np.testing.assert_array_equal(el.getStateVars(), stateVars)
        assert np.shares_memory(persistentKappa, el.getResultArray("kappa", 0, getPersistentView=True))
        np.testing.assert_array_equal(persistentKappa, kappa)
