"""NISTPArcLength with constraints: the constraint forces enter the residual as in NIST.

A cantilever whose right edge is tied to a reference point RP (linearizedRigidBody) is pulled at RP, by NIST with a
prescribed displacement of RP and by NISTPArcLength under indirect control (load at RP; linear elastic material) and
under dissipation control (prescribed displacement of RP scaled by the load parameter; gradient-enhanced CDP, which
provides the dissipation, loaded within its elastic range). The response is linear, so each arc length run ends in
the state of its NIST reference. With the constraint forces assembled into the internal instead of the external forces,
the arc length solver diverged.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.utils.inputfileparser import parseInputFile

# within the elastic range of the CDP material (max. bending stress about 0.45 MPa)
URP = -0.1

MATERIALS = {
    "linearElastic": dict(
        elType="CPE4\nelProvider=edelweiss",
        material="*material, name=linearelastic, id=theMaterial, provider=edelweiss\n30000.0, 0.2",
    ),
    # E, nu, fcy, fcu, fbu, ftu, Df, Ah, Bh, Ch, Dh, As, epsilonF, l, m, omegaMax, derivativeMethod, dTThreshold, eta
    "GCDP": dict(
        elType="GCPE8R",
        material="*material, name=GCDP, id=theMaterial\n"
        "30000.0, 0.2, 10.0, 31.0, 36.0, 2.7, 0.85, 0.08, 0.003, 2.0, 1e-6, 0.75, 0.0051, 20.0, 1.0, 0.999, 0, 0.0, 0.0",
    ),
}

MODEL = """
{material}

*section, name=section1, thickness=100.0, material=theMaterial, type=plane
all

*modelGenerator, generator=planeRectQuad, name=gen
x0=0, l=1000
y0=0, h=100
elType={elType}
nX=20
nY=4

*node, nSet=RP
99999, 1000.0, 50.0

*constraint, type=linearizedRigidBody, name=tie
nSet=gen_right
referencePoint=RP

*job, name=job, domain=2d
*solver, solver={solver}, name=theSolver

*fieldOutput
>>perNode, name=U, elSet=all, field=displacement, result=U
>>perNode, name=URP, nSet=RP, field=displacement, result=U, f(x)='x[0,1]'
>>perNode, name=PRP, nSet=RP, field=displacement, result=P, f(x)='x[0,1]'
>>perNode, name=PRight, nSet=gen_right, field=displacement, result=P, f(x)='sum(x[:,1])'

*step, solver=theSolver
maxInc={maxInc}, minInc=1e-4, maxNumInc=100, maxIter=10, stepLength=1
>>dirichlet, name=left, nSet=gen_left, field=displacement, 1=0, 2=0
{actions}
"""

NIST = dict(solver="NIST", maxInc=1.0, actions=f">>dirichlet, name=pull, nSet=RP, field=displacement, 2={URP}")

INDIRECT = dict(
    solver="NISTPArcLength",
    maxInc=0.25,
    actions=f""">>options, name=theSolver, arcLengthController=control
>>indirectcontrol, name=control, dof1='model.nodeSets["RP"][0].fields["displacement"]', cVector1='[0,1]', dof2='model.nodeSets["RP"][0].fields["displacement"]', cVector2='[0,0]', L={URP}
>>nodeforces, name=pull, nSet=RP, components='0,-100', field=displacement""",
)

# elastic range: nothing is dissipated, the displacement limit governs every increment
DISSIPATION = dict(
    solver="NISTPArcLength",
    maxInc=0.25,
    actions=f""">>options, name=theSolver, arcLengthController=control, stopCondition='fieldOutputs["URP"] < {URP}+1e-9'
>>dissipationcontrol, name=control, elSet=all, dissipation=1.0, displacement={abs(URP)}
>>dirichlet, name=pull, nSet=RP, field=displacement, 2={URP}""",
)


def _simulate(tmp_path, case, material):
    deck = tmp_path / "test.inp"
    deck.write_text(MODEL.format(**case, **MATERIALS[material]))
    model, fieldOutputController = finiteElementSimulation(parseInputFile(str(deck)), verbose=False, suppressPlots=True)
    return {
        name: np.asarray(fieldOutputController.fieldOutputs[name].getLastResult())
        for name in ("U", "URP", "PRP", "PRight")
    }


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    """NIST results per material."""
    return {material: _simulate(tmp_path_factory.mktemp("nist"), NIST, material) for material in MATERIALS}


def _assertSameState(result, reference):
    assert result["URP"] == pytest.approx(URP, rel=1e-6)
    np.testing.assert_allclose(result["U"], reference["U"], rtol=1e-6, atol=1e-9 * np.abs(reference["U"]).max())
    # the internal forces as in NIST: the reaction on the tied nodes, none at the reference point (no elements)
    assert result["PRight"] == pytest.approx(reference["PRight"], rel=1e-6)
    assert result["PRP"] == pytest.approx(0.0, abs=1e-9 * abs(reference["PRight"]))


def test_indirect_control_with_a_rigid_body_constraint_matches_nist(tmp_path, reference):
    _assertSameState(_simulate(tmp_path, INDIRECT, "linearElastic"), reference["linearElastic"])


def test_dissipation_control_with_a_rigid_body_constraint_matches_nist(tmp_path, reference):
    _assertSameState(_simulate(tmp_path, DISSIPATION, "GCDP"), reference["GCDP"])


# hanging nodes (multi-point constraint by DOF elimination): the AMR test case with GC3D20R elements and the AT2 phase
# field, pulled at x1 by NIST and by NISTPArcLength under indirect control of the displacement of x1
HANGING_NODES = Path(__file__).parents[1] / "testfiles" / "marmot" / "AMR_TwoFieldGC3D20R"

HANGING_NODES_STEP = """*step, solver=S
maxInc=0.25, minInc=1e-4, maxNumInc=50, maxIter=25,
>>options, name=S, arcLengthController=control
>>indirectcontrol, name=control, dof1='model.nodeSets["x1"][0].fields["displacement"]', cVector1='[1,0,0]', dof2='model.nodeSets["x1"][0].fields["displacement"]', cVector2='[0,0,0]', L=0.0002
"""


def _simulateHangingNodes(tmp_path, arcLength):
    deck = (HANGING_NODES / "test.inp").read_text()
    if arcLength:
        deck = deck.replace("*solver, solver=NIST, name=S\n", "*solver, solver=NISTPArcLength, name=S\n")
        deck = deck.replace("*step, solver=S\nmaxInc=1.0, minInc=1e-4, maxNumInc=50, maxIter=25,\n", HANGING_NODES_STEP)
        assert "NISTPArcLength" in deck and "indirectcontrol" in deck
    (tmp_path / "hanging.dat").write_text((HANGING_NODES / "hanging.dat").read_text())
    (tmp_path / "test.inp").write_text(deck)
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        model, fieldOutputController = finiteElementSimulation(
            parseInputFile(str(tmp_path / "test.inp")), verbose=False, suppressPlots=True
        )
    finally:
        os.chdir(cwd)
    return {name: np.asarray(fieldOutputController.fieldOutputs[name].getLastResult()) for name in ("U", "D")}


def test_indirect_control_with_hanging_nodes_matches_nist(tmp_path):
    for sub in ("nist", "arc"):
        (tmp_path / sub).mkdir()
    reference = _simulateHangingNodes(tmp_path / "nist", arcLength=False)
    result = _simulateHangingNodes(tmp_path / "arc", arcLength=True)
    for name in ("U", "D"):
        np.testing.assert_allclose(result[name], reference[name], rtol=1e-5, atol=1e-6 * np.abs(reference[name]).max())
