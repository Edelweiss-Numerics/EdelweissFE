"""Embedded region constraint (perfect bond): Marmot truss elements (T*) and beams (B*) embedded in host
continuum elements by master-slave DOF elimination, verified against analytical solutions; the rotation tying of
embedded beams (rotations=free|axis|host).

Needs the Marmot modules TrussElement and BeamElement.
"""

import math

import numpy as np
import pytest

pytest.importorskip("edelweissfe.elements.marmotelement.element")

from edelweissfe.drivers.inputfiledrivensimulation import (  # noqa: E402
    finiteElementSimulation,
)
from edelweissfe.utils.inputfileparser import parseInputFile  # noqa: E402


def _marmotHas(elementType: str) -> bool:
    from edelweissfe.elements.marmotelement.element import MarmotElementWrapper

    try:
        MarmotElementWrapper(elementType, 1)
        return True
    except NotImplementedError:
        return False


pytestmark = pytest.mark.skipif(
    not all(_marmotHas(t) for t in ("T3D2", "T2D2CR", "B2D2", "B2D3", "B3D2")),
    reason="Marmot is built without the TrussElement / BeamElement modules",
)

E_STEEL, NU_STEEL, E_CONCRETE, NU_CONCRETE = 200000.0, 0.3, 30000.0, 0.2
A_STEEL = 0.01
G_STEEL = E_STEEL / (2 * (1 + NU_STEEL))


def _run(tmp_path, deck: str, *results):
    path = tmp_path / "test.inp"
    path.write_text(deck)
    model, fieldOutputController = finiteElementSimulation(parseInputFile(str(path)), suppressPlots=True)
    return model, [fieldOutputController.fieldOutputs[r] for r in results]


def _bar(firstLabel: int, start, end, nElements: int, elType: str = "T2D2"):
    """*node and *element blocks of a straight bar, and its first and last node labels."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    nodes = "\n".join(
        f"{firstLabel + i}, " + ", ".join(f"{c:.12g}" for c in start + (end - start) * i / nElements)
        for i in range(nElements + 1)
    )
    elements = "\n".join(f"{firstLabel + i}, {firstLabel + i}, {firstLabel + i + 1}" for i in range(nElements))
    return f"*node\n{nodes}\n\n*element, type={elType}\n{elements}\n", firstLabel, firstLabel + nElements


def _concreteStrip(nX=8, nY=2, elType="CPE4"):
    return f"""*material, name=linearelastic, id=concrete, provider=marmot
{E_CONCRETE}, {NU_CONCRETE}

*material, name=linearelastic, id=steel, provider=marmot
{E_STEEL}, 0.3

*modelGenerator, generator=planeRectQuad, name=gen
l=4
h=1
nX={nX}
nY={nY}
elType={elType}

*section, name=concreteSection, thickness=1.0, material=concrete, type=plane
gen_all
"""


def _rebarsInStrip(nRebarElements=5, y=0.37):
    bar, first, last = _bar(100001, (0.0, y), (4.0, y), nRebarElements)
    return f"""{bar}
*elSet, elSet=rebars, generate=True
{first}, {last - 1}, 1

*nSet, nSet=rebarRight
{last}

*section, name=rebarSection, area={A_STEEL}, material=steel, type=truss
rebars
"""


_UNIAXIAL_STEP = """*job, name=job, domain=2d
*solver, solver=NIST, name=theSolver

*fieldOutput
>>perNode, name=RF, nSet=gen_right, field=displacement, result=P, f(x)='sum(x[:,0])', saveHistory=True,
>>perNode, name=RFrebar, nSet=rebarRight, field=displacement, result=P, f(x)='sum(x[:,0])', saveHistory=True,
>>perElement, name=rebarStrain, elSet=rebars, result=strain, quadraturePoint=0, saveHistory=True,

*step, solver=theSolver
maxInc=1.0, minInc=1e-8, maxNumInc=10, maxIter=10, stepLength=1
>>dirichlet, name=left, nSet=gen_left, field=displacement, 1=0
>>dirichlet, name=corner, nSet=gen_leftBottom, field=displacement, 2=0
>>dirichlet, name=right, nSet=gen_right, field=displacement, 1=0.004
"""

#: uniaxial tension of the reinforced plane strain strip at the strain 1e-3: concrete + steel
REACTION_PERFECT_BOND = (E_CONCRETE / (1 - NU_CONCRETE**2) * 1.0 + E_STEEL * A_STEEL) * 1e-3


def test_embedded_region_uniaxial_tension(tmp_path):
    """perfect bond: the non-matching rebar follows the host strain, the reaction is concrete + steel"""
    deck = (
        _concreteStrip()
        + _rebarsInStrip()
        + """
*constraint, type=embeddedRegion, name=embedded
embeddedElSet=rebars, hostElSet=gen_all
"""
        + _UNIAXIAL_STEP
    )
    _, (RF, RFrebar, strain) = _run(tmp_path, deck, "RF", "RFrebar", "rebarStrain")
    np.testing.assert_allclose(strain.getLastResult(), 1e-3, rtol=1e-10)
    # the end node of the rebar is a slave: its share of the reaction is reported on it
    np.testing.assert_allclose(RF.getLastResult() + RFrebar.getLastResult(), REACTION_PERFECT_BOND, rtol=1e-10)


def test_embedded_region_rejects_unembedded_nodes(tmp_path):
    deck = (
        _concreteStrip()
        + _rebarsInStrip().replace("100001, 0, 0.37", "100001, -0.5, 0.37")
        + """
*constraint, type=embeddedRegion, name=embedded
embeddedElSet=rebars, hostElSet=gen_all
"""
        + _UNIAXIAL_STEP
    )
    with pytest.raises(ValueError, match="outside of all host elements"):
        _run(tmp_path, deck, "RF")


def _line(firstLabel: int, start, end, nElements: int, elType: str, quadratic: bool = None):
    """*node and *element blocks of a straight line of 2-node (or 3-node: end, end, mid, the mid nodes labeled from
    firstLabel + 50000) elements, and its first and last node labels."""
    if quadratic is None:
        quadratic = elType.endswith("3") or elType in ("T2D3", "T3D3")
    start, end = np.asarray(start, float), np.asarray(end, float)

    def node(label, s):
        return f"{label}, " + ", ".join(f"{c:.12g}" for c in start + (end - start) * s)

    nodes = [node(firstLabel + i, i / nElements) for i in range(nElements + 1)]
    if quadratic:
        nodes += [node(firstLabel + 50000 + i, (i + 0.5) / nElements) for i in range(nElements)]
    elements = "\n".join(
        f"{firstLabel + i}, {firstLabel + i}, {firstLabel + i + 1}"
        + (f", {firstLabel + 50000 + i}" if quadratic else "")
        for i in range(nElements)
    )
    nodes = "\n".join(nodes)
    return f"*node\n{nodes}\n\n*element, type={elType}\n{elements}\n", firstLabel, firstLabel + nElements


def _strip(nX=8, nY=2, elType="CPE4", E=E_CONCRETE, lX=4.0, lY=1.0):
    return f"""*material, name=linearelastic, id=concrete, provider=marmot
{E}, {NU_CONCRETE}

*material, name=linearelastic, id=steel, provider=marmot
{E_STEEL}, {NU_STEEL}

*modelGenerator, generator=planeRectQuad, name=gen
l={lX}
h={lY}
nX={nX}
nY={nY}
elType={elType}

*section, name=concreteSection, thickness=1.0, material=concrete, type=plane
gen_all
"""


@pytest.mark.parametrize("rotations", ["free", "axis", "host"])
def test_embedded_beam_follows_uniform_host_strain(tmp_path, rotations):
    """an inclined beam embedded in a strip in uniaxial stress (plane strain): the beam takes the host strain along
    its axis; it stays straight (zero curvature) with free rotations or rotations tied to the host material line
    along the axis, but is bent if tied to the host rotation (zero here), as the uniform strain rotates the inclined
    material line"""
    start, end = np.array([0.3, 0.2]), np.array([3.7, 0.8])
    line, first, last = _line(100001, start, end, 5, "B2D2")
    deck = (
        _strip()
        + f"""{line}
*elSet, elSet=beams, generate=True
{first}, {last - 1}, 1

*section, name=beamSection, type=beam, profile=generic, area=1e-8, I=1e-8, material=steel
beams

*constraint, type=embeddedRegion, name=embedded
embeddedElSet=beams, hostElSet=gen_all, rotations={rotations}

*job, name=job, domain=2d
*solver, solver=NIST, name=theSolver

*fieldOutput
>>perElement, name=strain, elSet=beams, result=axial strain, quadraturePoint=0, saveHistory=True,
>>perElement, name=kappa, elSet=beams, result=curvature, quadraturePoint=0, saveHistory=True,

*step, solver=theSolver
maxInc=1.0, minInc=1e-8, maxNumInc=10, maxIter=10, stepLength=1
>>dirichlet, name=left, nSet=gen_left, field=displacement, 1=0
>>dirichlet, name=corner, nSet=gen_leftBottom, field=displacement, 2=0
>>dirichlet, name=right, nSet=gen_right, field=displacement, 1=0.004
"""
    )
    _, (strain, kappa) = _run(tmp_path, deck, "strain", "kappa")
    t = (end - start) / np.linalg.norm(end - start)
    # the beam is negligibly stiff (E A / (E_c A_c) ~ 7e-5): the host stays in the uniform strain of the plain strip
    epsXX = 1e-3
    epsYY = -NU_CONCRETE / (1 - NU_CONCRETE) * epsXX
    curvature = np.asarray(kappa.getLastResult()).ravel()
    # the material line rotates by theta = t x (eps t); a beam element of length l with the chord rotated by theta,
    # but both nodes at zero rotation (rotations=host), has the curvature 6 theta / l at its ends (here at the
    # first quadrature point, (6 - 12 s) theta / l with s = (1 - sqrt(3/5)) / 2)
    theta = t[0] * t[1] * (epsYY - epsXX)
    elementLength = np.linalg.norm(end - start) / 5
    s = (1 - math.sqrt(0.6)) / 2
    curvatureOfHostRotation = (6 - 12 * s) * theta / elementLength
    if rotations == "host":
        np.testing.assert_allclose(curvature, curvatureOfHostRotation, rtol=1e-3)
        return
    np.testing.assert_allclose(strain.getLastResult(), t[0] ** 2 * epsXX + t[1] ** 2 * epsYY, rtol=1e-4)
    assert np.abs(curvature).max() < 1e-4 * np.abs(curvatureOfHostRotation)


@pytest.mark.parametrize("rotations, elType", [("axis", "B2D2"), ("free", "B2D2"), ("axis", "B2D3")])
def test_embedded_cantilever_clamped_in_stiff_block(tmp_path, rotations, elType):
    """a cantilever embedded with its root in a (practically rigid) block: with the rotations tied to the host, the
    free part deflects as a cantilever clamped at the block face, P L^3 / (3 E I); with free rotations the root is
    not clamped (only the nodal displacements are tied) and the beam deflects more"""
    b, h, L, P = 10.0, 10.0, 200.0, 1.0
    I = b * h**3 / 12  # noqa: E741
    embedded, _, _ = _line(100001, (50.0, 50.0), (100.0, 50.0), 2, elType)
    free, _, tip = _line(100003, (100.0, 50.0), (100.0 + L, 50.0), 8, elType)
    free = free.replace("100003, 100, 50\n", "")  # the face node is the last node of the embedded part
    deck = (
        _strip(nX=4, nY=4, E=E_STEEL * 1e4, lX=100.0, lY=100.0)
        + f"""{embedded}
{free}
*elSet, elSet=beams, generate=True
100001, 100010, 1

*nSet, nSet=tip
{tip}

*section, name=beamSection, type=beam, profile=rectangle, b={b}, h={h}, material=steel
beams

*constraint, type=embeddedRegion, name=embedded
embeddedElSet=beams, hostElSet=gen_all, rotations={rotations}, allowUnembedded=True

*job, name=job, domain=2d
*solver, solver=NIST, name=theSolver

*fieldOutput
>>perNode, name=u, nSet=tip, field=displacement, result=U, saveHistory=True,

*step, solver=theSolver
maxInc=1.0, minInc=1e-8, maxNumInc=10, maxIter=10, stepLength=1
>>dirichlet, name=left, nSet=gen_left, field=displacement, 1=0, 2=0
>>nodeforces, name=P, nSet=tip, field=displacement, 2={P}
"""
    )
    _, (u,) = _run(tmp_path, deck, "u")
    deflection = u.getLastResult().ravel()[1]
    clamped = P * L**3 / (3 * E_STEEL * I)
    if rotations == "axis":
        np.testing.assert_allclose(deflection, clamped, rtol=1e-5)
    else:
        assert deflection > 1.05 * clamped


def test_embedded_cantilever_clamped_in_stiff_block_3d(tmp_path):
    """3D: a cantilever (B3D2) embedded in a stiff C3D8 block with tied rotations: biaxial bending and torsion of the
    free part as clamped at the block face"""
    b, h, L = 10.0, 20.0, 200.0
    Iz, Iy = b * h**3 / 12, h * b**3 / 12  # h along the local y axis (orientation (0, 1, 0)), b along z
    J = h * b**3 * (1 / 3 - 0.21 * b / h * (1 - b**4 / (12 * h**4)))
    Py, Pz, T = 1.0, 0.5, 20.0
    embedded, _, _ = _line(100001, (50.0, 50.0, 50.0), (100.0, 50.0, 50.0), 2, "B3D2")
    free, _, tip = _line(100003, (100.0, 50.0, 50.0), (100.0 + L, 50.0, 50.0), 6, "B3D2")
    free = free.replace("100003, 100, 50, 50\n", "")
    deck = f"""*material, name=linearelastic, id=block, provider=marmot
{E_STEEL * 1e4}, 0.2

*material, name=linearelastic, id=steel, provider=marmot
{E_STEEL}, {NU_STEEL}

*modelGenerator, generator=boxGen, name=gen
lX=100
lY=100
lZ=100
nX=2
nY=2
nZ=2
elType=C3D8

*section, name=blockSection, material=block, type=solid
gen_all

{embedded}
{free}
*elSet, elSet=beams, generate=True
100001, 100008, 1

*nSet, nSet=tip
{tip}

*section, name=beamSection, type=beam, profile=rectangle, b={b}, h={h}, n1x=0, n1y=1, n1z=0, nY=4, nZ=4, material=steel
beams

*constraint, type=embeddedRegion, name=embedded
embeddedElSet=beams, hostElSet=gen_all, rotations=axis, allowUnembedded=True

*job, name=job, domain=3d
*solver, solver=NIST, name=theSolver

*fieldOutput
>>perNode, name=u, nSet=tip, field=displacement, result=U, saveHistory=True,
>>perNode, name=r, nSet=tip, field=rotation, result=U, saveHistory=True,

*step, solver=theSolver
maxInc=1.0, minInc=1e-8, maxNumInc=10, maxIter=10, stepLength=1
>>dirichlet, name=left, nSet=gen_left, field=displacement, 1=0, 2=0, 3=0
>>nodeforces, name=P, nSet=tip, field=displacement, 2={Py}, 3={Pz}
>>nodeforces, name=T, nSet=tip, field=rotation, 1={T}
"""
    _, (u, r) = _run(tmp_path, deck, "u", "r")
    u, r = u.getLastResult().ravel(), r.getLastResult().ravel()
    np.testing.assert_allclose(u[1], Py * L**3 / (3 * E_STEEL * Iz), rtol=1e-5)
    np.testing.assert_allclose(u[2], Pz * L**3 / (3 * E_STEEL * Iy), rtol=1e-5)
    np.testing.assert_allclose(r[0], T * L / (G_STEEL * J), rtol=1e-5)


@pytest.mark.parametrize("shape, nDim", [("quad4", 2), ("quad8", 2), ("hexa8", 3), ("hexa20", 3)])
def test_rotation_operator(shape, nDim):
    """the rotation operators of the embedded region constraint: a rigid rotation gives the rotation itself (both
    modes); a uniform strain leaves the axis mode at t x (eps t) and the host mode at zero"""
    from types import SimpleNamespace

    from edelweissfe.constraints.embeddedregion import rotationOperator
    from edelweissfe.utils.embedding import HostPoint
    from edelweissfe.utils.isoparametricshapes import NODE_PARAMETRIC_COORDINATES

    rng = np.random.default_rng(1)
    xiNodes = NODE_PARAMETRIC_COORDINATES[shape]
    A = np.eye(nDim) * 2.0 + 0.2 * rng.random((nDim, nDim))  # an affinely distorted element
    X = xiNodes @ A.T + 1.0
    element = SimpleNamespace(ensightType=shape, nodes=[SimpleNamespace(coordinates=x) for x in X])
    point = HostPoint(element, 0.3 * np.ones(nDim) - 0.1)
    t = np.ones(nDim) / math.sqrt(nDim)

    if nDim == 2:
        w = 1e-3
        W = np.array([[0.0, -w], [w, 0.0]])
        rotation = np.array([w])
    else:
        rotation = np.array([1e-3, -2e-3, 0.5e-3])
        W = np.array(
            [[0.0, -rotation[2], rotation[1]], [rotation[2], 0.0, -rotation[0]], [-rotation[1], rotation[0], 0.0]]
        )
    eps = 1e-3 * np.diag(np.arange(1, nDim + 1, dtype=float))

    for mode in ("axis", "host"):
        C = rotationOperator(point, mode, [t])
        np.testing.assert_allclose(np.einsum("rai,ai->r", C, X @ W.T), rotation, atol=1e-15)
        theta = np.einsum("rai,ai->r", C, X @ eps.T)
        if mode == "host":
            np.testing.assert_allclose(theta, 0.0, atol=1e-15)
        elif nDim == 2:
            np.testing.assert_allclose(theta, [t[0] * (eps @ t)[1] - t[1] * (eps @ t)[0]], atol=1e-15)
        else:
            np.testing.assert_allclose(theta, np.cross(t, eps @ t), atol=1e-15)


# ---------------------------------------------------------------------------------------------------------------------
# section points (edelweissfe.utils.beamsections)
# ---------------------------------------------------------------------------------------------------------------------
