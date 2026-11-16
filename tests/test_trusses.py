"""Truss (bar) elements: the Marmot truss elements T2D2 ... T3D3 (small strain) and T2D2CR ... T3D3CR (finite strain,
co-rotated) with the truss section, verified against analytical solutions.

Needs the Marmot module TrussElement.
"""

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
    not all(_marmotHas(t) for t in ("T3D2", "T2D2CR", "T3D2CR")),
    reason="Marmot is built without the TrussElement module",
)

E_STEEL, A_STEEL = 200000.0, 0.01


def _run(tmp_path, deck: str, *results):
    path = tmp_path / "test.inp"
    path.write_text(deck)
    model, fieldOutputController = finiteElementSimulation(parseInputFile(str(path)), suppressPlots=True)
    return model, [fieldOutputController.fieldOutputs[r] for r in results]


def test_truss_small_strain_inclined(tmp_path):
    """an inclined linear elastic truss: reaction = EA/L x axial elongation"""
    deck = f"""*material, name=linearelastic, id=steel, provider=marmot
{E_STEEL}, 0.3

*node
1, 0.0, 0.0
2, 3.0, 4.0

*element, type=T2D2
1, 1, 2

*elSet, elSet=bar
1

*nSet, nSet=left
1
*nSet, nSet=right
2

*section, name=s, area={A_STEEL}, material=steel, type=truss
bar

*job, name=job, domain=2d
*solver, solver=NIST, name=theSolver

*fieldOutput
>>perNode, name=RF, nSet=right, field=displacement, result=P, saveHistory=True,
>>perElement, name=N, elSet=bar, result=normal force, quadraturePoint=0, saveHistory=True,

*step, solver=theSolver
maxInc=1.0, minInc=1e-8, maxNumInc=10, maxIter=10, stepLength=1
>>dirichlet, name=left, nSet=left, field=displacement, 1=0, 2=0
>>dirichlet, name=right, nSet=right, field=displacement, 1=0.003, 2=0.004
"""
    _, (RF, N) = _run(tmp_path, deck, "RF", "N")
    axialForce = E_STEEL * A_STEEL * 0.005 / 5.0
    np.testing.assert_allclose(RF.getLastResult().ravel(), axialForce * np.array([0.6, 0.8]), rtol=1e-10)
    np.testing.assert_allclose(N.getLastResult().ravel(), axialForce, rtol=1e-10)


def _neoHookeUniaxialNominalStress(stretch: float, K: float, G: float) -> float:
    """the uniaxial nominal (first Piola-Kirchhoff) stress of Marmot's CompressibleNeoHooke (Pence-Gou potential B),
    psi = G/2 (I1 J^(-2/3) - 3) + K/8 (J - 1/J)^2, for F = diag(stretch, mu, mu) with the lateral stretch mu from
    dpsi/dmu = 0 (bisection): P = dpsi/dstretch, independent of Marmot's implementation"""

    def derivatives(mu):
        J, I1 = stretch * mu**2, stretch**2 + 2 * mu**2
        dPsi_dJ = -G / 3 * I1 * J ** (-5 / 3) + K / 4 * (J - 1 / J) * (1 + 1 / J**2)
        dPsi_dI1 = G / 2 * J ** (-2 / 3)
        return dPsi_dI1 * 2 * stretch + dPsi_dJ * mu**2, dPsi_dI1 * 4 * mu + dPsi_dJ * 2 * stretch * mu

    lo, hi = 0.05, 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if derivatives(mid)[1] < 0 else (lo, mid)
    return derivatives(0.5 * (lo + hi))[0]


def test_truss_finite_strain_frame_invariance(tmp_path):
    """a Neo-Hooke truss (area 2) stretched by 1.5, once along x and once rotated by 90 degrees: the normal force is
    the closed-form uniaxial Neo-Hooke response 2 P(1.5), equal in both, and the end force acts along the current
    axis"""

    def deck(u):
        return f"""*material, name=compressibleneohooke, id=rubber, provider=marmot
3000.0, 1000.0, 1e-9

*node
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0

*element, type=T3D2CR
1, 1, 2

*elSet, elSet=bar
1

*nSet, nSet=left
1
*nSet, nSet=right
2

*section, name=s, area=2.0, material=rubber, type=truss
bar

*job, name=job, domain=3d
*solver, solver=NIST, name=theSolver

*fieldOutput
>>perNode, name=RF, nSet=right, field=displacement, result=P, saveHistory=True,
>>perElement, name=N, elSet=bar, result=normal force, quadraturePoint=0, saveHistory=True,

*step, solver=theSolver
maxInc=0.1, minInc=1e-8, maxNumInc=100, maxIter=15, stepLength=1
>>dirichlet, name=left, nSet=left, field=displacement, 1=0, 2=0, 3=0
>>dirichlet, name=right, nSet=right, field=displacement, 1={u[0]}, 2={u[1]}, 3={u[2]}
"""

    _, (RFx, Nx) = _run(tmp_path, deck((0.5, 0.0, 0.0)), "RF", "N")
    _, (RFy, Ny) = _run(tmp_path, deck((-1.0, 1.5, 0.0)), "RF", "N")
    N = Nx.getLastResult().item()
    np.testing.assert_allclose(N, 2.0 * _neoHookeUniaxialNominalStress(1.5, 3000.0, 1000.0), rtol=1e-9)
    np.testing.assert_allclose(Ny.getLastResult().item(), N, rtol=1e-10)
    np.testing.assert_allclose(RFx.getLastResult().ravel(), [N, 0, 0], atol=1e-9 * N)
    np.testing.assert_allclose(RFy.getLastResult().ravel(), [0, N, 0], atol=1e-9 * N)


def test_truss_quadratic_3d_through_the_section(tmp_path):
    """a T3D3 bar (area 0.01) along x, pulled by 0.005 over L = 5 (its mid node held transversally, the zero-energy mode
    of a quadratic truss): N = EA Delta / L, uniform along the bar"""
    deck = f"""*material, name=linearelastic, id=steel, provider=marmot
{E_STEEL}, 0.3

*node
1, 0.0, 0.0, 0.0
2, 5.0, 0.0, 0.0
3, 2.5, 0.0, 0.0

*element, type=T3D3
1, 1, 2, 3

*elSet, elSet=bar
1

*nSet, nSet=left
1
*nSet, nSet=right
2
*nSet, nSet=mid
3

*section, name=s, area={A_STEEL}, material=steel, type=truss
bar

*job, name=job, domain=3d
*solver, solver=NIST, name=theSolver

*fieldOutput
>>perNode, name=RF, nSet=right, field=displacement, result=P, saveHistory=True,
>>perElement, name=N0, elSet=bar, result=normal force, quadraturePoint=0, saveHistory=True,
>>perElement, name=N1, elSet=bar, result=normal force, quadraturePoint=1, saveHistory=True,

*step, solver=theSolver
maxInc=1.0, minInc=1e-8, maxNumInc=10, maxIter=10, stepLength=1
>>dirichlet, name=left, nSet=left, field=displacement, 1=0, 2=0, 3=0
>>dirichlet, name=mid, nSet=mid, field=displacement, 2=0, 3=0
>>dirichlet, name=right, nSet=right, field=displacement, 1=0.005, 2=0, 3=0
"""
    _, (RF, N0, N1) = _run(tmp_path, deck, "RF", "N0", "N1")
    axialForce = E_STEEL * A_STEEL * 0.005 / 5.0
    np.testing.assert_allclose(RF.getLastResult().ravel(), [axialForce, 0, 0], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose([N0.getLastResult().item(), N1.getLastResult().item()], axialForce, rtol=1e-10)


def test_truss_snap_through_equilibrium(tmp_path):
    """the regression deck TrussSnapThroughNeoHooke along its path: the apex force is F = 2 N sin(theta) of the two
    compressed bars (theta from the current geometry, the supports moving apart), the bars shorten monotonically"""
    import pathlib

    deck = (pathlib.Path(__file__).parent.parent / "testfiles/marmot/TrussSnapThroughNeoHooke/test.inp").read_text()
    _, (F, N, uSupport) = _run(tmp_path, deck, "F", "N", "uSupport")
    t = np.asarray(F.getTimeHistory())
    forces = np.asarray(F.getResultHistory()).ravel()
    normal = np.asarray(N.getResultHistory()).reshape(len(t), -1)
    supports = np.asarray(uSupport.getResultHistory()).reshape(len(t), 2, 2)
    yApex = 1.0 - 1.0 * t  # the apex is driven from y = 1 to y = 0
    halfSpan = 10.0 - supports[:, 0, 0]  # the left support moves outwards (negative x)
    sinTheta = yApex / np.hypot(halfSpan, yApex)
    assert np.all(normal[t > 0, 0] < 0) and np.allclose(normal[:, 0], normal[:, 1], rtol=1e-10)  # symmetric, compressed
    np.testing.assert_allclose(forces, 2 * normal[:, 0] * sinTheta, rtol=1e-8, atol=1e-10)
