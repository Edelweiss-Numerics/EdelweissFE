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


def _bar(firstLabel: int, start, end, nElements: int, elType: str = "T2D2"):
    """*node and *element blocks of a straight bar, and its first and last node labels."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    nodes = "\n".join(
        f"{firstLabel + i}, " + ", ".join(f"{c:.12g}" for c in start + (end - start) * i / nElements)
        for i in range(nElements + 1)
    )
    elements = "\n".join(f"{firstLabel + i}, {firstLabel + i}, {firstLabel + i + 1}" for i in range(nElements))
    return f"*node\n{nodes}\n\n*element, type={elType}\n{elements}\n", firstLabel, firstLabel + nElements


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


def test_truss_finite_strain_frame_invariance(tmp_path):
    """a Neo-Hooke truss stretched by 1.5, once along x and once rotated by 90 degrees: equal normal force, the end
    force along the current axis"""

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
    assert N > 0
    np.testing.assert_allclose(Ny.getLastResult().item(), N, rtol=1e-10)
    np.testing.assert_allclose(RFx.getLastResult().ravel(), [N, 0, 0], atol=1e-9 * N)
    np.testing.assert_allclose(RFy.getLastResult().ravel(), [0, N, 0], atol=1e-9 * N)
