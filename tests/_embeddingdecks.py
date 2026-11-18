"""Shared input-deck helpers of the embedding tests (embedded region, embedded bond): Marmot availability, running
a deck, and the deck blocks of straight lines of bar elements and of a plane strain host strip."""

import numpy as np

from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.utils.inputfileparser import parseInputFile

E_STEEL, NU_STEEL, A_STEEL = 200000.0, 0.3, 0.01
E_CONCRETE, NU_CONCRETE = 30000.0, 0.2


def _marmotHas(elementType: str) -> bool:
    """Whether Marmot is built with the element ``elementType``."""
    from edelweissfe.elements.marmotelement.element import MarmotElementWrapper

    try:
        MarmotElementWrapper(elementType, 1)
        return True
    except NotImplementedError:
        return False


def _run(tmp_path, deck: str, *results):
    """Run ``deck`` in ``tmp_path``; the model and the field outputs named ``results``."""
    path = tmp_path / "test.inp"
    path.write_text(deck)
    model, fieldOutputController = finiteElementSimulation(parseInputFile(str(path)), suppressPlots=True)
    return model, [fieldOutputController.fieldOutputs[r] for r in results]


def _line(firstLabel: int, start, end, nElements: int, elType: str = "T2D2", quadratic: bool = None):
    """*node and *element blocks of a straight line of 2-node (or 3-node: end, end, mid, the mid nodes labeled from
    firstLabel + 50000) elements, and its first and last node labels."""
    if quadratic is None:
        quadratic = elType.endswith("3")
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


def _strip(nX=8, nY=2, elType="CPE4", E=E_CONCRETE, lX=4.0, lY=1.0, origin=(0.0, 0.0)):
    """materials and a plane strain strip [0, lX] x [0, lY] (shifted by ``origin``) of host elements ``gen_all``"""
    return f"""*material, name=linearelastic, id=concrete, provider=marmot
{E}, {NU_CONCRETE}

*material, name=linearelastic, id=steel, provider=marmot
{E_STEEL}, {NU_STEEL}

*modelGenerator, generator=planeRectQuad, name=gen
l={lX}
h={lY}
x0={origin[0]}
y0={origin[1]}
nX={nX}
nY={nY}
elType={elType}

*section, name=concreteSection, thickness=1.0, material=concrete, type=plane
gen_all
"""


def _barsInStrip(nBarElements=5, y=0.37):
    """a horizontal line of T2D2 bars ``rebars`` at height ``y`` across the strip of :func:`_strip`, its truss section,
    and the node set ``rebarRight`` of its right end"""
    bar, first, last = _line(100001, (0.0, y), (4.0, y), nBarElements, "T2D2")
    return f"""{bar}
*elSet, elSet=rebars, generate=True
{first}, {last - 1}, 1

*nSet, nSet=rebarRight
{last}

*section, name=rebarSection, area={A_STEEL}, material=steel, type=truss
rebars
"""


#: the field output of the bar strains in :data:`_UNIAXIAL_STEP`; ``.replace`` it away for decks without it
_REBAR_STRAIN_OUTPUT = (
    ">>perElement, name=rebarStrain, elSet=rebars, result=strain, quadraturePoint=0, saveHistory=True,\n"
)

#: uniaxial tension of the strip of :func:`_strip` with the bars of :func:`_barsInStrip`, to the strain 1e-3
_UNIAXIAL_STEP = f"""*job, name=job, domain=2d
*solver, solver=NIST, name=theSolver

*fieldOutput
>>perNode, name=RF, nSet=gen_right, field=displacement, result=P, f(x)='sum(x[:,0])', saveHistory=True,
>>perNode, name=RFrebar, nSet=rebarRight, field=displacement, result=P, f(x)='sum(x[:,0])', saveHistory=True,
{_REBAR_STRAIN_OUTPUT}
*step, solver=theSolver
maxInc=1.0, minInc=1e-8, maxNumInc=10, maxIter=10, stepLength=1
>>dirichlet, name=left, nSet=gen_left, field=displacement, 1=0
>>dirichlet, name=corner, nSet=gen_leftBottom, field=displacement, 2=0
>>dirichlet, name=right, nSet=gen_right, field=displacement, 1=0.004
"""

#: uniaxial tension of the reinforced plane strain strip at the strain 1e-3: concrete + steel
REACTION_PERFECT_BOND = (E_CONCRETE / (1 - NU_CONCRETE**2) * 1.0 + E_STEEL * A_STEEL) * 1e-3
