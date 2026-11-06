"""Bond-slip of embedded reinforcement: the embeddedBond generator with the Marmot bond elements
(EB<nDim>D<nBarNodes><host>, small slip; EBLS<nDim>D<nBarNodes><host>W<n>, large slip) and bond-slip laws, for truss
(T*) and beam (B*) bars, verified against analytical pull-out solutions and against the perfect bond of the
embeddedRegion constraint.

Needs the Marmot modules TrussElement, BeamElement, EmbeddedBondElement, LinearElasticBondSlip and
ModelCode2010BondSlip.
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
    not all(_marmotHas(t) for t in ("T2D2", "T3D2", "B2D2", "EB2D2Q4", "EB3D2H8")),
    reason="Marmot is built without the TrussElement / BeamElement / EmbeddedBondElement modules",
)

E_STEEL, NU_STEEL, A_STEEL, E_CONCRETE, NU_CONCRETE = 200000.0, 0.3, 0.01, 30000.0, 0.2
PERIMETER = 0.3545  # circular bar of area 0.01


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


@pytest.mark.parametrize("elType", ["CPE4", "CPE8"])
def test_stiff_bond_reproduces_perfect_bond(tmp_path, elType):
    """bond-slip with a very stiff linear bond law converges to the perfect bond solution"""
    deck = (
        _concreteStrip(elType=elType)
        + _rebarsInStrip()
        + """
*material, name=LinearElasticBondSlip, id=bond, provider=marmot
1e8, 1e8

*modelGenerator, generator=embeddedBond, name=rebarBond, executeAfterManualGeneration=True
rebarElSet=rebars
hostElSet=gen_all
material=bond
perimeter=0.3545
"""
        + _UNIAXIAL_STEP
    )
    model, (RF, RFrebar) = _run(tmp_path, deck, "RF", "RFrebar")
    # the free rebar end carries no load, the bond transfers it to the host; the short shear lag at the free rebar
    # ends (1/beta = 0.0075 << 4) makes the response only slightly softer than with perfect bond
    assert abs(RFrebar.getLastResult()) < 1e-9
    assert RF.getLastResult() < REACTION_PERFECT_BOND
    np.testing.assert_allclose(RF.getLastResult(), REACTION_PERFECT_BOND, rtol=3e-4)
    # 8 x 2 host elements, the rebar at y = 0.37 crosses all 8 lower ones, split at the 5 rebar elements' nodes
    assert len(model.elementSets["rebarBond_bond"]) == 8 + 4


def _pullOutDeck(bondMaterial: str, nRebarElements: int, steps: str, L=4.0):
    bar, first, last = _bar(100001, (0.0, 0.37), (L, 0.37), nRebarElements)
    return (
        _concreteStrip(nX=5)
        + f"""{bar}
*elSet, elSet=rebars, generate=True
{first}, {last - 1}, 1

*nSet, nSet=loadedEnd
{last}

*section, name=rebarSection, area={A_STEEL}, material=steel, type=truss
rebars

{bondMaterial}

*modelGenerator, generator=embeddedBond, name=rebarBond, executeAfterManualGeneration=True
rebarElSet=rebars
hostElSet=gen_all
material=bond
perimeter={PERIMETER}

*job, name=job, domain=2d
*solver, solver=NIST, name=theSolver

*fieldOutput
>>perNode, name=F, nSet=loadedEnd, field=displacement, result=P, f(x)='sum(x[:,0])', saveHistory=True,
>>perNode, name=s, nSet=loadedEnd, field=displacement, result=U, f(x)='sum(x[:,0])', saveHistory=True,

{steps}
"""
    )


def test_pull_out_linear_bond_vs_analytical(tmp_path):
    """pull-out from a rigid host with a linear bond law: F / u = EA beta tanh(beta L), beta = sqrt(K_t p / EA)"""
    Kt = 1000.0
    deck = _pullOutDeck(
        f"*material, name=LinearElasticBondSlip, id=bond, provider=marmot\n{Kt}, 1e6",
        40,
        """*step, solver=theSolver
maxInc=1.0, minInc=1e-8, maxNumInc=10, maxIter=10, stepLength=1
>>dirichlet, name=host, nSet=gen_all, field=displacement, 1=0, 2=0
>>dirichlet, name=pull, nSet=loadedEnd, field=displacement, 1=0.01
""",
    )
    _, (F,) = _run(tmp_path, deck, "F")
    EA = E_STEEL * A_STEEL
    beta = math.sqrt(Kt * PERIMETER / EA)
    np.testing.assert_allclose(F.getLastResult(), EA * beta * math.tanh(beta * 4.0) * 0.01, rtol=5e-4)


def test_pull_out_model_code_2010(tmp_path):
    """pull-out of a short, stiff bar from a rigid host: nearly uniform slip, so the pull-out force follows the
    bond-slip envelope x perimeter x embedded length, through the peak, the softening and the residual branch"""
    tauMax, L = 2.5 * math.sqrt(30.0), 0.5
    mc2010 = f"*material, name=ModelCode2010BondSlip, id=bond, provider=marmot\n{tauMax}, 1.0, 2.0, 10.0, 0.4, {0.4 * tauMax}, 200.0, 1000.0"
    deck = _pullOutDeck(
        mc2010,
        4,
        """*step, solver=theSolver
maxInc=0.02, minInc=1e-8, maxNumInc=1000, maxIter=15, stepLength=1
>>dirichlet, name=host, nSet=gen_all, field=displacement, 1=0, 2=0
>>dirichlet, name=pull, nSet=loadedEnd, field=displacement, 1=12.0
""",
        L=L,
    ).replace(
        f"area={A_STEEL}", "area=100.0"
    )  # practically rigid bar: uniform slip
    _, (F, s) = _run(tmp_path, deck, "F", "s")

    slip = np.asarray(s.getResultHistory()).ravel()
    force = np.asarray(F.getResultHistory()).ravel()

    def envelope(x):
        s0 = (tauMax / (200.0 * 1.0**0.4)) ** (1 / 0.6)
        return np.select(
            [x <= s0, x <= 1.0, x <= 2.0, x <= 10.0],
            [200.0 * x, tauMax * np.abs(x) ** 0.4, tauMax, tauMax - 0.6 * tauMax * (x - 2.0) / 8.0],
            0.4 * tauMax,
        )

    np.testing.assert_allclose(force, envelope(slip) * PERIMETER * L, rtol=2e-3, atol=1e-6)
    assert force.max() == pytest.approx(tauMax * PERIMETER * L, rel=1e-6)


@pytest.mark.parametrize("barType", ["T3D2", "T3D3"])
def test_bond_3d_inclined_bar_matches_embedded_region(tmp_path, barType):
    """3D: an inclined bar crossing hexahedra, split at the host boundaries (splitBars=True); a stiff bond-slip law
    converges to perfect bond of the split bar"""

    def deck(bondStiffness, perfectBond):
        bar, first, last = _bar(100001, (0.15, 0.2, 0.3), (1.85, 0.75, 0.65), 3, "T3D2")
        if barType == "T3D3":  # quadratic bars: add the mid nodes
            lines = bar.split("*element, type=T3D2\n")[1].strip().splitlines()
            nodes, elements = [], []
            for i, line in enumerate(lines):
                a, b = (int(v) for v in line.split(",")[1:])
                xa = np.array([float(v) for v in bar.splitlines()[1 + a - first].split(",")[1:]])
                xb = np.array([float(v) for v in bar.splitlines()[1 + b - first].split(",")[1:]])
                nodes.append(f"{200001 + i}, " + ", ".join(f"{c:.12g}" for c in 0.5 * (xa + xb)))
                elements.append(f"{first + i}, {a}, {b}, {200001 + i}")
            bar = bar.split("*element")[0] + "\n".join(nodes) + "\n\n*element, type=T3D3\n" + "\n".join(elements)
        coupling = (
            "*constraint, type=embeddedRegion, name=embedded\nembeddedElSet=rebars, hostElSet=gen_all"
            if perfectBond
            else ""
        )
        return f"""*material, name=linearelastic, id=concrete, provider=marmot
{E_CONCRETE}, {NU_CONCRETE}

*material, name=linearelastic, id=steel, provider=marmot
{E_STEEL}, 0.3

*material, name=LinearElasticBondSlip, id=bond, provider=marmot
{bondStiffness}, {bondStiffness}

*modelGenerator, generator=boxGen, name=gen
lX=2
lY=1
lZ=1
nX=4
nY=2
nZ=2
elType=C3D8

*section, name=concreteSection, material=concrete, type=solid
gen_all

{bar}

*elSet, elSet=rebars, generate=True
{first}, {last - 1}, 1

*nSet, nSet=rebarNodes, generate=True
{first}, {last}, 1

*section, name=rebarSection, area={A_STEEL}, material=steel, type=truss
rebars

*modelGenerator, generator=embeddedBond, name=rebarBond, executeAfterManualGeneration=True
rebarElSet=rebars
hostElSet=gen_all
material=bond
perimeter=0.3545
splitBars=True

{coupling}

*job, name=job, domain=3d
*solver, solver=NIST, name=theSolver

*fieldOutput
>>perNode, name=uRebar, nSet=rebarNodes, field=displacement, result=U, saveHistory=True,
>>perNode, name=uHost, elSet=gen_all, field=displacement, result=U, saveHistory=True,

*step, solver=theSolver
maxInc=1.0, minInc=1e-8, maxNumInc=10, maxIter=10, stepLength=1
>>dirichlet, name=left, nSet=gen_left, field=displacement, 1=0, 2=0, 3=0
>>dirichlet, name=right, nSet=gen_right, field=displacement, 1=0.002, 2=0.001, 3=-0.0005
"""

    # perfect bond of the split bars: zero bond stiffness, embedded region constraint
    _, (uRebarPerfect, uHostPerfect) = _run(tmp_path, deck(0.0, True), "uRebar", "uHost")
    model, (uRebarBond, uHostBond) = _run(tmp_path, deck(1e9, False), "uRebar", "uHost")

    rebars = model.elementSets["rebars"]
    assert len(rebars) > 3  # split at host boundaries
    assert all(len(el.nodes) == (2 if barType == "T3D2" else 3) for el in rebars)
    scale = 0.002
    np.testing.assert_allclose(uRebarBond.getLastResult(), uRebarPerfect.getLastResult(), atol=1e-4 * scale)
    np.testing.assert_allclose(uHostBond.getLastResult(), uHostPerfect.getLastResult(), atol=1e-4 * scale)


def test_bond_law_property_change_and_shared_nodes(tmp_path):
    """a changeMaterialProperty step action drives the bond-slip law of the generated bond elements; a bar sharing
    nodes with its host is rejected by the bond generator"""
    Kt = 1000.0
    deck = _pullOutDeck(
        f"*material, name=LinearElasticBondSlip, id=bond, provider=marmot\n{Kt}, 1e6",
        8,
        """*step, solver=theSolver
maxInc=1.0, minInc=1e-8, maxNumInc=10, maxIter=10, stepLength=1
>>dirichlet, name=host, nSet=gen_all, field=displacement, 1=0, 2=0
>>dirichlet, name=pull, nSet=loadedEnd, field=displacement, 1=0.01
>>changeMaterialProperty, name=chProp, material=bond, index=0, f(t)='1000 + 3000 * t'
""",
    )
    _, (F,) = _run(tmp_path, deck, "F")
    _, (FRef,) = _run(
        tmp_path, deck.replace(f"\n{Kt}, 1e6", "\n4000.0, 1e6").replace(">>changeMaterialProperty", "**"), "F"
    )
    np.testing.assert_allclose(F.getLastResult(), FRef.getLastResult(), rtol=1e-10)

    # a bar along the bottom edge of the strip, using the host nodes
    shared = (
        _concreteStrip(nX=2, nY=1)
        + """*element, type=T2D2
100001, 1, 3

*elSet, elSet=rebars
100001

*section, name=rebarSection, area=0.01, material=steel, type=truss
rebars

*material, name=LinearElasticBondSlip, id=bond, provider=marmot
1000.0, 1e6

*modelGenerator, generator=embeddedBond, name=rebarBond, executeAfterManualGeneration=True
rebarElSet=rebars
hostElSet=gen_all
material=bond
perimeter=0.3545
"""
        + _UNIAXIAL_STEP.replace(">>perNode, name=RFrebar", "**").replace(">>perElement, name=rebarStrain", "**")
    )
    with pytest.raises(ValueError, match="are nodes of host element"):
        _run(tmp_path, shared, "RF")


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


def _beamPullOutDeck(barSection: str, elType: str, bond: str, generatorOptions: str = ""):
    """a bar along y = 0.37 through a rigid strip (x = 0..4), 10 elements, pulled at x = 4"""
    bar, first, last = _line(100001, (0.0, 0.37), (4.0, 0.37), 10, elType)
    return (
        _strip(nX=5)
        + f"""{bar}
*elSet, elSet=rebars, generate=True
{first}, {last - 1}, 1

*nSet, nSet=loadedEnd
{last}

*section, name=rebarSection, {barSection}, material=steel
rebars

*material, name=LinearElasticBondSlip, id=bond, provider=marmot
{bond}

*modelGenerator, generator=embeddedBond, name=rebarBond, executeAfterManualGeneration=True
rebarElSet=rebars
hostElSet=gen_all
material=bond
perimeter=0.3545
{generatorOptions}

*job, name=job, domain=2d
*solver, solver=NIST, name=theSolver

*fieldOutput
>>perNode, name=F, nSet=loadedEnd, field=displacement, result=P, f(x)='sum(x[:,0])', saveHistory=True,

*step, solver=theSolver
maxInc=0.25, minInc=1e-8, maxNumInc=100, maxIter=10, stepLength=1
>>dirichlet, name=host, nSet=gen_all, field=displacement, 1=0, 2=0
>>dirichlet, name=pull, nSet=loadedEnd, field=displacement, 1=0.01
"""
    )


@pytest.mark.parametrize(
    "generatorOptions, beamType, trussType",
    [("", "B2D2", "T2D2"), ("", "B2D3", "T2D3")],
)
def test_bond_slip_pull_out_beam_equals_truss(tmp_path, generatorOptions, beamType, trussType):
    """axial pull-out of a beam with bond-slip gives the force of a truss of the same area
    and the analytical F = E A beta tanh(beta L) u; the beam rotations stay zero"""
    A, Kt = 0.01, 1000.0
    _, (FTruss,) = _run(
        tmp_path, _beamPullOutDeck(f"type=truss, area={A}", trussType, f"{Kt}, 1e6", generatorOptions), "F"
    )
    _, (FBeam,) = _run(
        tmp_path,
        _beamPullOutDeck(f"type=beam, profile=generic, area={A}, I=1e-5", beamType, f"{Kt}, 1e6", generatorOptions),
        "F",
    )
    np.testing.assert_allclose(FBeam.getLastResult(), FTruss.getLastResult(), rtol=1e-9)
    EA = E_STEEL * A
    beta = math.sqrt(Kt * 0.3545 / EA)
    np.testing.assert_allclose(FBeam.getLastResult(), EA * beta * math.tanh(beta * 4.0) * 0.01, rtol=5e-3)


def test_bond_slip_split_beams_match_perfect_bond(tmp_path):
    """an inclined beam in a strip bent by a transverse end displacement: with splitBars=True, the beam is re-meshed
    with B2D2 elements at the host crossings, and a stiff bond converges to perfect bond (displacements tied, rotations
    free) of the split beam"""

    def deck(bondStiffness, perfectBond):
        line, first, last = _line(100001, (0.3, 0.2), (3.7, 0.8), 3, "B2D2")
        coupling = (
            "*constraint, type=embeddedRegion, name=embedded\nembeddedElSet=beams, hostElSet=gen_all"
            if perfectBond
            else ""
        )
        return (
            _strip()
            + f"""{line}
*elSet, elSet=beams, generate=True
{first}, {last - 1}, 1

*section, name=beamSection, type=beam, profile=generic, area=0.01, I=1e-4, material=steel
beams

*material, name=LinearElasticBondSlip, id=bond, provider=marmot
{bondStiffness}, {bondStiffness}

*modelGenerator, generator=embeddedBond, name=beamBond, executeAfterManualGeneration=True
rebarElSet=beams
hostElSet=gen_all
material=bond
perimeter=0.3545
splitBars=True

{coupling}

*job, name=job, domain=2d
*solver, solver=NIST, name=theSolver

*fieldOutput
>>perNode, name=uBeam, elSet=beams, field=displacement, result=U, saveHistory=True,
>>perNode, name=rBeam, elSet=beams, field=rotation, result=U, saveHistory=True,
>>perNode, name=uHost, elSet=gen_all, field=displacement, result=U, saveHistory=True,

*step, solver=theSolver
maxInc=1.0, minInc=1e-8, maxNumInc=10, maxIter=10, stepLength=1
>>dirichlet, name=left, nSet=gen_left, field=displacement, 1=0, 2=0
>>dirichlet, name=right, nSet=gen_right, field=displacement, 2=0.01
"""
        )

    model, (uPerfect, rPerfect, hostPerfect) = _run(tmp_path, deck(0.0, True), "uBeam", "rBeam", "uHost")
    model, (uBond, rBond, hostBond) = _run(tmp_path, deck(1e10, False), "uBeam", "rBeam", "uHost")
    beams = model.elementSets["beams"]
    assert len(beams) > 3 and all(el.elType == "B2D2" for el in beams)
    np.testing.assert_allclose(uBond.getLastResult(), uPerfect.getLastResult(), atol=1e-6 * 0.01)
    np.testing.assert_allclose(hostBond.getLastResult(), hostPerfect.getLastResult(), atol=1e-6 * 0.01)
    # the rotations follow from the bending of the beam between the bond points; the finite bond stiffness shows most
    rScale = np.abs(rPerfect.getLastResult()).max()
    assert rScale > 1e-4
    np.testing.assert_allclose(rBond.getLastResult(), rPerfect.getLastResult(), atol=1e-4 * rScale)


# ---------------------------------------------------------------------------------------------------------------------
# large slip (largeSlip=True): Marmot EBLS<nDim>D<nBarNodes><host>W<n> elements
# ---------------------------------------------------------------------------------------------------------------------

largeSlip = pytest.mark.skipif(
    not _marmotHas("EBLS2D2Q4W1"), reason="Marmot is built without the EmbeddedLargeSlipBondElement"
)


def _reverseOddElements(elementBlock: str) -> str:
    """swap the end nodes of every second element in the *element block of a *node + *element text: a chain with mixed orientation"""
    lines = elementBlock.splitlines()
    inElements = False
    for i, line in enumerate(lines):
        if line.startswith("*"):
            inElements = line.lower().startswith("*element")
            continue
        parts = [p.strip() for p in line.split(",")]
        if inElements and len(parts) >= 3 and int(parts[0]) % 2 == 0:
            parts[1], parts[2] = parts[2], parts[1]
            lines[i] = ", ".join(parts)
    return "\n".join(lines) + "\n"


def _largeSlipPullOutDeck(
    bondMaterial: str,
    generatorOptions: str,
    pull: float,
    maxInc: float = 0.05,
    elType: str = "T2D2",
    section: str = "area=100.0, type=truss",
    reverse: bool = False,
    nElements: int = 18,
):
    """a rigid bar along y = 0.37 from x = 0 to 9 (18 elements), bonded in the strip x = 0..4 (rigid host)"""
    bar, first, last = _line(100001, (0.0, 0.37), (9.0, 0.37), nElements, elType)
    if reverse:
        bar = _reverseOddElements(bar)
    midNodes = f"\n{first + 50000}, {last - 1 + 50000}, 1" if elType.endswith("3") else ""
    return (
        _concreteStrip(nX=5)
        + f"""{bar}
*elSet, elSet=rebars, generate=True
{first}, {last - 1}, 1

*nSet, nSet=barNodes, generate=True
{first}, {last}, 1{midNodes}

*nSet, nSet=loadedEnd
{last}

*section, name=rebarSection, {section}, material=steel
rebars

{bondMaterial}

*modelGenerator, generator=embeddedBond, name=rebarBond, executeAfterManualGeneration=True
rebarElSet=rebars
hostElSet=gen_all
material=bond
perimeter={PERIMETER}
{generatorOptions}

*job, name=job, domain=2d
*solver, solver=NIST, name=theSolver

*fieldOutput
>>perNode, name=F, nSet=loadedEnd, field=displacement, result=P, f(x)='sum(x[:,0])', saveHistory=True,
>>perNode, name=s, nSet=loadedEnd, field=displacement, result=U, f(x)='sum(x[:,0])', saveHistory=True,

*step, solver=theSolver
maxInc={maxInc}, minInc=1e-8, maxNumInc=10000, maxIter=15, stepLength=1
>>dirichlet, name=host, nSet=gen_all, field=displacement, 1=0, 2=0
>>dirichlet, name=barY, nSet=barNodes, field=displacement, 2=0
>>dirichlet, name=pull, nSet=loadedEnd, field=displacement, 1={pull}
"""
    )


_MC2010 = (
    "*material, name=ModelCode2010BondSlip, id=bond, provider=marmot\n13.69, 1.0, 2.0, 10.0, 0.4, 5.48, 200.0, 1e4"
)


@largeSlip
def test_large_slip_equals_small_slip_for_small_slip(tmp_path):
    """for slips small compared to the bonded length, the large-slip bond gives the small-slip response"""
    _, (FSmall,) = _run(tmp_path, _largeSlipPullOutDeck(_MC2010, "", 0.02, 0.25), "F")
    _, (FLarge,) = _run(tmp_path, _largeSlipPullOutDeck(_MC2010, "largeSlip=True\nmaxSlip=1.0", 0.02, 0.25), "F")
    np.testing.assert_allclose(
        np.asarray(FLarge.getResultHistory()), np.asarray(FSmall.getResultHistory()), rtol=2e-2, atol=1e-9
    )


def _completePullOut(tmp_path, **deckOptions):
    Kt = 10.0
    deck = _largeSlipPullOutDeck(
        f"*material, name=LinearElasticBondSlip, id=bond, provider=marmot\n{Kt}, 1e4",
        "largeSlip=True\nmaxSlip=6.0\nnIntegrationPoints=3",
        6.0,
        **deckOptions,
    )
    model, (F, s) = _run(tmp_path, deck, "F", "s")
    return Kt, model, np.asarray(s.getResultHistory()).ravel(), np.asarray(F.getResultHistory()).ravel()


@largeSlip
def test_large_slip_complete_pull_out(tmp_path):
    """a rigid bar pulled completely out of a rigid host with a linear bond law: the bonded length shrinks as
    L_b - s, so F = K_t s p (L_b - s), and the force vanishes once the bar has left the host"""
    Lb = 4.0
    Kt, model, slip, force = _completePullOut(tmp_path)
    assert all(el.elType.startswith("EBLS2D2Q4W") for el in model.elementSets["rebarBond_bond"])
    for target in (0.5, 1.0, 2.0, 3.0):
        i = np.argmin(np.abs(slip - target))
        expected = Kt * slip[i] * PERIMETER * (Lb - slip[i])
        # the bond fades smoothly over the tributary parts at the bar end: a small deviation from the exact length
        np.testing.assert_allclose(force[i], expected, rtol=3e-2, err_msg=f"at s = {slip[i]}")
    assert np.all(np.abs(force[slip > Lb + 0.1]) < 1e-6)  # Newton tolerance
    assert force.max() == pytest.approx(Kt * PERIMETER * Lb**2 / 4, rel=3e-2)


@largeSlip
@pytest.mark.parametrize(
    "elType, section, reverse",
    [
        ("T2D2", "area=100.0, type=truss", True),
        ("T2D3", "area=100.0, type=truss", False),
        ("T2D3", "area=100.0, type=truss", True),
        ("B2D2", "type=beam, profile=generic, area=100.0, I=1.0", False),
        ("B2D3", "type=beam, profile=generic, area=100.0, I=1.0", True),
    ],
)
def test_large_slip_chains_of_quadratic_reversed_and_beam_bars(tmp_path, elType, section, reverse):
    """the complete pull-out does not depend on the bar element type (quadratic, beam) or on the orientation of the
    bar elements in the chain"""
    _, _, slipRef, forceRef = _completePullOut(tmp_path)
    _, model, slip, force = _completePullOut(tmp_path, elType=elType, section=section, reverse=reverse)
    nBarNodes = 3 if elType.endswith("3") else 2
    assert all(el.elType.startswith(f"EBLS2D{nBarNodes}Q4W") for el in model.elementSets["rebarBond_bond"])
    np.testing.assert_allclose(np.interp(slipRef, slip, force), forceRef, rtol=1e-6, atol=1e-6 * forceRef.max())


@largeSlip
@pytest.mark.parametrize(
    "generatorOptions, nElements, branched, match",
    [
        ("largeSlip=True", 18, False, "requires a positive maxSlip"),
        ("largeSlip=True\nmaxSlip=-1.0", 18, False, "requires a positive maxSlip"),
        ("largeSlip=True\nmaxSlip=6.0", 180, False, "needs a window of"),
        ("largeSlip=True\nmaxSlip=1.0", 18, True, "unbranched"),
    ],
)
def test_large_slip_invalid_input(tmp_path, generatorOptions, nElements, branched, match):
    """large slip needs a positive maxSlip within the registered windows (48 bar elements), and unbranched bars"""
    deck = _largeSlipPullOutDeck(_MC2010, generatorOptions, 0.01, nElements=nElements)
    if branched:
        # a third bar element joins the chain at its node 100005
        deck = deck.replace(
            "*elSet, elSet=rebars, generate=True\n100001, 100018, 1",
            "*node\n200001, 2.0, 0.9\n\n*element, type=T2D2\n100100, 100005, 200001\n\n*elSet, elSet=rebars\n"
            + ", ".join(str(label) for label in [*range(100001, 100019), 100100]),
            1,
        )
    with pytest.raises(ValueError, match=match):
        _run(tmp_path, deck, "F")
