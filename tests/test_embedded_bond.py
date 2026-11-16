"""Bond-slip of embedded reinforcement: the embeddedBond generator with the Marmot bond elements
(EB<nDim>D<nBarNodes><host>, small slip; EBLS<nDim>D<nBarNodes><host>W<n>, large slip) and bond-slip laws, for truss (T*) and beam (B*) bars, verified against analytical
pull-out solutions and against the perfect bond of the embeddedRegion constraint.

Needs the Marmot modules TrussElement, BeamElement, EmbeddedBondElement, LinearElasticBondSlip and
ModelCode2010BondSlip.
"""

import math
import pathlib

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
#: the orientation test: the largest difference of the force-slip histories, relative to the peak force (from the
#: fade over the tributary parts at the bar end; measured 0.10 % tilted, 0.26 % 3D)
ORIENTATION_ATOL = 0.005


def _run(tmp_path, deck: str, *results):
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


def _triangleStrip(nX=8, nY=2, lX=4.0, lY=1.0):
    """as :func:`_strip`, with CPE3 hosts: every grid cell split into two triangles, and the node sets of the
    generator (``gen_left``, ``gen_right``, ``gen_leftBottom``)"""
    label = {(i, j): 1 + i * (nY + 1) + j for i in range(nX + 1) for j in range(nY + 1)}
    nodes = "\n".join(f"{label[i, j]}, {lX * i / nX:.12g}, {lY * j / nY:.12g}" for (i, j) in label)
    triangles = []
    for i in range(nX):
        for j in range(nY):
            a, b, c, d = label[i, j], label[i + 1, j], label[i + 1, j + 1], label[i, j + 1]
            triangles += [(a, b, c), (a, c, d)]
    elements = "\n".join(f"{k + 1}, {a}, {b}, {c}" for k, (a, b, c) in enumerate(triangles))

    def nodeSet(name, members):
        return f"*nSet, nSet={name}\n" + ", ".join(str(m) for m in members) + "\n"

    return f"""*material, name=linearelastic, id=concrete, provider=marmot
{E_CONCRETE}, {NU_CONCRETE}

*material, name=linearelastic, id=steel, provider=marmot
{E_STEEL}, {NU_STEEL}

*node
{nodes}

*element, type=CPE3
{elements}

*elSet, elSet=gen_all, generate=True
1, {len(triangles)}, 1

{nodeSet("gen_left", [label[0, j] for j in range(nY + 1)])}
{nodeSet("gen_right", [label[nX, j] for j in range(nY + 1)])}
{nodeSet("gen_leftBottom", [label[0, 0]])}
*section, name=concreteSection, thickness=1.0, material=concrete, type=plane
gen_all
"""


def _barsInStrip(nBarElements=5, y=0.37):
    bar, first, last = _line(100001, (0.0, y), (4.0, y), nBarElements)
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

*step, solver=theSolver
maxInc=1.0, minInc=1e-8, maxNumInc=10, maxIter=10, stepLength=1
>>dirichlet, name=left, nSet=gen_left, field=displacement, 1=0
>>dirichlet, name=corner, nSet=gen_leftBottom, field=displacement, 2=0
>>dirichlet, name=right, nSet=gen_right, field=displacement, 1=0.004
"""

#: uniaxial tension of the reinforced plane strain strip at the strain 1e-3: concrete + steel
REACTION_PERFECT_BOND = (E_CONCRETE / (1 - NU_CONCRETE**2) * 1.0 + E_STEEL * A_STEEL) * 1e-3

_STIFF_LINEAR_BOND = f"""
*material, name=LinearElasticBondSlip, id=bond, provider=marmot
1e8, 1e8

*modelGenerator, generator=embeddedBond, name=rebarBond, executeAfterManualGeneration=True
embeddedElSet=rebars
hostElSet=gen_all
material=bond
perimeter={PERIMETER}
"""


@pytest.mark.parametrize("hosts", ["CPE4", "CPE8", "CPE3"])
def test_stiff_bond_reproduces_perfect_bond(tmp_path, hosts):
    """bond-slip with a very stiff linear bond law converges to the perfect bond solution, for linear, quadratic and
    triangular hosts"""
    strip = _triangleStrip() if hosts == "CPE3" else _strip(elType=hosts)
    model, (RF, RFrebar) = _run(tmp_path, strip + _barsInStrip() + _STIFF_LINEAR_BOND + _UNIAXIAL_STEP, "RF", "RFrebar")
    # the free rebar end carries no load, the bond transfers it to the host; the short shear lag at the free rebar
    # ends (1/beta = 0.0075 << 4) makes the response only slightly softer than with perfect bond
    assert abs(RFrebar.getLastResult()) < 1e-9
    assert RF.getLastResult() < REACTION_PERFECT_BOND
    np.testing.assert_allclose(RF.getLastResult(), REACTION_PERFECT_BOND, rtol=3e-4)
    if hosts != "CPE3":
        # 8 x 2 hosts, the bar at y = 0.37 crosses the 8 lower ones; its 5 elements are split at the host crossings
        assert len(model.elementSets["rebarBond_bond"]) == 8 + 4
    assert all(
        el.elType.startswith("EB2D2" + {"CPE4": "Q4", "CPE8": "Q8", "CPE3": "T3"}[hosts])
        for el in model.elementSets["rebarBond_bond"]
    )


_PULL = """*step, solver=theSolver
maxInc={maxInc}, minInc=1e-8, maxNumInc=1000, maxIter=15, stepLength=1
>>dirichlet, name=host, nSet=gen_all, field=displacement, 1=0, 2=0
>>dirichlet, name=pull, nSet=loadedEnd, field=displacement, 1={u}
"""


def _pullOutDeck(
    bond: str,
    nElements: int = 10,
    L: float = 4.0,
    elType: str = "T2D2",
    barSection: str = f"type=truss, area={A_STEEL}",
    generatorOptions: str = "",
    u: float = 0.01,
    maxInc: float = 1.0,
    outputs: str = "",
    origin=(0.0, 0.0),
):
    """a bar along y = 0.37 through a rigid strip (x = 0..4), ``nElements`` elements over the length ``L``, pulled at
    its end x = L; ``bond`` is the *material block of the bond-slip law"""
    x0, y0 = origin
    bar, first, last = _line(100001, (x0, y0 + 0.37), (x0 + L, y0 + 0.37), nElements, elType)
    return (
        _strip(nX=5, origin=origin)
        + f"""{bar}
*elSet, elSet=rebars, generate=True
{first}, {last - 1}, 1

*nSet, nSet=loadedEnd
{last}

*section, name=rebarSection, {barSection}, material=steel
rebars

{bond}

*modelGenerator, generator=embeddedBond, name=rebarBond, executeAfterManualGeneration=True
embeddedElSet=rebars
hostElSet=gen_all
material=bond
perimeter={PERIMETER}
{generatorOptions}

*job, name=job, domain=2d
*solver, solver=NIST, name=theSolver

*fieldOutput
>>perNode, name=F, nSet=loadedEnd, field=displacement, result=P, f(x)='sum(x[:,0])', saveHistory=True,
>>perNode, name=s, nSet=loadedEnd, field=displacement, result=U, f(x)='sum(x[:,0])', saveHistory=True,
{outputs}

{_PULL.format(maxInc=maxInc, u=u)}
"""
    )


def _linearBond(Kt, Kn=1e6):
    return f"*material, name=LinearElasticBondSlip, id=bond, provider=marmot\n** Kt, Kn\n{Kt}, {Kn}"


def _analyticalPullOutForce(Kt, u=0.01, L=4.0, A=A_STEEL):
    """F = EA beta tanh(beta L) u, beta = sqrt(Kt p / EA): an elastic bar pulled out of a rigid host"""
    EA = E_STEEL * A
    beta = math.sqrt(Kt * PERIMETER / EA)
    return EA * beta * math.tanh(beta * L) * u


def test_pull_out_linear_bond_vs_analytical(tmp_path):
    """pull-out from a rigid host with a linear bond law: F / u = EA beta tanh(beta L), beta = sqrt(K_t p / EA)"""
    _, (F,) = _run(tmp_path, _pullOutDeck(_linearBond(1000.0), 40), "F")
    np.testing.assert_allclose(F.getLastResult(), _analyticalPullOutForce(1000.0), rtol=5e-4)


def test_pull_out_linear_bond_converges_with_second_order(tmp_path):
    """the error of the pull-out force decreases with the square of the bar element length"""
    exact = _analyticalPullOutForce(1000.0)
    errors = []
    for n in (5, 10, 20, 40):
        _, (F,) = _run(tmp_path, _pullOutDeck(_linearBond(1000.0), n), "F")
        errors.append(abs(F.getLastResult() - exact))
    rates = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    assert np.all(rates > 1.9), rates


def test_pull_out_far_from_the_origin(tmp_path):
    """BOND-01: host and bar shifted by 1e5 element sizes give the force of the model at the origin"""
    deck = _pullOutDeck(_linearBond(1000.0), 10)
    _, (F,) = _run(tmp_path, deck, "F")
    _, (FFar,) = _run(tmp_path, _pullOutDeck(_linearBond(1000.0), 10, origin=(1e5, -3e5)), "F")
    np.testing.assert_allclose(FFar.getLastResult(), F.getLastResult(), rtol=1e-8)


def test_pull_out_model_code_2010(tmp_path):
    """pull-out of a short, stiff bar from a rigid host: nearly uniform slip, so the pull-out force follows the
    bond-slip envelope x perimeter x embedded length, through the peak, the softening and the residual branch"""
    tauMax, L = 2.5 * math.sqrt(30.0), 0.5
    mc2010 = (
        "*material, name=ModelCode2010BondSlip, id=bond, provider=marmot\n"
        f"{tauMax}, 1.0, 2.0, 10.0, 0.4, {0.4 * tauMax}, 200.0, 1000.0"
    )
    # practically rigid bar: uniform slip
    deck = _pullOutDeck(mc2010, 4, L=L, barSection="type=truss, area=100.0", u=12.0, maxInc=0.02)
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


def test_pull_out_example_peak(tmp_path):
    """the realistic pull-out example of the docs (Ø16 bar, bonded length 120 mm in a 150 mm concrete cube, MC2010
    good bond, fck = 30 MPa): the peak force is tauMax p L_b"""
    deck = (pathlib.Path(__file__).parent.parent / "examples/EmbeddedBondPullOut/test.inp").read_text()
    _, (F,) = _run(tmp_path, deck, "F")
    peak = np.max(np.asarray(F.getResultHistory()).ravel())
    # at the peak, the whole bonded length is on the plateau s1 <= s <= s2 of the envelope
    np.testing.assert_allclose(peak, 13.69 * 50.27 * 120.0, rtol=1e-6)


@pytest.mark.parametrize("barType", ["T3D2", "T3D3"])
def test_bond_3d_inclined_bar_matches_embedded_region(tmp_path, barType):
    """3D: an inclined bar crossing hexahedra, split at the host boundaries; a stiff bond-slip law converges to
    perfect bond of the split bar"""

    def deck(bondStiffness, perfectBond):
        bar, first, last = _line(100001, (0.15, 0.2, 0.3), (1.85, 0.75, 0.65), 3, barType)
        coupling = (
            "*constraint, type=embeddedRegion, name=embedded\nembeddedElSet=rebars, hostElSet=gen_all"
            if perfectBond
            else ""
        )
        return f"""*material, name=linearelastic, id=concrete, provider=marmot
{E_CONCRETE}, {NU_CONCRETE}

*material, name=linearelastic, id=steel, provider=marmot
{E_STEEL}, {NU_STEEL}

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
embeddedElSet=rebars
hostElSet=gen_all
material=bond
perimeter={PERIMETER}

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
    deck = _pullOutDeck(_linearBond(1000.0), 8).replace(
        ">>dirichlet, name=pull,",
        ">>changeMaterialProperty, name=chProp, material=bond, index=0, f(t)='1000 + 3000 * t'\n>>dirichlet, name=pull,",
    )
    _, (F,) = _run(tmp_path, deck, "F")
    _, (FRef,) = _run(tmp_path, _pullOutDeck(_linearBond(4000.0), 8), "F")
    np.testing.assert_allclose(F.getLastResult(), FRef.getLastResult(), rtol=1e-10)

    # a bar along the bottom edge of the strip, using the host nodes
    shared = (
        _strip(nX=2, nY=1)
        + f"""*element, type=T2D2
100001, 1, 3

*elSet, elSet=rebars
100001

*nSet, nSet=rebarRight
3

*section, name=rebarSection, area={A_STEEL}, material=steel, type=truss
rebars
"""
        + _STIFF_LINEAR_BOND
        + _UNIAXIAL_STEP
    )
    with pytest.raises(ValueError, match="are nodes of host element"):
        _run(tmp_path, shared, "RF")


def test_adaptive_refinement_of_bond_hosts_is_rejected(tmp_path):
    """BOND-06: hAdaptivity cannot refine the hosts of bond elements (the bond elements would keep the coarse host
    nodes): the dynamic refinement deck with an embedded, bonded bar stops with an error"""
    deck = (pathlib.Path(__file__).parent.parent / "testfiles/marmot/AMR_DynamicRefinement/test.inp").read_text()
    # the deck's element set "all" would also contain the bars (and give them the solid section): name the hosts
    deck = deck.replace("elSet=all", "elSet=host").replace("type=solid\nall", "type=solid\nhost")
    bar, first, last = _line(900001, (0.1, 0.3, 0.6), (1.9, 0.3, 0.6), 6, "T3D2")
    deck = deck.replace(
        "*job, name=amr, domain=3d",
        f"""*material, name=linearelastic, id=steel, provider=marmot
{E_STEEL}, {NU_STEEL}

*material, name=LinearElasticBondSlip, id=bond, provider=marmot
1000.0, 1000.0

{bar}
*elSet, elSet=bars, generate=True
{first}, {last - 1}, 1

*section, name=barSection, area=1e-3, material=steel, type=truss
bars

*modelGenerator, generator=embeddedBond, name=barBond, executeAfterManualGeneration=True
embeddedElSet=bars
hostElSet=host
material=bond
perimeter=0.1

*job, name=amr, domain=3d""",
    )
    with pytest.raises(ValueError, match="refining such hosts is not supported"):
        _run(tmp_path, deck)


@pytest.mark.parametrize("beamType, trussType", [("B2D2", "T2D2"), ("B2D3", "T2D3")])
def test_bond_slip_pull_out_beam_equals_truss(tmp_path, beamType, trussType):
    """axial pull-out of a beam with bond-slip gives the force of a truss of the same area and the analytical
    F = E A beta tanh(beta L) u; the rotations of the beam stay zero"""
    Kt = 1000.0
    _, (FTruss,) = _run(tmp_path, _pullOutDeck(_linearBond(Kt), elType=trussType, maxInc=0.25), "F")
    _, (FBeam, rotations) = _run(
        tmp_path,
        _pullOutDeck(
            _linearBond(Kt),
            elType=beamType,
            barSection=f"type=beam, profile=generic, area={A_STEEL}, Iz=1e-5",
            maxInc=0.25,
            outputs=">>perNode, name=rotations, elSet=rebars, field=rotation, result=U, saveHistory=True,",
        ),
        "F",
        "rotations",
    )
    np.testing.assert_allclose(FBeam.getLastResult(), FTruss.getLastResult(), rtol=1e-9)
    np.testing.assert_allclose(FBeam.getLastResult(), _analyticalPullOutForce(Kt), rtol=5e-3)
    assert np.abs(rotations.getLastResult()).max() < 1e-12


def test_bond_slip_split_beams_match_perfect_bond(tmp_path):
    """an inclined beam in a strip bent by a transverse end displacement: the beam is re-meshed with B2D2 elements
    at the host crossings (splitBars), and a stiff bond converges to perfect bond (displacements tied, rotations
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

*section, name=beamSection, type=beam, profile=generic, area=0.01, Iz=1e-4, material=steel
beams

*material, name=LinearElasticBondSlip, id=bond, provider=marmot
{bondStiffness}, {bondStiffness}

*modelGenerator, generator=embeddedBond, name=beamBond, executeAfterManualGeneration=True
embeddedElSet=beams
hostElSet=gen_all
material=bond
perimeter={PERIMETER}

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


def test_unsplit_bars_lock_with_a_stiff_bond(tmp_path):
    """splitBars=False: a bar element spanning several hosts with a stiff bond forces the hosts to follow its linear
    interpolation (locking): the transverse stiffness of the strip in bending is overestimated, while splitting
    reproduces perfect bond; documents the default"""

    def deck(splitBars):
        bar, first, last = _line(100001, (0.0, 0.5), (4.0, 0.5), 1)
        return (
            _strip(nX=8, nY=2)
            + f"""{bar}
*elSet, elSet=rebars, generate=True
{first}, {last - 1}, 1

*nSet, nSet=rebarRight
{last}

*section, name=rebarSection, area={A_STEEL}, material=steel, type=truss
rebars
"""
            + _STIFF_LINEAR_BOND
            + f"splitBars={splitBars}\n"
            + _UNIAXIAL_STEP.replace("field=displacement, 1=0.004", "field=displacement, 2=0.004").replace(
                "f(x)='sum(x[:,0])', saveHistory=True,\n>>perNode, name=RFrebar",
                "f(x)='sum(x[:,1])', saveHistory=True,\n>>perNode, name=RFrebar",
            )
        )

    _, (RFSplit, _) = _run(tmp_path, deck(True), "RF", "RFrebar")
    _, (RFUnsplit, _) = _run(tmp_path, deck(False), "RF", "RFrebar")
    # 2.6 times stiffer here
    assert abs(RFUnsplit.getLastResult()) > 1.5 * abs(RFSplit.getLastResult())


# ---------------------------------------------------------------------------------------------------------------------
# large slip (largeSlip=True): Marmot EBLS<nDim>D<nBarNodes><host>W<n> elements
# ---------------------------------------------------------------------------------------------------------------------

largeSlip = pytest.mark.skipif(
    not _marmotHas("EBLS2D2Q4W1"), reason="Marmot is built without the EmbeddedLargeSlipBondElement"
)


def _reverseOddElements(elementBlock: str) -> str:
    """swap the end nodes of every second element of a *node + *element text: a bar with mixed orientation"""
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
    bond: str,
    generatorOptions: str,
    pull: float,
    maxInc: float = 0.05,
    elType: str = "T2D2",
    section: str = "area=100.0, type=truss",
    reverse: bool = False,
    nElements: int = 18,
):
    """a rigid bar along y = 0.37 from x = 0 to 9 (18 elements), bonded in the strip x = 0..4 (rigid host), pulled at
    x = 9 along x"""
    bar, first, last = _line(100001, (0.0, 0.37), (9.0, 0.37), nElements, elType)
    if reverse:
        bar = _reverseOddElements(bar)
    # a node set takes one generate line: the mid nodes of quadratic bars (transverse zero-energy modes) get their own
    midNodes = (
        f"\n*nSet, nSet=barMidNodes, generate=True\n{first + 50000}, {last - 1 + 50000}, 1\n"
        if elType.endswith("3")
        else ""
    )
    midNodesFixed = (
        ">>dirichlet, name=barMidY, nSet=barMidNodes, field=displacement, 2=0\n" if elType.endswith("3") else ""
    )
    return (
        _strip(nX=5)
        + f"""{bar}
*elSet, elSet=rebars, generate=True
{first}, {last - 1}, 1

*nSet, nSet=barNodes, generate=True
{first}, {last}, 1
{midNodes}
*nSet, nSet=loadedEnd
{last}

*section, name=rebarSection, {section}, material=steel
rebars

{bond}

*modelGenerator, generator=embeddedBond, name=rebarBond, executeAfterManualGeneration=True
embeddedElSet=rebars
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
{midNodesFixed}>>dirichlet, name=pull, nSet=loadedEnd, field=displacement, 1={pull}
"""
    )


_MC2010 = (
    "*material, name=ModelCode2010BondSlip, id=bond, provider=marmot\n"
    "** tauMax, s1, s2, s3, alpha, tauF, K0, Kn\n13.69, 1.0, 2.0, 10.0, 0.4, 5.48, 200.0, 1e4"
)


@largeSlip
def test_large_slip_equals_small_slip_for_small_slip(tmp_path):
    """for slips small compared to the bonded length, the large-slip bond gives the small-slip response (to first
    order: up to the fade over the tributary part at the bar end)"""
    _, (FSmall,) = _run(tmp_path, _largeSlipPullOutDeck(_MC2010, "", 0.02, 0.25), "F")
    _, (FLarge,) = _run(tmp_path, _largeSlipPullOutDeck(_MC2010, "largeSlip=True\nmaxSlip=1.0", 0.02, 0.25), "F")
    np.testing.assert_allclose(
        np.asarray(FLarge.getResultHistory()), np.asarray(FSmall.getResultHistory()), rtol=2e-2, atol=1e-9
    )


def _completePullOut(tmp_path, **deckOptions):
    Kt = 10.0
    deck = _largeSlipPullOutDeck(
        f"*material, name=LinearElasticBondSlip, id=bond, provider=marmot\n** Kt, Kn\n{Kt}, 1e4",
        # no splitting: split quadratic bars get mid nodes without transverse support once they are pulled out
        "largeSlip=True\nmaxSlip=6.0\nnIntegrationPoints=3\nsplitBars=False",
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
        # the bond fades smoothly over the tributary parts at the bar end, not exactly at the bar end
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
        ("B2D2", "type=beam, profile=generic, area=100.0, Iz=1.0", False),
        ("B2D3", "type=beam, profile=generic, area=100.0, Iz=1.0", True),
    ],
)
def test_large_slip_bars_of_quadratic_reversed_and_beam_elements(tmp_path, elType, section, reverse):
    """the complete pull-out does not depend on the bar element type (quadratic, beam) or on the orientation of the
    bar elements along the bar"""
    _, _, slipRef, forceRef = _completePullOut(tmp_path)
    _, model, slip, force = _completePullOut(tmp_path, elType=elType, section=section, reverse=reverse)
    nBarNodes = 3 if elType.endswith("3") else 2
    assert all(el.elType.startswith(f"EBLS2D{nBarNodes}Q4W") for el in model.elementSets["rebarBond_bond"])
    np.testing.assert_allclose(np.interp(slipRef, slip, force), forceRef, rtol=1e-6, atol=1e-6 * forceRef.max())


def _orientedPullOutDeck(tmp_path, direction, exitPoint, nDim):
    """a rigid bar along ``direction`` (bonded length 120, 24 elements, free length 20 in one element) in a practically
    rigid block of 200 (2D, CPE4 10 x 10) or 150 (3D, C3D8 15 x 5 x 5) bearing on its loaded face x = max, exiting
    there at ``exitPoint``, pulled 30 along itself"""
    t = np.asarray(direction, float) / np.linalg.norm(direction)
    exitPoint = np.asarray(exitPoint, float)
    points = [exitPoint - 120.0 * t + 5.0 * i * t for i in range(25)] + [exitPoint + 20.0 * t]
    nodes = "\n".join(f"{100001 + i}, " + ", ".join(f"{c:.12g}" for c in x) for i, x in enumerate(points))
    # co-rotational trusses: the tension stiffens the bar transversely (geometric stiffness), so that bar nodes no
    # channel point holds any more (e.g., at short split pieces) are not a zero-energy mode
    elType = "T2D2CR" if nDim == 2 else "T3D2CR"
    elements = "\n".join(f"{100001 + i}, {100001 + i}, {100002 + i}" for i in range(25))
    if nDim == 2:
        host = """*modelGenerator, generator=planeRectQuad, name=gen
l=200
h=200
nX=10
nY=10
elType=CPE4

*section, name=concreteSection, thickness=1.0, material=concrete, type=plane
gen_all
"""
        fixHost, pull = "1=0, 2=0", ", ".join(f"{d + 1}={30.0 * c:.12g}" for d, c in enumerate(t))
    else:
        host = """*modelGenerator, generator=boxGen, name=gen
lX=150
lY=150
lZ=150
nX=15
nY=5
nZ=5
elType=C3D8

*section, name=concreteSection, material=concrete, type=solid
gen_all
"""
        fixHost, pull = "1=0, 2=0, 3=0", ", ".join(f"{d + 1}={30.0 * c:.12g}" for d, c in enumerate(t))
    along = " + ".join(f"sum(x[:,{d}]) * {c:.12g}" for d, c in enumerate(t))
    return f"""** a practically rigid block
*material, name=linearelastic, id=concrete, provider=marmot
1e9, {NU_CONCRETE}

*material, name=linearelastic/hughes-winget, id=steel, provider=marmot
{E_STEEL}, {NU_STEEL}

*material, name=LinearElasticBondSlip, id=bond, provider=marmot
** Kt, Kn
10.0, 1e4

{host}
*node
{nodes}

*element, type={elType}
{elements}

*elSet, elSet=rebars, generate=True
100001, 100025, 1

*nSet, nSet=loadedEnd
100026

*section, name=rebarSection, area=1e4, material=steel, type=truss
rebars

*modelGenerator, generator=embeddedBond, name=rebarBond, executeAfterManualGeneration=True
embeddedElSet=rebars
hostElSet=gen_all
material=bond
perimeter=50.27
largeSlip=True
maxSlip=35.0

*job, name=job, domain={nDim}d
*solver, solver=NIST, name=theSolver

*fieldOutput
>>perNode, name=F, nSet=loadedEnd, field=displacement, result=P, f(x)='{along}', saveHistory=True,
>>perNode, name=s, nSet=loadedEnd, field=displacement, result=U, f(x)='{along}', saveHistory=True,

*step, solver=theSolver
maxInc=0.02, minInc=1e-8, maxNumInc=1000, maxIter=15, stepLength=1
>>dirichlet, name=plate, nSet=gen_right, field=displacement, {fixHost}
>>dirichlet, name=pull, nSet=loadedEnd, field=displacement, {pull}
"""


@largeSlip
def test_large_slip_pull_out_does_not_depend_on_the_orientation(tmp_path):
    """the pull-out of a rigid bar from a rigid block gives the same force-slip history for a bar aligned with the mesh
    (along element edges), tilted by 45 degrees (through the mesh nodes, along element diagonals) and inclined in 3D
    -- up to the fade over the tributary parts at the bar end, which depend on where the hosts cut the bar"""
    histories = []
    for direction, exitPoint, nDim in (
        ((1.0, 0.0), (200.0, 100.0), 2),
        ((1.0, 1.0), (200.0, 160.0), 2),
        ((0.85, 0.40, 0.34), (150.0, 100.0, 90.0), 3),
    ):
        _, (F, s) = _run(tmp_path, _orientedPullOutDeck(tmp_path, direction, exitPoint, nDim), "F", "s")
        histories.append((np.asarray(s.getResultHistory()).ravel(), np.asarray(F.getResultHistory()).ravel()))
    slipRef, forceRef = histories[0]
    # F = K_t s p (L_b - s), at the end of the pull s = 30
    assert forceRef[-1] == pytest.approx(10.0 * 30.0 * 50.27 * (120.0 - 30.0), rel=5e-3)  # measured 0.21 %
    for slip, force in histories[1:]:
        np.testing.assert_allclose(np.interp(slipRef, slip, force), forceRef, atol=ORIENTATION_ATOL * forceRef.max())


@largeSlip
@pytest.mark.parametrize(
    "generatorOptions, nElements, change, match",
    [
        ("largeSlip=True", 18, None, "requires a positive maxSlip"),
        ("largeSlip=True\nmaxSlip=-1.0", 18, None, "requires a positive maxSlip"),
        ("maxSlip=1.0", 18, None, "maxSlip is an option of largeSlip"),
        ("largeSlip=True\nmaxSlip=6.0", 180, None, "but Marmot has no EBLS2D2Q4W"),
        ("largeSlip=True\nmaxSlip=1.0", 18, "branch", "unbranched"),
        ("largeSlip=True\nmaxSlip=1.0", 18, "kink", "kink of"),
        ("largeSlip=True\nmaxSlip=1.0", 18, "loop", "closed loop"),
    ],
)
def test_large_slip_invalid_input(tmp_path, generatorOptions, nElements, change, match):
    """large slip needs a positive maxSlip (and maxSlip needs large slip), a window Marmot registers, and straight or
    smoothly curved bars with two ends"""
    deck = _largeSlipPullOutDeck(_MC2010, generatorOptions, 0.01, nElements=nElements)
    elementSet = "*elSet, elSet=rebars, generate=True\n100001, 100018, 1"
    if change == "branch":
        # a third bar element joins the bar at its node 100005
        extra = "*node\n200001, 2.0, 0.9\n\n*element, type=T2D2\n100100, 100005, 200001\n\n"
        deck = deck.replace(
            elementSet,
            extra + "*elSet, elSet=rebars\n" + ", ".join(str(label) for label in [*range(100001, 100019), 100100]),
            1,
        )
    elif change == "kink":
        # the last node moves off the line: a kink of about 22 degrees at node 100018
        deck = deck.replace("100019, 9, 0.37", "100019, 9, 0.57", 1)
    elif change == "loop":
        # four bar elements around the square (1, 0.2) - (3, 0.8): a closed loop, alone in the set
        extra = (
            "*node\n300001, 1.0, 0.2\n300002, 3.0, 0.2\n300003, 3.0, 0.8\n300004, 1.0, 0.8\n\n*element, type=T2D2\n"
            "300001, 300001, 300002\n300002, 300002, 300003\n300003, 300003, 300004\n300004, 300004, 300001\n\n"
        )
        deck = deck.replace(elementSet, extra + "*elSet, elSet=rebars, generate=True\n300001, 300004, 1", 1).replace(
            ">>dirichlet, name=barY, nSet=barNodes", "** "
        )
    with pytest.raises(ValueError, match=match):
        _run(tmp_path, deck, "F")
