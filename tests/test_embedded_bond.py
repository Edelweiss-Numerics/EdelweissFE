"""Bond-slip of embedded reinforcement (small slip): the embeddedBond generator with the Marmot bond elements
(EB<nDim>D<nBarNodes><host>) and bond-slip laws, for truss (T*) and beam (B*) bars, verified against analytical
pull-out solutions and against the perfect bond of the embeddedRegion constraint.

Needs the Marmot modules TrussElement, BeamElement, EmbeddedBondElement, LinearElasticBondSlip and
ModelCode2010BondSlip.
"""

import math
import pathlib

import numpy as np
import pytest

pytest.importorskip("edelweissfe.elements.marmotelement.element")

from _embeddingdecks import (  # noqa: E402
    _REBAR_STRAIN_OUTPUT,
    _UNIAXIAL_STEP,
    A_STEEL,
    E_CONCRETE,
    E_STEEL,
    NU_CONCRETE,
    NU_STEEL,
    REACTION_PERFECT_BOND,
    _barsInStrip,
    _line,
    _marmotHas,
    _run,
    _strip,
)

pytestmark = pytest.mark.skipif(
    not all(_marmotHas(t) for t in ("T2D2", "T3D2", "B2D2", "EB2D2Q4", "EB3D2H8")),
    reason="Marmot is built without the TrussElement / BeamElement / EmbeddedBondElement modules",
)

PERIMETER = 0.3545  # circular bar of area 0.01


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
    deck = strip + _barsInStrip() + _STIFF_LINEAR_BOND + _UNIAXIAL_STEP.replace(_REBAR_STRAIN_OUTPUT, "")
    model, (RF, RFrebar) = _run(tmp_path, deck, "RF", "RFrebar")
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
        + _UNIAXIAL_STEP.replace(_REBAR_STRAIN_OUTPUT, "")
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
            + _UNIAXIAL_STEP.replace(_REBAR_STRAIN_OUTPUT, "")
            .replace("field=displacement, 1=0.004", "field=displacement, 2=0.004")
            .replace(
                "f(x)='sum(x[:,0])', saveHistory=True,\n>>perNode, name=RFrebar",
                "f(x)='sum(x[:,1])', saveHistory=True,\n>>perNode, name=RFrebar",
            )
        )

    _, (RFSplit, _) = _run(tmp_path, deck(True), "RF", "RFrebar")
    _, (RFUnsplit, _) = _run(tmp_path, deck(False), "RF", "RFrebar")
    # 2.6 times stiffer here
    assert abs(RFUnsplit.getLastResult()) > 1.5 * abs(RFSplit.getLastResult())


@pytest.mark.skipif(
    not all(_marmotHas(t) for t in ("T2D3", "EB2D3Q4")),
    reason="Marmot is built without 3-node trusses or their bond elements",
)
def test_split_bar3_in_a_model_without_the_node_set_all():
    """a model built in Python, without the node set 'all' the input file path creates: a T2D3 bar across two CPE4
    hosts is split at their common edge, its old mid node (on that edge) is removed from the model and from every
    node set, and no 'all' set is invented"""
    from edelweissfe.config.elementlibrary import getElementClass
    from edelweissfe.generators.embeddedbond import EmbeddedBondSchema, Generator
    from edelweissfe.journal.journal import Journal
    from edelweissfe.models.femodel import FEModel
    from edelweissfe.points.node import Node
    from edelweissfe.sets.elementset import ElementSet
    from edelweissfe.sets.nodeset import NodeSet

    model = FEModel(2)
    coordinates = {1: (0, 0), 2: (1, 0), 3: (2, 0), 4: (2, 1), 5: (1, 1), 6: (0, 1)}
    coordinates.update({101: (0.25, 0.5), 102: (1.75, 0.5), 103: (1.0, 0.5)})
    for label, xy in coordinates.items():
        model.nodes[label] = Node(label, np.array(xy, dtype=float))

    def element(elType, number, labels):
        el = getElementClass(elType, "marmot")(elType, number)
        el.setNodes([model.nodes[n] for n in labels])
        model.elements[number] = el
        return el

    hosts = [element("CPE4", 1, (1, 2, 5, 6)), element("CPE4", 2, (2, 3, 4, 5))]
    bar = element("T2D3", 101, (101, 102, 103))
    model.elementSets["hosts"] = ElementSet("hosts", hosts)
    model.elementSets["bars"] = ElementSet("bars", [bar])
    model.nodeSets["barNodes"] = NodeSet("barNodes", [model.nodes[n] for n in (101, 102, 103)])
    model.materials["bond"] = {"name": "LinearElasticBondSlip", "properties": np.array([1.0, 1.0])}
    oldMidNode = model.nodes[103]

    with model.topology.changes():
        model.topology.adoptSetupElementNumbers()
        model.topology.adoptSetupNodeNumbers()
        Generator(
            "bond",
            model,
            Journal(verbose=False),
            configuration=EmbeddedBondSchema(embeddedElSet="bars", hostElSet="hosts", material="bond", perimeter=1.0),
        )

    assert "all" not in model.nodeSets
    assert 103 not in model.nodes and oldMidNode not in model.nodeSets["barNodes"]
    newBars = list(model.elementSets["bars"])
    assert len(newBars) == 2 and all(el.elType == "T2D3" for el in newBars)
    assert len(model.elementSets["bond_bond"]) == 2
    assert all(node.label in model.nodes for el in model.elements.values() for node in el.nodes)
