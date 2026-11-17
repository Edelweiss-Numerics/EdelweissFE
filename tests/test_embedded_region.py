"""Embedded region constraint (perfect bond): Marmot truss elements (T*) and beams (B*) embedded in host
continuum elements by master-slave DOF elimination, verified against analytical solutions; the rotation tying of
embedded beams (rotations=free|axis|host).

Needs the Marmot modules TrussElement and BeamElement.
"""

import math

import numpy as np
import pytest

pytest.importorskip("edelweissfe.elements.marmotelement.element")

from _embeddingdecks import (  # noqa: E402
    A_STEEL,
    E_CONCRETE,
    E_STEEL,
    NU_CONCRETE,
    NU_STEEL,
    _line,
    _marmotHas,
    _run,
    _strip,
)

pytestmark = pytest.mark.skipif(
    not all(_marmotHas(t) for t in ("T3D2", "T2D2CR", "B2D2", "B2D2CR", "B2D3", "B3D2")),
    reason="Marmot is built without the TrussElement / BeamElement modules",
)

G_STEEL = E_STEEL / (2 * (1 + NU_STEEL))
#: the embedded cantilever with free rotations relative to the clamped one (test_embedded_cantilever_clamped...)
FREE_ROTATION_SOFTENING = 1.1094


def _rebarsInStrip(nRebarElements=5, y=0.37):
    bar, first, last = _line(100001, (0.0, y), (4.0, y), nRebarElements, "T2D2")
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


@pytest.mark.parametrize("hostType", ["CPE4", "CPE8R"])
def test_embedded_region_uniaxial_tension(tmp_path, hostType):
    """perfect bond: the non-matching rebar follows the host strain, the reaction is concrete + steel (linear and
    quadratic hosts)"""
    deck = (
        _strip(elType=hostType)
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
        _strip()
        + _rebarsInStrip().replace("100001, 0, 0.37", "100001, -0.5, 0.37")
        + """
*constraint, type=embeddedRegion, name=embedded
embeddedElSet=rebars, hostElSet=gen_all
"""
        + _UNIAXIAL_STEP
    )
    with pytest.raises(ValueError, match="outside of all host elements"):
        _run(tmp_path, deck, "RF")


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

*section, name=beamSection, type=beam, profile=generic, area=1e-8, Iz=1e-8, material=steel
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


@pytest.mark.parametrize("rotations, elType", [("axis", "B2D2"), ("host", "B2D2"), ("free", "B2D2"), ("axis", "B2D3")])
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
    if rotations in ("axis", "host"):  # the block barely deforms: both clamp the beam
        np.testing.assert_allclose(deflection, clamped, rtol=1e-5)
    else:  # free rotations: the root is clamped only by the tied displacements of the embedded nodes
        np.testing.assert_allclose(deflection / clamped, FREE_ROTATION_SOFTENING, rtol=1e-3)


def _beamInStripDeck(elType: str, options: str) -> str:
    """the strip with an embedded inclined beam of ``elType`` (2D), the constraint with ``options``, in tension"""
    beam, first, last = _line(100001, (0.3, 0.2), (3.7, 0.8), 3, elType)
    return (
        _strip()
        + f"""{beam}
*elSet, elSet=beams, generate=True
{first}, {last - 1}, 1

*nSet, nSet=rebarRight
{last}

*section, name=beamSection, type=beam, profile=rectangle, b=0.1, h=0.1, material=steel
beams

*constraint, type=embeddedRegion, name=embedded
embeddedElSet=beams, hostElSet=gen_all, {options}
"""
        + _UNIAXIAL_STEP.replace(
            ">>perElement, name=rebarStrain, elSet=rebars, result=strain, quadraturePoint=0, saveHistory=True,\n", ""
        )
    )


@pytest.mark.parametrize("rotations", ["axis", "host"])
def test_embedded_corotational_beam_rejects_tied_rotations(tmp_path, rotations):
    """the rotation ties are infinitesimal: a co-rotational beam (B2D2CR) may only be embedded with free rotations"""
    with pytest.raises(ValueError, match="co-rotational beam .* cannot be embedded with rotations=" + rotations):
        _run(tmp_path, _beamInStripDeck("B2D2CR", f"rotations={rotations}"), "RF")


def test_embedded_corotational_beam_with_free_rotations(tmp_path):
    """a co-rotational beam (B2D2CR) embedded with free rotations: no rotation is tied, and every beam node is at
    the interpolation of its host"""
    from edelweissfe.utils.embedding import HostElementLocator

    model, _ = _run(tmp_path, _beamInStripDeck("B2D2CR", "rotations=free"))
    constraint = model.multiPointConstraints["embedded"]
    assert constraint._rotationRecords == [] and len(constraint._records) == 4
    field = model.nodeFields["displacement"]
    U = {node: np.asarray(field["U"][k]) for k, node in enumerate(field.nodes)}
    assert max(np.abs(u).max() for u in U.values()) > 1e-3, "the strip must be loaded"
    locator = HostElementLocator(list(model.elementSets["gen_all"]))
    for el in model.elementSets["beams"]:
        for node in el.nodes:
            point = locator.locate(node.coordinates)
            interpolated = sum(N * U[m] for N, m in zip(point.shapeFunctions(), point.element.nodes))
            np.testing.assert_allclose(U[node], interpolated, rtol=0, atol=1e-12)


def test_embedded_region_rejects_an_unknown_rotations_mode(tmp_path):
    with pytest.raises(ValueError, match="unknown rotations mode 'clamped'"):
        _run(tmp_path, _beamInStripDeck("B2D2", "rotations=clamped"), "RF")


def test_embedded_region_rejects_tied_rotations_of_another_field(tmp_path):
    """the rotation ties act on the host displacements: another constrained field with tied rotations is rejected"""
    with pytest.raises(ValueError, match="needs field=displacement, not field=temperature"):
        _run(tmp_path, _beamInStripDeck("B2D2", "rotations=axis, field=temperature"), "RF")


def test_embedded_cantilever_clamped_in_stiff_block_3d(tmp_path):
    """3D: a cantilever (B3D2) embedded in a stiff C3D8 block with tied rotations: biaxial bending and torsion of the
    free part as clamped at the block face"""
    b, h, L = 10.0, 20.0, 200.0
    Iz, Iy = b * h**3 / 12, h * b**3 / 12  # h along the local y axis (orientation (0, 1, 0)), b along z
    # torsion constant of the solid rectangle (Roark, Formulas for Stress and Strain), as the beam section
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

*section, name=beamSection, type=beam, profile=rectangle, b={b}, h={h}, vx=0, vy=1, vz=0, nY=4, nZ=4, material=steel
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


@pytest.mark.parametrize(
    "shape, nDim",
    [
        ("quad4", 2),
        ("quad8", 2),
        ("tria3", 2),
        ("tria6", 2),
        ("hexa8", 3),
        ("hexa20", 3),
        ("tetra4", 3),
        ("tetra10", 3),
    ],
)
def test_rotation_operator(shape, nDim):
    """the closed-form rotation operators dTheta/dU of the embedded region constraint: a rigid rotation gives the
    rotation itself (both modes); a uniform strain leaves the axis mode at t x (eps t) and the host mode at zero"""
    from types import SimpleNamespace

    from edelweissfe.constraints.embeddedregion import _dN_dX, dTheta_dU
    from edelweissfe.utils.embedding import HostPoint
    from edelweissfe.utils.isoparametricshapes import NODE_PARAMETRIC_COORDINATES

    rng = np.random.default_rng(1)
    xiNodes = NODE_PARAMETRIC_COORDINATES[shape]
    A = np.eye(nDim) * 2.0 + 0.2 * rng.random((nDim, nDim))  # an affinely distorted element
    X = xiNodes @ A.T + 1.0
    element = SimpleNamespace(ensightType=shape, nodes=[SimpleNamespace(coordinates=x) for x in X])
    point = HostPoint(element, 0.2 * np.ones(nDim) + (0.0 if shape.startswith("t") else -0.1))
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
        C = dTheta_dU(_dN_dX(point), mode, t)
        np.testing.assert_allclose(np.einsum("rai,ai->r", C, X @ W.T), rotation, atol=1e-15)
        theta = np.einsum("rai,ai->r", C, X @ eps.T)
        if mode == "host":
            np.testing.assert_allclose(theta, 0.0, atol=1e-15)
        elif nDim == 2:
            np.testing.assert_allclose(theta, [t[0] * (eps @ t)[1] - t[1] * (eps @ t)[0]], atol=1e-15)
        else:
            np.testing.assert_allclose(theta, np.cross(t, eps @ t), atol=1e-15)


@pytest.mark.parametrize("reverse", [False, True])
def test_rotation_on_a_host_boundary_is_independent_of_the_host_order(reverse):
    """EMB-02: a beam node on the edge between two quad4 hosts ([0, 1] and [1, 2] x [0, 1]) with the nodal field
    u_y = -x^2 / 2 (non-uniform: the gradient jumps across the edge): the rotation of the axis e_x at (1, 0.5) is
    -0.5 in the left host and -1.5 in the right one; the constraint averages them to the continuous value -1.0, for
    either order of the hosts"""
    from types import SimpleNamespace

    from edelweissfe.constraints.embeddedregion import Constraint
    from edelweissfe.utils.embedding import HostElementLocator

    class Node:  # hashable, as the model's nodes
        def __init__(self, label, x, y):
            self.label, self.coordinates = label, np.array([x, y], dtype=float)

    def node(label, x, y):
        return Node(label, x, y)

    nodes = {(x, y): node(10 * x + y, x, y) for x in (0, 1, 2) for y in (0, 1)}
    left = SimpleNamespace(elNumber=1, ensightType="quad4", nodes=[nodes[p] for p in ((0, 0), (1, 0), (1, 1), (0, 1))])
    right = SimpleNamespace(elNumber=2, ensightType="quad4", nodes=[nodes[p] for p in ((1, 0), (2, 0), (2, 1), (1, 1))])
    hosts = [right, left] if reverse else [left, right]

    constraint = Constraint.__new__(Constraint)
    constraint._rotationMode = "axis"
    points = HostElementLocator(hosts).locateAll(np.array([1.0, 0.5]))
    assert len(points) == 2
    masters = constraint._rotationMasters(points, [np.array([1.0, 0.0])])
    uY = {n: -n.coordinates[0] ** 2 / 2 for n in nodes.values()}
    theta = sum(c[0] * (uY[m] if i == 1 else 0.0) for m, i, c in masters)
    np.testing.assert_allclose(theta, -1.0, rtol=1e-12)


def test_embedded_region_follows_adaptive_refinement(tmp_path):
    """EMB-03: the embedded bar of the dynamic refinement deck (two C3D20 hosts, one refined during the step): after
    the refinement the bar nodes are tied to the refined hosts, i.e. they move with the interpolation of the refined
    field -- not with the coarse elements the constraint was built on"""
    import pathlib

    from edelweissfe.utils.embedding import HostElementLocator

    deck = (pathlib.Path(__file__).parent.parent / "testfiles/marmot/AMR_DynamicRefinement/test.inp").read_text()
    # the deck's element set "all" would also contain the bars (and give them the solid section): name the hosts
    deck = deck.replace("elSet=all", "elSet=host").replace("type=solid\nall", "type=solid\nhost")
    bar, first, last = _line(900001, (0.1, 0.3, 0.6), (1.9, 0.3, 0.6), 6, "T3D2")
    deck = deck.replace(
        "*job, name=amr, domain=3d",
        f"""*material, name=linearelastic, id=steel, provider=marmot
{E_STEEL}, 0.3

{bar}
*elSet, elSet=bars, generate=True
{first}, {last - 1}, 1

*section, name=barSection, area=1e-3, material=steel, type=truss
bars

*constraint, type=embeddedRegion, name=embedded
embeddedElSet=bars, hostElSet=host

*job, name=amr, domain=3d""",
    )
    model, _ = _run(tmp_path, deck)
    hosts = list(model.elementSets["host"])
    assert len(hosts) > 2, "the deck must refine"
    field = model.nodeFields["displacement"]
    U = {node: np.asarray(field["U"][k]) for k, node in enumerate(field.nodes)}
    locator = HostElementLocator(hosts)
    barNodes = {n for el in model.elementSets["bars"] for n in el.nodes}
    scale = max(np.abs(u).max() for u in U.values())
    for node in barNodes:
        point = locator.locate(node.coordinates)
        interpolated = sum(N * U[m] for N, m in zip(point.shapeFunctions(), point.element.nodes))
        np.testing.assert_allclose(U[node], interpolated, atol=1e-10 * scale)


@pytest.mark.parametrize(
    "changedElementSets, rebuilt",
    [(set(), False), ({"otherSet"}, False), ({"host"}, True), ({"bars"}, True), ({"host", "bars"}, True)],
)
def test_embedded_region_refreshes_only_if_the_change_touches_its_element_sets(changedElementSets, rebuilt):
    """R2-EMB-01: the records are rebuilt if and only if the model change touches the host or the embedded element
    set (as hAdaptivity reports a refined host set in ``changedElementSets``)"""
    from edelweissfe.constraints.embeddedregion import Constraint, EmbeddedRegionSchema
    from edelweissfe.models.modelchange import ModelChange
    from edelweissfe.models.modelchangeobserver import ModelChangeType

    builds = []
    constraint = Constraint.__new__(Constraint)
    constraint._configuration = EmbeddedRegionSchema(embeddedElSet="bars", hostElSet="host")
    constraint._build = builds.append
    change = ModelChange(kind=ModelChangeType.REFINEMENT, changedElementSets=changedElementSets)
    assert constraint.refresh("model", change) is rebuilt
    assert builds == (["model"] if rebuilt else [])


def test_embedded_region_explicit_dynamics(tmp_path):
    """EMB-07: explicit dynamics (NED, masses of the embedded nodes folded onto the hosts): the embedded rebar of the
    reinforced strip follows the host field exactly, at the end of the explicit step every rebar node is at the
    interpolation of its host"""
    from edelweissfe.utils.embedding import HostElementLocator

    rebar, first, last = _line(100001, (0.0, 0.37), (4.0, 0.37), 5, "T2D2")
    deck = f"""*material, name=linearelastic, id=concrete, provider=marmot
{E_CONCRETE}, {NU_CONCRETE}, 2.4e-9

*material, name=linearelastic, id=steel, provider=marmot
{E_STEEL}, 0.3, 7.85e-9

*modelGenerator, generator=planeRectQuad, name=gen
l=4
h=1
nX=8
nY=2
elType=CPE4

*section, name=concreteSection, thickness=1.0, material=concrete, type=plane
gen_all

{rebar}
*elSet, elSet=rebars, generate=True
{first}, {last - 1}, 1

*section, name=rebarSection, area={A_STEEL}, material=steel, type=truss
rebars

*constraint, type=embeddedRegion, name=embedded
embeddedElSet=rebars, hostElSet=gen_all

*job, name=job, domain=2d
*solver, solver=NED, name=theSolver

*step, type=adaptiveForExplicitSimulations, solver=theSolver
maxInc=1e-3, minInc=1e-12, maxNumInc=100000, maxIter=25, stepLength=1e-3
>>dirichlet, name=left, nSet=gen_left, field=displacement, 1=0, 2=0
>>dirichlet, name=right, nSet=gen_right, field=displacement, 1=0.004
"""
    model, _ = _run(tmp_path, deck)
    field = model.nodeFields["displacement"]
    U = {node: np.asarray(field["U"][k]) for k, node in enumerate(field.nodes)}
    assert max(np.abs(u).max() for u in U.values()) > 1e-4, "the strip must be loaded"
    locator = HostElementLocator(list(model.elementSets["gen_all"]))
    for el in model.elementSets["rebars"]:
        for node in el.nodes:
            point = locator.locate(node.coordinates)
            interpolated = sum(N * U[m] for N, m in zip(point.shapeFunctions(), point.element.nodes))
            np.testing.assert_allclose(U[node], interpolated, rtol=0, atol=1e-12)
