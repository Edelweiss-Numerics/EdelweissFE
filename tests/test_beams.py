"""Beams: the Marmot Euler-Bernoulli beam elements (B2D2, B2D3, B3D2, B3D3) and their co-rotational
(finite rotation) variant B2D2CR with the beam section (section points of parametric profiles, polygons, point
files), verified against analytical solutions.

Needs the Marmot module BeamElement.
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
    not all(_marmotHas(t) for t in ("B2D2", "B2D3", "B3D2", "B3D3", "B2D2CR")),
    reason="Marmot is built without the BeamElement module",
)

E_STEEL, NU_STEEL, E_CONCRETE = 200000.0, 0.3, 30000.0
G_STEEL = E_STEEL / (2 * (1 + NU_STEEL))


def _run(tmp_path, deck: str, *results):
    path = tmp_path / "test.inp"
    path.write_text(deck)
    model, fieldOutputController = finiteElementSimulation(parseInputFile(str(path)), suppressPlots=True)
    return model, [fieldOutputController.fieldOutputs[r] for r in results]


def _line(firstLabel: int, start, end, nElements: int, elType: str, quadratic: bool = None):
    """*node and *element blocks of a straight line of 2-node (or 3-node: end, end, mid, the mid nodes labeled from
    firstLabel + 50000) elements, and its first and last node labels."""
    if quadratic is None:
        quadratic = elType.endswith("3") or elType in ("TR2D3", "TR3D3")
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


def _beamDeck(elType, start, end, nElements, section, domain, steps, extra=""):
    line, first, last = _line(1, start, end, nElements, elType)
    return f"""*material, name=linearelastic, id=steel, provider=marmot
{E_STEEL}, {NU_STEEL}

{line}
*elSet, elSet=beam, generate=True
{first}, {last - 1}, 1
*nSet, nSet=root
{first}
*nSet, nSet=tip
{last}

*section, name=s, type=beam, {section}, material=steel
beam
{extra}
*job, name=job, domain={domain}
*solver, solver=NIST, name=theSolver

*fieldOutput
>>perNode, name=u, nSet=tip, field=displacement, result=U, saveHistory=True,
>>perNode, name=r, nSet=tip, field=rotation, result=U, saveHistory=True,

*step, solver=theSolver
maxInc=1.0, minInc=1e-8, maxNumInc=10, maxIter=10, stepLength=1
{steps}
"""


def test_beam_cantilever_2d(tmp_path):
    """an inclined 2D cantilever (rectangle 10 x 20) under a tip force normal to it and a tip moment: exact tip
    deflection and rotation"""
    b, h, L, P, M = 10.0, 20.0, 500.0, 7.0, 1500.0
    I = b * h**3 / 12  # noqa: E741
    t, n = np.array([0.6, 0.8]), np.array([-0.8, 0.6])
    deck = _beamDeck(
        "B2D2",
        (0, 0),
        L * t,
        4,
        f"profile=rectangle, b={b}, h={h}",
        "2d",
        f""">>dirichlet, name=root, nSet=root, field=displacement, 1=0, 2=0
>>dirichlet, name=rootR, nSet=root, field=rotation, 1=0
>>nodeforces, name=P, nSet=tip, field=displacement, 1={P * n[0]}, 2={P * n[1]}
>>nodeforces, name=M, nSet=tip, field=rotation, 1={M}
""",
    )
    _, (u, r) = _run(tmp_path, deck, "u", "r")
    u, r = u.getLastResult().ravel(), r.getLastResult().item()
    EI = E_STEEL * I
    np.testing.assert_allclose(u @ n, P * L**3 / (3 * EI) + M * L**2 / (2 * EI), rtol=1e-10)
    np.testing.assert_allclose(r, P * L**2 / (2 * EI) + M * L / EI, rtol=1e-10)
    assert abs(u @ t) < 1e-12


def test_beam_cantilever_3d(tmp_path):
    """a 3D cantilever (circle, d = 20) along (1, 2, 2) / 3 under a tip force and a torque: exact deflection and
    twist (G J of the circle)"""
    d, L, P, T = 20.0, 300.0, 3.0, 500.0
    I, J = math.pi * d**4 / 64, math.pi * d**4 / 32  # noqa: E741
    t = np.array([1.0, 2.0, 2.0]) / 3
    force = P * np.array([2.0, -1.0, 0.0]) / math.sqrt(5)  # normal to the axis
    torque = T * t
    deck = _beamDeck(
        "B3D2",
        (0, 0, 0),
        L * t,
        3,
        f"profile=circle, d={d}, nRings=2, nSectors=8",
        "3d",
        f""">>dirichlet, name=root, nSet=root, field=displacement, 1=0, 2=0, 3=0
>>dirichlet, name=rootR, nSet=root, field=rotation, 1=0, 2=0, 3=0
>>nodeforces, name=P, nSet=tip, field=displacement, 1={force[0]}, 2={force[1]}, 3={force[2]}
>>nodeforces, name=T, nSet=tip, field=rotation, 1={torque[0]}, 2={torque[1]}, 3={torque[2]}
""",
    )
    _, (u, r) = _run(tmp_path, deck, "u", "r")
    u, r = u.getLastResult().ravel(), r.getLastResult().ravel()
    np.testing.assert_allclose(u, force * L**3 / (3 * E_STEEL * I), rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(
        r, np.cross(t, force) * L**2 / (2 * E_STEEL * I) + torque * L / (G_STEEL * J), rtol=1e-9, atol=1e-12
    )


def _rectanglesIntegrals(rectangles):
    """analytical A, Sy, Sz, Iy = int z^2, Iz = int y^2, Iyz of a union of rectangles (y0, y1, z0, z1)"""
    c = dict.fromkeys(["A", "Sy", "Sz", "Iy", "Iz", "Iyz"], 0.0)
    for y0, y1, z0, z1 in rectangles:
        Y1, Y2, Y3 = y1 - y0, (y1**2 - y0**2) / 2, (y1**3 - y0**3) / 3
        Z1, Z2, Z3 = z1 - z0, (z1**2 - z0**2) / 2, (z1**3 - z0**3) / 3
        c["A"] += Y1 * Z1
        c["Sy"] += Y2 * Z1
        c["Sz"] += Y1 * Z2
        c["Iy"] += Y1 * Z3
        c["Iz"] += Y3 * Z1
        c["Iyz"] += Y2 * Z2
    return c


def _check(points, expected, rtol=1e-12):
    got = points.integrals()
    scale = {"A": expected["A"], "Sy": 1.0, "Sz": 1.0}
    for key, value in expected.items():
        s = scale.get(key, max(expected.get("Iy", 1.0), expected.get("Iz", 1.0)))
        if key in ("Sy", "Sz"):
            s = expected["A"] * math.sqrt(max(expected.get("Iz", 1.0), 1.0) / expected["A"])
        assert abs(got[key] - value) <= rtol * s, f"{key}: {got[key]} != {value}"


@pytest.mark.parametrize(
    "rule, n", [("gauss", 2), ("gauss", 5), ("lobatto", 3), ("lobatto", 6), ("simpson", 3), ("simpson", 7)]
)
def test_section_points_parametric_shapes_exact(rule, n):
    """area and second moments of the parametric shapes are integrated exactly by every rule"""
    from edelweissfe.utils import beamsections as bs

    b, h = 10.0, 20.0
    _check(bs.rectangle(b, h, n, n, rule), _rectanglesIntegrals([(-h / 2, h / 2, -b / 2, b / 2)]))
    d, di = 30.0, 22.0
    for inner in (0.0, di):
        points = bs.tube(d, inner, n, 8, rule)
        I = math.pi * (d**4 - inner**4) / 64  # noqa: E741
        _check(points, {"A": math.pi * (d**2 - inner**2) / 4, "Sy": 0, "Sz": 0, "Iy": I, "Iz": I, "Iyz": 0})
    H, B, tf, tw = 200.0, 100.0, 10.0, 6.0
    rects = [
        (H / 2 - tf, H / 2, -B / 2, B / 2),
        (-H / 2, -H / 2 + tf, -B / 2, B / 2),
        (-H / 2 + tf, H / 2 - tf, -tw / 2, tw / 2),
    ]
    _check(bs.iProfile(H, B, tf, tw, n, n, n, rule), _rectanglesIntegrals(rects))


@pytest.mark.parametrize("order", [2, 4, 5])
def test_section_points_polygons_exact(order, tmp_path):
    """polygons: an L-profile (non-principal axes) and a hollow square (hole by the even-odd rule) from files"""
    from edelweissfe.utils import beamsections as bs

    a, b, t = 60.0, 40.0, 6.0
    lLoop = [(0, 0), (0, b), (t, b), (t, t), (a, t), (a, 0)]  # (y, z): leg along y (length a), leg along z (b)
    expected = _rectanglesIntegrals([(0, a, 0, t), (0, t, t, b)])
    _check(bs.polygon([lLoop], meshSize=7.0, order=order), expected)

    path = tmp_path / "square.txt"
    path.write_text("# outer\n0 0\n50 0\n50 50\n0 50\n\n# hole\n10, 10\n10, 40\n40, 40\n40, 10\n")
    hollow = bs.polygon(bs.readPolygon(str(path)), meshSize=5.0, order=order)
    outer, inner = _rectanglesIntegrals([(0, 50, 0, 50)]), _rectanglesIntegrals([(10, 40, 10, 40)])
    _check(hollow, {k: outer[k] - inner[k] for k in outer})


@pytest.mark.parametrize("rule", ["gauss", "lobatto", "simpson"])
def test_section_points_plastic_modulus_converges(rule):
    """the plastic section modulus of the points, sum |y| A, converges to b h^2 / 4 for every rule"""
    from edelweissfe.utils import beamsections as bs

    b, h = 10.0, 20.0
    errors = []
    for n in (3, 7, 15, 31):
        p = bs.rectangle(b, h, n, 1, rule)
        errors.append(abs((np.abs(p.y) * p.area).sum() / (b * h * h / 4) - 1))
    assert all(e1 < e0 for e0, e1 in zip(errors, errors[1:])), errors
    assert errors[-1] < 2e-3, errors


@pytest.mark.parametrize("rule, detects", [("lobatto", True), ("simpson", True), ("gauss", False)])
def test_first_yield_detected_by_outer_fiber_points(tmp_path, rule, detects):
    """a 2D cantilever under a tip moment (constant moment) with a perfectly plastic von Mises material: with points
    on the outer fibers (lobatto, simpson), the first yield occurs exactly at the elastic limit moment
    M_el = f_y b h^2 / 6; Gauss points lie inside and miss it"""
    b, h, fy, L = 10.0, 20.0, 300.0, 100.0
    Mel = fy * b * h * h / 6

    def dissipation(M):
        deck = _beamDeck(
            "B2D2",
            (0, 0),
            (L, 0),
            2,
            f"profile=rectangle, b={b}, h={h}, nY=5, nZ=1, rule={rule}",
            "2d",
            f""">>dirichlet, name=root, nSet=root, field=displacement, 1=0, 2=0
>>dirichlet, name=rootR, nSet=root, field=rotation, 1=0
>>nodeforces, name=M, nSet=tip, field=rotation, 1={M}
""",
        ).replace(
            "linearelastic, id=steel, provider=marmot\n200000.0, 0.3",
            f"VonMises, id=steel, provider=marmot\n200000.0, 0.3, {fy}, 0, 0, 1, 1e-9",
        )
        deck = deck.replace(
            "*fieldOutput\n", "*fieldOutput\n>>perElement, name=D, elSet=beam, result=dissipation, quadraturePoint=0,\n"
        )
        _, (D,) = _run(tmp_path, deck, "D")
        return float(np.abs(D.getLastResult()).max())

    assert dissipation(0.999 * Mel) == 0.0
    if detects:
        assert dissipation(1.002 * Mel) > 0.0
    else:
        assert dissipation(1.002 * Mel) == 0.0


def test_l_profile_polygon_cantilever_couples_bending(tmp_path):
    """a 3D cantilever with an L-profile polygon section (centroid on the axis): a tip force along the local y axis,
    which is not a principal axis, deflects the beam also along z, as from the analytical Iy, Iz, Iyz"""
    a, b, t, L, P = 60.0, 40.0, 6.0, 400.0, 3.0
    path = tmp_path / "L.txt"
    path.write_text("0 0\n0 40\n6 40\n6 6\n60 6\n60 0\n")
    c = _rectanglesIntegrals([(0, a, 0, t), (0, t, t, b)])
    yc, zc = c["Sy"] / c["A"], c["Sz"] / c["A"]
    Iy, Iz, Iyz = c["Iy"] - c["A"] * zc**2, c["Iz"] - c["A"] * yc**2, c["Iyz"] - c["A"] * yc * zc
    deck = _beamDeck(
        "B3D2",
        (0, 0, 0),
        (L, 0, 0),
        4,
        f"profile=polygon, polygonFile={path}, meshSize=6, J=5000, n1x=0, n1y=1, n1z=0",
        "3d",
        f""">>dirichlet, name=root, nSet=root, field=displacement, 1=0, 2=0, 3=0
>>dirichlet, name=rootR, nSet=root, field=rotation, 1=0, 2=0, 3=0
>>nodeforces, name=P, nSet=tip, field=displacement, 2={P}
""",
    )
    _, (u,) = _run(tmp_path, deck, "u")
    u = u.getLastResult().ravel()
    # [My, Mz] = E [[Iy, -Iyz], [-Iyz, Iz]] [kappa_y, kappa_z] with My = 0, Mz = P (L - x)
    kappa = np.linalg.solve(E_STEEL * np.array([[Iy, -Iyz], [-Iyz, Iz]]), [0.0, P])
    np.testing.assert_allclose(u[1], kappa[1] * L**3 / 3, rtol=1e-9)
    np.testing.assert_allclose(u[2], -kappa[0] * L**3 / 3, rtol=1e-9)
    assert abs(u[2]) > 0.1 * abs(u[1])  # the coupling is substantial


def test_composite_section_by_overlapping_elements(tmp_path):
    """one material per element: a steel tube filled with concrete as two beam elements on the same nodes; the
    cantilever deflection is P L^3 / (3 (E_s I_s + E_c I_c))"""
    d, di, L, P = 100.0, 90.0, 1000.0, 50.0
    line, first, last = _line(1, (0, 0), (L, 0), 4, "B2D2")
    core = "\n".join(f"{100 + i}, {first + i}, {first + i + 1}" for i in range(4))
    deck = f"""*material, name=linearelastic, id=steel, provider=marmot
{E_STEEL}, 0.3
*material, name=linearelastic, id=concrete, provider=marmot
{E_CONCRETE}, 0.2

{line}
*element, type=B2D2
{core}

*elSet, elSet=tube, generate=True
{first}, {last - 1}, 1
*elSet, elSet=core, generate=True
100, 103, 1
*nSet, nSet=root
{first}
*nSet, nSet=tip
{last}

*section, name=tubeSection, type=beam, profile=tube, d={d}, di={di}, nRings=2, nSectors=12, material=steel
tube
*section, name=coreSection, type=beam, profile=circle, d={di}, nRings=3, nSectors=12, material=concrete
core

*job, name=job, domain=2d
*solver, solver=NIST, name=theSolver

*fieldOutput
>>perNode, name=u, nSet=tip, field=displacement, result=U, saveHistory=True,

*step, solver=theSolver
maxInc=1.0, minInc=1e-8, maxNumInc=10, maxIter=10, stepLength=1
>>dirichlet, name=root, nSet=root, field=displacement, 1=0, 2=0
>>dirichlet, name=rootR, nSet=root, field=rotation, 1=0
>>nodeforces, name=P, nSet=tip, field=displacement, 2={P}
"""
    _, (u,) = _run(tmp_path, deck, "u")
    EI = E_STEEL * math.pi * (d**4 - di**4) / 64 + E_CONCRETE * math.pi * di**4 / 64
    np.testing.assert_allclose(u.getLastResult().ravel()[1], P * L**3 / (3 * EI), rtol=1e-10)


def _cantilever(elType, L, nElements, section, load, maxInc, extraOutput=""):
    nodes = "\n".join(f"{i + 1}, {L * i / nElements:.12g}, 0" for i in range(nElements + 1))
    elements = "\n".join(f"{i + 1}, {i + 1}, {i + 2}" for i in range(nElements))
    return f"""*material, name=linearelastic, id=steel, provider=marmot
{E_STEEL}, 0.3

*node
{nodes}
*element, type={elType}
{elements}
*elSet, elSet=beam, generate=True
1, {nElements}, 1
*nSet, nSet=root
1
*nSet, nSet=tip
{nElements + 1}

*section, name=s, type=beam, {section}, material=steel
beam

*job, name=job, domain=2d
*solver, solver=NIST, name=theSolver

*fieldOutput
>>perNode, name=u, nSet=tip, field=displacement, result=U, saveHistory=True,
>>perNode, name=r, nSet=tip, field=rotation, result=U, saveHistory=True,
{extraOutput}
*step, solver=theSolver
maxInc={maxInc}, minInc=1e-8, maxNumInc=1000, maxIter=15, stepLength=1
>>dirichlet, name=root, nSet=root, field=displacement, 1=0, 2=0
>>dirichlet, name=rootR, nSet=root, field=rotation, 1=0
{load}
"""


def test_roll_up_full_circle(tmp_path):
    """an end moment M = 2 pi EI / L rolls the cantilever up into a full circle: the tip returns to the root with the
    rotation 2 pi; on the way the tip follows x = R sin(L/R), y = R (1 - cos(L/R)), R = EI / M"""
    L, b, h, n = 100.0, 1.0, 2.0, 20
    EI = E_STEEL * b * h**3 / 12
    M = 2 * math.pi * EI / L
    deck = _cantilever(
        "B2D2CR",
        L,
        n,
        f"profile=rectangle, b={b}, h={h}, nY=2",
        f">>nodeforces, name=M, nSet=tip, field=rotation, 1={M}",
        0.05,
    )
    _, (u, r) = _run(tmp_path, deck, "u", "r")
    U, R_ = np.asarray(u.getResultHistory()).reshape(-1, 2), np.asarray(r.getResultHistory()).ravel()
    t = np.asarray(u.getTimeHistory())
    np.testing.assert_allclose(R_, 2 * math.pi * t, rtol=1e-8)  # tip rotation = M L / EI, exact
    m = t > 0
    Rad = L / (2 * math.pi * t[m])
    x, y = L + U[m, 0], U[m, 1]
    err = np.hypot(x - Rad * np.sin(L / Rad), y - Rad * (1 - np.cos(L / Rad))) / L
    assert err.max() < 0.5 * (math.pi / n) ** 2  # chords of the circle: O(h^2)
    assert abs(x[-1]) < 1e-6 * L and abs(y[-1]) < 1e-6 * L  # full circle: the tip at the root


@pytest.mark.parametrize("lam, ref", [(2.0, (0.49346, 0.16064, 0.78175)), (10.0, (0.81061, 0.55500, 1.43029))])
def test_elastica_tip_force(tmp_path, lam, ref):
    """cantilever with a dead tip force P normal to the axis, lambda = P L^2 / EI: tip deflection v/L, shortening
    (L - x)/L and rotation vs the elastica (Mattiasson; rotation by the elliptic integral)"""
    L, b, h, n = 100.0, 1.0, 1.0, 40
    EI = E_STEEL * b * h**3 / 12
    deck = _cantilever(
        "B2D2CR",
        L,
        n,
        f"profile=rectangle, b={b}, h={h}, nY=2",
        f">>nodeforces, name=P, nSet=tip, field=displacement, 2={lam * EI / L**2}",
        0.1,
    )
    _, (u, r) = _run(tmp_path, deck, "u", "r")
    U, R_ = u.getLastResult().ravel(), r.getLastResult().ravel()
    np.testing.assert_allclose([U[1] / L, -U[0] / L, R_[0]], ref, atol=1e-3)


def test_small_load_equals_BE2D2(tmp_path):
    """small tip load: B2D2CR = B2D2 (geometrically linear) up to the second order effects"""
    L, b, h, n, P = 100.0, 2.0, 4.0, 4, 1e-3
    results = []
    for elType in ("B2D2", "B2D2CR"):
        deck = _cantilever(
            elType,
            L,
            n,
            f"profile=rectangle, b={b}, h={h}",
            f">>nodeforces, name=P, nSet=tip, field=displacement, 2={P}",
            1.0,
        )
        _, (u,) = _run(tmp_path, deck, "u")
        results.append(u.getLastResult().ravel()[1])
    np.testing.assert_allclose(results[1], results[0], rtol=1e-8)
    np.testing.assert_allclose(results[0], P * L**3 / (3 * E_STEEL * b * h**3 / 12), rtol=1e-10)


def test_section_multiplicity(tmp_path):
    """multiplicity = k: a beam of k identical members, k times stiffer (A, I scaled, coordinates not)"""
    L, d, n, P = 50.0, 1.0, 4, 1e-4
    tips = []
    for k in (1.0, 3.0):
        deck = _cantilever(
            "B2D2CR",
            L,
            n,
            f"profile=circle, d={d}, nRings=2, nSectors=8, multiplicity={k}",
            f">>nodeforces, name=P, nSet=tip, field=displacement, 2={P}",
            1.0,
        )
        _, (u,) = _run(tmp_path, deck, "u")
        tips.append(u.getLastResult().ravel()[1])
    np.testing.assert_allclose(tips[0] / tips[1], 3.0, rtol=1e-9)
    np.testing.assert_allclose(tips[0], P * L**3 / (3 * E_STEEL * math.pi * d**4 / 64), rtol=1e-6)
