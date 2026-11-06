#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  ---------------------------------------------------------------------
#
#  _____    _      _              _         _____ _____
# | ____|__| | ___| |_      _____(_)___ ___|  ___| ____|
# |  _| / _` |/ _ \ \ \ /\ / / _ \ / __/ __| |_  |  _|
# | |__| (_| |  __/ |\ V  V /  __/ \__ \__ \  _| | |___
# |_____\__,_|\___|_| \_/\_/ \___|_|___/___/_|   |_____|
#
#
#  Unit of Strength of Materials and Structural Analysis
#  University of Innsbruck,
#  2017 - today
#
#  Matthias Neuner matthias.neuner@uibk.ac.at
#
#  This file is part of EdelweissFE.
#
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation; either
#  version 2.1 of the License, or (at your option) any later version.
#
#  The full text of the license can be found in the file LICENSE.md at
#  the top level directory of EdelweissFE.
#  ---------------------------------------------------------------------
"""
Integration points of beam cross sections ("section points"), for beam elements that integrate their section over
given points, e.g., the Marmot beams ``B<d>D<n>``.

A section is a set of points :math:`(y_i, z_i)` with areas (weights) :math:`A_i` in the section coordinates
:math:`(y, z)` of the beam. The generators cover

* parametric shapes with a selectable 1D rule and number of points per direction: ``rectangle`` (:math:`n_y
  \\times n_z`), ``circle`` and ``tube`` (rings :math:`\\times` sectors), ``iProfile`` (per flange and web),
* polygons with holes (even-odd rule), decomposed exactly into trapezoids, subdivided to a mesh size and
  integrated with a triangle rule of selectable order,
* explicit point lists.

The 1D rules (on :math:`[-1, 1]`): ``gauss`` (Gauss-Legendre, :math:`n \\geq 1`), ``lobatto`` (Gauss-Lobatto,
:math:`n \\geq 2`, points on the outer fibers) and ``simpson`` (composite Simpson, odd :math:`n \\geq 3`, points on
the outer fibers). With points on the outer fibers, the first yield of a section is detected exactly. The second
moments of area are integrated exactly by gauss with :math:`n \\geq 2`, lobatto with :math:`n \\geq 3` and simpson.
"""

from dataclasses import dataclass

import numpy as np

#: The 1D integration rules.
RULES = ("gauss", "lobatto", "simpson")


def lineRule(rule: str, n: int) -> tuple[np.ndarray, np.ndarray]:
    """A 1D integration rule on [-1, 1].

    Parameters
    ----------
    rule
        ``gauss``, ``lobatto`` or ``simpson``.
    n
        The number of points.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The points and the weights (summing to 2).
    """
    rule = rule.lower()
    if rule == "gauss":
        if n < 1:
            raise ValueError("A Gauss rule needs at least 1 point.")
        return np.polynomial.legendre.leggauss(n)
    if rule == "lobatto":
        if n < 2:
            raise ValueError("A Lobatto rule needs at least 2 points.")
        P = np.polynomial.legendre.Legendre.basis(n - 1)
        x = np.concatenate([[-1.0], np.sort(P.deriv().roots().real), [1.0]])
        w = 2.0 / (n * (n - 1) * P(x) ** 2)
        return x, w
    if rule == "simpson":
        if n < 3 or n % 2 == 0:
            raise ValueError("A Simpson rule needs an odd number of at least 3 points.")
        x = np.linspace(-1.0, 1.0, n)
        w = np.full(n, 2.0)
        w[1::2] = 4.0
        w[0] = w[-1] = 1.0
        return x, w * (2.0 / (n - 1)) / 3.0
    raise ValueError(f"Unknown integration rule '{rule}', use one of {RULES}.")


@dataclass(frozen=True)
class SectionPoints:
    """The integration points of a cross section."""

    #: The y coordinates.
    y: np.ndarray
    #: The z coordinates.
    z: np.ndarray
    #: The areas (weights).
    area: np.ndarray

    def __post_init__(self):
        if not (len(self.y) == len(self.z) == len(self.area)) or len(self.y) == 0:
            raise ValueError("A section needs points with y, z and area each.")
        if np.any(np.asarray(self.area) <= 0):
            raise ValueError("Section points must have positive areas.")

    @classmethod
    def fromArrays(cls, y, z, area) -> "SectionPoints":
        y, z, area = (np.asarray(v, dtype=float).ravel() for v in (y, z, area))
        keep = area > 1e-14 * area.max()  # e.g., the center of a circle in a Lobatto rule
        return cls(y[keep], z[keep], area[keep])

    def __len__(self):
        return len(self.y)

    def integrals(self) -> dict:
        """``A``, the first moments ``Sy`` = int y dA and ``Sz`` = int z dA, and the second moments ``Iy`` = int z^2 dA,
        ``Iz`` = int y^2 dA, ``Iyz`` = int y z dA."""
        y, z, a = self.y, self.z, self.area
        return {
            "A": a.sum(),
            "Sy": (y * a).sum(),
            "Sz": (z * a).sum(),
            "Iy": (z * z * a).sum(),
            "Iz": (y * y * a).sum(),
            "Iyz": (y * z * a).sum(),
        }

    def centroid(self) -> tuple[float, float]:
        c = self.integrals()
        return c["Sy"] / c["A"], c["Sz"] / c["A"]

    def shifted(self, dy: float, dz: float) -> "SectionPoints":
        return SectionPoints(self.y + dy, self.z + dz, self.area)

    def centered(self) -> "SectionPoints":
        """The points shifted such that their centroid lies on the beam axis."""
        yc, zc = self.centroid()
        return self.shifted(-yc, -zc)

    def __add__(self, other: "SectionPoints") -> "SectionPoints":
        return SectionPoints(
            np.concatenate([self.y, other.y]),
            np.concatenate([self.z, other.z]),
            np.concatenate([self.area, other.area]),
        )


def rectangle(b: float, h: float, nY: int = 5, nZ: int = 5, rule: str = "gauss", center=(0.0, 0.0)) -> SectionPoints:
    """A rectangle of height ``h`` (along y) and width ``b`` (along z): ``nY`` x ``nZ`` points of the rule
    (``nZ = 1``: a single point at the middle of the width, e.g., for 2D beams)."""
    yi, wy = lineRule(rule, nY)
    zi, wz = (np.zeros(1), np.full(1, 2.0)) if nZ == 1 else lineRule(rule, nZ)
    Y, Z = np.meshgrid(center[0] + 0.5 * h * yi, center[1] + 0.5 * b * zi, indexing="ij")
    W = np.outer(wy, wz) * 0.25 * b * h
    return SectionPoints.fromArrays(Y, Z, W)


def tube(
    d: float, di: float = 0.0, nRings: int = 3, nSectors: int = 8, rule: str = "gauss", center=(0.0, 0.0)
) -> SectionPoints:
    """A circle (``di = 0``) or a tube of outer diameter ``d`` and inner diameter ``di``: ``nRings`` points of the rule
    in the radial direction (weighted by the radius) x ``nSectors`` (at least 3) equally spaced angles, starting on
    the y axis (exact area and second moments for exact radial rules)."""
    if nSectors < 3:
        raise ValueError("A circular section needs at least 3 sectors.")
    if not 0 <= di < d:
        raise ValueError("The inner diameter of a tube must be smaller than its outer diameter.")
    ri, ro = 0.5 * di, 0.5 * d
    xi, w = lineRule(rule, nRings)
    r = 0.5 * (ri + ro) + 0.5 * (ro - ri) * xi
    phi = 2 * np.pi * np.arange(nSectors) / nSectors
    R, PHI = np.meshgrid(r, phi, indexing="ij")
    A = np.outer(w * 0.5 * (ro - ri) * r, np.full(nSectors, 2 * np.pi / nSectors))
    return SectionPoints.fromArrays(center[0] + R * np.cos(PHI), center[1] + R * np.sin(PHI), A)


def iProfile(
    h: float,
    b: float,
    tf: float,
    tw: float,
    nWeb: int = 5,
    nWidth: int = 5,
    nThickness: int = 2,
    rule: str = "gauss",
) -> SectionPoints:
    """A doubly symmetric I (H) profile of height ``h`` (along y), flange width ``b`` (along z), flange thickness ``tf``
    and web thickness ``tw``: per flange ``nThickness`` x ``nWidth`` points, in the web (between the flanges)
    ``nWeb`` x ``nThickness`` points."""
    hw = h - 2 * tf
    if hw <= 0 or tw > b:
        raise ValueError("Invalid I profile dimensions.")
    flange = rectangle(b, tf, nThickness, nWidth, rule)
    web = rectangle(tw, hw, nWeb, nThickness, rule)
    return flange.shifted(0.5 * (h - tf), 0.0) + flange.shifted(-0.5 * (h - tf), 0.0) + web


#: Symmetric triangle rules with positive weights (Dunavant), by polynomial degree: (barycentric points, weights).
_TRIANGLE_RULES = {
    1: ([(1 / 3, 1 / 3, 1 / 3)], [1.0]),
    2: ([(2 / 3, 1 / 6, 1 / 6), (1 / 6, 2 / 3, 1 / 6), (1 / 6, 1 / 6, 2 / 3)], [1 / 3] * 3),
}


def _orbit(a: float) -> list:
    b = 1 - 2 * a
    return [(b, a, a), (a, b, a), (a, a, b)]


_TRIANGLE_RULES[4] = (
    _orbit(0.445948490915965) + _orbit(0.091576213509771),
    [0.223381589678011] * 3 + [0.109951743655322] * 3,
)
_TRIANGLE_RULES[5] = (
    [(1 / 3, 1 / 3, 1 / 3)] + _orbit(0.470142064105115) + _orbit(0.101286507323456),
    [0.225] + [0.132394152788506] * 3 + [0.125939180544827] * 3,
)

#: The available orders (polynomial degrees) of the triangle rules.
TRIANGLE_ORDERS = tuple(sorted(_TRIANGLE_RULES))


def _trapezoids(loops: list) -> list:
    """Exact decomposition of a polygon with holes (even-odd rule) into trapezoids with sides parallel to the z axis:
    (y0, y1, zLeft0, zRight0, zLeft1, zRight1)."""
    edges = []
    for loop in loops:
        loop = np.asarray(loop, dtype=float)
        for p, q in zip(loop, np.roll(loop, -1, axis=0)):
            if p[0] != q[0]:  # edges parallel to z bound no slab
                edges.append((p, q) if p[0] < q[0] else (q, p))
    levels = np.unique(np.concatenate([np.asarray(loop, dtype=float)[:, 0] for loop in loops]))
    trapezoids = []
    for y0, y1 in zip(levels[:-1], levels[1:]):
        ym = 0.5 * (y0 + y1)
        crossing = [(p, q) for p, q in edges if p[0] <= ym <= q[0]]

        def zAt(edge, y):
            p, q = edge
            return p[1] + (q[1] - p[1]) * (y - p[0]) / (q[0] - p[0])

        crossing.sort(key=lambda e: zAt(e, ym))
        if len(crossing) % 2:
            raise ValueError("The section polygon is not closed or self-intersecting.")
        for left, right in zip(crossing[0::2], crossing[1::2]):
            trapezoids.append((y0, y1, zAt(left, y0), zAt(right, y0), zAt(left, y1), zAt(right, y1)))
    return trapezoids


def polygon(loops: list, meshSize: float = None, order: int = 2) -> SectionPoints:
    """A polygonal section, optionally with holes.

    The polygon is decomposed exactly into trapezoids (between the y levels of its vertices), which are subdivided
    to the ``meshSize``, split into triangles and integrated with a triangle rule of the polynomial degree
    ``order`` (1, 2, 4 or 5; 2 or more integrates the second moments of area exactly).

    Parameters
    ----------
    loops
        The boundary loops, each a sequence of vertices (y, z); the outer boundary and the holes (even-odd rule, the
        orientation does not matter).
    meshSize
        The largest size of the cells; default: a tenth of the larger extent of the section.
    order
        The polynomial degree of the triangle rule.
    """
    if order not in _TRIANGLE_RULES:
        raise ValueError(f"Triangle rules of the orders {TRIANGLE_ORDERS} are available.")
    allVertices = np.concatenate([np.asarray(loop, dtype=float) for loop in loops])
    if meshSize is None:
        meshSize = np.ptp(allVertices, axis=0).max() / 10
    bary, weights = (np.asarray(v, dtype=float) for v in _TRIANGLE_RULES[order])

    ys, zs, areas = [], [], []

    def addTriangle(a, b, c):
        area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
        if area <= 0:
            return
        corners = np.array([a, b, c])
        p = bary @ corners
        ys.extend(p[:, 0])
        zs.extend(p[:, 1])
        areas.extend(weights * area)

    for y0, y1, zl0, zr0, zl1, zr1 in _trapezoids(loops):
        nY = max(1, int(np.ceil((y1 - y0) / meshSize)))
        nZ = max(1, int(np.ceil(max(zr0 - zl0, zr1 - zl1) / meshSize)))
        for i in range(nY):
            s0, s1 = i / nY, (i + 1) / nY
            ya, yb = y0 + s0 * (y1 - y0), y0 + s1 * (y1 - y0)
            la, ra = zl0 + s0 * (zl1 - zl0), zr0 + s0 * (zr1 - zr0)
            lb, rb = zl0 + s1 * (zl1 - zl0), zr0 + s1 * (zr1 - zr0)
            for j in range(nZ):
                t0, t1 = j / nZ, (j + 1) / nZ
                p00 = (ya, la + t0 * (ra - la))
                p01 = (ya, la + t1 * (ra - la))
                p10 = (yb, lb + t0 * (rb - lb))
                p11 = (yb, lb + t1 * (rb - lb))
                addTriangle(p00, p01, p11)
                addTriangle(p00, p11, p10)
    return SectionPoints.fromArrays(ys, zs, areas)


def genericPoints(A: float, Iy: float, Iz: float) -> SectionPoints:
    """The fewest points reproducing the area and the (principal) second moments: 2 points at y = +- sqrt(Iz / A)
    if Iy = 0 (2D), else 4 points at (+- sqrt(Iz / A), +- sqrt(Iy / A)). Exact for elastic materials; for inelastic
    ones an idealized sandwich section."""
    ry = np.sqrt(Iz / A)
    if not Iy:
        return SectionPoints.fromArrays([-ry, ry], [0.0, 0.0], [A / 2, A / 2])
    rz = np.sqrt(Iy / A)
    return SectionPoints.fromArrays([-ry, ry, ry, -ry], [-rz, -rz, rz, rz], [A / 4] * 4)


def readPoints(fileName: str) -> SectionPoints:
    """Section points from a text file, one point per line: ``y, z, area`` (or ``y, area`` for 2D sections);
    lines starting with ``#`` or ``**`` are comments."""
    rows = []
    with open(fileName) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("**"):
                continue
            rows.append([float(v) for v in line.replace(",", " ").split()])
    data = np.array(rows, dtype=float)
    if data.ndim != 2 or data.shape[1] not in (2, 3):
        raise ValueError(f"{fileName}: expected lines 'y, z, area' or 'y, area'.")
    if data.shape[1] == 2:
        return SectionPoints.fromArrays(data[:, 0], np.zeros(len(data)), data[:, 1])
    return SectionPoints.fromArrays(data[:, 0], data[:, 1], data[:, 2])


def readPolygon(fileName: str) -> list:
    """Polygon loops from a text file: one vertex ``y, z`` per line, loops (outer boundary, holes) separated by empty
    lines; lines starting with ``#`` or ``**`` are comments."""
    loops, current = [], []
    with open(fileName) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or line.startswith("**"):
                continue
            if not line:
                if current:
                    loops.append(current)
                    current = []
                continue
            current.append([float(v) for v in line.replace(",", " ").split()])
    if current:
        loops.append(current)
    if not loops or any(len(loop) < 3 for loop in loops):
        raise ValueError(f"{fileName}: expected loops of at least 3 vertices 'y, z'.")
    return loops
