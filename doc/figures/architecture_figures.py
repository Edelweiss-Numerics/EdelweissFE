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
"""Regenerate the figures of doc/source/developer/architecture.rst:

* architecture_example.svg: the two-element example of the numbering and the locations;
* architecture_objects.svg: the holder, lifetime and creator of the objects of a run;
* discretesystem_change.svg: what the solver rebuilds for each Change;
* discretesystem_increment.svg: one increment of an implicit and of the explicit solver.

The SVG files are written by hand-placed primitives, without dependencies beyond the standard library, and are
committed with the documentation. Run it from anywhere after a change of the figures:

.. code-block:: bash

    python doc/figures/architecture_figures.py

The colours of the lifetimes (dark blue: per numbering, light blue: per locations) are shared by the figures, so that
they read the same way on the whole page.
"""

import pathlib
import sys
from html import escape

SANS = "DejaVu Sans, Helvetica, Arial, sans-serif"
MONO = "DejaVu Sans Mono, Menlo, Consolas, monospace"
INK, MUTED, LINE, GREY, BLUEFILL, BLUE = "#1f2328", "#57606a", "#8c959f", "#f6f8fa", "#ddf4ff", "#0969da"


class Svg:
    def __init__(self, w, h, alt):
        self.w, self.h, self.alt, self.parts = w, h, alt, []
        self.arrowColors = {"#6e7781"}

    def rect(self, x, y, w, h, fill="#ffffff", stroke=LINE, dash=False, sw=1.2, rx=6):
        d = ' stroke-dasharray="5 4"' if dash else ""
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{sw}"{d}/>'
        )

    def text(self, x, y, s, size=11, bold=False, mono=False, color=MUTED, anchor="start", italic=False):
        fam = MONO if mono else SANS
        st = ' font-style="italic"' if italic else ""
        self.parts.append(
            f'<text x="{x}" y="{y}" xml:space="preserve" font-family="{fam}" font-size="{size}" '
            f'font-weight="{"bold" if bold else "normal"}" fill="{color}" text-anchor="{anchor}"{st}>{escape(s)}</text>'
        )

    def line(self, d, color="#6e7781", sw=1.4, arrow=False, dash=False, dot=False):
        if arrow:
            self.arrowColors.add(color)
        m = f' marker-end="url(#arrow{color.lstrip("#")})"' if arrow else ""
        ds = ' stroke-dasharray="5 4"' if dash else (' stroke-dasharray="1.5 3"' if dot else "")
        self.parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}"{ds}{m}/>')

    def circle(self, x, y, r, fill="#ffffff", stroke=INK):
        self.parts.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>')

    def write(self, path):
        alt = escape(self.alt)
        head = (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" '
            f'height="{self.h}" role="img" aria-label="{alt}">\n<title>{alt}</title>\n'
            "<defs>"
            + "".join(
                f'<marker id="arrow{c.lstrip("#")}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
                f'markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>'
                for c in sorted(self.arrowColors)
            )
            + "</defs>\n"
            f'<rect x="0" y="0" width="{self.w}" height="{self.h}" fill="#ffffff"/>\n'
        )
        with open(path, "w") as f:
            f.write(head + "\n".join(self.parts) + "\n</svg>\n")


NUMFILL, NUMLINE = "#a8d1fb", "#0969da"  # per numbering, as in the objects figure
LOCFILL, LOCLINE = "#dbeefe", "#3d8fe0"  # per locations
TB, TL = 13, 12  # title and label size


def node(svg, x, y, w, title, lines, kind="plain", mono=True, h=None, right=None, mlines=()):
    """A box with a bold title and lines of 12 px text; kind: plain, kept (grey, dashed), num, loc, solve.
    Lines whose index is in mlines are set in monospace."""
    fill, stroke, dash, sw = {
        "plain": ("#ffffff", LINE, False, 1.2),
        "kept": (GREY, LINE, True, 1.2),
        "num": (NUMFILL, NUMLINE, False, 2),
        "loc": (LOCFILL, LOCLINE, False, 2),
        "solve": (BLUEFILL, BLUE, False, 2),
    }[kind]
    h = h or (36 + 17 * len(lines))
    svg.rect(x, y, w, h, fill=fill, stroke=stroke, dash=dash, sw=sw)
    svg.text(x + 12, y + 22, title, size=TB, bold=True, mono=mono, color=INK)
    if right:
        svg.text(x + w - 12, y + 22, right, size=TL, color=MUTED, anchor="end")
    for k, s in enumerate(lines):
        svg.text(x + 12, y + 41 + 17 * k, s, size=TL, color=INK if kind != "kept" else MUTED, mono=k in mlines)
    return h


def example(path):
    alt = (
        "A small example. Left: two quadrilateral elements, e1 with the nodes 1, 2, 3, 4 and e2 with the nodes 5, 6, "
        "7, 8, separated by a gap; node 2 faces the middle of the facet 5-8 of e2, and a contact constraint c couples "
        "node 2 with this facet once the gap closes. Below: the ID array, two DOFs per node, node 1: 0, 1 up to node "
        "8: 14, 15, and the location of c, 2, 3, 8, 9, 14, 15. Middle: the sparsity pattern of the system matrix, one "
        "cell per pair of nodes, before the contact: two separate blocks, nodes 1 to 4 and nodes 5 to 8. Right: the "
        "pattern with the contact, case (a): the pairs 2-5, 2-8, 5-2 and 8-2 are added. The numbering is the same in "
        "both; the locations and the pattern change: Change.LOCATIONS."
    )
    svg = Svg(1000, 404, alt)
    svg.text(16, 26, "Two elements, one contact constraint", size=TB, bold=True, color=INK)
    svg.text(16, 46, "every node: a displacement field with 2 components, so 16 DOFs", size=TL)

    s, g, ox, oy = 110, 80, 40, 130  # oy: the top of e2
    e1 = [(ox, oy + s / 2), (ox + s, oy + s / 2), (ox + s, oy - s / 2), (ox, oy - s / 2)]  # 1, 2, 3, 4
    e2 = [(ox + s + g, oy + s), (ox + 2 * s + g, oy + s), (ox + 2 * s + g, oy), (ox + s + g, oy)]  # 5, 6, 7, 8
    for quad, name in ((e1, "e1"), (e2, "e2")):
        d = "M" + " L".join(f"{x},{y}" for x, y in quad) + " Z"
        svg.parts.append(f'<path d="{d}" fill="{GREY}" stroke="{INK}" stroke-width="1.4"/>')
        cx = sum(p[0] for p in quad) / 4
        cy = sum(p[1] for p in quad) / 4
        svg.text(cx, cy + 5, name, size=14, bold=True, mono=True, color=INK, anchor="middle")
    offsets = {
        1: (-12, 18),
        2: (-4, 20),
        3: (12, -8),
        4: (-12, -8),
        5: (-12, 18),
        6: (12, 18),
        7: (12, -8),
        8: (-12, -8),
    }
    corners = {1: e1[0], 2: e1[1], 3: e1[2], 4: e1[3], 5: e2[0], 6: e2[1], 7: e2[2], 8: e2[3]}
    # the contact: node 2 on the middle of the facet 5-8
    x2, y2 = e1[1]
    xf = e2[0][0]
    svg.line(f"M{xf},{e2[0][1]} V{e2[3][1]}", color=BLUE, sw=3.5)
    svg.line(f"M{x2},{y2} H{xf}", color=BLUE, sw=2, dash=True)
    svg.circle(xf, y2, 3, fill=BLUE, stroke=BLUE)
    svg.text((x2 + xf) / 2, y2 - 8, "c", size=14, bold=True, mono=True, color=BLUE, anchor="middle")
    for n, (x, y) in corners.items():
        svg.circle(x, y, 4, fill=BLUE if n in (2, 5, 8) else INK, stroke=BLUE if n in (2, 5, 8) else INK)
        dx, dy = offsets[n]
        svg.text(
            x + dx,
            y + dy,
            str(n),
            size=TL,
            bold=True,
            mono=True,
            color=BLUE if n in (2, 5, 8) else INK,
            anchor="middle",
        )
    svg.text(16, oy + s + 48, "c couples node 2 and the facet 5-8, once the gap closes", size=TL, color=BLUE)

    # the ID array, and the location of c
    ty = oy + s + 84
    cw = 52
    svg.text(16, ty, "node", size=TL, color=MUTED)
    svg.text(16, ty + 22, "ID", size=TL, color=MUTED)
    svg.line(f"M16,{ty + 7} H{16 + 50 + 8 * cw}", color="#d0d7de", sw=1)
    for n in range(1, 9):
        cx = 16 + 50 + (n - 0.5) * cw
        col = BLUE if n in (2, 5, 8) else INK
        svg.text(cx, ty, str(n), size=TL, bold=True, mono=True, color=col, anchor="middle")
        svg.text(cx, ty + 22, f"{2 * n - 2}, {2 * n - 1}", size=TL, mono=True, color=col, anchor="middle")
    svg.text(16, ty + 52, "locations[c] = 2, 3, 8, 9, 14, 15", size=TL, mono=True, color=BLUE)
    svg.text(16, ty + 70, "the ID of the nodes 2, 5, 8 (indices start at 0)", size=TL, color=MUTED)

    # the patterns
    def pattern(px, py, title, sub, withContact):
        cs = 22
        svg.text(px, py - 34, title, size=TB, bold=True, color=INK)
        svg.text(px, py - 15, sub, size=TL, color=BLUE if withContact else MUTED)
        coupled = set()
        for blk in ((1, 2, 3, 4), (5, 6, 7, 8)):
            coupled |= {(i, j) for i in blk for j in blk}
        new = set()
        if withContact:
            new = {(i, j) for i in (2, 5, 8) for j in (2, 5, 8)} - coupled
        for i in range(1, 9):
            svg.text(px - 8, py + (i - 0.5) * cs + 4, str(i), size=TL, mono=True, anchor="end")
            svg.text(px + (i - 0.5) * cs, py + 8 * cs + 16, str(i), size=TL, mono=True, anchor="middle")
            for j in range(1, 9):
                x, y = px + (j - 1) * cs, py + (i - 1) * cs
                if (i, j) in new:
                    svg.rect(x, y, cs, cs, fill=BLUE, stroke="#ffffff", sw=1, rx=0)
                elif (i, j) in coupled:
                    svg.rect(x, y, cs, cs, fill="#8c959f", stroke="#ffffff", sw=1, rx=0)
                else:
                    svg.rect(x, y, cs, cs, fill="#ffffff", stroke="#d0d7de", sw=0.6, rx=0)

    pattern(500, 112, "Pattern before the contact", "two separate blocks", False)
    pattern(770, 112, "With the contact, case (a)", "blue: the entries c adds", True)
    svg.text(486, 330, "one cell: the 2x2 block of a pair of nodes; rows and columns: node numbers", size=TL)
    svg.text(486, 352, "Same numbering in both; new locations, new pattern: Change.LOCATIONS", size=TL, color=INK)
    svg.write(path)


def change(path):
    alt = (
        "What the solver rebuilds for each Change. NIST runs the topology update at the start of every increment: model "
        "modifiers may change the mesh, except on a retry, and model.updateConnectivity() reports the rest; the "
        "other solvers are listed on the page of the solver base class. Row 1, Change.NONE with an existing discrete "
        "system: the DofNumbering and the DofLocations are the same objects, and no hook is called. Row 2, "
        "Change.LOCATIONS, case (a): withNewLocations() keeps the DofNumbering and creates new DofLocations; "
        "onNewNumbering() is not called, so U, dU and P keep their values and the MPC condensation and the DOF layout "
        "of the linear solver are kept; onNewLocations() wraps the vectors with the new locations, creates a new "
        "SystemMatrixAssembly, and the linear solver drops its factorization. Row 3, Change.NUMBERING or MESH, cases "
        "(b) and (c), or no discrete system yet: forgetDiscreteSystem() and buildDiscreteSystem() create a new "
        "DofNumbering and new DofLocations; onNewNumbering() reads U, dU, P from the model, builds the MPC "
        "condensation and the DOF layout of the linear solver, then onNewLocations() follows. Dark blue: new per "
        "numbering; light blue: new per locations; grey and dashed: kept, or not called."
    )
    W = 1080
    svg = Svg(W, 100, alt)
    svg.rect(16, 12, W - 32, 58, fill=GREY, dash=True)
    svg.text(
        30,
        36,
        "NIST, at the start of every increment: the topology update gives a Change",
        size=TB,
        bold=True,
        color=INK,
    )
    svg.text(
        30,
        57,
        "model modifiers may change the mesh (not on a retry); model.updateConnectivity() reports the rest; "
        "other solvers: see the solver base class",
        size=TL,
    )

    cx, cw = 44, 206
    ax = cx + cw
    nx, nw = ax + 176, 150
    lx, lw = nx + nw + 14, 150
    hx = lx + lw + 40
    hw = W - 16 - hx
    GAP = 8

    rows = [
        dict(
            change=("Change.NONE", ["nothing changed, and", "a discrete system exists"]),
            method=None,
            numNew=False,
            locNew=False,
            hooks=[("onNewNumbering()", "kept", [], "not called"), ("onNewLocations()", "kept", [], "not called")],
        ),
        dict(
            change=("Change.LOCATIONS", ["case (a): an entity couples", "other nodes, or takes part", "or not"]),
            method=["withNewLocations()"],
            numNew=False,
            locNew=True,
            hooks=[
                (
                    "onNewNumbering()",
                    "kept",
                    ["U, dU, P keep their values;", "MPC condensation, DOF layout kept"],
                    "not called",
                ),
                (
                    "onNewLocations()",
                    "loc",
                    [
                        "vectors: same values, new LM;",
                        "new SystemMatrixAssembly;",
                        "linear solver drops its factorization",
                    ],
                    None,
                ),
            ],
        ),
        dict(
            change=(
                "Change.NUMBERING / MESH",
                ["cases (b) and (c), or no", "discrete system yet: the first", "increment of every step"],
            ),
            method=["forgetDiscreteSystem()", "buildDiscreteSystem()"],
            numNew=True,
            locNew=True,
            hooks=[
                (
                    "onNewNumbering()",
                    "num",
                    ["U, dU, P read from the model;", "MPC condensation;", "DOF layout of the linear solver"],
                    None,
                ),
                (
                    "onNewLocations()",
                    "loc",
                    [
                        "vectors: same values, new LM;",
                        "new SystemMatrixAssembly;",
                        "linear solver drops its factorization",
                    ],
                    None,
                ),
            ],
        ),
    ]
    y = 96
    centers = []
    for k, r in enumerate(rows):
        hs = [36 + 17 * len(lines) for _, _, lines, _ in r["hooks"]]
        rh = max(sum(hs) + GAP, 96)
        title, lines = r["change"]
        node(svg, cx, y, cw, title, lines, h=rh)
        node(
            svg,
            nx,
            y,
            nw,
            "DofNumbering",
            ["(the ID array)", "new object" if r["numNew"] else "the same object"],
            kind="num" if r["numNew"] else "kept",
            h=rh,
        )
        node(
            svg,
            lx,
            y,
            lw,
            "DofLocations",
            ["(the LM array)", "new object" if r["locNew"] else "the same object"],
            kind="loc" if r["locNew"] else "kept",
            h=rh,
        )
        cy = y + rh / 2
        centers.append(cy)
        if r["method"]:
            svg.line(f"M{ax},{cy} H{nx - 2}", arrow=True)
            for m, s in enumerate(r["method"]):
                svg.text(
                    (ax + nx) / 2,
                    cy - 8 - 17 * (len(r["method"]) - 1 - m),
                    s,
                    size=TL,
                    mono=True,
                    color=INK,
                    anchor="middle",
                )
            svg.line(f"M{lx + lw},{cy} H{hx - 2}", arrow=True)
            svg.text((lx + lw + hx) / 2, cy - 7, "calls", size=TL, anchor="middle")
        else:
            svg.text((ax + nx) / 2, cy + 4, "nothing to do", size=TL, italic=True, anchor="middle")
        hy = y
        for (name, kind, hl, right), h in zip(r["hooks"], hs):
            node(svg, hx, hy, hw, name, hl, kind=kind, right=right, h=h)
            hy += h + GAP
        y += rh
        if k == 2:
            svg.text(
                W - 16, y + 18, "onNewNumbering() first, then onNewLocations()", size=TL, italic=True, anchor="end"
            )
            y += 10
        y += 26
    # the Change, fanned out to the three rows
    svg.line(f"M28,70 V{centers[-1]}", sw=1.4)
    for c in centers:
        svg.line(f"M28,{c} H{cx - 2}", arrow=True)
    # legend
    ly = y + 4
    svg.rect(16, ly, W - 32, 56, fill=GREY)
    x = 30
    for kind, label in (("num", "new, per numbering"), ("loc", "new, per locations"), ("kept", "kept, or not called")):
        fill, stroke, dash = {
            "num": (NUMFILL, NUMLINE, False),
            "loc": (LOCFILL, LOCLINE, False),
            "kept": ("#ffffff", LINE, True),
        }[kind]
        svg.rect(x, ly + 12, 28, 16, fill=fill, stroke=stroke, dash=dash, sw=2 if not dash else 1.2, rx=3)
        svg.text(x + 36, ly + 25, label, size=TL, color=INK)
        x += 210
    svg.text(
        30,
        ly + 46,
        "The hooks are shown as NIST implements them; NID and NED hold more, e.g. NED builds its lumped "
        "mass and its critical time step in onNewNumbering().",
        size=TL,
    )
    svg.h = ly + 56 + 12
    svg.write(path)


def increment(path):
    alt = (
        "One increment. 1: prepareIncrement checks whether the ID or the LM array changed, and the solver rebuilds "
        "what depends on them. Left: the discrete system with its elements and constraints; each computes its internal "
        "force P_e and tangent K_e, and its location places them in the global system. Implicit lane (NIST, NID and "
        "their variants), Newton's method: residual and tangent, K_e into K[el], R = f_ext - sum of P_e, the MPC "
        "condensation of R; then the convergence check: in the first iteration the Dirichlet rows of R receive the "
        "prescribed increments, later they are zero and R and ddU are checked; if converged, the increment is "
        "accepted; otherwise solve and update: the MPC condensation of K in CSR format, identity rows for the "
        "Dirichlet DOFs, linSolver(K_, R) gives ddU, the slaves follow the masters, dU += ddU, and the next Newton "
        "iteration computes the entities again. Explicit lane (NED, NEDParallel): the net force R = f_ext - sum of "
        "P_e, without tangents, the MPC slave forces folded into the masters, zero in the Dirichlet rows; the velocity "
        "V += dt M^-1 R per DOF with the lumped mass, no matrix and no linear solve, the prescribed velocity on "
        "Dirichlet DOFs; the position dU = dt V, the slaves follow the masters; then the increment is accepted. "
        "3: acceptIncrement writes U, dU, P, V and A to the node fields (a quasi-static solver writes V = A = 0), and "
        "model.advanceToTime lets the entities accept their state; then the step writes the output, and the next "
        "increment begins."
    )
    W = 1120
    svg = Svg(W, 100, alt)
    svg.rect(16, 12, W - 32, 56, fill=GREY, dash=True)
    svg.text(30, 36, "1.  prepareIncrement():  did the ID or the LM array change?", size=TB, bold=True, color=INK)
    svg.text(
        30,
        56,
        "If so, the solver rebuilds what depends on them before the increment is solved "
        '(section "When the ID and the LM array change").',
        size=TL,
    )

    sx, sw_ = 16, 206  # the discrete system
    bw = 196
    bx = W - 16 - bw  # box 3
    lx0 = sx + sw_ + 34
    lw = bx - 24 - lx0
    nw = (lw - 2 * 14 - 2 * 26) / 3
    nxs = [lx0 + 14 + k * (nw + 26) for k in range(3)]

    # implicit lane
    iy = 88
    ih = 212
    svg.rect(lx0, iy, lw, ih, fill=GREY, dash=True)
    svg.text(
        lx0 + 14, iy + 24, "2.  Implicit: NIST, NID and their variants: Newton's method", size=TB, bold=True, color=INK
    )
    ny = iy + 52
    hI = 36 + 17 * 5
    node(
        svg,
        nxs[0],
        ny,
        nw,
        "residual, tangent",
        ["K_e into K[el]", "R = f_ext − ΣP_e", "MPC: R ← TᵀR"],
        mono=False,
        h=hI,
    )
    node(
        svg,
        nxs[1],
        ny,
        nw,
        "converged?",
        [
            "Dirichlet rows of R:",
            "1st iteration: prescribed",
            "increment; later 0.",
            "Later iterations:",
            "R and ddU small?",
        ],
        mono=False,
        h=hI,
    )
    node(
        svg,
        nxs[2],
        ny,
        nw,
        "solve, update",
        [
            "K_ = TᵀKT (MPC), CSR",
            "Dirichlet: identity rows",
            "linSolver(K_, R) → ddU",
            "slaves from masters",
            "dU += ddU",
        ],
        kind="solve",
        mono=False,
        h=hI,
        mlines=(2, 4),
    )
    cyI = ny + hI / 2
    for k in range(2):
        svg.line(f"M{nxs[k] + nw},{cyI} H{nxs[k + 1] - 2}", arrow=True)
    svg.text(nxs[1] + nw + 13, cyI - 8, "no", size=TL, anchor="middle")
    # back to the entities: the next Newton iteration
    yb = iy + ih - 14
    svg.line(f"M{nxs[2] + nw / 2},{ny + hI} V{yb} H{sx + sw_ + 2}", arrow=True)
    svg.text(
        (lx0 + nxs[2] + nw / 2) / 2 + 40,
        yb - 6,
        "next Newton iteration, at U_np = U_n + dU",
        size=TL,
        italic=True,
        anchor="middle",
    )
    # converged: to the acceptance
    yy = ny - 12
    svg.line(f"M{nxs[1] + nw / 2},{ny} V{yy} H{bx - 2}", color=BLUE, sw=1.8, arrow=True)
    svg.text(bx - 12, yy - 6, "yes: done", size=TL, color=BLUE, anchor="end", bold=True)

    # explicit lane
    ey = iy + ih + 22
    eh = 176
    svg.rect(lx0, ey, lw, eh, fill=GREY, dash=True)
    svg.text(
        lx0 + 14,
        ey + 24,
        "2.  Explicit dynamic: NED, NEDParallel: no Newton loop, no linear solve",
        size=TB,
        bold=True,
        color=INK,
    )
    ny2 = ey + 42
    hE = 36 + 17 * 4
    node(
        svg,
        nxs[0],
        ny2,
        nw,
        "net force",
        ["R = f_ext − ΣP_e", "elements: no K_e", "MPC: slave forces folded", "Dirichlet rows: 0"],
        mono=False,
        h=hE,
    )
    node(
        svg,
        nxs[1],
        ny2,
        nw,
        "velocity",
        ["V += Δt M⁻¹R, per DOF:", "M lumped, a diagonal:", "no matrix, no solve", "Dirichlet: prescribed V"],
        kind="solve",
        mono=False,
        h=hE,
    )
    node(
        svg,
        nxs[2],
        ny2,
        nw,
        "position",
        ["dU = Δt V,  U += dU", "slaves follow masters", "Δt below the stable Δt,", "computed per numbering"],
        mono=False,
        h=hE,
    )
    cyE = ny2 + hE / 2
    for k in range(2):
        svg.line(f"M{nxs[k] + nw},{cyE} H{nxs[k + 1] - 2}", arrow=True)
    svg.line(f"M{nxs[2] + nw},{cyE} H{bx - 2}", color=BLUE, sw=1.8, arrow=True)

    # the discrete system
    top, bot = iy, ey + eh
    svg.rect(sx, top, sw_, bot - top, fill=GREY)
    svg.text(sx + 12, top + 24, "the discrete system", size=TB, bold=True, color=INK)
    node(svg, sx + 12, top + 40, sw_ - 24, "system.elements", ["P_e, K_e: internal", "force, tangent"])
    node(svg, sx + 12, top + 124, sw_ - 24, "system.constraints", ["P_c, K_c: their force,", "tangent"])
    for k, s in enumerate(
        [
            "locations[el], the LM",
            "array of el, places each",
            "block in the global system.",
            "",
            "Loads are step actions:",
            "they add −f_ext at their",
            "DOFs, not entities.",
        ]
    ):
        svg.text(sx + 12, top + 228 + 17 * k, s, size=TL, color=MUTED)
    svg.line(f"M{sx + sw_},{cyI} H{nxs[0] - 2}", arrow=True)
    svg.line(f"M{sx + sw_},{cyE} H{nxs[0] - 2}", arrow=True)

    # 3. the accepted increment
    svg.rect(bx, top, bw, bot - top, fill="#ffffff", stroke=INK, sw=1.4)
    svg.text(bx + 12, top + 24, "3.  acceptIncrement()", size=TB, bold=True, color=INK)
    acc = [
        ("writeResultsToModel():", True),
        ("U, dU, P, V, A to the", False),
        ("node fields; quasi-static", False),
        ("solvers write V = A = 0", False),
        ("", False),
        ("model.advanceToTime():", True),
        ("the entities accept their", False),
        ("state (history variables)", False),
        ("", False),
        ("then the step writes the", False),
        ("output, and the next", False),
        ("increment begins", False),
    ]
    for k, (s, m) in enumerate(acc):
        svg.text(bx + 12, top + 50 + 18 * k, s, size=TL, mono=m, color=INK if m else MUTED)
    # the next increment
    yn = bot + 22
    svg.line(f"M{bx + bw / 2},{bot} V{yn} H{sx + sw_ / 2} V{bot + 2}", arrow=True)
    svg.text((sx + bx) / 2 + 60, yn - 6, "next increment", size=TL, italic=True, anchor="middle")
    svg.h = yn + 14
    svg.write(path)


LIFETIME = {
    "job": ("#e7eaee", "#6e7781", False, "per job"),
    "step": ("#f6e7c1", "#9a6700", False, "per step"),
    "numbering": ("#a8d1fb", "#0969da", False, "per numbering"),
    "locations": ("#dbeefe", "#3d8fe0", False, "per locations"),
    "increment": ("#fde0d6", "#c4432b", False, "per increment"),
    "iteration": ("#ffffff", "#8c959f", True, "per Newton iteration"),
}
CREATE = "#1a7f37"


def combined(path):
    alt = (
        "The objects of a run with the implicit static solver NIST. Nesting shows which object holds which; the fill "
        "shows how long an object lives: per job, per step, per numbering, per locations, per increment or per Newton "
        "iteration. The driver, finiteElementSimulation, holds the StepManager with the step actions, the current Step "
        "with its TimeStepper, the FEModel with nodes, node fields, elements and constraints, the output managers, the "
        "journal, and the solver. The solver holds its options (per job), its convergence criterion and its "
        "LinearSolver (per step), the multi-point-constraint condensation (per numbering), the SystemMatrixAssembly "
        "and the vectors U, dU, P (per locations, the values carried over), and its DiscreteSystem (per locations), "
        "which holds the DofLocations (per locations), which hold the DofNumbering (per numbering): the numbering "
        "outlives its holders. The LinearSolver holds the DOF layout of the model (per numbering) and its "
        "factorization (by default per iteration). The local variables of solveIncrement are R and U_np (per "
        "increment) and the condensed matrix K_ and the correction ddU (per iteration). Green: the creator, where it "
        "is not the holder: the StepManager creates each Step (generateSteps), the FEModel creates the DiscreteSystem "
        "(buildDiscreteSystem), the DiscreteSystem creates its successor (withNewLocations), and the DofLocations "
        "create the vectors (constructDofVector)."
    )
    W = 1060
    svg = Svg(W, 0, alt)
    T, N = 13, 11.5  # title, notes

    def box(x, y, w, h, life, title, attr=None, notes=(), sw=1.6, size=T):
        fill, stroke, dash, _ = LIFETIME[life] if life else ("none", "#8c959f", True, None)
        svg.rect(x, y, w, h, fill=fill, stroke=stroke, dash=dash, sw=sw, rx=7)
        ty = y + 21
        if title:
            svg.text(x + 12, ty, title, size=size, bold=True, mono=True, color=INK)
        if attr:
            ty += 17
            svg.text(x + 12, ty, attr, size=N, mono=True, color=MUTED)
        for n in notes:
            ty += 16
            green = n.endswith("()") or n.startswith("withNewLocations")
            svg.text(x + 12, ty, n, size=N, mono=green, color=CREATE if green else INK)
        return ty

    def inner(x, y, w, h, life, lines):
        """An untitled part of an object: the first line names it, the second (grey, mono) the method."""
        fill, stroke, dash, _ = LIFETIME[life]
        svg.rect(x, y, w, h, fill=fill, stroke=stroke, dash=dash, sw=1.6, rx=7)
        svg.text(x + 12, y + 20, lines[0], size=N, color=INK)
        if len(lines) > 1:
            svg.text(x + 12, y + 37, lines[1], size=N, mono=True, color=MUTED)

    arrows = []  # drawn last, on top of the boxes they cross

    def arrow(d):
        arrows.append(d)

    # legend
    svg.text(
        10,
        18,
        "Nesting: who holds whom.   Fill: how long it lives.   Green: who creates it, if not the holder.",
        size=12,
        color=INK,
    )
    pitch = (W - 20) / 6
    for i, life in enumerate(LIFETIME):
        fill, stroke, dash, label = LIFETIME[life]
        x = 10 + i * pitch
        svg.rect(x, 33, 24, 16, fill=fill, stroke=stroke, dash=dash, sw=1.6, rx=4)
        svg.text(x + 32, 46, label, size=N, color=INK)
    svg.text(
        10,
        74,
        "Change.NUMBERING and .MESH renew both blues, Change.LOCATIONS the light blue.   "
        "Grey monospace: the attribute in the code.",
        size=N,
        color=MUTED,
    )

    PAD, GAP = 16, 26  # inset of a box around what it holds; vertical gap between boxes
    LGAP, HGAP = 76, 64  # horizontal gaps the green arrows cross: left column to solver, between solver columns
    y0 = 92
    dx, dw = 10, W - 20
    lx, lw_ = dx + PAD, 270
    sx = lx + lw_ + LGAP
    sw_ = dx + dw - PAD - sx
    ix, iw = sx + PAD, (sw_ - 2 * PAD - HGAP) / 2
    rx, rw = ix + iw + HGAP, iw
    sy = y0 + 56

    yA = sy + 70
    yB = yA + 58 + GAP
    dy = yB + 180 + GAP
    hD = 286
    solverBottom = dy + hD + PAD
    driverBottom = solverBottom + PAD

    box(dx, y0, dw, driverBottom - y0, "job", "finiteElementSimulation", notes=["the driver"], size=14)

    # left column
    box(lx, sy, lw_, 112, "job", "StepManager", "stepManager")
    inner(lx + PAD, sy + 50, lw_ - 2 * PAD, 34, "job", ["step actions, shared with the solver"])
    svg.text(lx + 12, sy + 102, "generateSteps()", size=N, mono=True, color=CREATE)
    arrow(f"M{lx + 70},{sy + 108} V{yB - 2}")
    yT = yB
    box(lx, yT, lw_, 102, "step", "Step", "step")
    svg.rect(lx + PAD, yT + 52, lw_ - 2 * PAD, 34, fill=LIFETIME["step"][0], stroke=LIFETIME["step"][1], sw=1.6, rx=7)
    svg.text(lx + PAD + 12, yT + 74, "TimeStepper", size=T, bold=True, mono=True, color=INK)
    yM = yT + 102 + GAP
    hM = 142
    yb = box(
        lx,
        yM,
        lw_,
        hM,
        "job",
        "FEModel",
        "model",
        notes=[
            "nodes, node fields, elements,",
            "constraints, and their state;",
            "refers to the solvers and the",
            "field output controller",
            "buildDiscreteSystem()",
        ],
    )
    arrow(f"M{lx + 180},{yb - 4} H{ix - 2}")
    yO = yM + hM + GAP
    box(lx, yO, lw_, 34, "job", "output managers")
    box(lx, yO + 34 + 14, lw_, 34, "job", "journal")

    # the solver
    box(
        sx,
        sy,
        sw_,
        solverBottom - sy,
        "job",
        "NIST",
        "solvers[name]",
        notes=["the solver; one per *solver line"],
        sw=2.2,
        size=14,
    )
    box(ix, yA, 96, 58, "job", "options")
    box(ix + 96 + 14, yA, iw - 96 - 14, 58, "step", "convergence")
    svg.text(ix + 96 + 26, yA + 38, "criterion", size=T, bold=True, mono=True, color=INK)
    box(rx, yA, rw, 58, "numbering", "mpcTransformation", notes=["the MPC condensation"])

    box(ix, yB, iw, 180, "step", "LinearSolver", "self.linSolver")
    inner(ix + PAD, yB + 54, iw - 2 * PAD, 50, "numbering", ["DOF layout of the model", "setModel()"])
    inner(ix + PAD, yB + 114, iw - 2 * PAD, 50, "iteration", ["factorization: by default", ""])
    svg.text(ix + PAD + 12, yB + 151, "built in every solve", size=N, color=INK)

    box(rx, yB, rw, 80, "locations", "SystemMatrixAssembly", "self._K", notes=["new values in every iteration"])

    # the local variables of the Newton loop: held by no object
    yF, hF = yB + 80 + GAP, 194
    svg.rect(rx, yF, rw, hF, fill="none", stroke="#8c959f", dash=True, sw=1.4, rx=7)
    svg.text(rx + 12, yF + 21, "solveIncrement()", size=T, bold=True, mono=True, color=INK)
    svg.text(rx + 12, yF + 38, "local variables, held by no object", size=N, color=MUTED)
    box(rx + PAD, yF + 52, rw - 2 * PAD, 52, "increment", "R, U_np", notes=["residual; U_np = U + dU"])
    box(
        rx + PAD,
        yF + 116,
        rw - 2 * PAD,
        hF - 116 - PAD,
        "iteration",
        "K_, ddU",
        notes=["condensed matrix (CSR);", "Newton correction"],
    )

    # the discrete system: the numbering outlives its holders
    box(ix, dy, iw, hD, "locations", "DiscreteSystem", "self.system")
    svg.text(ix + 12, dy + 56, "withNewLocations()", size=N, mono=True, color=CREATE)
    svg.text(ix + 12, dy + 72, "on Change.LOCATIONS, creates the next", size=N, color=CREATE)
    svg.text(ix + 12, dy + 88, "DiscreteSystem: new DofLocations,", size=N, color=CREATE)
    svg.text(ix + 12, dy + 104, "the same DofNumbering", size=N, color=CREATE)
    yL = dy + 118
    hL = hD - 118 - PAD
    box(
        ix + PAD, yL, iw - 2 * PAD, hL, "locations", "DofLocations", ".locations   (LM)", notes=["constructDofVector()"]
    )
    box(
        ix + 2 * PAD,
        yL + 68,
        iw - 4 * PAD,
        hL - 68 - PAD,
        "numbering",
        "DofNumbering",
        ".numbering   (ID)",
        notes=["outlives its holders"],
    )

    # the vectors, beside the DofLocations that create them
    yc = yL + 51
    yV = yc - 40
    box(rx, yV, rw, 80, "locations", "U, dU, P", "self._U, self._dU, self._P", notes=["values carried over"])
    arrow(f"M{ix + 2 * PAD + 150},{yc} H{rx - 2}")

    for d in arrows:
        svg.line(d, color=CREATE, sw=2.2, arrow=True)

    svg.h = driverBottom + 10
    svg.write(path)


if __name__ == "__main__":
    out = (
        sys.argv[1] if len(sys.argv) > 1 else str(pathlib.Path(__file__).resolve().parents[1] / "source" / "developer")
    )
    example(out + "/architecture_example.svg")
    combined(out + "/architecture_objects.svg")
    change(out + "/discretesystem_change.svg")
    increment(out + "/discretesystem_increment.svg")
