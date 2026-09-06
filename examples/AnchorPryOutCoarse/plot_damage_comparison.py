#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare the four pry-out variants: damage contours at the common final displacement.

Run each of the four decks in its own directory, then::

    python plot_damage_comparison.py [--runs DIR ...] [--out FIGURE.png]

All four ramp the fixture plate to the same 1.5 mm, which is what makes the comparison meaningful --
implicit against explicit at equal load rather than at equal time, and NTS against GPTS contact on
an otherwise identical model.

Two panels per variant:

* **damage on the symmetry plane** at the last exported frame, as filled contours. The concrete is
  the block carrying the element-wise ``damage`` field -- note that the Ensight export contains one
  part per node set, 73 in all, and the concrete is *not* the largest of them by cell count.
* **load-displacement**, all four overlaid, so the contours have something quantitative beside them.

For the explicit runs it also reports the peak kinetic energy as a fraction of the external work
integral(RF dU). That ratio, not the fact that the run completed, is what says whether a mass-scaled
explicit run was quasi-static enough to be read as a static result.
"""

import argparse
import os

import matplotlib
import numpy as np
import pyvista as pv

matplotlib.use("Agg")  # noqa: E402  headless: chosen before pyplot binds a backend

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.tri import Triangulation  # noqa: E402

#: The four variants, in the order they are laid out in the figure.
VARIANTS = [
    ("implicit_nts", "implicit  ·  NTS"),
    ("implicit_gpts", "implicit  ·  GPTS"),
    ("explicit_nts", "explicit  ·  NTS"),
    ("explicit_gpts", "explicit  ·  GPTS"),
]


def concreteBlock(multiblock):
    """The block carrying the element-wise damage field.

    Deliberately not "the largest block": the Ensight writer emits a part for every node set, and
    several of those are larger than the concrete.
    """

    candidates = [
        multiblock[i]
        for i in range(len(multiblock))
        if multiblock[i] is not None and "damage" in getattr(multiblock[i], "cell_data", {})
    ]
    if not candidates:
        raise LookupError("no block carries a 'damage' field -- did the run export it?")
    return max(candidates, key=lambda b: b.n_cells)


def damageOnSymmetryPlane(caseFile, z=-1e-3):
    """Final-frame damage as (triangulation, nodal values, cell count)."""

    reader = pv.get_reader(caseFile)
    reader.set_active_time_point(len(reader.time_values) - 1)
    blk = concreteBlock(reader.read()).cell_data_to_point_data()

    sl = blk.slice(normal="z", origin=(0.0, 0.0, z)).triangulate()
    if sl.n_points == 0:
        raise ValueError("empty slice -- is the model still the z <= 0 half specimen?")
    pts = sl.points
    tri = Triangulation(pts[:, 0], pts[:, 1], sl.faces.reshape(-1, 4)[:, 1:])
    return tri, np.asarray(sl.point_data["damage"]), blk.n_cells


def history(runDir):
    """Applied displacement, reaction and peak damage against time."""

    def col(name):
        return np.loadtxt(os.path.join(runDir, name + ".csv"))

    u, rf, dmg = col("U_loading"), col("RF_loading"), col("maxDamage")
    return u[:, 0], u[:, 1], rf[:, 1], dmg[:, 1]


def quasiStaticity(runDir):
    """Peak kinetic energy as a fraction of the external work, or None if not reported.

    The external work is the integral of the reaction through the applied displacement, which needs
    no cooperation from the material -- GCDP never populates a strain energy, so the solver's own
    internal/kinetic split reads zero on the internal side and cannot answer this.
    """

    logFile = os.path.join(runDir, "run.log")
    if not os.path.exists(logFile):
        return None
    kinetic = []
    for line in open(logFile, errors="ignore"):
        if "kinetic" in line:
            for token in line.replace("|", " ").split():
                try:
                    kinetic.append(abs(float(token)))
                    break
                except ValueError:
                    continue
    if not kinetic:
        return None
    _, u, rf, _ = history(runDir)
    work = abs(np.trapezoid(rf, u)) if hasattr(np, "trapezoid") else abs(np.trapz(rf, u))
    return max(kinetic) / work if work > 0 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--runs",
        nargs="*",
        default=[v for v, _ in VARIANTS],
        help="the %d run directories, in the order %s" % (len(VARIANTS), ", ".join(v for v, _ in VARIANTS)),
    )
    parser.add_argument("--out", default="damage_comparison.png")
    args = parser.parse_args()

    # zip() would silently truncate to the shorter of the two and leave the rest of the panels
    # looking up keys that are not there, so say it here rather than raising a KeyError further in.
    if len(args.runs) != len(VARIANTS):
        parser.error(
            "--runs takes exactly %d directories, one per variant, in the order %s (got %d)"
            % (len(VARIANTS), ", ".join(v for v, _ in VARIANTS), len(args.runs))
        )

    dirs = dict(zip([v for v, _ in VARIANTS], args.runs))
    labels = dict(VARIANTS)

    fig = plt.figure(figsize=(15.5, 7.6), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.92])
    levels = np.linspace(0.0, 1.0, 21)
    contour = None

    for k, (key, _) in enumerate(VARIANTS):
        ax = fig.add_subplot(grid[k // 2, k % 2])
        runDir = dirs[key]
        try:
            tri, vals, nCells = damageOnSymmetryPlane(os.path.join(runDir, "esExport.case"))
        except Exception as exc:
            ax.text(
                0.5, 0.5, "%s\n%s" % (labels[key], exc), ha="center", va="center", transform=ax.transAxes, fontsize=9
            )
            ax.set_axis_off()
            continue
        contour = ax.tricontourf(tri, np.clip(vals, 0.0, 1.0), levels=levels, cmap="inferno_r")
        ax.tricontour(tri, np.clip(vals, 0.0, 1.0), levels=[0.05, 0.5, 0.95], colors="k", linewidths=0.4)
        ax.set_aspect("equal")
        ax.set_xlim(-140, 140)
        ax.set_ylim(-125, 20)
        try:
            finalU = history(runDir)[1][-1]
            ax.set_title("%s   ·   U = %.3f mm   ·   %d elements" % (labels[key], finalU, nCells), fontsize=10)
        except OSError:
            ax.set_title("%s   (%d elements)" % (labels[key], nCells), fontsize=10.5)
        ax.set_xlabel("x [mm]", fontsize=9)
        ax.set_ylabel("y [mm]", fontsize=9)

    if contour is not None:
        fig.colorbar(contour, ax=fig.axes[:4], shrink=0.72, label=r"damage $\omega$", pad=0.01)

    ax = fig.add_subplot(grid[:, 2])
    for key, label in VARIANTS:
        try:
            _, u, rf, _ = history(dirs[key])
        except OSError:
            continue
        style = "-" if key.startswith("implicit") else "--"
        ax.plot(u, np.abs(rf) / 1e3, style, lw=1.5, label=label)
    ax.set_xlabel("applied plate displacement U [mm]")
    ax.set_ylabel("reaction |RF| [kN]")
    ax.set_title("load-displacement", fontsize=10.5)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=8.5, frameon=False)

    notes = []
    for key, label in VARIANTS:
        if not key.startswith("explicit"):
            continue
        ratio = quasiStaticity(dirs[key])
        if ratio is not None:
            notes.append("%s: peak KE / external work = %.2e" % (label, ratio))
    if notes:
        ax.text(0.02, -0.13, "\n".join(notes), transform=ax.transAxes, fontsize=8, va="top", color="0.35")

    fig.suptitle(
        "Bonded anchor pry-out — damage at the common final displacement (nominal 1.5 mm)\n"
        "same mesh, materials, ties and live h-adaptivity throughout; only the solver and the "
        "contact formulation differ",
        fontsize=12,
    )
    fig.savefig(args.out, dpi=145)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
