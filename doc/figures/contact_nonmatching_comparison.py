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
"""Regenerate ``doc/source/contact_nonmatching_comparison.png``.

Self-contained: writes the eight models, runs them, and renders the figure. Run it from anywhere:

.. code-block:: bash

    python doc/figures/contact_nonmatching_comparison.py

Requires ``pyvista`` and ``matplotlib``, neither of which EdelweissFE itself depends on -- which is
why this lives here and is run by hand rather than at documentation build time, with the resulting
PNG committed alongside the other figures.

The model: box on box with deliberately NON-MATCHING interface meshes (3x3 against 4x4 elements,
two elements thick), nu = 0 and a uniform imposed compression, so the exact solution is a flat
interface transmitting a uniform normal stress of -360. Every feature the figure shows is therefore
discretization error, and the two rows differ in nothing but the contact constraint.
"""

import os
import pathlib
import shutil
import subprocess
import tempfile

import numpy as np
import pyvista as pv

SIGMA_EXACT = -360.0
SZZ = 2  # Voigt index of S33 in the element's stress vector
WARP = 60.0
#: Half-height of the slab of each block kept around the interface, see renderPanel.
SLAB = 0.30
N_LOWER, N_UPPER = 3, 4

CASES = [
    ("h20-h20", "hexa20 / hexa20", "C3D20", "C3D20"),
    ("h20slave-h8master", "hexa20 slave / hexa8 master", "C3D8", "C3D20"),
    ("h8slave-h20master", "hexa8 slave / hexa20 master", "C3D20", "C3D8"),
    ("h8-h8", "hexa8 / hexa8", "C3D8", "C3D8"),
]
CONSTRAINTS = [
    ("node", "node-to-surface", "nodeToDeformableSurfacePenalty"),
    ("gpts", "integrated (GPTS)", "surfaceToDeformableSurfacePenalty"),
]

TPL = """** Non-matching box-on-box contact, for the documentation figure.
** Lower block {nL}x{nL}x2, upper block {nU}x{nU}x2, so the interface meshes do not match.
** nu = 0 and a uniform imposed compression, so the exact interface state is a flat interface at
** uniform pressure -- every departure visible in the figure is discretization error.
*material, name=LinearElastic, id=le, provider=edelweiss
1.8e4, 0.0

*job, name=vizjob, domain=3d
*solver, solver=NIST, name=theSolver

*modelGenerator, generator=boxGen, name=lower
nX      ={nL}
nY      ={nL}
nZ      =2
x0      =0
y0      =0
z0      =0
lX      =1
lY      =1
lZ      =1
elProvider =edelweiss
elType  ={etL}

*modelGenerator, generator=boxGen, name=upper
nX      ={nU}
nY      ={nU}
nZ      =2
x0      =0
y0      =0
z0      =1.02
lX      =1
lY      =1
lZ      =1
elProvider =edelweiss
elType  ={etU}

*modelGenerator, generator=surfaceElementGenerator, name=gen1
surface = upper_back
name    = slaveSurf
triangulation = midside

*modelGenerator, generator=surfaceElementGenerator, name=gen2
surface = lower_front
name    = masterSurf
triangulation = midside

*section, name=section1, material=le, type=solid
lower_all
upper_all

*constraint, name=contact, type={ctype}
slaveSurface=slaveSurf_facets, masterSurface=masterSurf_facets, penalty=3e7, type=linear,
searchDistance=1.0, sliding=small

*fieldOutput
>>perNode, elSet=lower_all, field=displacement, result=U, name=dispLower
>>perNode, elSet=upper_all, field=displacement, result=U, name=dispUpper
>>perElement, elSet=lower_all, result=stress, name=stressLower, quadraturePoint=0
>>perElement, elSet=upper_all, result=stress, name=stressUpper, quadraturePoint=0
>>perNode, name=RFNormal, nSet=lower_back, field=displacement, result=P, f(x)='sum(x[:,2])', saveHistory=True
>>perNode, name=Uzspread, nSet=slaveSurf_nodes, field=displacement, result=U, f(x)='np.max(x[:,2])-np.min(x[:,2])', saveHistory=True

*output, type=ensight, name=esExport
>>perNode, fieldOutput=dispLower
>>perNode, fieldOutput=dispUpper
>>perElement, fieldOutput=stressLower
>>perElement, fieldOutput=stressUpper

*output, type=monitor, name=monitor
fieldOutput=RFNormal
fieldOutput=Uzspread

*step, solver=theSolver
maxInc=0.25, minInc=1e-6, maxNumInc=100, maxIter=25, stepLength=1
>>options, name=theSolver, extrapolation=off
>>dirichlet, name=fixLower, nSet=lower_back, field=displacement, 1=0.0, 2=0.0, 3=0.0
>>dirichlet, name=pushUpper, nSet=upper_front, field=displacement, 1=0.0, 2=0.0, 3=-0.06
"""


def runCases(workDir: pathlib.Path) -> dict:
    """Write and run every model, returning its run directory by (case, constraint) tag."""

    dirs = {}
    for tag, _, etL, etU in CASES:
        for ctag, _, ctype in CONSTRAINTS:
            d = workDir / f"{tag}_{ctag}"
            d.mkdir(parents=True, exist_ok=True)
            (d / "test.inp").write_text(TPL.format(nL=N_LOWER, nU=N_UPPER, etL=etL, etU=etU, ctype=ctype))
            log = subprocess.run(
                ["edelweissfe", "test.inp"],
                cwd=d,
                capture_output=True,
                text=True,
                env=dict(os.environ, MPLBACKEND="Agg"),
                timeout=1800,
            )
            (d / "run.log").write_text(log.stdout + log.stderr)
            if not list(d.glob("esExport*.case")):
                raise RuntimeError(f"{d.name} produced no Ensight output; see its run.log")
            dirs[(tag, ctag)] = d
    return dirs


def staticCase(runDir: str) -> str:
    """A single-step Ensight case pointing at the last written step.

    EdelweissFE writes two time sets -- one step of geometry, N steps of variables -- and VTK's
    EnSight reader exposes only the geometry set, so reading the transient case silently yields
    step 0, where every displacement is still zero. Rewriting the case as static avoids that.
    """

    d = pathlib.Path(runDir)
    src = sorted(d.glob("esExport*.case"))[-1]
    dataDir = src.stem
    last = max(int(f.suffix.split("_")[-1]) for f in (d / dataDir).glob("dispUpper.var_*"))
    lines = ["FORMAT", "type: ensight gold", "GEOMETRY", f"model: {dataDir}/geometry.geo_0000", "VARIABLE"]
    for name, kind in (
        ("dispLower", "vector per node"),
        ("dispUpper", "vector per node"),
        ("stressLower", "tensor symm per element"),
        ("stressUpper", "tensor symm per element"),
    ):
        lines.append(f"{kind}: {name} {dataDir}/{name}.var_{last:04d}")
    out = d / "final.case"
    out.write_text("\n".join(lines) + "\n")
    return str(out)


def blocks(runDir: str):
    """The two solid blocks and the master interface, in the reference configuration, as context.

    Reference rather than deformed: the interface relief is exaggerated by WARP to be visible at
    all, and deforming the blocks by the same factor would distort them grossly (the imposed
    compression is 6% of a block height). They are drawn translucent, so they read as the setting
    for the interface rather than as a result themselves -- and their edges are what show the two
    interface meshes to be non-matching, which is the whole premise of the comparison.
    """

    m = pv.read(staticCase(runDir))
    lower, upper = m["lower_all"], m["upper_all"]

    # the lower block's top face: the master interface, whose mesh does not match the slave's
    surf = lower.extract_surface(algorithm="dataset_surface")
    keep = np.flatnonzero(np.abs(surf.points[:, 2] - surf.points[:, 2].max()) < 1e-9)
    masterFace = surf.extract_points(keep, adjacent_cells=False).extract_surface(algorithm="dataset_surface")
    return lower, upper, masterFace


def interfaceSurface(runDir: str):
    """The slave interface, carrying the z-displacement deviation from the interface mean as point
    data 'dUz' and the transmitted normal stress as cell data 'szz'."""

    up = pv.read(staticCase(runDir))["upper_all"]
    up.point_data["uz"] = np.asarray(up.point_data["dispUpper"])[:, 2]
    up.cell_data["szz"] = np.asarray(up.cell_data["stressUpper"])[:, SZZ]

    surf = up.extract_surface(algorithm="dataset_surface")
    keep = np.flatnonzero(np.abs(surf.points[:, 2] - surf.points[:, 2].min()) < 1e-9)
    iface = surf.extract_points(keep, adjacent_cells=False).extract_surface(algorithm="dataset_surface")
    d = np.asarray(iface.point_data["uz"])
    iface.point_data["dUz"] = d - d.mean()
    return iface


def renderPanel(surface, context, clim) -> np.ndarray:
    """Render one interface, in the setting of its two blocks, to an RGB array.

    Framing is fixed and identical for every panel, and deliberately NOT cropped to content: a
    flat interface crops to a thin sliver while an undulating one fills the frame, which would
    silently rescale the panels against each other and destroy the comparison the figure exists to
    make.

    The blocks are drawn translucent and *without* their element edges -- a hexa20's faces
    triangulate into a mesh of diagonals that buries the figure in noise -- so their outlines come
    from feature edges instead. The master interface is added as a wireframe, because otherwise
    only the slave mesh would be visible and the non-matching discretization, which is the entire
    premise, would be invisible.

    Composed with matplotlib rather than laid out by pyvista because a pyvista subplot grid gives
    no control over where the colour bar and the annotations land, and they collided.
    """

    lower, upper, masterFace = context

    # Only a slab of each block is drawn. Whole blocks put the interface -- the entire subject of
    # the figure -- into about a sixth of the frame, and looking at the relief through a full unit
    # cube of translucent material washes its colour out. A slab still shows two distinct bodies
    # meeting, which is what was missing when only the interface was drawn.
    zInterface = float(masterFace.points[:, 2].max())
    lower = lower.clip("z", origin=(0.0, 0.0, zInterface - SLAB), invert=False)
    upper = upper.clip("-z", origin=(0.0, 0.0, zInterface + SLAB), invert=False)

    p = pv.Plotter(off_screen=True, window_size=(820, 560), border=False)
    p.enable_depth_peeling(number_of_peels=10, occlusion_ratio=0.0)
    for block, opacity in ((lower, 0.30), (upper, 0.22)):
        p.add_mesh(block, color="#c4ced6", opacity=opacity, show_edges=False, lighting=False)
        p.add_mesh(block.extract_feature_edges(), color="#5d6a74", line_width=1)
    p.add_mesh(masterFace, style="wireframe", color="#8a6a2f", line_width=2)
    p.add_mesh(
        surface.warp_by_scalar("dUz", factor=WARP),
        scalars="szz",
        cmap="RdBu_r",
        clim=clim,
        show_edges=True,
        edge_color="#2a2a2a",
        line_width=1,
        show_scalar_bar=False,
    )
    p.camera_position = [(2.55, -2.30, 1.78), (0.5, 0.5, zInterface), (0, 0, 1)]
    p.camera.zoom(1.42)
    p.set_background("white")
    img = p.screenshot(return_img=True)
    p.close()
    return img


def main(outPath: str, runDirs: dict):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors

    surfaces, contexts, amps = {}, {}, {}
    for (tag, ctag), d in runDirs.items():
        s = interfaceSurface(str(d))
        surfaces[(tag, ctag)] = s
        contexts[(tag, ctag)] = blocks(str(d))
        amps[(tag, ctag)] = float(np.abs(s.point_data["dUz"]).max())

    spread = max(float(np.abs(np.asarray(s.cell_data["szz"]) - SIGMA_EXACT).max()) for s in surfaces.values())
    clim = (SIGMA_EXACT - spread, SIGMA_EXACT + spread)

    fig = plt.figure(figsize=(15.0, 7.2), dpi=150)
    gs = fig.add_gridspec(2, 4, left=0.055, right=0.985, top=0.855, bottom=0.20, wspace=0.02, hspace=0.34)

    for j, (tag, title, _, _) in enumerate(CASES):
        for i, (ctag, rowName, _) in enumerate(CONSTRAINTS):
            ax = fig.add_subplot(gs[i, j])
            ax.imshow(renderPanel(surfaces[(tag, ctag)], contexts[(tag, ctag)], clim))
            ax.set_axis_off()
            if i == 0:
                ax.set_title(title, fontsize=12, pad=10)
            gain = amps[(tag, "node")] / amps[(tag, ctag)]
            note = f"max |$\\Delta u_z$| = {amps[(tag, ctag)]:.2e}"
            if ctag == "gpts":
                note += f"   ({gain:.0f}$\\times$ flatter)"
            ax.text(0.5, -0.04, note, transform=ax.transAxes, ha="center", va="top", fontsize=10)
            if j == 0:
                ax.text(
                    -0.06,
                    0.5,
                    rowName,
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=13,
                    fontweight="bold",
                )

    cax = fig.add_axes([0.32, 0.075, 0.36, 0.028])
    cb = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(*clim), cmap="RdBu_r"), cax=cax, orientation="horizontal")
    cb.set_label("transmitted normal stress  $S_{33}$    (exact: $-360$)", fontsize=11)
    cb.ax.tick_params(labelsize=9)

    fig.suptitle(
        "Non-matching box-on-box contact: the exact interface is flat at uniform $S_{33}=-360$, "
        f"so all relief is discretization error (warped $\\times${WARP:.0f})",
        fontsize=13,
        y=0.985,
    )
    fig.savefig(outPath, facecolor="white")
    print(f"wrote {outPath}")
    print(f"  S33 range: {clim[0]:.1f} .. {clim[1]:.1f}")
    for k, v in sorted(amps.items(), key=lambda kv: -kv[1]):
        print(f"  {k[0]:<20} {k[1]:<5} max|du_z| = {v:.3e}")


if __name__ == "__main__":
    target = pathlib.Path(__file__).resolve().parents[1] / "source" / "contact_nonmatching_comparison.png"
    work = pathlib.Path(tempfile.mkdtemp(prefix="edelweiss-contact-figure-"))
    try:
        main(str(target), runCases(work))
    finally:
        shutil.rmtree(work, ignore_errors=True)
