"""Figures for the GCDP fracture-energy calibration.

Reads what the runs of ``run_calibration.py`` left behind -- the exported load-displacement
histories and the Ensight export of the damage fields -- and writes the figures for the write-up
of this calibration study.

The Ensight export is read directly rather than through VTK. It is Ensight Gold *binary*, written
by ``edelweissfe/outputmanagers/ensight.py``, and its layout is fixed and small: 80-byte character
records, 4-byte integers, 4-byte floats, with node and element labels present because that writer
defaults to ``node id given`` / ``element id given``. A per-element scalar of a 40-element mesh is
404 bytes, which is exactly 80 + 80 + 4 + 80 + 40*4 -- the arithmetic is the confirmation that this
parser reads what that writer wrote.

Usage
-----
    python plot_results.py                  # every run present, into figures/
    python plot_results.py --figures DIR
"""

import argparse
import glob
import os
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

#: Cross-sectional area of the bar, for the fracture energy.
CROSS_SECTION = 25.0

#: Total prescribed elongation.
TOTAL_ELONGATION = 0.2

#: Duration over which it is applied, the same for every variant. Needed here only to convert an
#: elongation back into a time when counting wave transits.
EXPLICIT_STEP_LENGTH = 1.0

#: Length of the bar, for the position axis of the contour plots.
BAR_LENGTH = 100.0

#: The fracture energy of examples/AlphaP_AMR_Study, the independent reference this study is
#: measured against.
REFERENCE_GF = 0.098

#: The variants, in reporting order, with a colour and a line style each so that every figure
#: identifies them the same way.
VARIANT_STYLE = {
    "implicit": ("#000000", "-", 2.0),
    # the parabolic curves are drawn wide and the hyperbolic ones narrow on top of them: the two
    # schemes agree so closely that a single width hides one of them completely, and a figure that
    # looks like it is missing a curve is worse than one that shows the agreement
    "explicit-parabolic": ("#1f77b4", "-", 3.4),
    "explicit-hyperbolic": ("#d62728", "-", 1.4),
    "explicit-parabolic-bv": ("#1f77b4", "--", 3.4),
    "explicit-hyperbolic-bv": ("#d62728", "--", 1.4),
}

MESHES = (20, 40, 80, 160)


# ----------------------------------------------------------------------------------------------
# Ensight Gold binary reader
# ----------------------------------------------------------------------------------------------


def readC80(handle):
    """Read one 80-byte character record."""

    return handle.read(80).decode("utf-8", "replace").rstrip("\x00").strip()


def readInts(handle, count):
    """Read count 4-byte integers."""

    return np.fromfile(handle, dtype=np.int32, count=count)


def readFloats(handle, count):
    """Read count 4-byte floats."""

    return np.fromfile(handle, dtype=np.float32, count=count)


def readGeometry(path):
    """Read one Ensight Gold binary geometry file.

    Returns the element centroids along the bar axis, in the order the per-element variables are
    written in, which is the order the elements appear here.
    """

    with open(path, "rb") as handle:
        first = readC80(handle)

        # the geometry file opens with a "C Binary" marker record, the variable files do not --
        # which is why a per-element scalar comes out at exactly 80 + 80 + 4 + 80 + nElements*4
        if not first.startswith("C Binary"):
            raise ValueError("{:} is not an Ensight Gold binary geometry (opens with {!r})".format(path, first))

        readC80(handle)  # description line 1
        readC80(handle)  # description line 2
        nodeIdOption = readC80(handle)
        elementIdOption = readC80(handle)

        readC80(handle)  # "part"
        readInts(handle, 1)  # part number
        readC80(handle)  # part description
        readC80(handle)  # "coordinates"
        nNodes = int(readInts(handle, 1)[0])

        if nodeIdOption.endswith("given") or nodeIdOption.endswith("ignore"):
            readInts(handle, nNodes)

        # written as the transpose of an (nNodes, 3) array: all x, then all y, then all z
        coordinates = readFloats(handle, 3 * nNodes).reshape(3, nNodes).T

        elementType = readC80(handle)
        nElements = int(readInts(handle, 1)[0])

        if elementIdOption.endswith("given") or elementIdOption.endswith("ignore"):
            readInts(handle, nElements)

        nodesPerElement = {"hexa8": 8, "hexa20": 20, "quad4": 4, "quad8": 8}[elementType]
        connectivity = readInts(handle, nElements * nodesPerElement).reshape(nElements, nodesPerElement) - 1

    # the corner nodes come first in both node orderings, and for a regular brick their mean is the
    # centre; taking all 20 of a serendipity element would weight the mid-side nodes twice
    corners = connectivity[:, :8]

    return coordinates[corners, 0].mean(axis=1), nElements


def readElementScalar(path, nElements):
    """Read one per-element scalar of one time state."""

    with open(path, "rb") as handle:
        readC80(handle)  # variable description
        readC80(handle)  # "part"
        readInts(handle, 1)  # part number
        readC80(handle)  # element type
        values = readFloats(handle, nElements)

    return values


def readCaseTimes(path):
    """Read the time values of the variable time set (set 2) out of a .case file."""

    with open(path) as handle:
        text = handle.read()

    # the variable trend is time set 2; the block runs until the next time set or the next section
    match = re.search(r"time set:\s*2\b(.*?)(?=\ntime set:|\Z)", text, re.S)
    if match is None:
        return None

    block = match.group(1)
    steps = int(re.search(r"number of steps:\s*(\d+)", block).group(1))

    # the time values run to the end of the block, but the next *section* keyword (GEOMETRY,
    # VARIABLE) can follow them, so read floats until something is not one
    times = []
    for token in re.search(r"time values:(.*)", block, re.S).group(1).split():
        try:
            times.append(float(token))
        except ValueError:
            break

    return np.array(times[:steps])


# ----------------------------------------------------------------------------------------------
# runs
# ----------------------------------------------------------------------------------------------


def runDir(name):
    """The directory of one run, or None if it is not there."""

    path = os.path.join(HERE, name)

    return path if os.path.isdir(path) else None


def reachedEndpoint(path, elongation):
    """Whether a run got to the prescribed elongation, the same test the table's status uses."""

    logPath = os.path.join(path, "run.log")
    log = open(logPath).read() if os.path.exists(logPath) else ""

    if "Simulation failed" in log or "THE SOLUTION HAS DIVERGED" in log:
        return False

    return bool(elongation.size) and elongation[-1] >= 0.999 * TOTAL_ELONGATION


def crackSite(path, threshold=0.1):
    """Where the crack ended up: position of maximum damage, and the width of the damaged band."""

    trend = readFieldTrend(path, "omega")

    if trend is None:
        return None

    centroids, _, values = trend
    finite = np.isfinite(values).all(axis=1)

    if not finite.any():
        return None

    last = values[finite][-1]
    spacing = BAR_LENGTH / centroids.size

    return float(centroids[np.argmax(last)]), float((last > threshold).sum() * spacing)


def readHistory(path):
    """The load-displacement history of one run: elongation [mm], force [N], time [s]."""

    rfPath = os.path.join(path, "RF.csv")
    uPath = os.path.join(path, "U.csv")

    if not (os.path.exists(rfPath) and os.path.exists(uPath)):
        return None

    reaction = np.atleast_2d(np.loadtxt(rfPath))
    displacement = np.atleast_2d(np.loadtxt(uPath))

    if reaction.shape[0] < 2:
        return None

    # The two exports are written by the same finalizeIncrement, but a run that is still going --
    # or one killed between the two writes -- leaves them one row apart, which broadcasts into a
    # ValueError inside the trapezoidal rule rather than into anything legible.
    rows = min(reaction.shape[0], displacement.shape[0])

    return displacement[:rows, 1], np.abs(reaction[:rows, 1]), displacement[:rows, 0]


def fractureEnergy(elongation, force):
    """Dissipated work per unit fracture area, the quantity the study calibrates."""

    return float(np.trapezoid(force, elongation) / CROSS_SECTION)


def readFieldTrend(path, variable):
    """The full space-time history of one per-element field of one run.

    Returns (element centroids, elongations, values[time, element]).
    """

    exportDir = os.path.join(path, "esExport")
    casePath = os.path.join(path, "esExport.case")

    geometries = sorted(glob.glob(os.path.join(exportDir, "geometry.geo_*")))
    frames = sorted(glob.glob(os.path.join(exportDir, "{:}.var_*".format(variable))))

    if not geometries or not frames:
        return None

    centroids, nElements = readGeometry(geometries[0])
    values = np.array([readElementScalar(frame, nElements) for frame in frames])

    times = readCaseTimes(casePath)
    history = readHistory(path)

    if times is None or history is None or times.size < values.shape[0]:
        return None

    # the frames and the exported history are both written by finalizeIncrement, but the mapping
    # between them is established through the time axis rather than assumed to be index-for-index
    elongationOfTime = np.interp(times[: values.shape[0]], history[2], history[0])

    order = np.argsort(centroids)

    return centroids[order], elongationOfTime, values[:, order]


# ----------------------------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------------------------


def figureLoadDisplacement(figures, zoom=False):
    """Load-displacement curves, one panel per mesh, every variant overlaid."""

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)

    for axis, nX in zip(axes.flat, MESHES):
        for variant, (colour, style, width) in VARIANT_STYLE.items():
            path = runDir("run_{:}_nX{:}".format(variant, nX))
            history = readHistory(path) if path else None

            if history is None:
                continue

            elongation, force, _ = history
            axis.plot(elongation, force, color=colour, linestyle=style, linewidth=width, label=variant)

        axis.set_title("nX = {:}  (h = {:.3f} mm)".format(nX, BAR_LENGTH / nX))
        axis.grid(alpha=0.3)

        if zoom:
            axis.set_xlim(0.0, 0.02)
        else:
            axis.set_xlim(0.0, TOTAL_ELONGATION)
            axis.set_ylim(bottom=0.0)

    for axis in axes[1, :]:
        axis.set_xlabel("elongation [mm]")
    for axis in axes[:, 0]:
        axis.set_ylabel("reaction force [N]")

    axes[0, 0].legend(fontsize=8, loc="upper right")
    fig.suptitle(
        "Load-displacement, {:}".format("peak region" if zoom else "full history"),
        fontsize=13,
    )
    fig.tight_layout()

    name = "ld_peak_zoom.png" if zoom else "ld_by_mesh.png"
    fig.savefig(os.path.join(figures, name), dpi=130)
    plt.close(fig)

    return name


def figureMeshConvergence(figures, variants=("implicit", "explicit-hyperbolic", "explicit-hyperbolic-bv")):
    """One panel per variant, the meshes overlaid: which variant converges and which does not."""

    fig, axes = plt.subplots(1, len(variants), figsize=(5 * len(variants), 4.2), sharey=True)
    shades = plt.cm.viridis(np.linspace(0.1, 0.85, len(MESHES)))

    for axis, variant in zip(np.atleast_1d(axes), variants):
        for shade, nX in zip(shades, MESHES):
            path = runDir("run_{:}_nX{:}".format(variant, nX))
            history = readHistory(path) if path else None

            if history is None:
                continue

            elongation, force, _ = history
            axis.plot(
                elongation,
                force,
                color=shade,
                linewidth=1.4,
                label="nX = {:} ({:.5f})".format(nX, fractureEnergy(elongation, force)),
            )

        axis.set_title(variant)
        axis.set_xlabel("elongation [mm]")
        axis.set_xlim(0.0, TOTAL_ELONGATION)
        axis.set_ylim(bottom=0.0)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8, title="G_f [N/mm]", title_fontsize=8)

    np.atleast_1d(axes)[0].set_ylabel("reaction force [N]")
    fig.suptitle("Mesh convergence of the load-displacement response", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(figures, "ld_mesh_convergence.png"), dpi=130)
    plt.close(fig)

    return "ld_mesh_convergence.png"


def figureFractureEnergy(figures):
    """G_f against element size, with the independent reference."""

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    for variant, (colour, style, width) in VARIANT_STYLE.items():
        sizes, energies, peaks, complete = [], [], [], []

        for nX in MESHES:
            path = runDir("run_{:}_nX{:}".format(variant, nX))
            history = readHistory(path) if path else None

            if history is None:
                continue

            elongation, force, _ = history

            if not np.isfinite(force).all() or force.max() <= 0.0:
                continue

            sizes.append(BAR_LENGTH / nX)
            energies.append(fractureEnergy(elongation, force))
            peaks.append(force.max())
            complete.append(reachedEndpoint(path, elongation))

        if not sizes:
            continue

        axes[0].plot(sizes, energies, color=colour, linestyle=style, linewidth=width, label=variant)
        axes[1].plot(sizes, peaks, color=colour, linestyle=style, linewidth=width, label=variant)

        # a hollow marker is a run that stopped short of the endpoint: its G_f is an integral over
        # part of the history, so the point is on the figure but is not comparable
        for axis, values in zip(axes, (energies, peaks)):
            for size, value, finished in zip(sizes, values, complete):
                axis.plot(
                    size,
                    value,
                    color=colour,
                    marker="o",
                    markersize=6,
                    markerfacecolor=colour if finished else "white",
                    markeredgecolor=colour,
                )

    axes[0].axhline(REFERENCE_GF, color="grey", linestyle=":", label="AlphaP_AMR_Study ({:})".format(REFERENCE_GF))
    axes[0].plot([], [], color="grey", marker="o", markerfacecolor="white", linestyle="", label="stopped short")
    axes[0].set_ylabel("G_f [N/mm]")
    axes[0].set_title("Fracture energy")
    axes[1].set_ylabel("peak force [N]")
    axes[1].set_title("Peak force")

    for axis in axes:
        axis.set_xscale("log")
        axis.set_xlabel("element size h [mm]")
        axis.grid(alpha=0.3, which="both")
        axis.legend(fontsize=8)

    fig.suptitle("Mesh convergence. Only rows that reached the endpoint are comparable", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(figures, "gf_convergence.png"), dpi=130)
    plt.close(fig)

    return "gf_convergence.png"


def figureSpaceTime(figures, cases, variable, name, title):
    """Contours of one per-element damage field over the bar and over the loading history."""

    available = []
    for label, directory in cases:
        path = runDir(directory)
        trend = readFieldTrend(path, variable) if path else None

        if trend is not None:
            available.append((label, trend))

    if not available:
        return None

    fig, axes = plt.subplots(1, len(available), figsize=(3.6 * len(available) + 1.2, 4.6), sharey=True)
    axes = np.atleast_1d(axes)

    peak = max(np.nanmax(trend[2]) for _, trend in available)
    levels = np.linspace(0.0, peak if peak > 0.0 else 1.0, 21)

    mesh = None
    for axis, (label, (centroids, elongation, values)) in zip(axes, available):
        finite = np.isfinite(values).all(axis=1)
        mesh = axis.contourf(
            centroids,
            elongation[finite],
            np.clip(values[finite], levels[0], levels[-1]),
            levels=levels,
            cmap="inferno",
        )
        axis.set_title(label, fontsize=10)
        axis.set_xlabel("position along bar [mm]")

        if not finite.all():
            axis.axhline(
                elongation[finite].max(),
                color="cyan",
                linestyle="--",
                linewidth=1.2,
            )
            axis.text(
                2.0,
                elongation[finite].max(),
                " diverged",
                color="cyan",
                fontsize=8,
                va="bottom",
            )

    axes[0].set_ylabel("elongation [mm]")
    fig.colorbar(mesh, ax=axes.tolist(), label=variable, fraction=0.03)
    fig.suptitle(title, fontsize=12)
    fig.savefig(os.path.join(figures, name), dpi=130, bbox_inches="tight")
    plt.close(fig)

    return name


def figureFinalProfiles(figures, variable="omega"):
    """The damage profile along the bar at the endpoint, one panel per mesh."""

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)

    for axis, nX in zip(axes.flat, MESHES):
        for variant, (colour, style, width) in VARIANT_STYLE.items():
            path = runDir("run_{:}_nX{:}".format(variant, nX))
            trend = readFieldTrend(path, variable) if path else None

            if trend is None:
                continue

            centroids, _, values = trend
            finite = np.isfinite(values).all(axis=1)

            if not finite.any():
                continue

            axis.plot(
                centroids,
                values[finite][-1],
                color=colour,
                linestyle=style,
                linewidth=width,
                marker=".",
                markersize=3,
                label=variant,
            )

        axis.set_title("nX = {:}  (h = {:.3f} mm)".format(nX, BAR_LENGTH / nX))
        axis.grid(alpha=0.3)
        axis.set_xlim(0.0, BAR_LENGTH)

    for axis in axes[1, :]:
        axis.set_xlabel("position along bar [mm]")
    for axis in axes[:, 0]:
        axis.set_ylabel(variable)

    axes[0, 0].legend(fontsize=8)
    fig.suptitle("{:} along the bar at the last finite state".format(variable), fontsize=13)
    fig.tight_layout()

    name = "profile_{:}.png".format(variable)
    fig.savefig(os.path.join(figures, name), dpi=130)
    plt.close(fig)

    return name


def figureCrackSite(figures):
    """Where the crack forms, and how wide the damaged band is, against the element size.

    This is the figure that explains the fracture energies. The bar's only imperfection is a
    central element weakened by 0.7 %, which turns out to be a weaker trigger than the clamped end,
    so which of the two wins varies with the mesh -- and a crack at the boundary has only half of
    its non-local band inside the bar to dissipate in.
    """

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    for variant, (colour, style, width) in VARIANT_STYLE.items():
        sizes, positions, bands = [], [], []

        for nX in MESHES:
            path = runDir("run_{:}_nX{:}".format(variant, nX))
            site = crackSite(path) if path else None

            if site is None:
                continue

            sizes.append(BAR_LENGTH / nX)
            positions.append(site[0])
            bands.append(site[1])

        if not sizes:
            continue

        axes[0].plot(sizes, positions, color=colour, linestyle=style, linewidth=width, marker="o", label=variant)
        axes[1].plot(sizes, bands, color=colour, linestyle=style, linewidth=width, marker="o", label=variant)

    axes[0].axhline(BAR_LENGTH / 2.0, color="green", linestyle=":", label="weakened element (centre)")
    axes[0].axhline(0.0, color="grey", linestyle=":", label="clamped end")
    axes[0].set_ylabel("position of maximum damage [mm]")
    axes[0].set_ylim(-5.0, BAR_LENGTH)
    axes[0].set_title("Where the crack forms")
    axes[1].set_ylabel("width of the band with omega > 0.1 [mm]")
    axes[1].set_title("Width of the damaged band")

    for axis in axes:
        axis.set_xscale("log")
        axis.set_xlabel("element size h [mm]")
        axis.grid(alpha=0.3, which="both")
        axis.legend(fontsize=8)

    fig.suptitle("The imperfection loses to the boundary, and not always at the same mesh", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(figures, "crack_site.png"), dpi=130)
    plt.close(fig)

    return "crack_site.png"


#: Material and geometry of the bar, for the elastic reference line of the quasi-staticity figure.
YOUNGS_MODULUS = 30600.0
DENSITY = 2.4e-9


def figureQuasiStatic(figures):
    """Whether each configuration is quasi-static, read at the clamped end against statics.

    Every run of the study proper reports a kinetic share of 0.00 %, so the question is no longer
    whether the loading is slow enough -- it is how far the two knobs that buy the time increment,
    mass scaling and the non-local viscosity, can be pushed before the answer stops being right.
    """

    waveSpeedAt = lambda scaling: ((YOUNGS_MODULUS / (DENSITY * scaling)) ** 0.5)  # noqa: E731

    cases = [
        ("f = 70, eta = 1e-4 (the study)", "run_explicit-hyperbolic_nX40", "#d62728", 70.0),
        ("f = 7500, eta = 1e-3", "run_explicit-hyperbolic_nX40_eta1e-03_ms8e+03", "#2ca02c", 7500.0),
        ("f = 220000, eta = 6e-3", "run_explicit-hyperbolic_nX40_eta6e-03_ms2e+05", "#9467bd", 220000.0),
        ("implicit (reference)", "run_implicit_nX40", "#000000", None),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    for axis, window in zip(axes, (0.012, TOTAL_ELONGATION)):
        if window < TOTAL_ELONGATION:
            elongations = np.linspace(0.0, window, 50)
            axis.plot(
                elongations,
                YOUNGS_MODULUS * CROSS_SECTION / BAR_LENGTH * elongations,
                color="grey",
                linestyle=":",
                label="static elastic, E*A/L * u",
            )

        for label, directory, colour, scaling in cases:
            path = runDir(directory)
            history = readHistory(path) if path else None

            if history is None:
                continue

            elongation, force, _ = history
            inside = elongation <= window
            transits = ""
            if scaling is not None:
                transitTime = BAR_LENGTH / waveSpeedAt(scaling)
                transits = ", {:.0f} transits".format(
                    (elongation[np.argmax(force)] / TOTAL_ELONGATION * EXPLICIT_STEP_LENGTH) / transitTime
                )
            axis.plot(
                elongation[inside],
                force[inside],
                color=colour,
                linewidth=1.5,
                label="{:}  (F_max {:.2f}{:})".format(label, force.max(), transits),
            )

        axis.set_xlabel("elongation [mm]")
        axis.set_ylabel("reaction force at the clamped end [N]")
        axis.set_xlim(0.0, window)
        axis.set_ylim(0.0, 80.0)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)

    axes[0].set_title("Up to the peak")
    axes[1].set_title("Whole history")
    fig.suptitle(
        "How far the mass scaling / eta levers can be pushed before the peak force goes wrong (nX = 40)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(figures, "quasi_static.png"), dpi=130)
    plt.close(fig)

    return "quasi_static.png"


def figureDiagnostics(figures):
    """The three controls the study's configuration rests on.

    Each one asks whether a parameter that was changed for the sake of run time changed the answer.
    """

    etaCases = [
        ("eta = 1e-4 (the study)", "run_explicit-hyperbolic_nX40", "#d62728", "-"),
        ("eta = 1e-5 (control)", "run_diag_eta1em5_hyperbolic_nX40", "#ff7f0e", "--"),
        ("implicit (reference)", "run_implicit_nX40", "#000000", "-"),
    ]
    pseudoTimeCases = [
        ("implicit, stepLength = 1 s", "run_implicit_nX40", "#000000", "-"),
        ("implicit, stepLength = 1e-2 s", "run_diag_implicit_steplength1em2_nX40", "#7f7f7f", "--"),
    ]
    scalingCases = [
        ("f = 70, eta = 1e-4", "run_explicit-hyperbolic_nX40", "#d62728", "-"),
        ("f = 7500, eta = 1e-3", "run_explicit-hyperbolic_nX40_eta1e-03_ms8e+03", "#2ca02c", "-"),
        ("f = 220000, eta = 6e-3", "run_explicit-hyperbolic_nX40_eta6e-03_ms2e+05", "#9467bd", "-"),
        ("implicit (reference)", "run_implicit_nX40", "#000000", "--"),
    ]

    titles = (
        "Does raising eta change the answer?",
        "Is the implicit step length pseudo-time?",
        "How far can the levers be pushed?",
    )

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))

    for axis, group, title in zip(axes, (etaCases, pseudoTimeCases, scalingCases), titles):
        drawn = False
        for label, directory, colour, style in group:
            path = runDir(directory)
            history = readHistory(path) if path else None

            if history is None:
                continue

            elongation, force, _ = history
            axis.plot(
                elongation,
                force,
                color=colour,
                linestyle=style,
                linewidth=1.5,
                label="{:}  (G_f {:.5f}, F_max {:.2f})".format(label, fractureEnergy(elongation, force), force.max()),
            )
            drawn = True

        axis.set_title(title, fontsize=11)
        axis.set_xlabel("elongation [mm]")
        axis.set_ylabel("reaction force [N]")
        axis.set_xlim(0.0, TOTAL_ELONGATION)
        axis.set_ylim(bottom=0.0)
        axis.grid(alpha=0.3)

        if drawn:
            axis.legend(fontsize=8)

    fig.suptitle("Controls on the configuration, all at nX = 40", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(figures, "diagnostics.png"), dpi=130)
    plt.close(fig)

    return "diagnostics.png"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", default=os.path.join(HERE, "figures"))
    arguments = parser.parse_args()

    os.makedirs(arguments.figures, exist_ok=True)

    written = [
        figureLoadDisplacement(arguments.figures, zoom=False),
        figureLoadDisplacement(arguments.figures, zoom=True),
        figureMeshConvergence(arguments.figures),
        figureFractureEnergy(arguments.figures),
        figureCrackSite(arguments.figures),
        figureFinalProfiles(arguments.figures, "omega"),
        figureFinalProfiles(arguments.figures, "alphaP"),
        figureSpaceTime(
            arguments.figures,
            [
                ("explicit-parabolic", "run_explicit-parabolic_nX160"),
                ("explicit-hyperbolic", "run_explicit-hyperbolic_nX160"),
                ("explicit-hyperbolic-bv", "run_explicit-hyperbolic-bv_nX160"),
                ("implicit", "run_implicit_nX160"),
            ],
            "omega",
            "spacetime_omega_nX160.png",
            "Damage over the bar and over the loading history, nX = 160 (h = 0.625 mm)",
        ),
        figureSpaceTime(
            arguments.figures,
            [
                ("implicit nX = 20", "run_implicit_nX20"),
                ("implicit nX = 40", "run_implicit_nX40"),
                ("implicit nX = 80", "run_implicit_nX80"),
                ("implicit nX = 160", "run_implicit_nX160"),
            ],
            "omega",
            "spacetime_omega_implicit.png",
            "The implicit reference across meshes: where the damage goes",
        ),
        figureSpaceTime(
            arguments.figures,
            [
                ("explicit-hyperbolic nX = 20", "run_explicit-hyperbolic_nX20"),
                ("explicit-hyperbolic nX = 40", "run_explicit-hyperbolic_nX40"),
                ("explicit-hyperbolic nX = 80", "run_explicit-hyperbolic_nX80"),
                ("explicit-hyperbolic nX = 160", "run_explicit-hyperbolic_nX160"),
            ],
            "omega",
            "spacetime_omega_hyperbolic.png",
            "The hyperbolic explicit scheme across meshes",
        ),
        figureSpaceTime(
            arguments.figures,
            [
                ("explicit-hyperbolic", "run_explicit-hyperbolic_nX80"),
                ("explicit-hyperbolic-bv", "run_explicit-hyperbolic-bv_nX80"),
                ("implicit", "run_implicit_nX80"),
            ],
            "alphaP",
            "spacetime_alphap_nX80.png",
            "Hardening variable alphaP, nX = 80: what bulk viscosity does away from the crack",
        ),
        figureDiagnostics(arguments.figures),
        figureQuasiStatic(arguments.figures),
    ]

    for name in written:
        if name is not None:
            print("wrote {:}".format(os.path.join(os.path.basename(arguments.figures), name)))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
