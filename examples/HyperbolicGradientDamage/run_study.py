"""Calibration and validation of the non-local micro-inertia on a uniaxial tension bar.

Runs the same GCDP bar twice per mesh -- once with the gradient-enhanced field integrated as a
viscous, parabolic equation (the status quo), once with a micro-inertia making it a damped
hyperbolic one -- and tabulates what has to agree and what has to differ.

What has to AGREE (this is the validation):
  * the peak force, to within the discretisation difference between the two meshes;
  * the dissipated fracture energy per unit area, which is the calibrated quantity of
    ``examples/AlphaP_AMR_Study``;
  * the width of the localisation band, which is set by the internal length and by nothing the
    time integration does.

What has to DIFFER (this is the point):
  * the stable time increment, and how it scales with the element size. The parabolic limit falls
    off with h^2 and is checked by nothing, so the parabolic column is expected to fail outright on
    the finest mesh while the hyperbolic one is still governed by the mechanical limit.

The micro-inertia is not a free parameter here. Requiring that the zeroth-order reaction mode not
ring gives m_k <= eta^2 / 4, and the largest admissible value is the best one because the stable
increment grows with sqrt(m_k) -- so m_k = eta^2 / 4 for the eta the deck already uses.

Usage
-----
    python run_study.py                  # the full matrix
    python run_study.py --quick          # coarse meshes only, short ramp
    python run_study.py --meshes 20,320  # a specific set of meshes

Runs are placed in ``run_<scheme>_nX<n>/`` next to this script and are not re-run if their
``RF.csv`` is already there; delete the directory to force a re-run.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

#: The non-local viscosity the template's material uses, in seconds.
NONLOCAL_VISCOSITY = 1e-5

#: Critical damping of the reaction mode: the largest micro-inertia that does not ring, and
#: therefore the one that buys the most time step. See doc/pages/features/nonlocalmicroinertia.rst
#: in Marmot for the derivation.
MICRO_INERTIA = NONLOCAL_VISCOSITY**2 / 4.0

#: Cross-sectional area of the bar, for the fracture energy.
CROSS_SECTION = 25.0

PARABOLIC_FIELDS = """second-order-fields="displacement"
first-order-fields="nonlocal damage\""""

HYPERBOLIC_FIELDS = """second-order-fields="displacement", "nonlocal damage"
micro-inertia-fields="nonlocal damage\""""

HYPERBOLIC_PROPERTY = """*elementProperty, elSet=gen_all, propertyName=nonlocal micro inertia
{:.6e}""".format(
    MICRO_INERTIA
)

SCHEMES = {
    "parabolic": (PARABOLIC_FIELDS, ""),
    "hyperbolic": (HYPERBOLIC_FIELDS, HYPERBOLIC_PROPERTY),
}


def writeDeck(scheme, nX, stepLength, maxNumInc, runDir):
    """Substitute the template's placeholders and write the deck into ``runDir``."""

    with open(os.path.join(HERE, "template.inp")) as templateFile:
        deck = templateFile.read()

    solverFields, elementProperty = SCHEMES[scheme]

    deck = deck.replace("__NX__", str(nX))
    deck = deck.replace("__SOLVER_FIELDS__", solverFields)
    deck = deck.replace("__ELEMENT_PROPERTY__", elementProperty)
    deck = deck.replace("__STEP_LENGTH__", "{:.6e}".format(stepLength))
    deck = deck.replace("__MAX_NUM_INC__", str(maxNumInc))

    deckPath = os.path.join(runDir, "test.inp")
    with open(deckPath, "w") as deckFile:
        deckFile.write(deck)

    return deckPath


def run(scheme, nX, stepLength, maxNumInc, threads):
    """Run one case, or reuse a completed one, and return its run directory."""

    runDir = os.path.join(HERE, "run_{:}_nX{:}".format(scheme, nX))

    if os.path.exists(os.path.join(runDir, "RF.csv")):
        print("  reusing {:}".format(os.path.basename(runDir)))
        return runDir

    shutil.rmtree(runDir, ignore_errors=True)
    os.makedirs(runDir)
    writeDeck(scheme, nX, stepLength, maxNumInc, runDir)

    environment = dict(os.environ)
    environment["OMP_NUM_THREADS"] = str(threads)
    # Free-threaded build: the element loop is where the parallelism is, and the GIL would serialise
    # it. See the workspace CLAUDE.md.
    environment["PYTHON_GIL"] = "0"

    print("  running {:} ...".format(os.path.basename(runDir)), flush=True)
    with open(os.path.join(runDir, "run.log"), "w") as logFile:
        subprocess.run(
            ["edelweissfe", "test.inp"],
            cwd=runDir,
            stdout=logFile,
            stderr=subprocess.STDOUT,
            check=False,
        )

    return runDir


def summarise(runDir, nX):
    """Read one run's log and exported history back into a row of the table."""

    row = {"nX": nX, "h": 100.0 / nX}

    logPath = os.path.join(runDir, "run.log")
    log = open(logPath).read() if os.path.exists(logPath) else ""

    match = re.search(r"Critical time step for explicit dynamics:\s*([0-9.eE+-]+)", log)
    row["dt"] = float(match.group(1)) if match else float("nan")

    # The energy guard is what a parabolic run above its own (unchecked) stability limit trips:
    # the kinetic energy overtakes the external work, which is impossible, and the solver says so.
    row["energyCreated"] = "ENERGY IS BEING CREATED" in log
    row["incrementationFailed"] = "Incrementation failed" in log

    rfPath = os.path.join(runDir, "RF.csv")
    uPath = os.path.join(runDir, "U.csv")

    if not (os.path.exists(rfPath) and os.path.exists(uPath)):
        row["increments"] = 0
        row["peakForce"] = float("nan")
        row["fractureEnergy"] = float("nan")
        return row

    reaction = np.atleast_2d(np.loadtxt(rfPath))
    displacement = np.atleast_2d(np.loadtxt(uPath))

    row["increments"] = reaction.shape[0]

    force = np.abs(reaction[:, 1])
    elongation = displacement[:, 1]

    row["peakForce"] = float(np.max(force)) if force.size else float("nan")
    # Trapezoidal work done on the bar, per unit fracture area. The same quantity the deck's
    # fractureenergyintegrator reports; recomputed here so the table does not depend on parsing it.
    row["fractureEnergy"] = float(np.trapezoid(force, elongation) / CROSS_SECTION) if force.size > 1 else float("nan")

    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meshes", default="20,40,80,320", help="comma-separated nX values")
    parser.add_argument("--quick", action="store_true", help="coarse meshes and a short ramp")
    parser.add_argument("--step-length", type=float, default=1e-3, help="ramp duration in seconds")
    parser.add_argument("--max-num-inc", type=int, default=400000)
    parser.add_argument("--threads", type=int, default=8)
    arguments = parser.parse_args()

    meshes = [20, 40] if arguments.quick else [int(n) for n in arguments.meshes.split(",")]
    stepLength = 3e-4 if arguments.quick else arguments.step_length

    rows = {}
    for scheme in ("parabolic", "hyperbolic"):
        print("{:}:".format(scheme))
        for nX in meshes:
            runDir = run(scheme, nX, stepLength, arguments.max_num_inc, arguments.threads)
            rows[(scheme, nX)] = summarise(runDir, nX)

    header = "{:<12} {:>5} {:>8} {:>12} {:>11} {:>11} {:>12} {:>9}".format(
        "scheme", "nX", "h [mm]", "dt [s]", "increments", "F_max [N]", "G_f [N/mm]", "status"
    )
    print("\n" + header)
    print("-" * len(header))

    for nX in meshes:
        for scheme in ("parabolic", "hyperbolic"):
            row = rows[(scheme, nX)]
            if row["energyCreated"]:
                status = "energy!"
            elif row["incrementationFailed"]:
                status = "cutbacks"
            elif row["increments"]:
                status = "ok"
            else:
                status = "FAILED"
            print(
                "{:<12} {:>5} {:>8.3f} {:>12.4e} {:>11} {:>11.2f} {:>12.5f} {:>9}".format(
                    scheme,
                    row["nX"],
                    row["h"],
                    row["dt"],
                    row["increments"],
                    row["peakForce"],
                    row["fractureEnergy"],
                    status,
                )
            )

    print(
        "\nmicro-inertia m_k = eta^2/4 = {:.4e} s^2 (eta = {:.1e} s), damage wave speed "
        "c_k = 2*l/eta = {:.3e} mm/s".format(MICRO_INERTIA, NONLOCAL_VISCOSITY, 2 * 5.0 / NONLOCAL_VISCOSITY)
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
