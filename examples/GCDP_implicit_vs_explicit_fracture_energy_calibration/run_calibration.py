"""GCDP fracture-energy calibration: implicit against explicit, across all field-integration
variants.

One uniaxial tension bar, one loading history, five ways of integrating it:

    implicit                the reference. Elliptic non-local field, Newton-Raphson.
    explicit-parabolic      non-local field first order in time (its viscosity as the mass),
                            forward Euler. The status quo, and the one whose stability limit falls
                            off with h^2 and is checked by nothing.
    explicit-hyperbolic     non-local field second order in time (a micro-inertia as the mass, its
                            viscosity as the damping). Stability limit falls off with h.
    explicit-*-bv           the same two, with artificial bulk viscosity on the mechanical field.

The calibrated quantity is the dissipated work per unit fracture area, G_f. Everything here exists
to answer one question: does buying an explicit time step -- with a micro-inertia, with bulk
viscosity, or both -- change the fracture energy the model delivers?

Held fixed across every variant so that the comparison means something: the geometry, the material
(including a ZERO Duvaut-Lions viscosity, so the constitutive problem is rate-independent), the
weakened central element, and the total prescribed elongation of 0.2 mm -- the same endpoint the
reference deck of examples/AlphaP_AMR_Study uses, so G_f is directly comparable to its 0.098 N/mm.

Not free parameters:
  * m_k = eta^2/4. Requiring the zeroth-order reaction mode not to ring bounds the micro-inertia at
    eta^2/4, and the largest admissible value is the best one because dt grows with sqrt(m_k). So
    the micro-inertia follows from the viscosity the deck already had.
  * b1 = 0.06, b2 = 1.2, the Abaqus/Explicit defaults for bulk viscosity.

Usage
-----
    python run_calibration.py                        # implicit + all four explicit, nX = 20..160
    python run_calibration.py --quick                # nX = 20, 40
    python run_calibration.py --meshes 40,160
    python run_calibration.py --variants implicit,explicit-hyperbolic

Each case lands in run_<variant>_nX<n>/ and is skipped if its RF.csv is already there; delete the
directory to force a re-run.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

#: The non-local viscosity the template's material declares, in seconds.
NONLOCAL_VISCOSITY = 1e-5

#: Critical damping of the reaction mode -- the largest micro-inertia that does not ring, and
#: therefore the one that buys the most time step. Units of seconds squared.
MICRO_INERTIA = NONLOCAL_VISCOSITY**2 / 4.0

#: Abaqus/Explicit's default bulk viscosity coefficients: linear (Landshoff) and quadratic
#: (Von Neumann-Richtmyer).
BULK_VISCOSITY = (0.06, 1.2)

#: Cross-sectional area of the bar, for the fracture energy.
CROSS_SECTION = 25.0

#: Total prescribed elongation, matching examples/AlphaP_AMR_Study so G_f is comparable.
TOTAL_ELONGATION = 0.2

#: Duration of the explicit ramp. Long against the bar's own transit time (100 mm at ~3.8e6 mm/s is
#: 2.6e-5 s), so the run is quasi-static in the sense that matters: the reported kinetic fraction
#: below is the check on that, not an assumption.
EXPLICIT_STEP_LENGTH = 1e-3

IMPLICIT_SOLVER = """*solver, solver=NISTParallel, name=theSolver"""

EXPLICIT_SOLVER = """** Courant 0.1, not the 0.8 default: l/c is a linear-element formula and a 20-node element's
** highest free eigenfrequency lies far above it. See testfiles/marmot/GCDPNEDExplicit.
*solver, solver=NEDParallel, name=theSolver
{fields:}
courant-number=0.1
output-frequency=2000"""

PARABOLIC_FIELDS = """second-order-fields="displacement"
first-order-fields="nonlocal damage\""""

HYPERBOLIC_FIELDS = """second-order-fields="displacement, nonlocal damage"
micro-inertia-fields="nonlocal damage\""""

MICRO_INERTIA_PROPERTY = """*elementProperty, elSet=gen_all, propertyName=nonlocal micro inertia
{:.6e}""".format(
    MICRO_INERTIA
)

BULK_VISCOSITY_PROPERTY = """*elementProperty, elSet=gen_all, propertyName=bulk viscosity
{:}, {:}""".format(
    *BULK_VISCOSITY
)

IMPLICIT_STEP = """*step, solver=theSolver
maxInc=1e-2, minInc=1e-6, maxNumInc=10000, maxIter=25, stepLength=1
>>options, name=theSolver, extrapolation=linear
>>dirichlet, name=leftX, nSet=gen_left, field=displacement, 1=0.0
>>dirichlet, name=leftY, nSet=gen_left, field=displacement, 2=0.0
>>dirichlet, name=leftZ, nSet=gen_left, field=displacement, 3=0.0
>>dirichlet, name=right, nSet=gen_right, field=displacement, 1={elongation:}"""

EXPLICIT_STEP = """*step, type=adaptiveForExplicitSimulations, solver=theSolver
maxInc=1, minInc=1e-14, maxNumInc=2000000, maxIter=25, stepLength={stepLength:.6e}
>>dirichlet, name=leftX, nSet=gen_left, field=displacement, 1=0.0
>>dirichlet, name=leftY, nSet=gen_left, field=displacement, 2=0.0
>>dirichlet, name=leftZ, nSet=gen_left, field=displacement, 3=0.0
>>dirichlet, name=right, nSet=gen_right, field=displacement, 1={elongation:}"""


def variants():
    """The variant matrix: name -> (solver block, element properties, step block)."""

    explicit = {
        "explicit-parabolic": (PARABOLIC_FIELDS, []),
        "explicit-hyperbolic": (HYPERBOLIC_FIELDS, [MICRO_INERTIA_PROPERTY]),
        "explicit-parabolic-bv": (PARABOLIC_FIELDS, [BULK_VISCOSITY_PROPERTY]),
        "explicit-hyperbolic-bv": (HYPERBOLIC_FIELDS, [MICRO_INERTIA_PROPERTY, BULK_VISCOSITY_PROPERTY]),
    }

    matrix = {
        "implicit": (
            IMPLICIT_SOLVER,
            "",
            IMPLICIT_STEP.format(elongation=TOTAL_ELONGATION),
        )
    }

    for name, (fields, properties) in explicit.items():
        matrix[name] = (
            EXPLICIT_SOLVER.format(fields=fields),
            "\n\n".join(properties),
            EXPLICIT_STEP.format(stepLength=EXPLICIT_STEP_LENGTH, elongation=TOTAL_ELONGATION),
        )

    return matrix


def writeDeck(variant, nX, runDir):
    """Substitute the template's tokens for one variant and mesh."""

    with open(os.path.join(HERE, "template.inp")) as templateFile:
        deck = templateFile.read()

    solverBlock, elementProperties, stepBlock = variants()[variant]

    deck = deck.replace("__NX__", str(nX))
    deck = deck.replace("__SOLVER_BLOCK__", solverBlock)
    deck = deck.replace("__ELEMENT_PROPERTIES__", elementProperties)
    deck = deck.replace("__STEP_BLOCK__", stepBlock)

    with open(os.path.join(runDir, "test.inp"), "w") as deckFile:
        deckFile.write(deck)


def run(variant, nX, threads):
    """Run one case, or reuse a completed one, and return its run directory."""

    runDir = os.path.join(HERE, "run_{:}_nX{:}".format(variant, nX))

    if os.path.exists(os.path.join(runDir, "RF.csv")):
        print("  reusing {:}".format(os.path.basename(runDir)))
        return runDir

    shutil.rmtree(runDir, ignore_errors=True)
    os.makedirs(runDir)
    writeDeck(variant, nX, runDir)

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
            env=environment,
            stdout=logFile,
            stderr=subprocess.STDOUT,
            check=False,
        )

    return runDir


def summarise(runDir, variant, nX):
    """Read one run's log and exported history back into a row of the table."""

    row = {"variant": variant, "nX": nX, "h": 100.0 / nX}

    logPath = os.path.join(runDir, "run.log")
    log = open(logPath).read() if os.path.exists(logPath) else ""

    match = re.search(r"Critical time step for explicit dynamics:\s*([0-9.eE+-]+)", log)
    row["dt"] = float(match.group(1)) if match else float("nan")

    # Divergence, in the two forms it takes. The energy guard reports it where it can; a run on a
    # solver without the NaN check passes silently, so the force is the backstop.
    row["diverged"] = "THE SOLUTION HAS DIVERGED" in log
    row["energyCreated"] = "ENERGY IS BEING CREATED" in log

    # Quasi-staticity, measured rather than assumed: the kinetic share of the external work at the
    # last reporting increment. Blank for the implicit variant, which has no kinetic energy.
    kinetic = re.findall(r"\|kinetic\s*\|\s*[+-][0-9.e+-]+\s*\|\s*([0-9.]+) %", log)
    row["kineticShare"] = float(kinetic[-1]) if kinetic else float("nan")

    # NOT the number of rows in the exported history: a saveHistory field output is finalised once
    # per output-frequency increments. The performance table's own tally is the increment count.
    match = re.search(r"\|\s*increment\s*\|[^|]*\|\s*(\d+)\s*\|", log)
    if match:
        row["increments"] = int(match.group(1))
    else:
        reported = re.findall(r"increment (\d+):", log)
        row["increments"] = int(reported[-1]) if reported else 0

    match = re.search(r"\|\s*wall clock\s*\|\s*([0-9.]+)s", log)
    row["wallClock"] = float(match.group(1)) if match else float("nan")

    rfPath = os.path.join(runDir, "RF.csv")
    uPath = os.path.join(runDir, "U.csv")

    row["peakForce"] = float("nan")
    row["fractureEnergy"] = float("nan")
    row["finalForceShare"] = float("nan")

    if os.path.exists(rfPath) and os.path.exists(uPath):
        reaction = np.atleast_2d(np.loadtxt(rfPath))
        displacement = np.atleast_2d(np.loadtxt(uPath))

        force = np.abs(reaction[:, 1])
        elongation = displacement[:, 1]

        if force.size > 1:
            row["peakForce"] = float(np.max(force))
            # Trapezoidal work per unit fracture area -- the same quantity the deck's
            # fractureenergyintegrator reports, recomputed here so the table does not depend on
            # parsing it.
            row["fractureEnergy"] = float(np.trapezoid(force, elongation) / CROSS_SECTION)
            # How much force is still standing at the end. G_f is an integral over a softening
            # tail, so a run stopped while this is still appreciable has not measured all of it --
            # which is a property of the loading history, not of the integration scheme, and the
            # reason every variant here shares one endpoint.
            row["finalForceShare"] = float(force[-1] / row["peakForce"]) if row["peakForce"] > 0.0 else float("nan")

    return row


def status(row):
    """One word on whether the row can be believed."""

    if row["diverged"] or not np.isfinite(row["peakForce"]):
        return "DIVERGED"
    if row["energyCreated"]:
        return "energy!"
    if not row["increments"]:
        return "FAILED"
    return "ok"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meshes", default="20,40,80,160", help="comma-separated nX values")
    parser.add_argument("--variants", default=",".join(variants()), help="comma-separated variant names")
    parser.add_argument("--quick", action="store_true", help="coarse meshes only")
    parser.add_argument("--threads", type=int, default=8)
    arguments = parser.parse_args()

    meshes = [20, 40] if arguments.quick else [int(n) for n in arguments.meshes.split(",")]
    selected = [name.strip() for name in arguments.variants.split(",")]

    for name in selected:
        if name not in variants():
            raise SystemExit("unknown variant {:}; known: {:}".format(name, ", ".join(variants())))

    rows = {}
    for variant in selected:
        print("{:}:".format(variant))
        for nX in meshes:
            runDir = run(variant, nX, arguments.threads)
            rows[(variant, nX)] = summarise(runDir, variant, nX)

    header = "{:<24} {:>5} {:>7} {:>11} {:>10} {:>9} {:>10} {:>7} {:>7} {:>9}".format(
        "variant", "nX", "h [mm]", "dt [s]", "increments", "F_max [N]", "G_f [N/mm]", "F_end", "T/W_ext", "status"
    )
    print("\n" + header)
    print("-" * len(header))

    for nX in meshes:
        for variant in selected:
            row = rows[(variant, nX)]
            print(
                "{:<24} {:>5} {:>7.3f} {:>11} {:>10} {:>9} {:>10} {:>7} {:>7} {:>9}".format(
                    row["variant"],
                    row["nX"],
                    row["h"],
                    "{:.4e}".format(row["dt"]) if np.isfinite(row["dt"]) else "-",
                    row["increments"] or "-",
                    "{:.2f}".format(row["peakForce"]) if np.isfinite(row["peakForce"]) else "-",
                    "{:.5f}".format(row["fractureEnergy"]) if np.isfinite(row["fractureEnergy"]) else "-",
                    "{:.1%}".format(row["finalForceShare"]) if np.isfinite(row["finalForceShare"]) else "-",
                    "{:.2f}%".format(row["kineticShare"]) if np.isfinite(row["kineticShare"]) else "-",
                    status(row),
                )
            )
        print()

    print(
        "m_k = eta^2/4 = {:.4e} s^2 (eta = {:.1e} s, damage wave speed 2*l/eta = {:.3e} mm/s); "
        "bulk viscosity b1 = {:}, b2 = {:}; all variants pulled to {:} mm".format(
            MICRO_INERTIA, NONLOCAL_VISCOSITY, 2 * 5.0 / NONLOCAL_VISCOSITY, *BULK_VISCOSITY, TOTAL_ELONGATION
        )
    )
    print("F_end is the force still standing at the endpoint, as a share of peak: G_f has measured")
    print("all of the softening only where that is small. T/W_ext is the kinetic share, i.e. how")
    print("quasi-static the explicit run actually was.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
