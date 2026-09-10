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
weakened central element, the quarter-symmetry restraint, the strengthened end zones, the total
prescribed elongation of 0.2 mm -- the same endpoint the reference deck of examples/AlphaP_AMR_Study
uses, so G_f is directly comparable to its 0.098 N/mm -- and, since it turned out to matter more
than anything else, the DURATION over which that elongation is applied.

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

#: The non-local viscosity the material declares, in seconds. It is what bounds this problem.
#:
#: The stable increment of the hyperbolic scheme is set by the NON-LOCAL limit,
#: omega_max = sqrt((1 + C l^2/h^2)/m_k) with m_k = eta^2/4, in which no density appears -- measured
#: 8.85e-08 / 6.38e-08 / 3.43e-08 at nX = 20 / 40 / 80, unchanged by a 1e4 mass scaling. Since
#: dt scales with sqrt(m_k), i.e. with eta, and the load is now applied over 1 s, eta is the only
#: lever on a run length that would otherwise be 11 to 57 million increments per case.
#:
#: 1e-4 rather than the 1e-5 of the first study. What makes that affordable is the slower loading:
#: eta's viscous contribution grows with the rate, and over 1 s the rate is 1000x below the 1e-3 s
#: ramp on which eta was measured doing real work, so ten times more eta still leaves it about two
#: orders of magnitude below that. Verified rather than assumed -- see the eta control pair in
#: NOTES_stepLength1.md.
#:
#: KNOWN CONSEQUENCE: it also lifts the PARABOLIC scheme's own limit, dt <= 2 eta/(1 + C l^2/h^2),
#: to 1.30e-07 at nX = 160 -- above the mechanical CFL of 1.52e-08. At eta = 1e-5 that limit was
#: 1.30e-08, i.e. violated by the increment the solver actually took, which is why the parabolic
#: scheme went NaN at h = 0.625 mm. So this deck can no longer show that divergence: raising eta to
#: afford the increment is precisely the workaround the micro-inertia exists to remove. The
#: divergence is demonstrated at eta = 1e-5 in RESULTS.md; this deck answers the fracture-energy
#: question instead.
NONLOCAL_VISCOSITY = 1e-4

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

#: Duration of the load application, the SAME for every variant. An earlier version of this study
#: ramped the explicit variants over 1e-3 s while the implicit one took 1 s, which made the two
#: incomparable in a way no column of the table revealed: at 1e-3 s the bar sees only 1.2 wave
#: transits before the peak, the reaction at the clamped end is still 0.5 N where statics says
#: 27.6 N, and the reported peak force is the amplitude of a transient rather than a strength.
#: One step length for all of them is the only way the numbers mean the same thing.
EXPLICIT_STEP_LENGTH = 1.0

#: Reference density of the bar, as the material data of examples/AlphaP_AMR_Study gives it.
DENSITY = 2.4e-9

#: Elastic constants of the template's material (E = 30600, nu = 0.2), for the dilatational wave
#: speed that the parameter rule below needs.
LAMES_LAMBDA = 30600.0 * 0.2 / ((1.0 + 0.2) * (1.0 - 2.0 * 0.2))
SHEAR_MODULUS = 30600.0 / (2.0 * (1.0 + 0.2))

#: Gershgorin constant C of the element's non-local operator, CALIBRATED from the measured ratio of
#: the two limits rather than taken from the plan. With eta = 1e-4 and mass scaling 70 the measured
#: dt_nl/dt_mech is 1.258 at nX = 40 and 1.354 at nX = 80, approaching its asymptote from below as
#: h/l falls, and R = eta*c_d/(l*sqrt(C)) then gives C = 40. Using the plan's "nearer 24 than 12",
#: or inserting the courant number a second time, understates the floor about sixfold and reports a
#: configuration as safe while the non-local limit is in fact binding.
GERSHGORIN_CONSTANT = 40.0


def microInertia(nonlocalViscosity):
    """The largest micro-inertia that does not let the zeroth-order reaction mode ring."""

    return nonlocalViscosity**2 / 4.0


def mechanicallyDominatedViscosity(massScaling, gershgorinConstant=GERSHGORIN_CONSTANT):
    """The smallest non-local viscosity that leaves the MECHANICAL problem in charge of dt.

    Both limits of the hyperbolic scheme are linear in h once h << l, so their ratio is a property
    of the parameters and not of the mesh -- fix it once and it holds at every refinement level:

        dt_nl / dt_mech = eta * c_d / (l * sqrt(C))

    The h cancels, and so does the courant number, which the solver applies to both limits alike.
    Requiring that ratio to be at least one gives the floor returned here,

        eta >= sqrt(C) * l / c_d        <=>        m_k >= (C/4) * (l/c_d)^2

    i.e. the micro-inertia must exceed the SQUARE of the time a mechanical wave needs to cross one
    non-local length -- the only natural time scale the coupled problem offers.

    Mass scaling lowers c_d as 1/sqrt(f), so this floor RISES as sqrt(f): the two knobs have to move
    together. Moving either alone measures as exactly zero gain -- 8.852981e-08 at nX = 20 both
    scaled and unscaled, and 6.063391e-08 at nX = 40 at both eta = 1e-5 and eta = 1e-4.
    """

    waveSpeed = ((LAMES_LAMBDA + 2.0 * SHEAR_MODULUS) / (DENSITY * massScaling)) ** 0.5

    return gershgorinConstant**0.5 * NONLOCAL_LENGTH / waveSpeed


#: Factor the density of the EXPLICIT variants is scaled by. Central difference needs an increment
#: below 2/omega_max, so a load applied over 1 s costs 3.3e7 increments at nX = 80 -- about a day
#: per case, and a fortnight for the matrix. Mass scaling buys that back: every wave speed falls
#: with the square root of the factor, so the stable increment rises by the same 100 at 1e4.
#:
#: Mass scaling of the EXPLICIT variants' density. The second of the two levers on the increment,
#: and it only works together with the first: eta raises the NON-LOCAL limit, mass scaling raises
#: the MECHANICAL CFL, and dt is the smaller of the two. Measured at nX = 40 with eta = 1e-4:
#:
#:   no scaling   dt = 6.063391e-08   <- identical to eta = 1e-5; the mechanical CFL binds
#:   scaling 70   dt = 5.072997e-07   <- 8.37x, exactly sqrt(70)
#:
#: So raising eta on its own bought nothing at nX >= 40, and a 1e4 scaling on its own bought nothing
#: either (the non-local limit bound it instead). Together they give the whole factor.
#:
#: 70 rather than more, because of the ONE limit in this problem that nothing checks: the parabolic
#: scheme's own, dt <= 2 eta/(1 + C l^2/h^2), which at eta = 1e-4 and nX = 160 is about 1.30e-07
#: against the 1.269e-07 this scaling takes -- roughly 98 % of it, with C itself only known to be
#: "nearer 24 than 12". At 1e4 that limit was exceeded a hundredfold and the parabolic runs went
#: NaN while still reporting that they reached the endpoint. Treat the parabolic nX = 160 row as
#: suspect on stability grounds; the hyperbolic rows are bounded by a limit the code computes.
#:
#: Quasi-static regardless: kinetic energy goes as rho*v^2, and over 1 s the velocity is 1000x
#: below the 1e-3 s study that already sat at 5 %, so 70x the density leaves it about 7e-5 of that.
#: T/W_ext measures it.
MASS_SCALING = 70.0

#: Nominal tensile strength, and the factor by which the two end zones are raised above it.
#: A gradient-damage bar localises at its own boundaries: the non-local field has a zero-flux
#: condition there, which raises eps_bar for the same local strain, and that beat the 0.7 % weakened
#: central element in most of the runs of the first study -- implicitly at some meshes and not
#: others, so the fracture energy was measured on a boundary crack half the time and jumped around
#: by a factor of two. Strengthening one non-local length at each end removes the competition
#: instead of out-shouting it: unlike a deeper notch it leaves the peak force a property of the
#: material rather than of the imperfection.
END_ZONE_STRENGTH_FACTOR = 1.5

#: Non-local length of the material, which is how far the boundary effect reaches.
NONLOCAL_LENGTH = 5.0

#: Length of the bar.
BAR_LENGTH = 100.0

#: Field-output finalizations wanted per explicit run. NOT a cosmetic setting: output-frequency
#: gates fieldOutputController.finalizeIncrement(), so it is the sampling rate of the RF/U
#: histories G_f is integrated from -- and the peak force lies within the first ~5 % of the
#: prescribed elongation. At the solver's reporting cadence the first sample already sat deep in
#: the softening branch, which reported a post-peak force as F_max and integrated G_f over ten
#: points. ~1000 samples puts ~45 of them before the peak and keeps the transient export small.
EXPLICIT_OUTPUT_SAMPLES = 1000

#: Measured UNSCALED stable increment at nX = 40 (h = 2.5 mm) with eta = 1e-4, in seconds. Only
#: used to estimate the increment count *before* a run starts, which is what the output cadence has
#: to be derived from; the estimate is then dt * sqrt(massScaling), which reproduces the measured
#: 5.072997e-07 at a scaling of 70 exactly.
#:
#: At eta = 1e-4 this is the MECHANICAL CFL, and it scales as 40/nX. At nX = 20 the non-local limit
#: (8.85e-07) binds below the estimate instead, so the cadence there comes out slightly coarse --
#: the samples column reports what each run actually took.
DT_AT_NX40 = 6.063391e-08

IMPLICIT_SOLVER = """*solver, solver=NISTParallel, name=theSolver"""

EXPLICIT_SOLVER = """** Courant 0.1, not the 0.8 default: l/c is a linear-element formula and a 20-node element's
** highest free eigenfrequency lies far above it. See testfiles/marmot/GCDPNEDExplicit.
*solver, solver=NEDParallel, name=theSolver
{fields:}
courant-number=0.1
output-frequency={outputFrequency:}"""

PARABOLIC_FIELDS = """second-order-fields="displacement"
first-order-fields="nonlocal damage\""""

HYPERBOLIC_FIELDS = """second-order-fields="displacement, nonlocal damage"
micro-inertia-fields="nonlocal damage\""""


def microInertiaProperty(nonlocalViscosity):
    """The named element property that makes the non-local field second order in time."""

    return """*elementProperty, elSet=gen_all, propertyName=nonlocal micro inertia
{:.6e}""".format(
        microInertia(nonlocalViscosity)
    )


BULK_VISCOSITY_PROPERTY = """*elementProperty, elSet=gen_all, propertyName=bulk viscosity
{:}, {:}""".format(
    *BULK_VISCOSITY
)

#: The restraint, shared by every variant. Quarter-symmetry rather than a fully fixed left face:
#: x on the left face, y on the bottom, z on the front. It is the same rigid-body restraint, but it
#: leaves the loaded and supported faces free to contract laterally. Fixing all three components
#: over the whole left face suppresses the Poisson contraction exactly where the reaction is
#: measured, which raised the damage driving force there; with this restraint the implicit run
#: cracks where the imperfection is and converges in 116 increments instead of 2085 and a failure.
RESTRAINT = """>>dirichlet, name=leftX, nSet=gen_left, field=displacement, 1=0.0
>>dirichlet, name=bottomY, nSet=gen_bottom, field=displacement, 2=0.0
>>dirichlet, name=frontZ, nSet=gen_front, field=displacement, 3=0.0
>>dirichlet, name=right, nSet=gen_right, field=displacement, 1={elongation:}"""

#: maxInc is 1e-3 rather than 1e-2 because of the limit point. At 1e-2 the nX = 80 mesh walked into
#: the peak with a step it could not recover from: six "GCDPModel::computeStress: Return mapping
#: failed: all yield-surface" in a row, then "Incrementation failed", 8 increments in and 99 % of the
#: peak force still standing. The other three meshes cleared the same schedule, so this is the
#: return mapping at the limit point rather than anything mesh-specific in the setup. It costs
#: nothing to be careful here: the implicit leg runs in about a minute per mesh either way, and its
#: step length is pseudo-time -- MEASURED invariant, G_f = 0.098400 and F_max = 66.7378 at both 1 s
#: and 1e-2 s, so nothing about the answer depends on how finely this path is walked.
IMPLICIT_STEP = """*step, solver=theSolver
maxInc=1e-3, minInc=1e-8, maxNumInc=100000, maxIter=25, stepLength={stepLength:.6e}
>>options, name=theSolver, extrapolation=linear
{restraint:}"""

EXPLICIT_STEP = """*step, type=adaptiveForExplicitSimulations, solver=theSolver
maxInc=1, minInc=1e-14, maxNumInc=200000000, maxIter=25, stepLength={stepLength:.6e}
{restraint:}"""


#: The explicit variants: name -> (field declarations, element properties).
def explicitVariants(nonlocalViscosity):
    """The explicit variants: name -> (field declarations, element properties)."""

    microInertiaBlock = microInertiaProperty(nonlocalViscosity)

    return {
        "explicit-parabolic": (PARABOLIC_FIELDS, []),
        "explicit-hyperbolic": (HYPERBOLIC_FIELDS, [microInertiaBlock]),
        "explicit-parabolic-bv": (PARABOLIC_FIELDS, [BULK_VISCOSITY_PROPERTY]),
        "explicit-hyperbolic-bv": (HYPERBOLIC_FIELDS, [microInertiaBlock, BULK_VISCOSITY_PROPERTY]),
    }


#: Every variant, in the order the table reports them. Available without a mesh, which the solver
#: blocks are not.
VARIANT_NAMES = ("implicit",) + tuple(explicitVariants(NONLOCAL_VISCOSITY))


def stableIncrement(nX, massScaling):
    """Estimate of the stable time increment, before a run starts.

    Anchored on the measured value at nX = 40. The hyperbolic limit falls off linearly in h, and
    mass scaling raises it with the square root of the factor because it lowers every wave speed by
    the same amount.
    """

    return DT_AT_NX40 * (40.0 / nX) * massScaling**0.5


def explicitIncrements(nX, massScaling):
    """Estimate of the number of increments one explicit run will take."""

    return EXPLICIT_STEP_LENGTH / stableIncrement(nX, massScaling)


def explicitOutputFrequency(nX, massScaling):
    """The output cadence that samples one explicit run about EXPLICIT_OUTPUT_SAMPLES times."""

    return max(1, int(round(explicitIncrements(nX, massScaling) / EXPLICIT_OUTPUT_SAMPLES)))


def endZones(nX):
    """The element ranges within one non-local length of either end, and the section that
    strengthens them.

    Element labels run 1..nX along the bar for this mesh, so the zones are the leading and trailing
    ranges. At nX = 20 one element already IS the non-local length.
    """

    perZone = max(1, int(round(NONLOCAL_LENGTH / (BAR_LENGTH / nX))))

    return """*elSet, elSet=gen_endZones, generate=True
1, {:}, 1

*elSet, elSet=gen_endZoneRight, generate=True
{:}, {:}, 1

*section, name=sectionEnds, material=myMaterialStrong, type=solid
gen_endZones
gen_endZoneRight""".format(
        perZone, nX - perZone + 1, nX
    )


def variants(nX, massScaling=MASS_SCALING, nonlocalViscosity=NONLOCAL_VISCOSITY):
    """The variant matrix for one mesh: name -> (solver block, element properties, step block)."""

    restraint = RESTRAINT.format(elongation=TOTAL_ELONGATION)

    matrix = {
        "implicit": (
            IMPLICIT_SOLVER,
            "",
            IMPLICIT_STEP.format(stepLength=EXPLICIT_STEP_LENGTH, restraint=restraint),
        )
    }

    for name, (fields, properties) in explicitVariants(nonlocalViscosity).items():
        matrix[name] = (
            EXPLICIT_SOLVER.format(fields=fields, outputFrequency=explicitOutputFrequency(nX, massScaling)),
            "\n\n".join(properties),
            EXPLICIT_STEP.format(stepLength=EXPLICIT_STEP_LENGTH, restraint=restraint),
        )

    return matrix


def deckFor(variant, nX, massScaling=MASS_SCALING, nonlocalViscosity=NONLOCAL_VISCOSITY):
    """Substitute the template's tokens for one variant and mesh."""

    with open(os.path.join(HERE, "template.inp")) as templateFile:
        deck = templateFile.read()

    solverBlock, elementProperties, stepBlock = variants(nX, massScaling, nonlocalViscosity)[variant]

    # The implicit variant has no inertia at all, so mass scaling cannot reach it -- and must not
    # appear in its deck either, or an implicit run would be "reused" for one scaling and re-run for
    # the next while computing exactly the same thing.
    density = DENSITY * (1.0 if variant == "implicit" else massScaling)

    deck = deck.replace("__NX__", str(nX))
    deck = deck.replace("__DENSITY__", "{:.6e}".format(density))
    deck = deck.replace("__NONLOCAL_VISCOSITY__", "{:.6e}".format(nonlocalViscosity))
    deck = deck.replace("__END_ZONES__", endZones(nX))
    deck = deck.replace("__SOLVER_BLOCK__", solverBlock)
    deck = deck.replace("__ELEMENT_PROPERTIES__", elementProperties)
    deck = deck.replace("__STEP_BLOCK__", stepBlock)

    return deck


def run(variant, nX, threads, massScaling=MASS_SCALING, nonlocalViscosity=NONLOCAL_VISCOSITY):
    """Run one case, or reuse a completed one, and return its run directory.

    A completed run is reused only if it was produced by the deck this script writes *now*. A
    finished RF.csv alone is not enough: an earlier version of this study sampled its histories too
    sparsely to see the peak force, and a reuse check that looked only for the file would have
    carried those numbers into every table since. The deck is the run's identity.
    """

    # A non-default eta or mass scaling gets its own directory. Without that the deck check would
    # bounce the same case between two configurations, re-running and discarding it each time.
    suffix = ""
    if variant != "implicit":
        if nonlocalViscosity != NONLOCAL_VISCOSITY:
            suffix += "_eta{:.0e}".format(nonlocalViscosity)
        if massScaling != MASS_SCALING:
            suffix += "_ms{:.0e}".format(massScaling)

    runDir = os.path.join(HERE, "run_{:}_nX{:}{:}".format(variant, nX, suffix))
    deckPath = os.path.join(runDir, "test.inp")
    deck = deckFor(variant, nX, massScaling, nonlocalViscosity)

    if os.path.exists(os.path.join(runDir, "RF.csv")):
        if os.path.exists(deckPath) and open(deckPath).read() == deck:
            print("  reusing {:}".format(os.path.basename(runDir)))
            return runDir
        print("  re-running {:}: its deck is not the one this script writes".format(os.path.basename(runDir)))

    shutil.rmtree(runDir, ignore_errors=True)
    os.makedirs(runDir)
    with open(deckPath, "w") as deckFile:
        deckFile.write(deck)

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

    # An implicit run that runs out of increment reductions is neither of those: it stops, says so,
    # and leaves behind a perfectly well-formed history of the part of the step it did manage. The
    # first run of this study reported one of those as "ok" and compared its truncated G_f against
    # runs that had gone the whole way. Both halves are needed -- the message, and the elongation
    # actually reached -- because a run killed from outside writes no message at all.
    row["failed"] = "Simulation failed" in log

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
    row["samples"] = 0
    row["endElongation"] = float("nan")

    if os.path.exists(rfPath) and os.path.exists(uPath):
        reaction = np.atleast_2d(np.loadtxt(rfPath))
        displacement = np.atleast_2d(np.loadtxt(uPath))

        force = np.abs(reaction[:, 1])
        elongation = displacement[:, 1]
        # Reported, because a run sampled too sparsely to see its own peak still produces a
        # plausible-looking F_max and G_f. Anything in the low tens is not a measurement.
        row["samples"] = int(force.size)
        row["endElongation"] = float(elongation[-1])

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


#: Fraction of the prescribed elongation a run has to reach for its G_f to be the whole integral.
COMPLETENESS = 0.999


def status(row):
    """One word on whether the row can be believed."""

    if row["diverged"] or not np.isfinite(row["peakForce"]):
        return "DIVERGED"
    if not row["increments"]:
        return "FAILED"
    if row["failed"] or not row["endElongation"] >= COMPLETENESS * TOTAL_ELONGATION:
        return "INCOMPLETE"
    if row["energyCreated"]:
        return "energy!"
    return "ok"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meshes", default="20,40,80,160", help="comma-separated nX values")
    parser.add_argument("--variants", default=",".join(VARIANT_NAMES), help="comma-separated variant names")
    parser.add_argument("--quick", action="store_true", help="coarse meshes only")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--mass-scaling",
        type=float,
        default=MASS_SCALING,
        dest="massScaling",
        help=(
            "multiply the density of the EXPLICIT variants by this factor (default {:g}). The "
            "stable increment grows with its square root, so the default buys a factor of 100 in "
            "run time. Legitimate only while the run stays quasi-static, which the T/W_ext column "
            "measures rather than assumes; the implicit variant has no inertia and is never "
            "scaled. Pass 1 for the unscaled physics and a fortnight of compute.".format(MASS_SCALING)
        ),
    )
    parser.add_argument(
        "--nonlocal-viscosity",
        type=float,
        default=NONLOCAL_VISCOSITY,
        dest="nonlocalViscosity",
        help=(
            "non-local viscosity eta in seconds (default {:g}); the micro-inertia follows as "
            "eta^2/4. With --mass-scaling this is what sets the increment: eta raises the non-local "
            "limit, mass scaling the mechanical CFL, and dt is the smaller of the two. See "
            "mechanicallyDominatedViscosity() for the value that keeps the mechanical problem in "
            "charge -- and note the floor rises as sqrt(mass scaling).".format(NONLOCAL_VISCOSITY)
        ),
    )
    parser.add_argument(
        "--estimate",
        action="store_true",
        help="print the increment counts the selected runs would take, and run nothing",
    )
    arguments = parser.parse_args()

    if arguments.massScaling <= 0.0:
        raise SystemExit("--mass-scaling must be positive")

    meshes = [20, 40] if arguments.quick else [int(n) for n in arguments.meshes.split(",")]
    selected = [name.strip() for name in arguments.variants.split(",")]

    for name in selected:
        if name not in VARIANT_NAMES:
            raise SystemExit("unknown variant {:}; known: {:}".format(name, ", ".join(VARIANT_NAMES)))

    if arguments.estimate:
        floor = mechanicallyDominatedViscosity(arguments.massScaling)
        print(
            "step length {:} s, mass scaling {:g} (dt x {:.3g}), eta {:g} s (m_k {:.4e} s^2)".format(
                EXPLICIT_STEP_LENGTH,
                arguments.massScaling,
                arguments.massScaling**0.5,
                arguments.nonlocalViscosity,
                microInertia(arguments.nonlocalViscosity),
            )
        )
        print(
            "eta floor for a mechanically dominated increment: {:.3e} s -- {:}\n".format(
                floor,
                (
                    "SATISFIED, ratio {:.2f}".format(arguments.nonlocalViscosity / floor)
                    if arguments.nonlocalViscosity >= floor
                    else "NOT satisfied, the non-local limit will bind and the scaling is wasted"
                ),
            )
        )
        print("{:>5} {:>10} {:>14} {:>12}".format("nX", "dt [s]", "increments", "output every"))
        for nX in meshes:
            print(
                "{:>5} {:>10.4e} {:>14,.0f} {:>12}".format(
                    nX,
                    stableIncrement(nX, arguments.massScaling),
                    explicitIncrements(nX, arguments.massScaling),
                    explicitOutputFrequency(nX, arguments.massScaling),
                )
            )
        print("\nThe estimate is for the hyperbolic scheme; the parabolic one is finer on fine meshes.")
        return 0

    rows = {}
    for variant in selected:
        print("{:}:".format(variant))
        for nX in meshes:
            runDir = run(variant, nX, arguments.threads, arguments.massScaling, arguments.nonlocalViscosity)
            rows[(variant, nX)] = summarise(runDir, variant, nX)

    header = "{:<24} {:>5} {:>7} {:>11} {:>10} {:>7} {:>9} {:>10} {:>7} {:>7} {:>9}".format(
        "variant",
        "nX",
        "h [mm]",
        "dt [s]",
        "increments",
        "samples",
        "F_max [N]",
        "G_f [N/mm]",
        "F_end",
        "T/W_ext",
        "status",
    )
    print("\n" + header)
    print("-" * len(header))

    for nX in meshes:
        for variant in selected:
            row = rows[(variant, nX)]
            print(
                "{:<24} {:>5} {:>7.3f} {:>11} {:>10} {:>7} {:>9} {:>10} {:>7} {:>7} {:>9}".format(
                    row["variant"],
                    row["nX"],
                    row["h"],
                    "{:.4e}".format(row["dt"]) if np.isfinite(row["dt"]) else "-",
                    row["increments"] or "-",
                    row["samples"] or "-",
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
        "bulk viscosity b1 = {:}, b2 = {:}; every variant pulled to {:} mm over {:g} s".format(
            microInertia(arguments.nonlocalViscosity),
            arguments.nonlocalViscosity,
            2 * NONLOCAL_LENGTH / arguments.nonlocalViscosity,
            *BULK_VISCOSITY,
            TOTAL_ELONGATION,
            EXPLICIT_STEP_LENGTH,
        )
    )
    print(
        "End zones of {:g} mm strengthened by {:g}x so that the crack forms at the weakened central "
        "element and not at a boundary.".format(NONLOCAL_LENGTH, END_ZONE_STRENGTH_FACTOR)
    )
    if arguments.massScaling != 1.0:
        print(
            "Density of the explicit variants mass scaled by {:g} (dt x {:.3g}). Read T/W_ext "
            "before believing any of it.".format(arguments.massScaling, arguments.massScaling**0.5)
        )
    incomplete = [row for row in rows.values() if status(row) == "INCOMPLETE"]
    if incomplete:
        print(
            "\nSTOPPED SHORT of the prescribed {:} mm, so G_f is an integral over part of the".format(TOTAL_ELONGATION)
        )
        print("history and is not comparable with the rows that finished:")
        for row in incomplete:
            print(
                "  {:}_nX{:} reached {:.4f} mm ({:.0%}), force there {:.1%} of peak".format(
                    row["variant"],
                    row["nX"],
                    row["endElongation"],
                    row["endElongation"] / TOTAL_ELONGATION,
                    row["finalForceShare"],
                )
            )
        print()

    print("F_end is the force still standing at the endpoint, as a share of peak: G_f has measured")
    print("all of the softening only where that is small. T/W_ext is the kinetic share, i.e. how")
    print("quasi-static the explicit run actually was.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
