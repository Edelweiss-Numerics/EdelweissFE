"""Dissipation control (``stepactions/dissipationcontrol.py``) through snap-backs, across a refinement and a restart,
and its edge cases.

The decks ``testfiles/marmot/DissipationControlSnapBacks`` (with a displacement limit) and
``DissipationControlSnapBacksThreshold`` (load parameter control up to a dissipation threshold) pull a bar of plane
strain elements in uniaxial strain. Three weak elements soften and harden again, one after the other, so that the
load-displacement path snaps back three times; an indirect displacement control would have to switch its controlled
element at every snap-back. Every element is in a homogeneous state, so that the path is known in closed form: for
von Mises plasticity in uniaxial strain (Poisson's ratio zero), an element yields at the axial stress
:math:`s = f_y(\\kappa) + E \\kappa / 2` with the hardening variable :math:`\\kappa`, which is its axial plastic strain,
and the end displacement is :math:`u = L s / E + h \\sum \\kappa`. Each run is checked against this path, increment by
increment against its constraint, and for the three snap-backs.
"""

from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy.optimize import brentq

from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.stepactions.dissipationcontrol import StepAction as DissipationControl
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.inputfileparser import parseInputFile

TESTFILES = Path(__file__).parents[1] / "testfiles" / "marmot"

TOLERANCE = DissipationControl.constraintTolerance


def _simulate(deck: Path):
    """The parsed deck, which the simulation consumes, and the model and the field outputs after the run."""
    model, fieldOutputController = finiteElementSimulation(parseInputFile(str(deck)), verbose=False, suppressPlots=True)
    return parseInputFile(str(deck)), model, fieldOutputController


def _history(fieldOutputController, name: str) -> np.ndarray:
    return np.asarray(fieldOutputController.fieldOutputs[name].getResultHistory())


def _controllerOptions(inputFile) -> dict:
    return inputFile["step"][0]["moduleoptions"]["dissipationcontrol"][0]


class _AnalyticalBar:
    """The equilibrium path of the bar: a sequence of phases, in which either one element softens (all others unload
    elastically), or all elements at their yield stress harden together."""

    def __init__(self, inputFile, model):
        materials = {m["id"]: np.array(m["datalines"][0].split(","), dtype=float) for m in inputFile["material"]}
        self.E = next(iter(materials.values()))[0]
        assert all(p[0] == self.E and p[1] == 0.0 for p in materials.values()), "the solution needs one E and nu = 0"

        # yield stress, linear hardening, exponential softening and its rate, element by element
        self.elements = [
            materials[section["material"]][2:6]
            for section in inputFile["section"]
            for setName in section["datalines"]
            for _ in model.elementSets[setName]
        ]
        x = np.array([node.coordinates for node in model.nodes.values()])
        self.L = np.ptp(x[:, 0])
        self.h = self.L / len(self.elements)
        self.A = np.ptp(x[:, 1]) * float(inputFile["section"][0]["thickness"])

    def yieldStress(self, p, kappa):
        return p[0] + p[1] * kappa + p[2] * (1 - np.exp(-p[3] * kappa)) + 0.5 * self.E * kappa

    def hardeningModulus(self, p, kappa):
        return p[1] + p[2] * p[3] * np.exp(-p[3] * kappa) + 0.5 * self.E

    def path(self, sEnd: float, n: int = 2000) -> np.ndarray:
        """Points (end displacement, reaction force) of the path up to the axial stress sEnd."""

        kappa = np.zeros(len(self.elements))
        s = 0.0
        points = [(0.0, 0.0)]

        def record():
            points.append((self.L * s / self.E + self.h * kappa.sum(), self.A * s))

        while True:
            softening = [i for i, p in enumerate(self.elements) if self.hardeningModulus(p, kappa[i]) < 0]
            nextPeak = min((self.yieldStress(self.elements[i], kappa[i]) for i in softening), default=np.inf)

            # all elements at their yield stress harden, until the next softening element yields
            for sTrial in np.linspace(s, min(nextPeak, sEnd), n)[1:]:
                for i, p in enumerate(self.elements):
                    if i not in softening and self.yieldStress(p, kappa[i]) < sTrial:
                        kappa[i] = brentq(lambda k: self.yieldStress(p, k) - sTrial, kappa[i], 1.0)
                s = sTrial
                record()
            if nextPeak > sEnd:
                return np.array(points)

            # it softens alone, down to the minimum of its yield stress
            j = min(softening, key=lambda i: self.yieldStress(self.elements[i], kappa[i]))
            p = self.elements[j]
            kappaMin = brentq(lambda k: self.hardeningModulus(p, k), kappa[j], 1.0)
            for k in np.linspace(kappa[j], kappaMin, n)[1:]:
                kappa[j] = k
                s = self.yieldStress(p, k)
                record()


def _distanceToPath(points: np.ndarray, path: np.ndarray) -> np.ndarray:
    """The distance of every point to the polyline, both scaled by the extent of the polyline."""

    scale = np.abs(path).max(axis=0)
    a, b = path[:-1] / scale, path[1:] / scale
    ab = b - a
    distances = []
    for p in points / scale:
        t = np.clip(((p - a) * ab).sum(axis=1) / np.maximum((ab * ab).sum(axis=1), 1e-300), 0.0, 1.0)
        distances.append(np.linalg.norm(a + t[:, None] * ab - p, axis=1).min())
    return np.array(distances)


def _numberOfSnapBacks(u: np.ndarray) -> int:
    """The number of turns of the end displacement from growing to shrinking."""

    du = np.diff(u)
    du = du[np.abs(du) > 1e-9 * np.abs(u).max()]
    return int(np.sum((du[:-1] > 0) & (du[1:] < 0)))


def _nonzeroIncrements(fieldOutputController):
    """Step progress increments and the histories at their ends, without the zero increments."""

    t = np.ravel(fieldOutputController.fieldOutputs["D"].getTimeHistory())
    D = np.ravel(_history(fieldOutputController, "D"))
    U = _history(fieldOutputController, "U").reshape(len(t), -1)
    u = np.ravel(_history(fieldOutputController, "u"))
    nonzero = np.flatnonzero(np.diff(t) > 0.0) + 1
    return (
        np.diff(t)[nonzero - 1],
        np.diff(D)[nonzero - 1],
        D[nonzero - 1],
        U[nonzero] - U[nonzero - 1],
        np.diff(u)[nonzero - 1],
    )


@pytest.fixture(scope="module", params=["DissipationControlSnapBacks", "DissipationControlSnapBacksThreshold"])
def snapBackRun(request):
    return _simulate(TESTFILES / request.param / "test.inp")


def test_the_path_through_three_snap_backs_is_the_analytical_one(snapBackRun):
    inputFile, model, fieldOutputController = snapBackRun
    u = np.ravel(_history(fieldOutputController, "u"))
    F = np.ravel(_history(fieldOutputController, "F"))

    assert _numberOfSnapBacks(u) == 3
    # past the minimum of the third snap-back, back on a hardening branch
    assert u[-1] > u[-2]

    path = _AnalyticalBar(inputFile, model).path(sEnd=1.3)
    assert _numberOfSnapBacks(path[:, 0]) == 3
    # backward Euler leaves the peaks a little early: within 1 % of the extent of the path
    assert _distanceToPath(np.c_[u, F], path).max() < 1e-2


def test_every_increment_meets_its_constraint(snapBackRun):
    inputFile, model, fieldOutputController = snapBackRun
    options = _controllerOptions(inputFile)
    dp, dD, DAtStart, dU, du = _nonzeroIncrements(fieldOutputController)
    dTau = dp * options["dissipation"]

    onDissipationTarget = np.abs(dD - dTau) <= TOLERANCE * dTau

    if options["displacement"] is not None:
        dUMax = dp * options["displacement"]
        dUNorm = np.abs(dU).max(axis=1)
        onDissipationTarget &= dUNorm <= (1 + TOLERANCE) * dUMax
        onDisplacementLimit = (np.abs(dUNorm - dUMax) <= TOLERANCE * dUMax) & (dD <= (1 + TOLERANCE) * dTau)
        assert np.all(onDissipationTarget | onDisplacementLimit)
    else:
        # the load parameter scales the prescribed end displacement of 1.0
        loadControlled = DAtStart <= options["threshold"]
        assert np.allclose(du[loadControlled], dp[loadControlled] * options["loadparameter"], rtol=1e-10)
        assert np.all(onDissipationTarget[~loadControlled])
        assert loadControlled[0] and not loadControlled[-1]

    # most of the path is controlled by the dissipation
    assert onDissipationTarget.sum() > 40


def test_restart_resumes_the_dissipation_control(tmp_path):
    """A run interrupted in the second snap-back and resumed from its checkpoint ends as the uninterrupted one: the
    controller carries the dissipation at the increment start and the last increment (its predictor and its forward
    direction) in the checkpoint."""

    deck = (TESTFILES / "DissipationControlSnapBacks" / "test.inp").read_text()
    step = deck.index("*step")
    checkpointBaseName = tmp_path / "ckpt"

    def write(name, text):
        (tmp_path / name).write_text(text)
        return tmp_path / name

    _, reference, _ = _simulate(write("full.inp", deck))

    restartOutput = f"*output, type=restart, name=restart\nwriteInterval=1, baseName={checkpointBaseName}\n\n"
    truncated = deck[:step] + restartOutput + deck[step:].replace("maxNumInc=1000", "maxNumInc=60")
    _simulate(write("truncated.inp", truncated))

    def checkpointTime(path):
        with h5py.File(path, "r") as f:
            return f.attrs["time"]

    checkpoint = max(tmp_path.glob("ckpt_*.h5"), key=checkpointTime)
    _, resumed, _ = _simulate(write("resumed.inp", deck[:step] + f"*restart, readFrom={checkpoint}\n\n" + deck[step:]))

    np.testing.assert_allclose(
        resumed.nodeFields["displacement"]["U"], reference.nodeFields["displacement"]["U"], rtol=1e-8, atol=1e-12
    )


def test_refinement_rebuilds_the_buffers_and_rebases_the_dissipation():
    """The refinement changes the controlled element set; afterwards, every increment dissipates its share again.
    Without the dissipation at the increment start taken from the transferred state, the increment after the
    refinement cannot meet its target."""

    inputFile, model, fieldOutputController = _simulate(TESTFILES / "AMR_DissipationControl" / "test.inp")
    options = _controllerOptions(inputFile)
    assert len(model.elements) > 2, "the stress marker never refined"

    t = np.ravel(fieldOutputController.fieldOutputs["D"].getTimeHistory())
    D = np.ravel(_history(fieldOutputController, "D"))
    assert t[-1] == pytest.approx(1.0)

    nonzero = np.flatnonzero(np.diff(t) > 0.0)
    dTau = np.diff(t)[nonzero] * options["dissipation"]
    dD = np.diff(D)[nonzero]
    # the field output counts the change of the summed dissipation by the state transfer to the refinement's increment
    overshooting = dD > (1 + TOLERANCE) * dTau
    assert overshooting.sum() <= 1
    assert np.sum(np.abs(dD - dTau) <= TOLERANCE * dTau) > 10


# The edge cases, with stand-ins for the elements and the model


class _Element:
    def __init__(self, label: int, nDof: int = 2, dissipation: float = 0.0, providesEnergyGradients: bool = True):
        self.elNumber = label
        self.elType = "stand-in"
        self.nDof = nDof
        self.dissipation = np.array([dissipation])
        self.providesEnergyGradients = providesEnergyGradients
        self.assigned = None

    def getNumberOfQuadraturePoints(self):
        return 1

    def getResultArray(self, result, quadraturePoint, getPersistentView=True):
        assert result == "dissipation"
        return self.dissipation

    def assignEnergyGradientOutput(self, dElasticEnergy_dQ, dDissipation_dQ):
        self.assigned = dDissipation_dQ


class _ElementSet(list):
    name = "controlled"
    _version = 0


class _Topology:
    version = 0

    def registerMeshDependent(self, consumer):
        pass


class _Model:
    topology = _Topology()


class _System:
    locations = None
    numbering = None


class _Journal:
    def message(self, *args, **kwargs):
        pass


def _controller(elements, **options):
    return DissipationControl(
        "control", _ElementSet(elements), options.pop("dissipation", 1.0), _Model(), _Journal(), **options
    )


def _timeStep(stepProgressIncrement):
    return TimeStep(1, stepProgressIncrement, 0.5, stepProgressIncrement, 0.5, 0.5)


def test_an_element_without_energy_gradients_is_rejected_at_setup():
    with pytest.raises(ValueError, match="element 7 .* provides no energy gradients"):
        _controller([_Element(1), _Element(7, providesEnergyGradients=False)], displacement=1.0)


@pytest.mark.parametrize(
    "options, message",
    [
        (dict(dissipation=0.0, displacement=1.0), "dissipation must be positive"),
        (dict(displacement=-1.0), "displacement must be positive"),
        (dict(), "either displacement or loadParameter"),
    ],
)
def test_invalid_options_are_rejected_at_setup(options, message):
    with pytest.raises(ValueError, match=message):
        _controller([_Element(1)], **options)


def test_the_checkpoint_carries_the_state_between_increments():
    """The resumed run above takes the dissipation at the start again in the zero increment it starts with, and its
    increments are as large as the restored time stepper says; whatever else the controller carries from one
    increment to the next must be in the checkpoint."""

    controller = _controller([_Element(1)], displacement=1.0)
    controller.D_n, controller.dLambdaLast, controller.dDLast, controller.dUNormLast = 1.0, 0.1, 0.2, 0.3
    controller.dULast, controller.dULastKeys = np.array([0.3, -0.1]), np.array([[1, 0], [1, 1]])

    resumed = _controller([_Element(1)], displacement=1.0)
    resumed.setRestartData(controller.getRestartData())

    for name in ("D_n", "dLambdaLast", "dDLast", "dUNormLast", "dULast", "dULastKeys"):
        np.testing.assert_array_equal(getattr(resumed, name), getattr(controller, name), err_msg=name)


def test_the_base_element_provides_no_energy_gradients():
    from edelweissfe.config.elementlibrary import getElementClass

    element = getElementClass("CPE4", "edelweiss")("CPE4", 1)
    assert not element.providesEnergyGradients
    with pytest.raises(NotImplementedError, match="element 1"):
        element.assignEnergyGradientOutput(np.zeros(8), np.zeros(8))


def test_the_gradient_output_is_assigned_only_while_controlling():
    elements = [_Element(1), _Element(2)]
    controller = _controller(elements, displacement=1.0)
    assert all(el.assigned is None for el in elements)

    controller.beginControl(_Model())
    assert all(el.assigned is not None and el.assigned.size == el.nDof for el in elements)

    controller.endControl(_Model())
    assert all(el.assigned is None for el in elements)


def test_a_zero_increment_rebases_the_dissipation_and_keeps_the_predictor():
    element = _Element(1, dissipation=1.0)
    controller = _controller([element], displacement=1.0)
    controller.beginControl(_Model())
    controller.D_n = 1.0
    controller.dDLast, controller.dLambdaLast = 0.1, 0.2

    # plastic flow in the re-equilibration after a model change
    element.dissipation[0] = 1.5
    controller.finishIncrement(None, None, 0.0, _timeStep(0.0), _System())

    assert controller.D_n == 1.5
    assert (controller.dDLast, controller.dLambdaLast) == (0.1, 0.2)


def test_an_elastic_iterate_without_a_secant_follows_the_load_parameter():
    """Dissipation controlled (without displacement), an elastic iterate fixes no load parameter, and without a last
    increment that dissipated, there is no secant either: the load parameter rule moves the iterate instead of a
    division by the vanishing dissipation rate."""

    element = _Element(1, dissipation=1.0)
    controller = _controller([element], loadParameter=2.0, threshold=0.5)
    controller.beginControl(_Model())
    controller.D_n = 1.0
    controller.F = 1.0
    controller.dLambda = 0.0
    assert controller.isDissipationControlled() and controller.dDLast == 0.0

    # elastic: no dissipation gradient
    controller._linearization = (1.0, np.zeros(2))
    ddLambda = controller.computeDDLambda(np.zeros(2), np.zeros(2), np.ones(2), _timeStep(0.1), _System())

    assert ddLambda == pytest.approx(0.1 * 2.0)
