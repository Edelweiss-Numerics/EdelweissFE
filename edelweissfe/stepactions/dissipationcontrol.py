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
Dissipation controller for the NISTPArcLength solver (Gutiérrez 2004; Verhoosel, Remmers, Gutiérrez 2009).
"""

from dataclasses import dataclass

import numpy as np

from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.numerics.dofmanager import DofManager, DofVector
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.stepactions.base.arclengthcontrollerbase import ArcLengthControllerBase
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.caseinsensitivedict import CaseInsensitiveDict
from edelweissfe.utils.misc import withoutParserBookkeepingKeys
from edelweissfe.utils.schema import buildSchemaFromOptions, schemaField


@dataclass(frozen=True)
class DissipationControlSchema:
    """The scalar options of the ``dissipationcontrol`` keyword.

    ``elSet`` names an existing model object and is resolved by :meth:`StepAction.fromStepActionDefinition`
    before the schema is built; it is declared here only to document the grammar, as is ``name``.
    """

    name: str | None = schemaField(
        description="Name of the step action, by which the solver option arcLengthController selects it.",
        dtype=str,
        default=None,
        required=True,
        structuralOnly=True,
    )
    elSet: str | None = schemaField(
        description="Element set whose quadrature point result 'dissipation' is summed.",
        dtype=str,
        default="all",
        structuralOnly=True,
    )
    dissipation: float | None = schemaField(
        description="Energy to be dissipated over the step.",
        dtype=float,
        default=None,
        required=True,
    )
    displacement: float | None = schemaField(
        description="Maximum nodal displacement increment over the step; enables the combined constraint.",
        dtype=float,
        default=None,
    )
    loadParameter: float | None = schemaField(
        description="Load parameter increment over the step before dissipating (without displacement).",
        dtype=float,
        default=None,
    )
    threshold: float | None = schemaField(
        description="Dissipated energy beyond which the dissipation controlled phase starts (without displacement).",
        dtype=float,
        default=0.0,
    )


class StepAction(ArcLengthControllerBase):
    """Dissipation controller for the NISTPArcLength solver, for loading by prescribed displacements
    (Dirichlet BCs scaled by the load parameter).

    The dissipation constraint ``D(u) - D_n = dTau`` uses the dissipation ``D`` summed from the quadrature point
    result ``dissipation`` over ``elSet``, and is linearized consistently with the gradients of the elements'
    dissipation wrt their dofs (``computeEnergyGradients``, which requires a material providing energy tangents).

    With ``displacement``, an increment ends where the first of the dissipation increment ``dTau`` and the maximum
    nodal displacement increment ``dUMax`` is reached. The displacement limit governs the elastic range, where the
    dissipation constraint is singular, and the final failure, where the dissipation hardly constrains the
    deformation anymore; the dissipation selects the forward direction along the path. Without ``displacement``,
    the load parameter is controlled directly until the dissipated energy exceeds ``threshold``.

    Parameters
    ----------
    name
        The name of this step action.
    elSet
        The element set whose dissipation is summed.
    dissipation
        The energy to be dissipated over the step.
    model
        The model tree.
    journal
        The journal object for logging.
    displacement
        The maximum nodal displacement increment over the step.
    loadParameter
        The load parameter increment over the step in the load parameter controlled phase (without displacement).
    threshold
        The dissipated energy beyond which the dissipation controlled phase starts (without displacement).
    """

    identification = "DissipationControl"

    #: Option schema for this step action, consumed by OptionSchemaProvider's registry.
    schema = DissipationControlSchema

    #: Relative tolerance of the increments of dissipation and displacement.
    constraintTolerance = 1e-3

    #: Maximum ratio of predicted and last increments, against a nearly vanishing last increment.
    maxPredictorRatio = 10.0

    #: A zero prediction after a cutback may follow an elastic unloading branch.
    extrapolatesAfterCutback = True

    def __init__(
        self,
        name: str,
        elSet: ElementSet,
        dissipation: float,
        model: FEModel,
        journal: Journal,
        displacement: float = None,
        loadParameter: float = None,
        threshold: float = 0.0,
    ):
        self.name = name
        self.journal = journal
        self.model = model

        # converged dissipation at the start of the increment
        self.D_n = None
        # current reaction conjugate to the load parameter, for the log
        self.F = None
        self.dLambda = 0.0
        # dissipation of the iterate and its gradient, see _linearizationOfIterate
        self._linearization = None
        # load parameter, dissipation and displacement increments of the last converged increment
        self.dLambdaLast = 0.0
        self.dDLast = 0.0
        self.dUNormLast = 0.0
        self.dULast = None

        self.updateStepAction(elSet, dissipation, displacement, loadParameter, threshold)

    @classmethod
    def fromStepActionDefinition(cls, name, definition, jobInfo, model, fieldOutputController, journal):
        """Build this controller from a parsed ``>>dissipationcontrol`` definition."""

        definition = CaseInsensitiveDict(withoutParserBookkeepingKeys(definition))
        definition.pop("name", None)
        elSet = model.elementSets[definition.pop("elSet", "all")]
        configuration = buildSchemaFromOptions(cls.schema, definition)

        return cls(
            name,
            elSet,
            configuration.dissipation,
            model,
            journal,
            displacement=configuration.displacement,
            loadParameter=configuration.loadParameter,
            threshold=configuration.threshold,
        )

    def updateStepActionFromDefinition(self, definition, jobInfo, model, fieldOutputController, journal):
        """Update from a parsed ``>>dissipationcontrol`` definition re-declared in a later step."""

        definition = CaseInsensitiveDict(withoutParserBookkeepingKeys(definition))
        definition.pop("name", None)
        elSet = model.elementSets[definition.pop("elSet", "all")]
        configuration = buildSchemaFromOptions(self.schema, definition)

        self.updateStepAction(
            elSet,
            configuration.dissipation,
            configuration.displacement,
            configuration.loadParameter,
            configuration.threshold,
        )

    def updateStepAction(
        self, elSet: ElementSet, dissipation: float, displacement: float, loadParameter: float, threshold: float
    ):
        """Control a new element set and new targets.

        Parameters
        ----------
        elSet
            The element set whose dissipation is summed.
        dissipation
            The energy to be dissipated over the step.
        displacement
            The maximum nodal displacement increment over the step, or None.
        loadParameter
            The load parameter increment over the step in the load parameter controlled phase (without displacement).
        threshold
            The dissipated energy beyond which the dissipation controlled phase starts (without displacement).
        """

        if displacement is None and loadParameter is None:
            raise ValueError(f"{self.identification}: either displacement or loadParameter is required")

        self.elSet = elSet
        self.dissipation = dissipation
        self.displacement = displacement
        self.loadParameter = loadParameter
        self.threshold = threshold

        # persistent views on the trial state, updated by the elements in every iteration
        self._dissipationViews = [
            el.getResultArray("dissipation", qp, getPersistentView=True)
            for el in elSet
            for qp in range(el.getNumberOfQuadraturePoints())
        ]
        # buffers for the element energy gradients
        self._gradientBuffers = {el: (np.empty(el.nDof), np.empty(el.nDof)) for el in elSet}

    def computeDissipation(self) -> float:
        """The dissipated energy of the current (trial) state."""

        return sum(view[0] for view in self._dissipationViews)

    def computeDissipationGradient(self, dofManager: DofManager) -> np.ndarray:
        """The gradient of the dissipated energy of the current (trial) state wrt the solution vector.

        Interim solution: the elements' gradients are assembled here in a second (serial) pass over the elements,
        after the solver's element loop. Once the energy gradients are an output of the element evaluation in Marmot
        (interface change, see MarmotElement::computeEnergyGradients), they should be assembled in the solver's
        element loop together with the residual.
        """

        gradient = np.zeros(dofManager.nDof)
        for el, (dElasticEnergy_dQ, dDissipation_dQ) in self._gradientBuffers.items():
            el.computeEnergyGradients(dElasticEnergy_dQ, dDissipation_dQ)
            gradient[dofManager.idcsOfElementsInDofVector[el]] += dDissipation_dQ
        return gradient

    @staticmethod
    def _displacementIdcs(dofManager: DofManager) -> np.ndarray:
        """The indices of the displacement field in the solution vector."""

        return dofManager.idcsOfFieldsInDofVector["displacement"]

    def setSystemState(
        self, reaction: DofVector, referenceIncrement: DofVector, loadParameter: float, loadParameterIncrement: float
    ):
        """Receive the state of the iterate from the solver.

        Parameters
        ----------
        reaction
            The internal forces of the elements and the constraints.
        referenceIncrement
            The change of the prescribed degrees of freedom over the step (zero elsewhere).
        loadParameter
            The load parameter at the start of the increment.
        loadParameterIncrement
            The increment of the load parameter of the iterate.
        """

        self.F = referenceIncrement.dot(reaction)
        self.dLambda = loadParameterIncrement
        self._linearization = None

        if self.D_n is None:
            # first call of the step, at the converged start state
            self.D_n = self.computeDissipation()

    def _linearizationOfIterate(self, dofManager: DofManager) -> tuple[float, np.ndarray]:
        """The dissipation of the iterate and its gradient, evaluated in the first call after :meth:`setSystemState`:
        in later calls of the iteration (line search), the elements hold trial states."""

        if self._linearization is None:
            self._linearization = (self.computeDissipation(), self.computeDissipationGradient(dofManager))
        return self._linearization

    def computeDDLambda(self, dU, ddU_0, ddU_f, timeStep: TimeStep, dofManager: DofManager):
        """Compute the increment of the arc length load parameter for the current iteration.

        Parameters
        ----------
        dU
            The current increment of the solution vector.
        ddU_0
            The correction of the solution vector due to the dead and the current reference load.
        ddU_f
            The correction of the solution vector due to the unit reference load.
        timeStep
            The current time step.
        dofManager
            The dof manager of the current equation system.

        Returns
        -------
        float
            The increment of the load parameter.
        """

        if self.displacement is None and not self.isDissipationControlled():
            return timeStep.stepProgressIncrement * self.loadParameter - self.dLambda

        dTau = timeStep.stepProgressIncrement * self.dissipation

        # linearized dissipation increment along ddU = ddU_0 + ddLambda ddU_f: dD_0 + ddLambda dD_f
        D, dD_dU = self._linearizationOfIterate(dofManager)
        dD_0 = D - self.D_n + dD_dU.dot(ddU_0)
        dD_f = dD_dU.dot(ddU_f)

        if self.displacement is None:
            # singular for an elastic iterate, e.g. in the first iteration from a zero increment: secant predictor
            if abs(dD_f) < 1e-6 * abs(self.F) and self.dDLast > 0.0:
                return min((dTau - D + self.D_n) / self.dDLast, self.maxPredictorRatio) * self.dLambdaLast
            return (dTau - dD_0) / dD_f

        # interval of ddLambda within the displacement limit, |a_i + ddLambda b_i| <= dUMax for all nodal dofs
        idcs = self._displacementIdcs(dofManager)
        a = (dU + ddU_0)[idcs]
        b = ddU_f[idcs]
        dUMax = timeStep.stepProgressIncrement * self.displacement

        moving = np.abs(b) > 1e-14 * np.abs(b).max()
        bounds = np.sort(np.stack([(-dUMax - a[moving]) / b[moving], (dUMax - a[moving]) / b[moving]]), axis=0)
        lower, upper = bounds[0].max(), bounds[1].min()
        if lower > upper:
            # the displacement limit cannot be met along ddU_f: closest approach
            return 0.5 * (lower + upper)

        # the dissipation target within the limit, else the end of the interval towards it
        if abs(dD_f) * (upper - lower) > 1e-9 * dTau:
            return float(np.clip((dTau - dD_0) / dD_f, lower, upper))

        # no dissipation gradient (elastic): continue in the direction of the last increment
        def forwardness(ddLambda):
            if self.dULast is None or not self.dULast.any():
                return ddLambda
            return (a + ddLambda * b).dot(self.dULast)

        return upper if forwardness(upper) >= forwardness(lower) else lower

    def isDissipationControlled(self) -> bool:
        """Whether the current increment is dissipation controlled (without displacement)."""

        return self.D_n is not None and self.D_n > self.threshold

    def isConstraintSatisfied(self, timeStep: TimeStep, dU, dofManager: DofManager) -> bool:
        """Check the (nonlinear) constraint for the current iterate; the load parameter constraint is linear and
        always satisfied after the first correction.

        Parameters
        ----------
        timeStep
            The current time step.
        dU
            The current increment of the solution vector.
        dofManager
            The dof manager of the current equation system.

        Returns
        -------
        bool
            True if the increment ends at the first of the dissipation and displacement targets.
        """

        if self.displacement is None and not self.isDissipationControlled():
            return True

        tol = self.constraintTolerance
        dTau = timeStep.stepProgressIncrement * self.dissipation
        dD = self.computeDissipation() - self.D_n
        if self.displacement is None:
            return abs(dD - dTau) <= tol * dTau

        dUMax = timeStep.stepProgressIncrement * self.displacement
        dUNorm = np.abs(dU[self._displacementIdcs(dofManager)]).max()
        onDissipationTarget = abs(dD - dTau) <= tol * dTau and dUNorm <= (1 + tol) * dUMax
        onDisplacementLimit = abs(dUNorm - dUMax) <= tol * dUMax and dD <= (1 + tol) * dTau
        return onDissipationTarget or onDisplacementLimit

    def getExtrapolationFactor(self, timeStep: TimeStep, prevTimeStep: TimeStep) -> float:
        """Scale factor of the last converged increment for the predictor of the current one: the ratio of target and
        last increments of the quantity reached first.

        Parameters
        ----------
        timeStep
            The current time step.
        prevTimeStep
            The time step of the last converged increment.

        Returns
        -------
        float
            The scale factor.
        """

        stepRatio = timeStep.stepProgressIncrement / prevTimeStep.stepProgressIncrement

        factors = []
        if (self.displacement is not None or self.isDissipationControlled()) and self.dDLast > 0.0:
            factors.append(timeStep.stepProgressIncrement * self.dissipation / self.dDLast)
        if self.displacement is not None and self.dUNormLast > 0.0:
            factors.append(timeStep.stepProgressIncrement * self.displacement / self.dUNormLast)

        return min(factors + [self.maxPredictorRatio * stepRatio]) if factors else stepRatio

    def finishIncrement(self, U, dU, dLambda, timeStep: TimeStep, dofManager):
        """Store the dissipation and displacement increments of a converged increment.

        Parameters
        ----------
        U
            The current solution vector.
        dU
            The current increment of the solution vector.
        dLambda
            The increment of the load parameter.
        timeStep
            The current time step.
        dofManager
            The dof manager of the current equation system.
        """

        D = self.computeDissipation()
        idcs = self._displacementIdcs(dofManager)
        self.dULast = np.array(dU[idcs], copy=True)
        self.dUNormLast = np.abs(self.dULast).max()

        if self.displacement is None:
            phase = "dissipation" if self.isDissipationControlled() else "load parameter"
        else:
            reached = (D - self.D_n) / (timeStep.stepProgressIncrement * self.dissipation)
            phase = "dissipation" if reached >= 1 - self.constraintTolerance else "displacement"
        self.journal.message(
            f"{phase} controlled: D={D:.6e} (dD={D - self.D_n:.3e}), max dU={self.dUNormLast:.3e}, F={self.F:.6e}, "
            f"dLambda={dLambda:.3e}",
            self.identification,
        )
        self.dLambdaLast = dLambda
        self.dDLast = D - self.D_n
        self.D_n = D

    def applyAtStepEnd(self, model):
        """Restart the bookkeeping of the start value in a subsequent step.

        Parameters
        ----------
        model
            The current state of the model.
        """

        self.D_n = None
