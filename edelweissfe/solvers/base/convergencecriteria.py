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
"""Field-wise convergence criteria for the Newton-Raphson iterations of the implicit solvers.

Two criteria are available, selected by the solver option ``convergenceCriterion``:

``legacy`` (default)
    EdelweissFE's own criterion. Field by field, it compares the largest flux residual :math:`r_{\\max}` with the
    spatial average flux :math:`\\bar{q}` of the *current* iteration, computed from the element fluxes only, and the
    largest correction :math:`c_{\\max}` with an absolute tolerance and with a tolerance relative to
    :math:`\\Delta u_{\\max}`, the largest increment of the field in the current increment. A field is converged if

    .. math::

        \\left( r_{\\max} \\leq R_a \\;\\text{or}\\; r_{\\max} \\leq R \\, \\bar{q} \\right)
        \\quad \\text{and} \\quad
        \\left( c_{\\max} < C_a \\;\\text{or}\\; c_{\\max} < C \\, \\Delta u_{\\max} \\right),

    with the tolerances of the field:

    * :math:`R`, the flux residual tolerance relative to the spatial average flux (``fluxResidualTolerance``),
      replaced by ``fluxResidualToleranceAlternative`` from iteration :math:`I` (``iterationsForAlternativeTolerance``)
      on, where a Newton iteration that converges only slowly may stop at a looser tolerance;
    * :math:`R_a`, the absolute flux residual tolerance (``absoluteFluxResidualTolerance``): a floor for a field whose
      flux is so small that a relative tolerance would ask for the round-off of the residual;
    * :math:`C_a`, the absolute correction tolerance (``fieldCorrectionTolerance``);
    * :math:`C`, the correction tolerance relative to the largest increment (``relativeFieldCorrectionTolerance``);
      the default 0 accepts a correction by its absolute size only.

    The tolerances of each field are set with ``*updateConfiguration``, for instance

    .. code-block:: edelweiss

        *updateConfiguration, configuration=absoluteFluxResidualTolerance
        displacement=1e-10

    and default to :mod:`edelweissfe.config.phenomena`. Three rules, set by solver options, are off by default:

    * Zero flux (``zeroFluxThreshold``, :math:`\\bar{q}_0`): a field whose spatial average flux does not exceed
      :math:`\\bar{q}_0` carries no flux its residual could be compared with, so its flux residual counts as
      converged. An example is a nonlocal damage field before the onset of damage.
    * Zero increment (``zeroIncrementThreshold``, :math:`\\Delta u_0`): a field whose largest increment does not
      exceed :math:`\\Delta u_0` has not moved, so there is no increment to relate its correction to; its correction
      counts as converged.
    * Minimum iterations (``minimumIterations``, :math:`n`): no flux residual counts as converged before iteration
      :math:`n`, so an increment takes at least :math:`n` Newton iterations. With :math:`n = 0`, a prediction that
      already satisfies the criterion is accepted as it is. The zero flux rule needs :math:`n \\geq 1`: before the
      first correction, the elements of a model at rest report no flux at all.

    The scalar variables (e.g., Lagrange multipliers) are checked like a field, with the tolerances of the entry
    ``scalar variables``. No element flux acts on them, so their flux residual is compared with the tolerances
    directly, as if their spatial average flux were one.

    Two parameter sets of this criterion are in use: EdelweissFE's defaults, which are those of the original
    criterion, and the settings of EdelweissMeshfree's solvers (for every field):

    ======================================  ==============================  ==================
    parameter                               EdelweissFE                     EdelweissMeshfree
    ======================================  ==============================  ==================
    ``fluxResidualTolerance``               per field, see phenomena        1e-4
    ``fluxResidualToleranceAlternative``    per field, see phenomena        1e-3
    ``iterationsForAlternativeTolerance``   15                              5
    ``absoluteFluxResidualTolerance``       1e-7, scalar variables 0        1e-14
    ``fieldCorrectionTolerance``            per field, see phenomena        1e-14
    ``relativeFieldCorrectionTolerance``    0                               1e-3
    ``zeroFluxThreshold``                   off                             1e-9
    ``zeroIncrementThreshold``              off                             1e-13
    ``minimumIterations``                   0                               1
    ======================================  ==============================  ==================

    EdelweissMeshfree also counts the residual growths of a diverging iteration differently: its solvers set the
    solver options ``countFirstResidualAsGrowth`` and ``countGrowthOfConvergedFields`` to False.

``abaqus``
    The criterion of Abaqus/Standard ("Convergence criteria for nonlinear problems", Abaqus Analysis User's
    Guide), with its default parameters as printed in the ``.msg`` file of an Abaqus run without ``*CONTROLS``.
    A field is converged if

    .. math::

        r_{\\max} \\leq R_n \\, \\tilde{q} \\quad \\text{and} \\quad c_{\\max} \\leq C_n \\, \\Delta u_{\\max},

    with :math:`R_n` replaced by :math:`R_p` from iteration :math:`I_p` on. :math:`\\Delta u_{\\max}` is the
    largest increment of the field in the current increment, so the correction criterion is *relative*.
    :math:`\\tilde{q}` is the *time average* flux: the spatial average flux :math:`\\bar{q}` averaged over all
    converged increments of the step in which the field carried flux, and the current iteration. The spatial
    average :math:`\\bar{q}` includes the fluxes of constraints (e.g., contact forces). Until a field has carried
    flux, :math:`\\tilde{q}` is the time average flux of the previous step or, in the first step, the initial value
    :math:`\\tilde{q}^0` (Abaqus' "initial value of time average force", ``1e-2``).

    A field whose current spatial average flux is below :math:`\\varepsilon \\, \\tilde{q}` carries no flux; it is
    converged if :math:`c_{\\max} \\leq C_\\varepsilon \\, \\Delta u_{\\max}`. A field with
    :math:`r_{\\max} \\leq R_l \\, \\tilde{q}` is converged irrespective of the correction (linear case).

    The criterion is formulated for fields; the scalar variables are checked by the ``legacy`` criterion.

    Not implemented: the estimate of the next correction from the convergence rate (Abaqus accepts an increment
    if it predicts a small enough correction), and the detection of inactive regions for the spatial average.
    Both make Abaqus' criterion *more* permissive than this implementation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from edelweissfe.numerics.dofmanager import DofManager, DofVector


@dataclass
class FieldConvergence:
    """The convergence state of one field in one iteration.

    Parameters
    ----------
    fluxResidual
        The largest absolute flux residual :math:`r_{\\max}`.
    fluxResidualConverged
        True if the flux residual criterion is satisfied.
    correction
        The largest absolute correction :math:`c_{\\max}`.
    correctionConverged
        True if the correction criterion is satisfied.
    indexOfLargestResidual
        The index of :math:`r_{\\max}` in the dof vector of the field.
    """

    fluxResidual: float
    fluxResidualConverged: bool
    correction: float
    correctionConverged: bool
    indexOfLargestResidual: int


class ConvergenceCriterionBase(ABC):
    """Base class for the field-wise convergence criteria of the implicit solvers."""

    def startStep(self, previousCriterion: "ConvergenceCriterionBase | None"):
        """Called at the beginning of each step.

        Parameters
        ----------
        previousCriterion
            The criterion of the previous step, if any.
        """

    def timeAveragedFluxesAtStepEnd(self) -> dict[str, float]:
        """The time average fluxes at the end of the step, used as reference by the criterion of the next step.

        Returns
        -------
        dict[str,float]
            The time average flux per field; empty for criteria without time averaging.
        """
        return {}

    def acceptIncrement(self):
        """Called once an increment is converged and accepted."""

    @abstractmethod
    def checkField(
        self,
        field: str,
        R: np.ndarray,
        ddU: np.ndarray | None,
        dU: np.ndarray,
        spatialAveragedFlux: float,
        iterationCounter: int,
    ) -> FieldConvergence:
        """Check the convergence of a single field.

        Parameters
        ----------
        field
            The name of the field.
        R
            The residual of the field.
        ddU
            The current correction of the field, None if no correction was computed yet in this increment.
        dU
            The current increment of the field.
        spatialAveragedFlux
            The spatial average flux :math:`\\bar{q}` of the field in the current iteration.
        iterationCounter
            The current iteration number.

        Returns
        -------
        FieldConvergence
            The convergence state of the field.
        """

    @abstractmethod
    def checkScalarVariables(
        self, R: np.ndarray, ddU: np.ndarray | None, dU: np.ndarray, iterationCounter: int
    ) -> FieldConvergence:
        """Check the convergence of the scalar variables (e.g., Lagrange multipliers), together as one column.

        Parameters
        ----------
        R
            The residual of the scalar variables.
        ddU
            The current correction of the scalar variables, None if no correction was computed yet in this increment.
        dU
            The current increment of the scalar variables.
        iterationCounter
            The current iteration number.

        Returns
        -------
        FieldConvergence
            The convergence state of the scalar variables.
        """

    @abstractmethod
    def computeSpatialAveragedFluxes(
        self, F: DofVector, FConstraints: DofVector, theDofManager: DofManager
    ) -> dict[str, float]:
        """Compute the spatial average flux of every field.

        Parameters
        ----------
        F
            The accumulated absolute element fluxes.
        FConstraints
            The accumulated absolute constraint fluxes.
        theDofManager
            The DofManager.

        Returns
        -------
        dict[str,float]
            The spatial average flux of every field.
        """


class LegacyConvergenceCriterion(ConvergenceCriterionBase):
    """EdelweissFE's own criterion; see the module documentation.

    Parameters
    ----------
    fluxResidualTolerances
        :math:`R` per field: the flux residual tolerances relative to the spatial average flux.
    fluxResidualTolerancesAlternative
        The relative flux residual tolerances per field from iteration :math:`I` on.
    absoluteFluxResidualTolerances
        :math:`R_a` per field: the absolute flux residual tolerances.
    fieldCorrectionTolerances
        :math:`C_a` per field: the absolute correction tolerances.
    relativeFieldCorrectionTolerances
        :math:`C` per field: the correction tolerances relative to the largest increment of the field.
    iterationsForAlternativeTolerance
        :math:`I`, the iteration from which on the alternative flux residual tolerances apply.
    zeroFluxThreshold
        :math:`\\bar{q}_0`, the spatial average flux up to which a field carries no flux; None for no zero flux rule.
    zeroIncrementThreshold
        :math:`\\Delta u_0`, the largest increment up to which a field has not moved; None for no zero increment rule.
    minimumIterations
        :math:`n`, the iteration before which no flux residual counts as converged.
    """

    def __init__(
        self,
        fluxResidualTolerances: dict[str, float],
        fluxResidualTolerancesAlternative: dict[str, float],
        absoluteFluxResidualTolerances: dict[str, float],
        fieldCorrectionTolerances: dict[str, float],
        relativeFieldCorrectionTolerances: dict[str, float],
        iterationsForAlternativeTolerance: int,
        zeroFluxThreshold: float | None,
        zeroIncrementThreshold: float | None,
        minimumIterations: int,
    ):
        self.fluxResidualTolerances = fluxResidualTolerances
        self.fluxResidualTolerancesAlternative = fluxResidualTolerancesAlternative
        self.absoluteFluxResidualTolerances = absoluteFluxResidualTolerances
        self.fieldCorrectionTolerances = fieldCorrectionTolerances
        self.relativeFieldCorrectionTolerances = relativeFieldCorrectionTolerances
        self.iterationsForAlternativeTolerance = iterationsForAlternativeTolerance
        self.zeroFluxThreshold = zeroFluxThreshold
        self.zeroIncrementThreshold = zeroIncrementThreshold
        self.minimumIterations = minimumIterations

    def checkField(self, field, R, ddU, dU, spatialAveragedFlux, iterationCounter):
        indexOfMax = np.argmax(np.abs(R))
        fluxResidual = abs(R[indexOfMax])
        correction = np.linalg.norm(ddU, np.inf) if ddU is not None else 0.0

        return FieldConvergence(
            fluxResidual,
            self.fluxResidualConverged(field, fluxResidual, spatialAveragedFlux, iterationCounter),
            correction,
            self.correctionConverged(field, correction, dU),
            indexOfMax,
        )

    def checkScalarVariables(self, R, ddU, dU, iterationCounter):
        # No element flux acts on the scalar variables: their residual is compared with the tolerances directly.
        return self.checkField("scalar variables", R, ddU, dU, 1.0, iterationCounter)

    def fluxResidualConverged(
        self, field: str, fluxResidual: float, spatialAveragedFlux: float, iterationCounter: int
    ) -> bool:
        """The flux residual part of the criterion.

        Parameters
        ----------
        field
            The name of the field.
        fluxResidual
            The largest absolute flux residual :math:`r_{\\max}` of the field.
        spatialAveragedFlux
            The spatial average flux :math:`\\bar{q}` of the field.
        iterationCounter
            The current iteration number.

        Returns
        -------
        bool
            True if the flux residual is converged.
        """

        if iterationCounter < self.minimumIterations:
            return False

        if fluxResidual <= self.absoluteFluxResidualTolerances[field]:
            return True

        # 'not above' rather than 'at most': a field without any element flux (a NaN average) carries no flux either
        if self.zeroFluxThreshold is not None and not spatialAveragedFlux > self.zeroFluxThreshold:
            return True

        relativeTolerances = (
            self.fluxResidualTolerances
            if iterationCounter < self.iterationsForAlternativeTolerance
            else self.fluxResidualTolerancesAlternative
        )

        return fluxResidual <= relativeTolerances[field] * spatialAveragedFlux

    def correctionConverged(self, field: str, correction: float, dU: np.ndarray) -> bool:
        """The correction part of the criterion.

        Parameters
        ----------
        field
            The name of the field.
        correction
            The largest absolute correction :math:`c_{\\max}` of the field.
        dU
            The current increment of the field.

        Returns
        -------
        bool
            True if the correction is converged.
        """

        if correction < self.fieldCorrectionTolerances[field]:
            return True

        largestIncrement = np.linalg.norm(dU, np.inf)

        if self.zeroIncrementThreshold is not None and not largestIncrement > self.zeroIncrementThreshold:
            return True

        return correction < self.relativeFieldCorrectionTolerances[field] * largestIncrement

    def computeSpatialAveragedFluxes(self, F, FConstraints, theDofManager):
        return {
            field: np.linalg.norm(F[theDofManager.idcsOfFieldsInDofVector[field]], 1) / nDof
            for field, nDof in theDofManager.nAccumulatedNodalFluxesFieldwise.items()
        }


class AbaqusConvergenceCriterion(ConvergenceCriterionBase):
    """The criterion of Abaqus/Standard; see the module documentation.

    Parameters
    ----------
    scalarVariableCriterion
        The criterion that checks the scalar variables, which Abaqus' criterion does not cover: the ``legacy`` one.
    residualTolerance
        :math:`R_n`, the flux residual tolerance relative to the time average flux.
    correctionTolerance
        :math:`C_n`, the correction tolerance relative to the largest increment of the field.
    residualToleranceAlternative
        :math:`R_p`, the flux residual tolerance used from iteration :math:`I_p` on.
    iterationsForAlternativeTolerance
        :math:`I_p`.
    zeroFluxThreshold
        :math:`\\varepsilon`, below which a field is considered to carry no flux, relative to the time average flux.
    zeroFluxCorrectionTolerance
        :math:`C_\\varepsilon`, the correction tolerance of a field without flux.
    linearResidualTolerance
        :math:`R_l`, below which a field is converged irrespective of its correction.
    initialTimeAveragedFlux
        :math:`\\tilde{q}^0`, the time average flux of a field that has not carried flux yet, in the first step.
    """

    def __init__(
        self,
        scalarVariableCriterion: ConvergenceCriterionBase,
        residualTolerance: float = 5e-3,
        correctionTolerance: float = 1e-2,
        residualToleranceAlternative: float = 2e-2,
        iterationsForAlternativeTolerance: int = 9,
        zeroFluxThreshold: float = 1e-5,
        zeroFluxCorrectionTolerance: float = 1e-3,
        linearResidualTolerance: float = 1e-8,
        initialTimeAveragedFlux: float = 1e-2,
    ):
        self.scalarVariableCriterion = scalarVariableCriterion
        self.residualTolerance = residualTolerance
        self.correctionTolerance = correctionTolerance
        self.residualToleranceAlternative = residualToleranceAlternative
        self.iterationsForAlternativeTolerance = iterationsForAlternativeTolerance
        self.zeroFluxThreshold = zeroFluxThreshold
        self.zeroFluxCorrectionTolerance = zeroFluxCorrectionTolerance
        self.linearResidualTolerance = linearResidualTolerance
        self.initialTimeAveragedFlux = initialTimeAveragedFlux

        # Per field: the sum and the number of the spatial average fluxes of the converged increments with flux.
        self._sumOfConvergedSpatialAveragedFluxes = {}
        self._nIncrementsWithFlux = {}
        # Per field: the time average flux of the previous step, the reference for the zero flux check in the
        # first increment of a step.
        self._timeAveragedFluxOfPreviousStep = {}
        # Per field: the spatial average flux of the last checked iteration, recorded if the increment is accepted.
        self._spatialAveragedFluxOfLastIteration = {}

    def startStep(self, previousCriterion):
        if previousCriterion is not None:
            self._timeAveragedFluxOfPreviousStep = previousCriterion.timeAveragedFluxesAtStepEnd()

    def timeAveragedFluxesAtStepEnd(self):
        fields = self._timeAveragedFluxOfPreviousStep.keys() | self._nIncrementsWithFlux.keys()
        return {field: self._timeAveragedFluxOfConvergedIncrements(field) for field in fields}

    def acceptIncrement(self):
        for field, spatialAveragedFlux in self._spatialAveragedFluxOfLastIteration.items():
            if self._carriesFlux(field, spatialAveragedFlux):
                self._sumOfConvergedSpatialAveragedFluxes[field] = (
                    self._sumOfConvergedSpatialAveragedFluxes.get(field, 0.0) + spatialAveragedFlux
                )
                self._nIncrementsWithFlux[field] = self._nIncrementsWithFlux.get(field, 0) + 1
        self._spatialAveragedFluxOfLastIteration.clear()

    def _timeAveragedFluxOfConvergedIncrements(self, field: str) -> float:
        nIncrements = self._nIncrementsWithFlux.get(field, 0)
        if nIncrements == 0:
            return self._timeAveragedFluxOfPreviousStep.get(field, self.initialTimeAveragedFlux)
        return self._sumOfConvergedSpatialAveragedFluxes[field] / nIncrements

    def _carriesFlux(self, field: str, spatialAveragedFlux: float) -> bool:
        return spatialAveragedFlux > self.zeroFluxThreshold * self._timeAveragedFluxOfConvergedIncrements(field)

    def timeAveragedFlux(self, field: str, spatialAveragedFlux: float) -> float:
        """The time average flux :math:`\\tilde{q}`, including the current iteration if the field carries flux.

        Parameters
        ----------
        field
            The name of the field.
        spatialAveragedFlux
            The spatial average flux :math:`\\bar{q}` of the field in the current iteration.

        Returns
        -------
        float
            The time average flux.
        """
        if not self._carriesFlux(field, spatialAveragedFlux):
            return self._timeAveragedFluxOfConvergedIncrements(field)

        nIncrements = self._nIncrementsWithFlux.get(field, 0)
        return (self._sumOfConvergedSpatialAveragedFluxes.get(field, 0.0) + spatialAveragedFlux) / (nIncrements + 1)

    def checkField(self, field, R, ddU, dU, spatialAveragedFlux, iterationCounter):
        self._spatialAveragedFluxOfLastIteration[field] = spatialAveragedFlux

        indexOfMax = np.argmax(np.abs(R))
        fluxResidual = abs(R[indexOfMax])
        correction = np.linalg.norm(ddU, np.inf) if ddU is not None else 0.0
        largestIncrement = np.linalg.norm(dU, np.inf)
        timeAveragedFlux = self.timeAveragedFlux(field, spatialAveragedFlux)

        if not self._carriesFlux(field, spatialAveragedFlux):
            # A field without flux and without residual (e.g., a nonlocal field before the onset of damage) has
            # nothing left to iterate on, even if its correction is pure round-off of the size of its increment.
            correctionConverged = fluxResidual == 0.0 or (
                ddU is not None and correction <= self.zeroFluxCorrectionTolerance * largestIncrement
            )
            return FieldConvergence(fluxResidual, True, correction, correctionConverged, indexOfMax)

        if fluxResidual <= self.linearResidualTolerance * timeAveragedFlux:
            return FieldConvergence(fluxResidual, True, correction, True, indexOfMax)

        residualTolerance = (
            self.residualTolerance
            if iterationCounter < self.iterationsForAlternativeTolerance
            else self.residualToleranceAlternative
        )

        return FieldConvergence(
            fluxResidual,
            fluxResidual <= residualTolerance * timeAveragedFlux,
            correction,
            ddU is not None and correction <= self.correctionTolerance * largestIncrement,
            indexOfMax,
        )

    def checkScalarVariables(self, R, ddU, dU, iterationCounter):
        return self.scalarVariableCriterion.checkScalarVariables(R, ddU, dU, iterationCounter)

    def computeSpatialAveragedFluxes(self, F, FConstraints, theDofManager):
        return {
            field: (
                np.linalg.norm(F[theDofManager.idcsOfFieldsInDofVector[field]], 1)
                + np.linalg.norm(FConstraints[theDofManager.idcsOfFieldsInDofVector[field]], 1)
            )
            / nDof
            for field, nDof in theDofManager.nAccumulatedNodalFluxesFieldwise.items()
        }
