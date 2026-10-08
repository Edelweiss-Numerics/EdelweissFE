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
    The original EdelweissFE criterion. A field is converged if

    .. math::

        r_{\\max} \\leq \\max(R \\, \\bar{q}, 10^{-7}) \\quad \\text{and} \\quad c_{\\max} < C,

    with the *absolute* correction tolerance :math:`C` (``fieldCorrectionTolerance``), the residual tolerance
    :math:`R` (``fluxResidualTolerance``, ``fluxResidualToleranceAlternative`` from iteration 15 on), and the
    spatial average flux :math:`\\bar{q}` of the *current* iteration, computed from the element fluxes only.

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
    average :math:`\\bar{q}` includes the fluxes of constraints (e.g., contact forces).

    A field whose current spatial average flux is below :math:`\\varepsilon \\, \\tilde{q}` carries no flux; it is
    converged if :math:`c_{\\max} \\leq C_\\varepsilon \\, \\Delta u_{\\max}`. A field with
    :math:`r_{\\max} \\leq R_l \\, \\tilde{q}` is converged irrespective of the correction (linear case).

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
    """The original EdelweissFE criterion; see the module documentation.

    Parameters
    ----------
    fluxResidualTolerances
        The relative flux residual tolerances per field.
    fluxResidualTolerancesAlt
        The relative flux residual tolerances per field from iteration 15 on.
    fieldCorrectionTolerances
        The absolute correction tolerances per field.
    """

    def __init__(self, fluxResidualTolerances: dict, fluxResidualTolerancesAlt: dict, fieldCorrectionTolerances: dict):
        self.fluxResidualTolerances = fluxResidualTolerances
        self.fluxResidualTolerancesAlt = fluxResidualTolerancesAlt
        self.fieldCorrectionTolerances = fieldCorrectionTolerances

    def checkField(self, field, R, ddU, dU, spatialAveragedFlux, iterationCounter):
        fluxResidualTolerances = (
            self.fluxResidualTolerances if iterationCounter < 15 else self.fluxResidualTolerancesAlt
        )

        indexOfMax = np.argmax(np.abs(R))
        fluxResidual = abs(R[indexOfMax])
        correction = np.linalg.norm(ddU, np.inf) if ddU is not None else 0.0

        return FieldConvergence(
            fluxResidual,
            fluxResidual <= max(fluxResidualTolerances[field] * spatialAveragedFlux, 1e-7),
            correction,
            correction < self.fieldCorrectionTolerances[field],
            indexOfMax,
        )

    def computeSpatialAveragedFluxes(self, F, FConstraints, theDofManager):
        return {
            field: max(1e-10, np.linalg.norm(F[theDofManager.idcsOfFieldsInDofVector[field]], 1) / nDof)
            for field, nDof in theDofManager.nAccumulatedNodalFluxesFieldwise.items()
        }


class AbaqusConvergenceCriterion(ConvergenceCriterionBase):
    """The criterion of Abaqus/Standard; see the module documentation.

    Parameters
    ----------
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
    """

    def __init__(
        self,
        residualTolerance: float = 5e-3,
        correctionTolerance: float = 1e-2,
        residualToleranceAlternative: float = 2e-2,
        iterationsForAlternativeTolerance: int = 9,
        zeroFluxThreshold: float = 1e-5,
        zeroFluxCorrectionTolerance: float = 1e-3,
        linearResidualTolerance: float = 1e-8,
    ):
        self.residualTolerance = residualTolerance
        self.correctionTolerance = correctionTolerance
        self.residualToleranceAlternative = residualToleranceAlternative
        self.iterationsForAlternativeTolerance = iterationsForAlternativeTolerance
        self.zeroFluxThreshold = zeroFluxThreshold
        self.zeroFluxCorrectionTolerance = zeroFluxCorrectionTolerance
        self.linearResidualTolerance = linearResidualTolerance

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
            return self._timeAveragedFluxOfPreviousStep.get(field, 0.0)
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

    def computeSpatialAveragedFluxes(self, F, FConstraints, theDofManager):
        return {
            field: (
                np.linalg.norm(F[theDofManager.idcsOfFieldsInDofVector[field]], 1)
                + np.linalg.norm(FConstraints[theDofManager.idcsOfFieldsInDofVector[field]], 1)
            )
            / nDof
            for field, nDof in theDofManager.nAccumulatedNodalFluxesFieldwise.items()
        }
