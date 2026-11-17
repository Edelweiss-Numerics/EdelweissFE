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
Dissipation controller for the NISTPArcLength solver, after the dissipation-based arc length method of Gutiérrez
(2004) and Verhoosel, Remmers and Gutiérrez (2009), for loading by prescribed displacements (Dirichlet boundary
conditions scaled by the load parameter).

The controller adds the equation

.. math::

    g(\\boldsymbol{U}) = D(\\boldsymbol{U}) - D_n - \\Delta \\tau = 0

for the increment of the load parameter :math:`\\Delta\\lambda`. Here, :math:`D` is the dissipated energy of the
material, summed from the quadrature point result ``dissipation`` over the elements of ``elSet`` (instead of the
expressions of Verhoosel et al. in terms of the load and the displacements, which assume a particular material),
:math:`D_n` its value at the start of the increment, and :math:`\\Delta\\tau = \\Delta p \\, \\tau` the share of the
increment (of size :math:`\\Delta p`) in the dissipation ``dissipation`` of the step. Linearized along the correction
:math:`\\Delta\\Delta\\boldsymbol{U} = \\Delta\\Delta\\boldsymbol{U}_0 + \\Delta\\Delta\\lambda \\, \\Delta\\Delta
\\boldsymbol{U}_f`, the correction of the load parameter is

.. math::

    \\Delta\\Delta\\lambda = \\frac{\\Delta\\tau - (D - D_n) - \\nabla D \\cdot \\Delta\\Delta\\boldsymbol{U}_0}
        {\\nabla D \\cdot \\Delta\\Delta\\boldsymbol{U}_f},

with :math:`D` and its gradient :math:`\\nabla D` of the iterate. The elements write their share of the gradient in
every evaluation of their kernels (:meth:`~edelweissfe.elements.base.baseelement.BaseElement.assignEnergyGradientOutput`),
which needs Marmot elements and materials that provide the energy gradients. The solver accepts an increment once its
residual has converged and :math:`|D - D_n - \\Delta\\tau| \\le \\epsilon \\, \\Delta\\tau`, with the relative
tolerance :math:`\\epsilon` = ``constraintTolerance``.

The dissipation equation is singular where nothing dissipates. Two ways cover these parts of the path:

* With ``displacement`` (the combined constraint, a heuristic), every increment ends at the first of the dissipation
  target and the limit :math:`\\Delta u_{\\max} = \\Delta p \\, u_{\\max}` of the largest nodal displacement increment,

  .. math::

      \\max_i \\left| \\Delta U_i + \\Delta\\Delta U_{0,i} + \\Delta\\Delta\\lambda \\, \\Delta\\Delta U_{f,i} \\right|
          \\le \\Delta u_{\\max}

  over all displacement degrees of freedom :math:`i`. This bounds :math:`\\Delta\\Delta\\lambda` to an interval; the
  dissipation target is taken within it, and otherwise an end of the interval: the one where the linearized
  dissipation does not decrease, as an increment cannot release dissipated energy (at the onset of a snap-back, the
  end in the direction of the last converged increment does), and if both ends or neither qualify, the end whose
  increment points in the direction of the last converged increment. The displacement limit thus governs the
  elastic range and the final failure, where the dissipation hardly constrains the deformation anymore.
* Without ``displacement``, the load parameter is controlled directly, :math:`\\Delta\\lambda = \\Delta p \\,
  \\lambda_{\\mathrm{step}}` with :math:`\\lambda_{\\mathrm{step}}` = ``loadParameter``, until the dissipation at
  the start of an increment exceeds ``threshold``; then the dissipation is. An elastic iterate in the dissipation
  controlled phase (the first one from a zero prediction) is moved by the secant of the last increment, or, without
  one, by the load parameter rule.

References
----------
Gutiérrez, M. A. (2004). Energy release control for numerical simulations of failure in quasi-brittle solids.
Communications in Numerical Methods in Engineering, 20(1), 19-29.

Verhoosel, C. V., Remmers, J. J. C., Gutiérrez, M. A. (2009). A dissipation-based arc-length method for robust
simulation of brittle and ductile failure. International Journal for Numerical Methods in Engineering, 77(9),
1290-1321.
"""

from dataclasses import dataclass

import numpy as np

from edelweissfe.journal.journal import Journal
from edelweissfe.models.discretesystem import DiscreteSystem
from edelweissfe.models.femodel import FEModel
from edelweissfe.models.meshdependent import MeshDependent
from edelweissfe.numerics.doflocations import DofLocations
from edelweissfe.numerics.dofnumbering import DofNumbering
from edelweissfe.numerics.dofvector import DofVector
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


class StepAction(ArcLengthControllerBase, MeshDependent):
    """Dissipation controller for the NISTPArcLength solver; see the module documentation for the equations.

    The elements of ``elSet`` write the gradients of their dissipation to one buffer while the controller controls a
    step (:meth:`beginControl` to :meth:`endControl`). A refinement that changes the set rebuilds the buffer, and the
    dissipation at the start of the next increment is taken anew from the transferred state (:meth:`refresh`), as it is
    after a zero increment (:meth:`finishIncrement`), e.g. the re-equilibration after a model change.

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

    #: The dissipation rate along the reference load, :math:`\nabla D \cdot \Delta\Delta U_f`, below which the iterate
    #: counts as elastic (without displacement), relative to the work rate of the reference load: then the linearized
    #: dissipation fixes no load parameter.
    elasticDissipationRate = 1e-6

    #: The change of the linearized dissipation across the interval of the displacement limit, relative to the target,
    #: below which it sets no direction (an elastic predictor).
    minDissipationAcrossInterval = 1e-3

    #: The response of a displacement to the reference load, relative to the largest one, below which it is round-off
    #: and bounds no load parameter.
    negligibleResponse = 1e-14

    #: A zero prediction after a cutback may follow an elastic unloading branch.
    extrapolatesAfterCutback = True

    #: Carried from one increment to the next: the dissipation at the start of the increment and the last accepted
    #: increment (predictor, increment size); F and dLambda are set again in every iteration by setSystemState.
    checkpointedState = {
        "D_n": float,
        "dLambdaLast": float,
        "dDLast": float,
        "dUNormLast": float,
        "dULast": np.ndarray,
        "dULastKeys": np.ndarray,
    }

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
        # (node label, component) of each entry of dULast, to remap it after the dofs were renumbered (refinement)
        self.dULastKeys = None
        # whether the controller controls the current step, see beginControl
        self._controlling = False
        # views on the dissipation of the elements, the elements writing their energy gradients to the buffers, and
        # the buffers' indices in the dof vector of the locations they were found in
        self._dissipationViews = []
        self._gradientElements = []
        self._gradientIdcs = None
        self._gradientIdcsLocations = None
        # (numbering, keys) of the displacement dofs, see _displacementKeys
        self._displacementKeysOfNumbering = (None, None)

        self.updateStepAction(elSet, dissipation, displacement, loadParameter, threshold)

        # told of every change of the mesh, see refresh
        model.topology.registerMeshDependent(self)

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

        Raises
        ------
        ValueError
            If dissipation or displacement is not positive, neither displacement nor loadParameter is given, or an
            element of the set provides no energy gradients.
        """

        if dissipation <= 0.0:
            raise ValueError(f"{self.identification} {self.name}: dissipation must be positive, got {dissipation}")
        if displacement is not None and displacement <= 0.0:
            raise ValueError(f"{self.identification} {self.name}: displacement must be positive, got {displacement}")
        if displacement is None and loadParameter is None:
            raise ValueError(f"{self.identification} {self.name}: either displacement or loadParameter is required")

        for el in elSet:
            if not el.providesEnergyGradients:
                raise ValueError(
                    f"{self.identification} {self.name}: element {el.elNumber} ({el.elType}) of element set {elSet.name} "
                    "provides no energy gradients"
                )

        self.elSet = elSet
        self.dissipation = dissipation
        self.displacement = displacement
        self.loadParameter = loadParameter
        self.threshold = threshold

        if self._controlling:
            self._buildElementViews()

    def beginControl(self, model: FEModel):
        """Start controlling a step: the elements of the set write their energy gradients from now on. The dissipation
        at the start of the step is taken at the first iterate, which a resumed step restores instead.

        Parameters
        ----------
        model
            The model tree.
        """

        self._controlling = True
        self.D_n = None
        self._buildElementViews()

    def endControl(self, model: FEModel):
        """Stop controlling at the end of a step: the elements stop writing their energy gradients, which a following
        step without this controller would pay for in every evaluation.

        Parameters
        ----------
        model
            The model tree.
        """

        for el in self._gradientElements:
            el.assignEnergyGradientOutput(None, None)
        self._gradientElements = []
        self._dissipationViews = []
        self._gradientIdcsLocations = None
        self._controlling = False
        self.D_n = None

    def _buildElementViews(self):
        """Persistent views on the elements' dissipation and the buffers their energy gradients are written to, for
        the current members of the element set (rebuilt after a refinement changed the set)."""

        elements = list(self.elSet)

        # persistent views on the trial state, updated by the elements in every iteration
        self._dissipationViews = [
            el.getResultArray("dissipation", qp, getPersistentView=True)
            for el in elements
            for qp in range(el.getNumberOfQuadraturePoints())
        ]

        # elements that left the set stop writing to the old buffers
        for el in set(self._gradientElements).difference(elements):
            el.assignEnergyGradientOutput(None, None)

        # one contiguous buffer per gradient; each element writes to its slice in every kernel evaluation. The elements
        # write both gradients: the one of the elastic energy is not used here, but needs a buffer of its own.
        offsets = np.cumsum([0] + [el.nDof for el in elements])
        self._dElasticEnergy_dQ = np.zeros(offsets[-1])
        self._dDissipation_dQ = np.zeros(offsets[-1])
        for el, begin, end in zip(elements, offsets[:-1], offsets[1:]):
            el.assignEnergyGradientOutput(self._dElasticEnergy_dQ[begin:end], self._dDissipation_dQ[begin:end])
        self._gradientElements = elements
        self._gradientIdcsLocations = None

        self._recordSetVersion(self.elSet)

    def _refreshIfElementsChanged(self):
        """Rebuild the views and buffers if the element set was mutated (e.g. by h-adaptivity)."""

        if self._checkSetChanged(self.elSet):
            self._buildElementViews()

    def refresh(self, model: FEModel, change) -> bool:
        """After a change of the mesh, before the first evaluation of the next increment: rebuild the views and buffers
        if the element set changed, and take the dissipation at the start of the next increment from the state the
        refinement transferred to the new elements, which may differ from the one of the old elements; see
        :class:`~edelweissfe.models.meshdependent.MeshDependent`.

        Parameters
        ----------
        model
            The model tree.
        change
            The change of the mesh.

        Returns
        -------
        bool
            False: the controller keeps no record that enters the discrete system.
        """

        if not self._controlling or not self._checkSetChanged(self.elSet):
            return False

        self._buildElementViews()
        if self.D_n is not None:
            self._rebaseDissipation("mesh change")
        return False

    def _rebaseDissipation(self, reason: str):
        """Take the dissipation at the start of the next increment from the current state of the elements.

        Parameters
        ----------
        reason
            Why, for the log.
        """

        D = self.computeDissipation()
        if D != self.D_n:
            self.journal.message(
                f"{reason}: dissipation at the increment start changed from {self.D_n:.6e} to {D:.6e}",
                self.identification,
            )
        self.D_n = D

    def computeDissipation(self) -> float:
        """The dissipated energy of the current (trial) state."""

        self._refreshIfElementsChanged()
        return sum(view[0] for view in self._dissipationViews)

    def computeDissipationGradient(self, locations: DofLocations) -> np.ndarray:
        """The gradient of the dissipated energy of the current (trial) state wrt the solution vector.

        The elements wrote their gradients in the last evaluation of their kernels; only the scatter remains.

        Parameters
        ----------
        locations
            The locations of the elements in the current discrete system.

        Returns
        -------
        np.ndarray
            The gradient.
        """

        self._refreshIfElementsChanged()
        # the solver gets new locations with every change of the connectivity or of the numbering
        if self._gradientIdcsLocations is not locations:
            self._gradientIdcs = np.concatenate(
                [locations[el] for el in self._gradientElements] + [np.empty(0, dtype=int)]
            )
            self._gradientIdcsLocations = locations
        return np.bincount(self._gradientIdcs, weights=self._dDissipation_dQ, minlength=locations.numbering.nDof)

    @staticmethod
    def _displacementIdcs(numbering: DofNumbering) -> slice:
        """The indices of the displacement field in the solution vector."""

        return numbering.idcsOfFieldsInDofVector["displacement"]

    def _displacementKeys(self, numbering: DofNumbering) -> np.ndarray:
        """(node label, component) of each displacement dof, in the order of :meth:`_displacementIdcs`; kept for the
        numbering it was made for."""

        cachedNumbering, keys = self._displacementKeysOfNumbering
        if cachedNumbering is numbering:
            return keys

        keyOfIndex = {}
        for fieldVariable, idcs in numbering.idcsOfFieldVariablesInDofVector.items():
            if fieldVariable.field == "displacement":
                for component, index in enumerate(idcs):
                    keyOfIndex[int(index)] = (fieldVariable.node.label, component)
        displacementIdcs = range(*self._displacementIdcs(numbering).indices(numbering.nDof))
        keys = np.array([keyOfIndex[i] for i in displacementIdcs], dtype=np.int64)

        self._displacementKeysOfNumbering = (numbering, keys)
        return keys

    def _lastIncrementOnCurrentDofs(self, numbering: DofNumbering) -> np.ndarray:
        """The last displacement increment on the current displacement dofs; remapped by node and component if the
        dofs were renumbered, zero on nodes created since."""

        keys = self._displacementKeys(numbering)
        if np.array_equal(self.dULastKeys, keys):
            return self.dULast
        last = {tuple(key): value for key, value in zip(self.dULastKeys, self.dULast)}
        return np.array([last.get(tuple(key), 0.0) for key in keys])

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
            # first call of the step, in the zero increment at its converged start state
            self.D_n = self.computeDissipation()

    def _linearizationOfIterate(self, locations: DofLocations) -> tuple[float, np.ndarray]:
        """The dissipation of the iterate and its gradient, evaluated in the first call after :meth:`setSystemState`:
        in later calls of the iteration (line search), the elements hold trial states."""

        if self._linearization is None:
            self._linearization = (self.computeDissipation(), self.computeDissipationGradient(locations))
        return self._linearization

    def _loadParameterRule(self, timeStep: TimeStep) -> float:
        """The correction of the load parameter that makes the increment its share of ``loadParameter``."""

        return timeStep.stepProgressIncrement * self.loadParameter - self.dLambda

    def computeDDLambda(self, dU, ddU_0, ddU_f, timeStep: TimeStep, system: DiscreteSystem) -> float:
        """Compute the correction of the load parameter for the current iteration.

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
        system
            The current discrete system.

        Returns
        -------
        float
            The correction of the load parameter.
        """

        # A zero increment (the equilibrium check at the start of a step, the re-equilibration after a model change)
        # has no arc length to control: its targets are zero, so round-off alone would violate them. Iterate at a
        # fixed load parameter.
        if timeStep.stepProgressIncrement == 0.0:
            return 0.0

        if self.displacement is None and not self.isDissipationControlled():
            return self._loadParameterRule(timeStep)

        dTau = timeStep.stepProgressIncrement * self.dissipation

        # linearized dissipation increment along ddU = ddU_0 + ddLambda ddU_f: dD_0 + ddLambda dD_f
        D, dD_dU = self._linearizationOfIterate(system.locations)
        dD_0 = D - self.D_n + dD_dU.dot(ddU_0)
        dD_f = dD_dU.dot(ddU_f)

        if self.displacement is None:
            if abs(dD_f) <= self.elasticDissipationRate * abs(self.F):
                # an elastic iterate, e.g. the first one from a zero prediction: the secant of the last increment, and
                # without one the load parameter rule, which moves the iterate until it dissipates
                if self.dDLast > 0.0:
                    return min((dTau - D + self.D_n) / self.dDLast, self.maxPredictorRatio) * self.dLambdaLast
                return self._loadParameterRule(timeStep)
            return (dTau - dD_0) / dD_f

        # interval of ddLambda within the displacement limit, |a_i + ddLambda b_i| <= dUMax for all nodal dofs
        idcs = self._displacementIdcs(system.numbering)
        a = (dU + ddU_0)[idcs]
        b = ddU_f[idcs]
        dUMax = timeStep.stepProgressIncrement * self.displacement

        moving = np.abs(b) > self.negligibleResponse * np.abs(b).max()
        bounds = np.sort(np.stack([(-dUMax - a[moving]) / b[moving], (dUMax - a[moving]) / b[moving]]), axis=0)
        lower, upper = bounds[0].max(), bounds[1].min()
        if lower > upper:
            # the displacement limit cannot be met along ddU_f: closest approach
            return 0.5 * (lower + upper)

        # the dissipation target within the limit, unless the linearized dissipation hardly changes across the interval
        if abs(dD_f) * (upper - lower) > self.minDissipationAcrossInterval * dTau:
            target = (dTau - dD_0) / dD_f
            if lower <= target <= upper:
                return float(target)

        # target out of reach or no dissipation gradient: an end of the interval. An increment cannot release
        # dissipated energy, so an end where the linearized dissipation decreases is no candidate (at the onset of a
        # snap-back, the end in the direction of the last increment)
        candidates = [end for end in (lower, upper) if dD_0 + end * dD_f >= 0.0]
        if len(candidates) != 1:
            # both ends dissipate, or neither: the one in the direction of the last increment (the linearized
            # dissipation may grow backwards, e.g. by crushing in a fully cracked tail)
            candidates = [lower, upper]
        dULast = None if self.dULast is None else self._lastIncrementOnCurrentDofs(system.numbering)

        def forwardness(ddLambda):
            if dULast is None or not dULast.any():
                return ddLambda
            return (a + ddLambda * b).dot(dULast)

        return max(candidates, key=forwardness)

    def isDissipationControlled(self) -> bool:
        """Whether the current increment is dissipation controlled (without displacement)."""

        return self.D_n is not None and self.D_n > self.threshold

    def isConstraintSatisfied(self, timeStep: TimeStep, dU, system: DiscreteSystem) -> bool:
        """Check the (nonlinear) constraint for the current iterate; the load parameter constraint is linear and
        always satisfied after the first correction.

        Parameters
        ----------
        timeStep
            The current time step.
        dU
            The current increment of the solution vector.
        system
            The current discrete system.

        Returns
        -------
        bool
            True if the increment ends at the first of the dissipation and displacement targets.
        """

        # nothing is controlled in a zero increment, see computeDDLambda
        if timeStep.stepProgressIncrement == 0.0:
            return True

        if self.displacement is None and not self.isDissipationControlled():
            return True

        tol = self.constraintTolerance
        dTau = timeStep.stepProgressIncrement * self.dissipation
        dD = self.computeDissipation() - self.D_n
        if self.displacement is None:
            return abs(dD - dTau) <= tol * dTau

        dUMax = timeStep.stepProgressIncrement * self.displacement
        dUNorm = np.abs(dU[self._displacementIdcs(system.numbering)]).max()
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

    def finishIncrement(self, U, dU, dLambda, timeStep: TimeStep, system: DiscreteSystem):
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
        system
            The current discrete system.
        """

        # A zero increment controls nothing, but may change the dissipation, e.g. by plastic flow or damage in the
        # re-equilibration after a model change. The next increment starts from its state; the record of the last
        # increment (the predictor) is kept.
        if timeStep.stepProgressIncrement == 0.0:
            self._rebaseDissipation("zero increment")
            return

        D = self.computeDissipation()

        idcs = self._displacementIdcs(system.numbering)
        self.dULast = np.array(dU[idcs], copy=True)
        self.dULastKeys = self._displacementKeys(system.numbering)
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
