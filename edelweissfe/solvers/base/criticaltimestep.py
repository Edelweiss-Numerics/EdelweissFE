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
"""The critical time step of an explicit analysis, estimated part by part.

The central difference method is stable for :math:`\\Delta t \\le 2/\\omega_\\mathrm{max}`, the critical time step,
with :math:`\\omega_\\mathrm{max}^2` the largest eigenvalue of :math:`\\boldsymbol{M}^{-1}\\boldsymbol{K}`. The explicit
dynamic solver ``NED`` never solves that eigenvalue problem. It asks the parts of the model for an estimate of their
own critical time step -- a lower bound of it, from an upper bound of :math:`\\omega_\\mathrm{max}^2` --, takes the
smallest, and runs at the fraction ``courant-number`` of it. The parts have disjoint degrees of freedom, so the
smallest of their critical time steps is the model's.

A part gives its estimate as a :class:`CriticalTimeStepEstimate`. A penalty constraint -- a contact, a support, a tie
held by a penalty -- owns no degrees of freedom: it adds stiffness to the ones of the parts it couples. It has no
critical time step of its own to take the minimum with; it adds to the parts' estimates at the degrees of freedom it
couples: a part gives its estimate for them as a :class:`CriticalTimeStepEstimateOfTheDofs`, to which the row sums of
the constraints' mass-scaled stiffness are added.

An estimate costs something, and how much its value can change while the analysis runs depends on the part, so every
part (:class:`CriticalTimeStepOfAPart`) comes with its own policy of when its estimate is taken anew:

* :class:`CriticalTimeStepOfTheElements`: the finite elements, estimated **once per mesh**;
* :class:`CriticalTimeStepOfEntitiesWithOwnInertia`: the entities that carry their own inertia (the particles of a
  meshfree model), estimated anew **when the margin of the last estimate over the time increment in use may be used
  up**;
* :class:`CriticalTimeStepWithThePenaltyConstraints`: the parts with the penalty constraints, estimated **after every
  evaluation of the net force**, from the stiffness the constraints had in it. This third value is not the constraints'
  own critical time step (see the class).

The theory, and what the estimates do not see, are explained in :doc:`/theory/criticaltimestep`.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from edelweissfe.numerics.dofvector import DofVector


@dataclass
class CriticalTimeStepEstimate:
    """A part's estimate of its own critical time step, as the part gives it. A derived class also holds what the
    estimate was made from, so that what is added to the part's stiffness can be added to the estimate
    (:class:`CriticalTimeStepEstimateOfTheDofs`).

    Its one field, ``value``, is the critical time step of the part alone, not scaled by a courant number: a lower
    bound of :math:`2/\\omega_\\mathrm{max}` of the part's degrees of freedom; infinite for a part without stiffness.
    """

    value: float


@dataclass(eq=False)
class CriticalTimeStepEstimateOfTheDofs(CriticalTimeStepEstimate, ABC):
    """A part's estimate in a form the penalty constraints can be added to: made from upper bounds of the squared
    frequencies :math:`\\omega^2` of the part's degrees of freedom. A penalty constraint adds, at every degree of
    freedom :math:`i` it couples, the row sum :math:`r_i` of the absolute value of its stiffness scaled by the lumped
    mass on both sides, :math:`r_i = \\sum_j |K_{ij}| / \\sqrt{M_i M_j}`; the part's estimate with these additions is a
    lower bound of the critical time step of the part and the constraints together (see :doc:`/theory/criticaltimestep`,
    "Penalty constraints").

    Besides ``value``, its field ``dofs`` holds the degrees of freedom of the part, indices into the numbering it was
    estimated on.
    """

    dofs: np.ndarray

    @abstractmethod
    def criticalTimeStep(self, rowSumsOfThePenaltyConstraints: np.ndarray) -> float:
        """The critical time step of the part with the penalty constraints at its degrees of freedom: ``value`` where
        no constraint couples them.

        Parameters
        ----------
        rowSumsOfThePenaltyConstraints
            The row sums :math:`r_i` of the penalty constraints, one per degree of freedom of the numbering; zero where
            no constraint acts.

        Returns
        -------
        float
            The critical time step, at most ``value``.
        """


@dataclass(eq=False)
class CriticalTimeStepEstimateFromTheBoundOfEachDof(CriticalTimeStepEstimateOfTheDofs):
    """A part's estimate from one upper bound of the squared frequency per degree of freedom, :math:`\\omega_i^2`: the
    critical time step with the penalty constraints is :math:`2 / \\sqrt{\\max_i (\\omega_i^2 + r_i)}` over the degrees
    of freedom the constraints couple, or the part's own where that is smaller. The finite elements give it, with the
    largest squared frequency of the elements at each degree of freedom
    (:meth:`CriticalTimeStepOfTheElements.estimateFromTheBoundOfEachDof`).

    Besides ``value`` and ``dofs``, its field ``squaredFrequencies`` holds the upper bound of the squared frequency of
    each degree of freedom, in the order of ``dofs``.
    """

    squaredFrequencies: np.ndarray

    def criticalTimeStep(self, rowSumsOfThePenaltyConstraints: np.ndarray) -> float:
        rowSums = rowSumsOfThePenaltyConstraints[self.dofs]
        coupled = rowSums > 0.0
        if not coupled.any():
            return self.value
        return min(self.value, 2.0 / np.sqrt(np.max(self.squaredFrequencies[coupled] + rowSums[coupled])))


class CriticalTimeStepOfAPart:
    """The critical time step of one part of a model as last estimated (:attr:`lastEstimate`). A derived class says how
    the part is estimated, and when its estimate is taken anew (its policy).

    Parameters
    ----------
    part
        What the part is, for the journal.
    """

    def __init__(self, part: str):
        self.part = part
        self.reset()

    def reset(self):
        """Forget the estimate, at the start of every step: a material property may have changed with the step. The next
        estimate is due at once."""

        self.forget()
        #: How often the part was estimated in the step.
        self.numberOfEstimates = 0

    def forget(self):
        """Forget the last estimate, without counting an estimate: the model has no such part any more, e.g. the
        discrete system has lost its last penalty constraint. Its critical time step is infinite until it is estimated
        again."""

        #: The last estimate; None before the first one, and for a model without such a part.
        self.lastEstimate = None

    @property
    def value(self) -> float:
        """The critical time step of the part as last estimated, not scaled by a courant number; infinite before the
        first estimate, and for a model without such a part."""

        return np.inf if self.lastEstimate is None else self.lastEstimate.value

    def _take(self, estimate: CriticalTimeStepEstimate | None):
        """Take a new estimate.

        Parameters
        ----------
        estimate
            The estimate of the part; None if the model has no such part.
        """

        self.lastEstimate = estimate
        self.numberOfEstimates += 1


class CriticalTimeStepOfTheElements(CriticalTimeStepOfAPart):
    """The critical time step of the finite elements, the smallest of the elements' own estimates
    (``computeCriticalTimeStepForExplicitDynamics``): the largest eigenfrequency of an assembled mesh is bounded by the
    largest of its elements, so the smallest critical time step of an element is an estimate of the mesh's.

    **Policy: once per mesh, and at the start of every step.** An element's estimate depends on the mesh, on its lumped
    mass and on the wave speed of its material at the state of the estimate; a finite-strain element of Marmot takes its
    reference geometry with the wave speed at its current deformation. The policy assumes that within a step neither
    the geometry nor the wave speed changes enough to use up the margin that the courant number leaves. A softening
    material only raises the true limit; the assumption fails where elements are compressed far at finite strain, or
    where their material stiffens with the deformation (a hyperelastic solid in compression, a compacting material).
    There, take a lower ``courant-number``, or shorter steps, since every step estimates anew. So the elements are
    estimated anew when the mesh changes (a refinement) and at the start of every step, since a material property may
    have changed with the step; not for a new numbering of the same mesh, and never within an increment loop that does
    not change the mesh.
    """

    def __init__(self):
        super().__init__("the finite elements")

    def reset(self):
        super().reset()
        #: The version of the mesh the estimate is of (:attr:`~edelweissfe.models.topologypipeline.TopologyPipeline.version`).
        self._mesh = None
        #: The elements and their critical time steps, as last estimated.
        self._elements = []
        self._criticalTimeStepOfElement = np.zeros(0)
        #: The estimate from the bound of each degree of freedom (:meth:`estimateFromTheBoundOfEachDof`), and the
        #: numbering and the multi-point constraints it is of.
        self._estimateFromTheBoundOfEachDof = None
        self._numberingOfThatEstimate = None
        self._mpcTransformationOfThatEstimate = None

    def isDue(self, system) -> bool:
        """Whether the mesh is not the one of the last estimate.

        Parameters
        ----------
        system
            The discrete system.

        Returns
        -------
        bool
            True if the elements are to be estimated anew.
        """

        return self._mesh != system.model.topology.version

    def estimate(self, system, U: DofVector):
        """Estimate the critical time step of the elements of the discrete system
        (:attr:`~edelweissfe.models.discretesystem.DiscreteSystem.elementsWithNodalInertia`): the smallest of the
        elements' own. Each element's is kept, for the bound of each degree of freedom the penalty constraints add to
        (:meth:`estimateFromTheBoundOfEachDof`).

        Parameters
        ----------
        system
            The discrete system.
        U
            The solution, which an element may need for its current deformation.
        """

        self._elements = list(system.elementsWithNodalInertia)
        self._criticalTimeStepOfElement = np.array(
            [element.computeCriticalTimeStepForExplicitDynamics(U[element]) for element in self._elements]
        )
        self._mesh = system.model.topology.version
        self._estimateFromTheBoundOfEachDof = None
        self._take(CriticalTimeStepEstimate(np.min(self._criticalTimeStepOfElement, initial=np.inf)))

    def estimateFromTheBoundOfEachDof(
        self, system, mpcTransformation=None
    ) -> CriticalTimeStepEstimateFromTheBoundOfEachDof:
        """The last estimate in the form the penalty constraints add to: at each degree of freedom of the elements, the
        bound of its squared frequency, the largest squared frequency of the elements there, :math:`\\omega_i^2 =
        \\max_{e \\ni i} (2 / \\Delta t_e)^2`. Formed anew for every numbering, from the elements' critical time steps
        of the last estimate; its ``value`` is the smallest of them.

        With multi-point constraints, a slave moves with its masters, and an element at the slave acts on the masters
        (through :math:`T^T K T`): a master takes the largest squared frequency of the elements at it and at its slaves
        (:meth:`~edelweissfe.numerics.mpctransformation.MultiPointConstraintTransformation.foldLargestOntoMasters`),
        where the row sums of the penalty constraints, folded onto the masters as well, meet it (see
        :doc:`/theory/criticaltimestep`, "With multi-point constraints").

        Parameters
        ----------
        system
            The discrete system, whose numbering the estimate is of.
        mpcTransformation
            The condensation of the multi-point constraints of the discrete system; None if it has none.

        Returns
        -------
        CriticalTimeStepEstimateFromTheBoundOfEachDof
            The estimate.
        """

        if (
            self._estimateFromTheBoundOfEachDof is None
            or self._numberingOfThatEstimate is not system.numbering
            or self._mpcTransformationOfThatEstimate is not mpcTransformation
        ):
            dofsOfElement = [np.asarray(system.locations[element]) for element in self._elements]
            dofs = np.concatenate([np.zeros(0, dtype=np.intp)] + dofsOfElement)
            squaredFrequencyOfElement = (2.0 / self._criticalTimeStepOfElement) ** 2
            squaredFrequencies = np.zeros(system.numbering.nDof)
            np.maximum.at(
                squaredFrequencies, dofs, np.repeat(squaredFrequencyOfElement, [len(d) for d in dofsOfElement])
            )
            dofs = np.unique(dofs)
            if mpcTransformation is not None:
                ofTheElements = np.zeros(system.numbering.nDof)
                ofTheElements[dofs] = 1.0
                dofs = np.flatnonzero(mpcTransformation.foldLargestOntoMasters(ofTheElements))
                squaredFrequencies = mpcTransformation.foldLargestOntoMasters(squaredFrequencies)
            self._estimateFromTheBoundOfEachDof = CriticalTimeStepEstimateFromTheBoundOfEachDof(
                np.min(self._criticalTimeStepOfElement, initial=np.inf), dofs, squaredFrequencies[dofs]
            )
            self._numberingOfThatEstimate = system.numbering
            self._mpcTransformationOfThatEstimate = mpcTransformation
        return self._estimateFromTheBoundOfEachDof


class CriticalTimeStepOfEntitiesWithOwnInertia(CriticalTimeStepOfAPart):
    """The critical time step of the entities that carry their own inertia (the particles of a meshfree model), a property
    of their assembled system at their current configuration
    (:meth:`~edelweissfe.models.discretesystem.DiscreteSystem.criticalTimeStepOfEntitiesWithOwnInertia`).

    **Policy: when the margin may be used up.** It depends on the state of the entities -- their stiffness, their
    positions, their lumped inertia and damping -- and changes from increment to increment, but an estimate costs a pass
    over all of them. After an estimate :math:`v`, the solver therefore keeps, in every increment, the critical time
    step at worst, :math:`L`: a lower bound of the entities' critical time step now, :math:`n` increments after the
    estimate,

    .. math::

        L = v \\, \\min\\left(1, \\sqrt{\\alpha}, \\beta\\right) (1 - \\delta)^n ,

    and estimates anew as soon as the increment in use is no longer a ``courant-number`` fraction of it.

    * :math:`\\alpha = \\min_i M_i / M_i^0 \\le 1` is the smallest ratio of a lumped inertia now to the same inertia at
      the estimate (1 if none fell), over the degrees of freedom that had one then and have one now, and
      :math:`\\beta` the same for the damping of the degrees of freedom integrated by forward Euler (a damping, no
      inertia). This part is exact: the estimate scales the stiffness with :math:`\\boldsymbol{M}^{-1/2}` on both
      sides, so with the stiffness held, a fall of the masses to :math:`\\alpha` times raises the bound of
      :math:`\\omega_\\mathrm{max}^2` at most :math:`1/\\alpha` times and lowers :math:`2/\\omega_\\mathrm{max}` at most
      to :math:`\\sqrt{\\alpha}` times; the forward Euler limit :math:`2/\\lambda_\\mathrm{max}(\\boldsymbol{C}^{-1}
      \\boldsymbol{K})` is linear in the damping, so it falls at most to :math:`\\beta` times. The masses are known in
      every increment: the solver assembles them anyway.
    * :math:`\\delta` (option ``critical-time-step-decrease-per-increment``) is the largest relative fall of the estimate
      per increment that its stiffness may cause. This part is an assumption, not a theorem: the tangent of a material
      can change abruptly, and the stiffness of a particle changes with its deformation. It rests on what an increment
      is -- a fraction of the time a wave needs to cross an entity, so the configuration moves by a fraction of an entity
      -- and on measurements, and it is documented with them in :doc:`/theory/criticaltimestep`.

    The estimate is also taken anew when a degree of freedom has an inertia, or a damping alone, that it did not have at
    the last estimate (a node that a particle reaches for the first time: a new row whose mass starts from zero), after
    a new numbering, and in the first evaluation of a step. A degree of freedom that loses its inertia leaves the system
    and can only raise its critical time step.

    Where the entities set the time increment, the margin is used up after one increment, and they are estimated in
    every increment; where another part sets it, they are estimated only every so many increments, as many as their
    margin allows at the rate :math:`\\delta`.

    **Carried weights** (option ``carry-critical-time-step-weights``). The estimate of the entities is made from a
    Collatz-Wielandt bound with weights constant per field
    (:meth:`~edelweissfe.models.discretesystem.DiscreteSystem.criticalTimeStepOfEntitiesWithOwnInertia`). With carried
    weights, every estimate takes the weights the last one left and leaves the next ones, one step of the power
    iteration towards the Perron vector, where the bound is exact. Any positive weights bound the largest eigenvalue
    from above, but weights refined at another configuration can bound it less tightly than the weights constant per
    field they started from when the stiffness changed faster than the weights followed, e.g. where entities reach
    nodes whose weights the power iteration has made small. So an estimate with carried weights below the estimate the
    weights started from starts them anew: the entities are estimated again from the weights constant per field, and
    the larger of the two estimates is taken. Every estimate is thus at least the one the weights started from: with
    the time increment lowered only, it never falls below the smallest estimate with weights constant per field of
    the step; followed (``follow-critical-time-step``), it can fall below the estimate without carried weights of the
    same configuration, down to the one the weights started from. The weights start anew -- constant per field -- also
    with every numbering (the nodes they belong to may have changed) and every step (a material property may have
    changed), and after a resume: they are not part of the checkpoint.

    Parameters
    ----------
    decreasePerIncrement
        :math:`\\delta`, in :math:`(0, 1]`; 1 estimates in every increment.
    carryWeights
        Whether the weights of the estimate are carried from one estimate to the next.
    """

    def __init__(self, decreasePerIncrement: float, carryWeights: bool = False):
        if not 0.0 < decreasePerIncrement <= 1.0:
            raise ValueError(
                "The decrease of the critical time step per increment must lie in (0, 1], not {:}.".format(
                    decreasePerIncrement
                )
            )
        self.decreasePerIncrement = decreasePerIncrement
        self.carryWeights = carryWeights
        super().__init__("the entities that carry their own inertia")

    def reset(self):
        super().reset()
        #: The numbering, the number of the increment, and the lumped inertia and damping of the last estimate.
        self._numbering = None
        self._increment = None
        self._inertia = None
        self._damping = None
        #: With carried weights, the weights the last estimate left for the next one, a vector of its numbering, zero
        #: where a degree of freedom has none yet; None without carried weights, and before the first estimate on a
        #: numbering.
        self._weightsOfTheEstimate = None
        #: With carried weights, the critical time step of the estimate the weights started from, the first one after
        #: they were set to zero, with weights constant per field.
        self._valueTheWeightsStartedFrom = np.inf

    def estimate(self, system, lumpedInertia: DofVector, lumpedDamping: DofVector, forEachChunk, increment: int):
        """Estimate the critical time step of the entities at their current configuration; with carried weights, from
        the weights the last estimate left, started anew where it falls below the estimate they started from (see the
        class documentation, "Carried weights").

        Parameters
        ----------
        system
            The discrete system.
        lumpedInertia
            The lumped inertia of the entities at their current configuration, as the solver has just assembled it.
        lumpedDamping
            Their lumped damping, likewise.
        forEachChunk
            The solver's loop over chunks of independent entities.
        increment
            The number of the increment just accepted.
        """

        if self.carryWeights and self._numbering is not system.numbering:
            self._weightsOfTheEstimate = np.zeros(system.numbering.nDof)
        weights = self._weightsOfTheEstimate
        startsTheWeights = weights is not None and not weights.any()

        estimate = system.criticalTimeStepOfEntitiesWithOwnInertia(lumpedInertia, lumpedDamping, forEachChunk, weights)
        if estimate is not None and weights is not None:
            if startsTheWeights:
                self._valueTheWeightsStartedFrom = estimate.value
            elif estimate.value < self._valueTheWeightsStartedFrom:
                weights[:] = 0.0
                anew = system.criticalTimeStepOfEntitiesWithOwnInertia(
                    lumpedInertia, lumpedDamping, forEachChunk, weights
                )
                self._valueTheWeightsStartedFrom = anew.value
                if anew.value > estimate.value:
                    estimate = anew
        self._take(estimate)
        self._numbering = system.numbering
        self._increment = increment
        self._inertia = np.array(lumpedInertia)
        self._damping = np.array(lumpedDamping)

    def criticalTimeStepAtWorst(self, increment: int, lumpedInertia: DofVector, lumpedDamping: DofVector) -> float:
        """The critical time step at worst, :math:`L`, after the increment ``increment``, :math:`n` increments after the
        last estimate: a lower bound of the entities' critical time step now (see the class documentation). It holds
        for the degrees of freedom that had an inertia (or a damping alone) at the last estimate; whether there are
        others, :meth:`isDue` asks first.

        Parameters
        ----------
        increment
            The number of the increment just accepted.
        lumpedInertia
            The lumped inertia of the entities now.
        lumpedDamping
            Their lumped damping now.

        Returns
        -------
        float
            The critical time step at worst.
        """

        massFactor, stiffnessFactor = self._fallSinceTheEstimate(increment, lumpedInertia, lumpedDamping)
        return self.value * massFactor * stiffnessFactor

    def _fallSinceTheEstimate(
        self, increment: int, lumpedInertia: DofVector, lumpedDamping: DofVector
    ) -> tuple[float, float]:
        """The factors :math:`\\min(1, \\sqrt{\\alpha}, \\beta)` and :math:`(1 - \\delta)^n` by which the critical time
        step of the entities may have fallen since the last estimate, from their masses and from their stiffness (see
        the class documentation).

        Parameters
        ----------
        increment
            The number of the increment just accepted.
        lumpedInertia
            The lumped inertia of the entities now.
        lumpedDamping
            Their lumped damping now.

        Returns
        -------
        tuple[float, float]
            The two factors, each in (0, 1].
        """

        inertia, damping = np.asarray(lumpedInertia), np.asarray(lumpedDamping)
        withInertia = (inertia > 0.0) & (self._inertia > 0.0)
        withDampingAlone = ~(inertia > 0.0) & (damping > 0.0) & ~(self._inertia > 0.0) & (self._damping > 0.0)
        alpha = np.min(inertia[withInertia] / self._inertia[withInertia], initial=1.0)
        beta = np.min(damping[withDampingAlone] / self._damping[withDampingAlone], initial=1.0)
        return min(1.0, np.sqrt(alpha), beta), (1.0 - self.decreasePerIncrement) ** (increment - self._increment)

    def isDue(
        self,
        system,
        increment: int,
        lumpedInertia: DofVector,
        lumpedDamping: DofVector,
        criticalTimeStepInUse: float,
        rowSumsOfThePenaltyConstraints: np.ndarray | None = None,
    ) -> bool:
        """Whether the entities are to be estimated anew after the increment ``increment``.

        Parameters
        ----------
        system
            The discrete system.
        increment
            The number of the increment just accepted.
        lumpedInertia
            The lumped inertia of the entities now.
        lumpedDamping
            Their lumped damping now.
        criticalTimeStepInUse
            The critical time step the time increment in use is the ``courant-number`` fraction of.
        rowSumsOfThePenaltyConstraints
            The row sums of the penalty constraints in this increment
            (:class:`CriticalTimeStepWithThePenaltyConstraints`), if the model has penalty constraints.

        Returns
        -------
        bool
            True in the first evaluation of a step, after a new numbering, when a degree of freedom has an inertia (or a
            damping alone) it did not have at the last estimate, and when the critical time step at worst
            (:meth:`criticalTimeStepAtWorst`) has fallen below the one in use -- alone, or with the penalty constraints
            that couple the entities' degrees of freedom: their row sums are those of now, the entities' stiffness may
            have grown by the square of the same factor as in :meth:`criticalTimeStepAtWorst`.
        """

        if self._numbering is not system.numbering:
            return True
        inertia, damping = np.asarray(lumpedInertia), np.asarray(lumpedDamping)
        hadInertia = self._inertia > 0.0
        hadDampingAlone = ~hadInertia & (self._damping > 0.0)
        hasInertia = inertia > 0.0
        hasDampingAlone = ~hasInertia & (damping > 0.0)
        if np.any(hasInertia & ~hadInertia) or np.any(hasDampingAlone & ~hadDampingAlone):
            return True
        if self.criticalTimeStepAtWorst(increment, lumpedInertia, lumpedDamping) < criticalTimeStepInUse:
            return True
        if rowSumsOfThePenaltyConstraints is None or not np.any(
            rowSumsOfThePenaltyConstraints[self.lastEstimate.dofs] > 0.0
        ):
            return False
        # With the entities' stiffness grown by 1/fall^2, their critical time step with the constraints is fall times
        # the one with the constraints' row sums scaled by fall^2.
        massFactor, stiffnessFactor = self._fallSinceTheEstimate(increment, lumpedInertia, lumpedDamping)
        fall = massFactor * stiffnessFactor
        return (
            fall * self.lastEstimate.criticalTimeStep(fall**2 * rowSumsOfThePenaltyConstraints) < criticalTimeStepInUse
        )

    def withThePenaltyConstraints(self, rowSumsOfThePenaltyConstraints: np.ndarray) -> float:
        """The critical time step of the entities with the penalty constraints that couple their degrees of freedom,
        from the last estimate (:meth:`CriticalTimeStepEstimateOfTheDofs.criticalTimeStep`).

        Parameters
        ----------
        rowSumsOfThePenaltyConstraints
            The row sums of the penalty constraints, one per degree of freedom.

        Returns
        -------
        float
            The critical time step; infinite in a model without such entities.
        """

        return (
            np.inf if self.lastEstimate is None else self.lastEstimate.criticalTimeStep(rowSumsOfThePenaltyConstraints)
        )


class CriticalTimeStepWithThePenaltyConstraints(CriticalTimeStepOfAPart):
    """The critical time step of the parts with the penalty constraints -- contacts, supports and ties held by a
    penalty: the estimate of each part (:class:`CriticalTimeStepEstimateOfTheDofs`) with the row sums of the
    constraints added at the degrees of freedom they couple,

    .. math::

        \\omega_\\mathrm{max}^2 \\le \\max_i \\left(\\omega_i^2 + r_i\\right), \\qquad
        r_i = \\frac{1}{\\sqrt{M_i}} \\sum_j |K^\\mathrm{c}_{ij}| \\frac{1}{\\sqrt{M_j}} ,

    with :math:`\\omega_i^2` the part's bound of the squared frequency at the degree of freedom :math:`i` and
    :math:`\\boldsymbol{K}^\\mathrm{c}` the stiffness of the constraints
    (:meth:`~edelweissfe.constraints.base.constraintbase.ConstraintBase.addBoundOfAbsoluteStiffnessTimes`). A constraint
    has no critical time step of its own: it raises the frequencies of the degrees of freedom it couples, so it lowers
    the critical time step only where it acts, and only by what it adds there.

    **The value is not the constraints' own.** It is the critical time step of the parts *with* the constraints' rows
    added at the degrees of freedom they couple, so it is never above the parts' own estimates as they were when it was
    taken; and a part estimated anew is estimated before it (the elements of a new mesh, the entities right before it),
    so it is never above their current ones either. The model's critical time step is nevertheless the minimum of all
    three (``NED._criticalTimeStepOfTheModel``), so that it needs no knowledge of which part has constraints: without
    constraints this value is infinite -- before their first estimate in a step, and from the estimate that finds the
    discrete system without any (:meth:`~CriticalTimeStepOfAPart.forget`), e.g. once the last particle has left a wall,
    so that the time increment it lowered can follow the parts again.

    **Policy: after every evaluation of the net force.** Increment :math:`n+1` integrates the net force
    :math:`\\boldsymbol{f}_n`, evaluated at the end of increment :math:`n`, over the average of its own time
    increment and the one before, :math:`\\tfrac{1}{2}(\\Delta t_n + \\Delta t_{n+1})`. :math:`\\Delta t_{n+1}` is set
    right after :math:`\\boldsymbol{f}_n`, so it has to be within the critical time step of the stiffness that force
    was evaluated with. The stiffness of a constraint follows the state of the contact -- which of its points are
    closed, and how deep --, which changes from increment to increment, and bounding it costs about as much as
    evaluating its force. So it is estimated right after the net force is, at the same state and on the same
    connectivity: at the start of a step and after every increment; and after a refinement, at the state on the new
    mesh, before the first increment on it. :math:`\\Delta t_n` was set with :math:`\\boldsymbol{f}_{n-1}`: stiffness
    that is new in :math:`\\boldsymbol{f}_n` -- a point that closed in increment :math:`n`, a point a contact search
    paired at its start -- is half an increment late, since the average still holds an increment taken without it
    (:doc:`/theory/criticaltimestep`, "What the estimate does not see").
    """

    def __init__(self):
        super().__init__("the parts with the penalty constraints")

    def estimate(
        self,
        system,
        rowSumsOfThePenaltyConstraints: np.ndarray,
        elements: CriticalTimeStepOfTheElements,
        entities: CriticalTimeStepOfEntitiesWithOwnInertia,
        mpcTransformation=None,
    ):
        """Estimate the critical time step of the parts with the penalty constraints, from the last estimates of the
        parts.

        Parameters
        ----------
        system
            The discrete system.
        rowSumsOfThePenaltyConstraints
            The row sums :math:`r_i` of the penalty constraints, one per degree of freedom of the numbering, folded onto
            the masters of multi-point constraints.
        elements
            The critical time step of the finite elements, as last estimated.
        entities
            The critical time step of the entities that carry their own inertia, as last estimated.
        mpcTransformation
            The condensation of the multi-point constraints of the discrete system; None if it has none
            (:meth:`CriticalTimeStepOfTheElements.estimateFromTheBoundOfEachDof`).
        """

        self._take(
            CriticalTimeStepEstimate(
                min(
                    elements.estimateFromTheBoundOfEachDof(system, mpcTransformation).criticalTimeStep(
                        rowSumsOfThePenaltyConstraints
                    ),
                    entities.withThePenaltyConstraints(rowSumsOfThePenaltyConstraints),
                )
            )
        )
