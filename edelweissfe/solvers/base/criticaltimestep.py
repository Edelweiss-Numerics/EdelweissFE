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
own critical time step -- a lower bound of it --, takes the smallest, and runs at the fraction ``courant-number`` of it.
The parts have disjoint degrees of freedom, so the smallest of their critical time steps is the model's.

An estimate costs something, and how much its value can change while the analysis runs depends on the part, so every
part comes with its own policy of when its estimate is taken anew:

* :class:`CriticalTimeStepOfTheElements`: the finite elements, estimated **once per mesh**;
* :class:`CriticalTimeStepOfEntitiesWithOwnInertia`: the entities that carry their own inertia (the particles of a
  meshfree model), estimated anew **when the margin of the last estimate over the time increment in use may be used up**.

The theory, and what is not estimated, is explained in :doc:`/theory/criticaltimestep`.
"""

import numpy as np

from edelweissfe.numerics.dofvector import DofVector


class CriticalTimeStepBound:
    """The estimate of the critical time step of one part of a model, as the part gives it: its value, and, in a derived
    class, what it was estimated from.

    Attributes
    ----------
    value : float
        The critical time step of the part alone.
    """

    value: float


class CriticalTimeStepEstimate:
    """The critical time step of one part of a model, :math:`2/\\omega_\\mathrm{max}` of the part's degrees of freedom,
    as last estimated: a lower bound, not scaled by a courant number. A derived class says how the part is estimated,
    and when its estimate is taken anew (its policy).

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

        #: The last estimate; infinite before the first one, and for a part without degrees of freedom.
        self.value = np.inf
        #: How often the part was estimated in the step.
        self.numberOfEstimates = 0

    def _take(self, value: float | None):
        """Take a new estimate.

        Parameters
        ----------
        value
            The critical time step of the part; None if the model has no such part.
        """

        self.value = np.inf if value is None else value
        self.numberOfEstimates += 1


class CriticalTimeStepOfTheElements(CriticalTimeStepEstimate):
    """The critical time step of the finite elements, the smallest of the elements' own
    (``computeCriticalTimeStepForExplicitDynamics``): the largest eigenfrequency of an assembled mesh is bounded by the
    largest of its elements, so the smallest critical time step of an element bounds the mesh's.

    **Policy: once per mesh.** An element's estimate depends on its geometry, its lumped mass and the largest wave speed
    of its material, not on the state the material is in: a softening material only raises the true limit. So it is
    estimated anew only when the mesh changes (a refinement), and at the start of every step; not for a new numbering of
    the same mesh, and never within an increment loop that does not change the mesh.
    """

    def __init__(self):
        super().__init__("the finite elements")

    def reset(self):
        super().reset()
        #: The version of the mesh the estimate is of (:attr:`~edelweissfe.models.topologypipeline.TopologyPipeline.version`).
        self._mesh = None

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
        (:attr:`~edelweissfe.models.discretesystem.DiscreteSystem.elementsWithNodalInertia`).

        Parameters
        ----------
        system
            The discrete system.
        U
            The solution, which an element may need for its current geometry.
        """

        self._take(
            min(
                (
                    element.computeCriticalTimeStepForExplicitDynamics(U[element])
                    for element in system.elementsWithNodalInertia
                ),
                default=np.inf,
            )
        )
        self._mesh = system.model.topology.version


class CriticalTimeStepOfEntitiesWithOwnInertia(CriticalTimeStepEstimate):
    """The critical time step of the entities that carry their own inertia (the particles of a meshfree model), a property
    of their assembled system at their current configuration
    (:meth:`~edelweissfe.models.discretesystem.DiscreteSystem.criticalTimeStepOfEntitiesWithOwnInertia`).

    **Policy: when the margin may be used up.** It depends on the state of the entities -- their stiffness, their
    positions, their lumped inertia and damping -- and changes from increment to increment, but an estimate costs a pass
    over all of them. After an estimate :math:`v`, the solver therefore keeps, in every increment, a lower bound of what
    the estimate could be now, :math:`n` increments later,

    .. math::

        L = v \\, \\min\\left(1, \\sqrt{\\alpha}, \\beta\\right) (1 - \\delta)^n ,

    and estimates anew as soon as the increment in use is no longer a ``courant-number`` fraction of it.

    * :math:`\\alpha = \\min_i M_i / M_{0,i}` is the largest relative fall of the lumped inertia since the estimate, over
      the degrees of freedom that had one then and have one now, and :math:`\\beta` the same for the damping of the
      degrees of freedom integrated by forward Euler (a damping, no inertia). This part is exact: the estimate scales the
      stiffness with :math:`\\boldsymbol{M}^{-1/2}` on both sides, so with the stiffness held, a fall of the masses by
      :math:`\\alpha` raises :math:`\\omega_\\mathrm{max}^2` at most by :math:`1/\\alpha` and lowers :math:`2/\\omega_\\mathrm{max}`
      at most by :math:`\\sqrt{\\alpha}`; the forward Euler limit :math:`2/\\lambda_\\mathrm{max}(\\boldsymbol{C}^{-1}
      \\boldsymbol{K})` by :math:`\\beta`. The masses are known in every increment: the solver assembles them anyway.
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

    **Carried weights** (option ``carry-critical-time-step-weights``). The estimate of the entities is a
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
        #: The last estimate; None before the first one, and for a model without such entities.
        self.bound = None
        #: The numbering, the number of the increment, and the lumped inertia and damping of the last estimate.
        self._numbering = None
        self._increment = None
        self._inertia = None
        self._damping = None
        #: With carried weights, the weights the last estimate left for the next one, a vector of its numbering, zero
        #: where a degree of freedom has none yet; None without carried weights, and before the first estimate on a
        #: numbering.
        self._weightsOfTheBound = None
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
            self._weightsOfTheBound = np.zeros(system.numbering.nDof)
        weights = self._weightsOfTheBound
        startsTheWeights = weights is not None and not weights.any()

        bound = system.criticalTimeStepOfEntitiesWithOwnInertia(lumpedInertia, lumpedDamping, forEachChunk, weights)
        if bound is not None and weights is not None:
            if startsTheWeights:
                self._valueTheWeightsStartedFrom = bound.value
            elif bound.value < self._valueTheWeightsStartedFrom:
                weights[:] = 0.0
                anew = system.criticalTimeStepOfEntitiesWithOwnInertia(
                    lumpedInertia, lumpedDamping, forEachChunk, weights
                )
                self._valueTheWeightsStartedFrom = anew.value
                if anew.value > bound.value:
                    bound = anew
        self.bound = bound
        self._take(None if self.bound is None else self.bound.value)
        self._numbering = system.numbering
        self._increment = increment
        self._inertia = np.array(lumpedInertia)
        self._damping = np.array(lumpedDamping)

    def lowerBound(self, increment: int, lumpedInertia: DofVector, lumpedDamping: DofVector) -> float:
        """The lower bound :math:`L` of the critical time step of the entities after the increment ``increment``, from the
        last estimate (see the class documentation). It holds for the degrees of freedom that had an inertia (or a
        damping alone) at the last estimate; whether there are others, :meth:`isDue` asks first.

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
            The lower bound.
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

        Returns
        -------
        bool
            True in the first evaluation of a step, after a new numbering, when a degree of freedom has an inertia (or a
            damping alone) it did not have at the last estimate, and when the lower bound of the critical time step
            (:meth:`lowerBound`) has fallen below the one in use.
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
        return self.lowerBound(increment, lumpedInertia, lumpedDamping) < criticalTimeStepInUse
