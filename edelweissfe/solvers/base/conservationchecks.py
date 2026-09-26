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
"""What a topology change ought to leave alone, checked the same way by every dynamic solver.

A refinement carries the kinematic state onto new nodes by isoparametric interpolation and
reassembles (or re-lumps) every per-element coefficient on the new mesh. Three statements follow,
in decreasing strength, and they hold for the explicit and the implicit dynamic solver alike:

* **Every field's own total coefficient -- a mass, a viscosity, a micro-inertia -- is conserved
  exactly.** The children of a refined element tile it and carry the same value, so this is a
  geometric identity; the quadrature that assembles it is exact only to a polynomial order, which
  admits a small change on a distorted element. Checked per field, never summed across fields: a
  violation in a numerically small field would otherwise hide inside a large one. Violating it
  raises (:class:`ConservationCheck`).
* **Linear momentum is conserved exactly for a spatially uniform velocity field**, because the
  shape functions are a partition of unity; for a general field the discrepancy is second order in
  the velocity gradient across the refined parent -- discretisation error, not a defect. Reported
  per spatial component (:func:`linearMomentum`), never enforced.
* **Kinetic energy is not conserved** in general and is the most sensitive of the three, being
  quadratic in the interpolation error. Reported as a relative jump; more than roughly a percent is
  a reason to look at the transfer rather than to believe the physics.

What differs between the solvers is only how their mass is represented -- a lumped vector or a
consistent matrix -- so each computes its own totals and its own mass-weighted velocity, and hands
them to the functions here.
"""

import numpy as np

from edelweissfe.models.femodel import FEModel
from edelweissfe.numerics.dofmanager import DofManager

#: Default tolerance on the relative change, across one topology change, of a field's total
#: coefficient. Conservation is a geometric identity, but the assembly is Gauss quadrature, exact
#: only to a polynomial order, and a distorted hexa20's Jacobian is not polynomial. Measured on the
#: anchor pry-out, one live refinement moves the total mass by 4.63e-08 relative; a real refinement
#: or assembly error would be O(1).
CONSERVATION_TOLERANCE = 1e-6

#: Tolerance on the accumulated drift of any one total over a whole step: a single change being
#: within tolerance does not bound a run with hundreds of refinements. At the measured 4.63e-08 per
#: change this permits over two thousand of them.
CUMULATIVE_DRIFT_TOLERANCE = 1e-4


def linearMomentum(
    massTimesVelocity: np.ndarray, dofManager: DofManager, fieldNames: list[str], model: FEModel
) -> np.ndarray:
    """The linear momentum of the given fields, per spatial component.

    Per component, not summed over them: adding a momentum's x, y and z contributions produces a
    number with no physical meaning and would hide a component-wise error behind a cancellation.
    Across fields the sum is legitimate -- a system's momentum is the sum of its parts'. A field
    occupies a contiguous slice of the dof vector, node-major with the component innermost (what
    ``writeNodeFieldToDofVector``'s flatten establishes), so reshaping the slice recovers the
    per-node vectors.

    Parameters
    ----------
    massTimesVelocity
        The mass applied to the velocity -- :math:`m \\odot v` for a lumped mass, :math:`M v` for a
        consistent one -- as a vector over the dof manager's degrees of freedom.
    dofManager
        The manager that vector is indexed by.
    fieldNames
        The fields whose inertia is a mass; any other has no linear momentum.
    model
        The model tree, for the fields' spatial dimension.

    Returns
    -------
    np.ndarray
        The momentum, one entry per spatial component; empty when no field is given.

    Raises
    ------
    ValueError
        If two fields differ in spatial dimension, so that their momenta have no common components.
    """

    total = None
    for fieldName in fieldNames:
        indices = dofManager.idcsOfFieldsInDofVector[fieldName]
        dimension = model.nodeFields[fieldName].dimension
        contribution = np.asarray(massTimesVelocity[indices]).reshape((-1, dimension)).sum(axis=0)
        # numpy would broadcast a shape (1,) contribution onto every component of a shape (3,)
        # total instead of refusing to add them
        if total is not None and contribution.shape != total.shape:
            raise ValueError(
                "Field {:} has dimension {:} against {:} for the fields before it, so their momenta "
                "have no common components to add.".format(fieldName, contribution.shape[0], total.shape[0])
            )
        total = contribution if total is None else total + contribution

    return total if total is not None else np.zeros(0)


def formatMomentumAndKineticEnergy(
    momentumBefore: np.ndarray, momentumAfter: np.ndarray, kineticBefore: float, kineticAfter: float
) -> str:
    """The momentum and kinetic-energy part of a topology-change report, in one wording for every
    solver.

    Parameters
    ----------
    momentumBefore, momentumAfter
        Per-component momentum before and after the change.
    kineticBefore, kineticAfter
        Kinetic energy before and after the change.

    Returns
    -------
    str
        The largest momentum component change against the largest component before, and the kinetic
        energy before, after and as a signed relative jump.
    """

    momentumChange = float(np.max(np.abs(momentumAfter - momentumBefore))) if momentumBefore.size else 0.0
    momentumScale = float(np.max(np.abs(momentumBefore))) if momentumBefore.size else 0.0
    kineticJump = abs(kineticAfter - kineticBefore) / kineticBefore if kineticBefore > 0.0 else 0.0

    return "largest momentum component change {:.3e} (of {:.3e}); kinetic energy {:.6e} -> {:.6e} ({:+.2f} %)".format(
        momentumChange,
        momentumScale,
        kineticBefore,
        kineticAfter,
        kineticJump * 100.0 * (1.0 if kineticAfter >= kineticBefore else -1.0),
    )


class ConservationCheck:
    """The per-change check of a conserved total, and the drift those changes accumulate over a
    step.

    Each total is checked against the tolerance on its own and raises on a violation. What a single
    check cannot bound is many individually tolerable changes adding up; the check sums them per
    total and warns once per step when the sum exceeds :data:`CUMULATIVE_DRIFT_TOLERANCE` --
    reported, not raised, because what accumulates there is quadrature error rather than a violated
    invariant, and aborting a multi-hour run on an accumulated heuristic is out of proportion.

    Parameters
    ----------
    journal
        The journal to report to.
    identification
        The owning solver's identification, for the journal.
    """

    def __init__(self, journal, identification: str):
        self.journal = journal
        self.identification = identification
        self.reset()

    def reset(self):
        """Start a new step: forget the accumulated drift and which totals were warned about."""

        #: Summed relative drift of each total over the step's topology changes, by label.
        self._cumulativeDrift = {}
        #: The labels whose cumulative-drift warning has fired this step.
        self._warned = set()

    def check(self, label: str, before: float, after: float, tolerance: float) -> float:
        """Check one total for conservation across a topology change.

        Parameters
        ----------
        label
            What the total is, e.g. ``"mass of field 'displacement'"``; used in the messages and as
            the key of its accumulated drift.
        before, after
            The total before and after the change.
        tolerance
            The largest admissible relative change.

        Returns
        -------
        float
            The relative change, for the caller to report.

        Raises
        ------
        RuntimeError
            If the relative change exceeds ``tolerance``.
        """

        relativeChange = abs(after - before) / before if before > 0.0 else 0.0
        self._cumulativeDrift[label] = self._cumulativeDrift.get(label, 0.0) + relativeChange

        if relativeChange > tolerance:
            raise RuntimeError(
                "A topology change did not conserve the total {:}: {:e} became {:e}, a relative change "
                "of {:e} against a tolerance of {:e}. The children of a refined element tile it and "
                "carry the same value, so it is conserved geometrically; the quadrature that assembles "
                "it is exact only up to a polynomial order, which admits a small change. A violation of "
                "this size is not quadrature -- it means the refinement or the assembly is "
                "wrong.".format(label, before, after, relativeChange, tolerance)
            )

        if self._cumulativeDrift[label] > CUMULATIVE_DRIFT_TOLERANCE and label not in self._warned:
            self._warned.add(label)
            self.journal.message(
                "The accumulated relative drift of the total {:} over this step has reached {:e}, above "
                "the tolerance of {:e}. Each individual topology change was within its own bound, so "
                "this is many small quadrature changes adding up rather than one bad refinement; the "
                "model's {:} is no longer the one the step started with.".format(
                    label, self._cumulativeDrift[label], CUMULATIVE_DRIFT_TOLERANCE, label
                ),
                self.identification,
                1,
            )

        return relativeChange
