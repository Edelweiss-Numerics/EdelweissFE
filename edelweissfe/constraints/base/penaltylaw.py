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
The laws of the penalty springs of the contact constraints: which springs a contact point gets, and how hard they pull.

A contact constraint is made of three pieces: its *contact points* (where the slave surface is, see
:mod:`~edelweissfe.constraints.base.contactpoints`), the *geometry* that pairs each of them with a point of the master
surface and measures where it is relative to that point, and the *law* that turns this into penalty springs
(:class:`~edelweissfe.constraints.base.penaltysprings.PenaltySprings`). The law is a separate object, held by the
constraint. It owns what differs from one law to the next, and nothing else: **its springs**
(:meth:`PenaltyLaw.springs`). Given, for the paired points, their gaps along the frozen normals, the normals and the
penalty times the area of each point, it returns the springs it puts there -- their points, directions, forces,
stiffnesses and stored energies.

The constraint owns what every law shares: the contact points and where they are, the search and the pairing, the
relative position and the gap, the force and the tangent of the springs on the nodes, the bound of their stiffness for
the critical time step (:mod:`~edelweissfe.constraints.base.penaltysprings`), the state of the points in a checkpoint,
and the outputs, which it derives from the springs alone. So a new law takes its place without touching the
constraint.

The laws of contact are :class:`ContactPenaltyLaw`: a unilateral normal spring at every point whose gap is closed.
Its normal force follows from the gap alone (:meth:`ContactPenaltyLaw.normalForce`), which is also what the node-based
contacts (one gap per slave node) ask of it. :class:`LinearPenaltyLaw` and :class:`QuadraticPenaltyLaw`.

What a friction law would implement (not implemented): a :class:`PenaltyLaw` whose :meth:`~PenaltyLaw.springs` returns
the normal spring of a contact law (from ``gaps``, as :class:`ContactPenaltyLaw` does) and, at the closed points, the
tangential springs: along the tangential part :math:`(\\boldsymbol{I} - \\bar{\\boldsymbol{n}} \\otimes
\\bar{\\boldsymbol{n}})\\, \\boldsymbol{d}` of the displacement of the point from where it rests
:math:`\\boldsymbol{d}`, of the tangential penalty, their force limited to :math:`\\mu |f_n|` by a radial return. A
contact point rests on its master point; a sticking point rests where it stuck, its *stick point*, which is its history:
a slip moves it. Three things then join, all independent of the law: the law is given the displacements of the points
from where they rest, the constraint keeps the stick point of every point, promoted in
:meth:`~edelweissfe.constraints.base.constraintbase.ConstraintBase.acceptLastState` (not in every iteration), and a
point paired anew with another master point keeps its stick point where it is in space, so that the history follows the
point.

Sign convention: the gap :math:`g` is negative when the surfaces penetrate, and the force :math:`f` of a spring
carries the sign of its stretch, i.e. :math:`f_n \\le 0` in contact. The stiffness :math:`\\mathrm{d}f/\\mathrm{d}g` is
then positive for :math:`g < 0`. ``penaltyTimesArea`` is the penalty modulus times the area the contact point
represents: a nodal tributary area in node-based formulations, a quadrature weight in integrated ones.
"""

from abc import ABC, abstractmethod

import numpy as np

from edelweissfe.constraints.base.penaltysprings import PenaltySprings


class PenaltyLaw(ABC):
    """The law of the penalty springs of a contact constraint: the springs it puts at the paired contact points. See
    the module documentation for what a law owns and what the constraint owns."""

    #: The name of the law, as the input file spells it (the ``type`` option of the contact constraints).
    name: str

    @abstractmethod
    def springs(self, penaltyTimesArea: np.ndarray, gaps: np.ndarray, normals: np.ndarray) -> list[PenaltySprings]:
        """The penalty springs at the paired contact points.

        Parameters
        ----------
        penaltyTimesArea
            The penalty modulus times the area of each point.
        gaps
            The gap of each point along its normal, negative in contact, as the law sees it (beyond an initial overlap
            the constraint does not count).
        normals
            The unit normal of the master at each point, frozen at the pairing, (nPoints, nDim).

        Returns
        -------
        list[PenaltySprings]
            The springs, each set along one direction per point; empty if there is none.
        """


class ContactPenaltyLaw(PenaltyLaw):
    """Contact: a unilateral normal spring. A point gets one spring, along its normal, while its gap is closed
    (negative); its normal force follows from the gap alone. Every method works element-wise on arrays of points as
    well as on single points."""

    def isClosed(self, gap: float | np.ndarray) -> bool | np.ndarray:
        """Whether the contact points with these gaps transmit a force: while they penetrate.

        Parameters
        ----------
        gap
            The gap(s), negative in contact.

        Returns
        -------
        bool or numpy.ndarray
            True where the gap is closed.
        """

        return gap < 0.0

    @abstractmethod
    def normalForce(
        self, penaltyTimesArea: float | np.ndarray, gap: float | np.ndarray
    ) -> tuple[float | np.ndarray, float | np.ndarray]:
        """The normal force at closed contact points and its derivative with respect to the gap.

        Parameters
        ----------
        penaltyTimesArea
            The penalty modulus times the area of the contact point(s).
        gap
            The gap(s), negative in contact.

        Returns
        -------
        tuple
            The normal force(s) :math:`f_n` and the stiffness :math:`\\mathrm{d}f_n/\\mathrm{d}g`.
        """

    @abstractmethod
    def storedEnergy(self, penaltyTimesArea: float | np.ndarray, gap: float | np.ndarray) -> float | np.ndarray:
        """The energy stored in the penalty springs of closed contact points, :math:`\\Pi(g)` with
        :math:`\\mathrm{d}\\Pi/\\mathrm{d}g = f_n`: the potential whose gradient is the contact force.

        Parameters
        ----------
        penaltyTimesArea
            The penalty modulus times the area of the contact point(s).
        gap
            The gap(s), negative in contact.

        Returns
        -------
        float or numpy.ndarray
            The stored energy, non-negative.
        """

    def springs(self, penaltyTimesArea, gaps, normals):
        """One spring along the normal at every closed point, its force and stiffness :meth:`normalForce`, its
        energy :meth:`storedEnergy`; see :meth:`PenaltyLaw.springs`."""

        closed = np.flatnonzero(self.isClosed(gaps))
        if not len(closed):
            return []
        penaltyTimesAreaOfClosed, gapsOfClosed = penaltyTimesArea[closed], gaps[closed]
        forces, stiffness = self.normalForce(penaltyTimesAreaOfClosed, gapsOfClosed)
        return [
            PenaltySprings(
                closed, normals[closed], forces, stiffness, self.storedEnergy(penaltyTimesAreaOfClosed, gapsOfClosed)
            )
        ]


class LinearPenaltyLaw(ContactPenaltyLaw):
    """:math:`f_n = k A \\, g`: a linear spring with the constant stiffness :math:`k A`, which jumps from zero when the
    gap closes."""

    name = "linear"

    def normalForce(self, penaltyTimesArea, gap):
        return penaltyTimesArea * gap, penaltyTimesArea

    def storedEnergy(self, penaltyTimesArea, gap):
        return 0.5 * penaltyTimesArea * gap**2


class QuadraticPenaltyLaw(ContactPenaltyLaw):
    """:math:`f_n = -\\tfrac{1}{2} k A \\, g^2`: a spring that stiffens with the penetration, its stiffness
    :math:`-k A \\, g` rising continuously from zero when the gap closes."""

    name = "quadratic"

    def normalForce(self, penaltyTimesArea, gap):
        return -0.5 * penaltyTimesArea * gap**2, -penaltyTimesArea * gap

    def storedEnergy(self, penaltyTimesArea, gap):
        return -penaltyTimesArea * gap**3 / 6.0


#: The penalty laws of contact, by the name the input file gives them.
penaltyLaws = {law.name: law for law in (LinearPenaltyLaw, QuadraticPenaltyLaw)}


def penaltyLaw(contactType: str) -> PenaltyLaw:
    """The penalty law an input file names, case-insensitively.

    Parameters
    ----------
    contactType
        The ``type`` option of a contact constraint: 'linear' or 'quadratic'.

    Returns
    -------
    PenaltyLaw
        The law.
    """

    law = penaltyLaws.get(contactType.lower())
    if law is None:
        raise ValueError(
            f"Constraint type '{contactType.lower()}' is not supported. Use "
            + " or ".join(f"'{name}'" for name in penaltyLaws)
            + "."
        )
    return law()
