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
"""The base of the arc length controllers of the indirect displacement control, whose controlled quantity is a linear
combination of degrees of freedom."""

from abc import abstractmethod

import numpy as np

from edelweissfe.numerics.dofmanager import DofManager, DofVector
from edelweissfe.stepactions.base.arclengthcontrollerbase import ArcLengthControllerBase
from edelweissfe.timesteppers.timestep import TimeStep


class IndirectDisplacementControlBase(ArcLengthControllerBase):
    """The base of the arc length controllers of the indirect displacement control (Jirásek and Bažant): the controlled
    quantity is a linear combination of degrees of freedom,

    .. math::

        q = \\boldsymbol{c} \\cdot \\boldsymbol{U},

    with the weights :math:`\\boldsymbol{c}` of the controlled degrees of freedom (:attr:`cVector`), e.g. the opening of
    a crack between two nodes, or the mean contraction of a ring of nodes. Over its step, :math:`q` changes by
    :math:`\\Delta L`, and every increment by its share, :math:`\\Delta p \\, \\Delta L` for an increment of the step
    progress :math:`\\Delta p`. The controller's equation for the corrected increment of the solution,

    .. math::

        \\boldsymbol{c} \\cdot (\\Delta \\boldsymbol{U} + \\Delta\\Delta \\boldsymbol{U}_0
        + \\Delta\\Delta\\lambda \\, \\Delta\\Delta \\boldsymbol{U}_f) = \\Delta p \\, \\Delta L,

    is linear in the correction of the load parameter :math:`\\Delta\\Delta\\lambda`, which :meth:`computeDDLambda`
    solves it for.

    With a weight of the load parameter :math:`c_\\lambda` (``cLambda``), the controlled quantity is
    :math:`q + c_\\lambda \\, \\lambda`: the increment of the load parameter of the iterate,
    :math:`\\Delta\\lambda + \\Delta\\Delta\\lambda`, enters the left-hand side of the equation times
    :math:`c_\\lambda`. This keeps the equation regular while the controlled degrees of freedom do not respond to the
    load parameter -- in a strength reduction, as long as no point yields -- where the load parameter then advances by
    :math:`\\Delta L / c_\\lambda` over the step. The load parameter of a step starts from zero, so the target of a
    declaration is the one of :math:`q` as without the weight.

    The target ``L`` of a declaration -- the construction of the controller, or its re-declaration in a later step,
    which updates it -- is, with ``absolute``, the value of :math:`q` at the end of the step: the change of the step is
    :math:`\\Delta L = L - q`, with :math:`q` of the model when the controller is declared, i.e. of the state the step
    starts from (:meth:`declareTarget`). A controller first declared in a later step aims at ``L`` as well, and so does
    one whose quantity changed over steps it did not control. Without ``absolute``, ``L`` is the change
    :math:`\\Delta L` itself. A step that does not re-declare the controller changes :math:`q` by the :math:`\\Delta L`
    of its last declaration once more.

    A controller of this kind names its controlled degrees of freedom and their weights: :attr:`cVector`,
    :meth:`controlledDofIndices` and :meth:`controlledValuesOfModel`.

    Parameters
    ----------
    name
        The name of this step action.
    absolute
        Whether the target ``L`` of a declaration is the value of the controlled quantity at the end of the step, or its
        change over the step. Fixed at construction: a re-declaration cannot change it.
    cLambda
        The weight of the load parameter in the controlled quantity. Fixed at construction.
    """

    #: The change of the controlled quantity over the step, :math:`\\Delta L`, which a declaration derives from its
    #: target and the model; a resumed run declares the controller anew on the restored model, so a checkpoint holds it.
    checkpointedState = {"changeOverStep": float}

    #: The weights :math:`\\boldsymbol{c}` of the controlled degrees of freedom, in the order of
    #: :meth:`controlledDofIndices` and :meth:`controlledValuesOfModel`.
    cVector: np.ndarray

    def __init__(self, name: str, absolute: bool, cLambda: float = 0.0):
        self.name = name
        #: The weight of the load parameter in the controlled quantity.
        self.cLambda = cLambda
        #: Whether the target ``L`` of a declaration is the value of the controlled quantity at the end of the step.
        self.absolute = absolute
        #: The change of the controlled quantity over the step, :math:`\\Delta L`; see :meth:`declareTarget`.
        self.changeOverStep = 0.0

    @abstractmethod
    def controlledDofIndices(self, dofManager: DofManager) -> np.ndarray:
        """The indices of the controlled degrees of freedom in the dof vector of an equation system.

        Parameters
        ----------
        dofManager
            The dof manager of the equation system.

        Returns
        -------
        np.ndarray
            The indices, in the order of :attr:`cVector`.
        """

    @abstractmethod
    def controlledValuesOfModel(self) -> np.ndarray:
        """The values of the controlled degrees of freedom in the model, as its node fields hold them.

        Returns
        -------
        np.ndarray
            The values, in the order of :attr:`cVector`.
        """

    def controlledQuantity(self) -> float:
        """The controlled quantity of the model, :math:`q = \\boldsymbol{c} \\cdot \\boldsymbol{U}`.

        Returns
        -------
        float
            The controlled quantity.
        """

        return self.cVector.dot(self.controlledValuesOfModel())

    def declareTarget(self, L: float):
        """Declare the target of the step, from which the change of the controlled quantity over the step,
        :attr:`changeOverStep`, follows: :math:`\\Delta L = L - q`, with the controlled quantity :math:`q` of the model
        now, if ``absolute`` is set, else :math:`\\Delta L = L`. Called with the controlled degrees of freedom and their
        weights in place.

        Parameters
        ----------
        L
            The target: the value of the controlled quantity at the end of the step with ``absolute``, else its change
            over the step.
        """

        self.changeOverStep = L - self.controlledQuantity() if self.absolute else L

    def computeDDLambda(
        self,
        dU: DofVector,
        ddU_0: DofVector,
        ddU_f: DofVector,
        timeStep: TimeStep,
        dofManager: DofManager,
        dLambda: float,
    ) -> float:
        """The correction of the load parameter that makes the corrected increment change the controlled quantity by the
        share of the increment, :math:`\\Delta p \\, \\Delta L`; see
        :meth:`~edelweissfe.stepactions.base.arclengthcontrollerbase.ArcLengthControllerBase.computeDDLambda`.

        Parameters
        ----------
        dU
            The current increment of the solution.
        ddU_0
            The correction of the solution caused by the out-of-balance force.
        ddU_f
            The correction of the solution caused by the reference load alone.
        timeStep
            The increment being solved, whose size :math:`\\Delta p` is ``timeStep.stepProgressIncrement``.
        dofManager
            The dof manager of the current equation system.
        dLambda
            The current increment of the load parameter, which enters with the weight ``cLambda``.

        Returns
        -------
        float
            The correction of the load parameter.
        """

        idcs = self.controlledDofIndices(dofManager)

        dL = timeStep.stepProgressIncrement * self.changeOverStep

        return (dL - self.cVector.dot(dU[idcs] + ddU_0[idcs]) - self.cLambda * dLambda) / (
            self.cVector.dot(ddU_f[idcs]) + self.cLambda
        )
