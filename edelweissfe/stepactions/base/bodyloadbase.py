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

from abc import abstractmethod

import numpy as np

from edelweissfe.numerics.dofmanager import DofManager, DofVector
from edelweissfe.numerics.systemmatrixassembly import SystemMatrixBlocks
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.stepactions.base.loadbase import LoadBase
from edelweissfe.timesteppers.timestep import TimeStep


class BodyLoadBase(LoadBase):
    """A load per unit volume on a set of elements, such as gravity: each element integrates it over its volume,
    see :meth:`~edelweissfe.elements.base.baseelement.BaseElement.computeBodyForce`."""

    @property
    @abstractmethod
    def elementSet(self) -> ElementSet:
        """The elements this load is acting on

        Returns
        -------
        list
            The list of elements.
        """

    # @property
    # @abstractmethod
    # def loadType(self) -> str:
    #     """Return the type of load (e.g., bodyforce, ... )

    #     Returns
    #     -------
    #     float
    #         The magnituide.
    #     """
    #     pass

    @abstractmethod
    def getCurrentLoad(self, timeStep: TimeStep) -> np.ndarray:
        """Return the current magnitude for this body load.

        Returns
        -------
        np.ndarray
            The magnitude.
        """

    def assembleLoad(
        self,
        P: DofVector,
        K: SystemMatrixBlocks | None,
        U_np: DofVector,
        dU: DofVector,
        timeStep: TimeStep,
        dofManager: DofManager,
    ):
        """Let every element of the element set add the residual contribution of the body force, the negative of
        the external load, and its tangent; see :meth:`~edelweissfe.stepactions.base.loadbase.LoadBase.assembleLoad`.

        Parameters
        ----------
        P
            The global residual contribution vector to augment.
        K
            The global system matrix to augment; None for an explicit solver, which needs no tangent.
        U_np
            The current global solution.
        dU
            The global solution increment.
        timeStep
            The current time step.
        dofManager
            The DOF manager of the discrete system.
        """

        time = timeStep.totalTime
        dT = timeStep.timeIncrement

        force = self.getCurrentLoad(timeStep)
        for el in self.elementSet:
            Pe = np.zeros(el.nDof)
            Ke = K[el] if K is not None else np.zeros(el.nDof * el.nDof)

            el.computeBodyForce(Pe, Ke, force, U_np[el], time, dT)

            P[el] += Pe
