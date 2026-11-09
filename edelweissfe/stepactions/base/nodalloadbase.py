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

from edelweissfe.numerics.dofmanager import DofManager, DofVector, VIJSystemMatrix
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.stepactions.base.loadbase import LoadBase
from edelweissfe.timesteppers.timestep import TimeStep


class NodalLoadBase(LoadBase):
    """Concentrated loads on the degrees of freedom of a field on a node set, such as node forces."""

    @property
    @abstractmethod
    def field(self) -> str:
        pass

    @property
    @abstractmethod
    def nodeSet(self) -> NodeSet:
        pass

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
        K: VIJSystemMatrix | None,
        U_np: DofVector,
        dU: DofVector,
        timeStep: TimeStep,
        dofManager: DofManager,
    ):
        """Subtract the current node loads from the residual contribution at the degrees of freedom of the field
        on the node set; a dead load, so without a tangent. See
        :meth:`~edelweissfe.stepactions.base.loadbase.LoadBase.assembleLoad`.

        Parameters
        ----------
        P
            The global residual contribution vector to augment.
        K
            The global system matrix; untouched.
        U_np
            The current global solution.
        dU
            The global solution increment.
        timeStep
            The current time step.
        dofManager
            The DOF manager of the equation system, which locates the degrees of freedom of the node set.
        """

        dofIndices = dofManager.idcsOfFieldsOnNodeSetsInDofVector[self.field][self.nodeSet]
        P[dofIndices] -= self.getCurrentLoad(timeStep).flatten()
