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

from edelweissfe.config.phenomena import rateFluxFields
from edelweissfe.numerics.doflocations import DofLocations
from edelweissfe.numerics.dofvector import DofVector
from edelweissfe.numerics.systemmatrixassembly import SystemMatrixBlocks
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
        """Return the current loads of the nodes.

        Returns
        -------
        np.ndarray
            The loads, one row per node of the set, in its order, with the components of the field.
        """

    def assembleLoad(
        self,
        P: DofVector,
        K: SystemMatrixBlocks | None,
        U_np: DofVector,
        dU: DofVector,
        timeStep: TimeStep,
        locations: DofLocations,
    ):
        """Subtract the current node loads from the residual contribution at the degrees of freedom of the field
        on the node set; a dead load, so without a tangent. See
        :meth:`~edelweissfe.stepactions.base.loadbase.LoadBase.assembleLoad`.

        Nothing is added in a zero time increment for a field whose fluxes are rates
        (:data:`~edelweissfe.config.phenomena.rateFluxFields`, e.g. ``temperature``): no time passes, so a point
        source supplies nothing, like the distributed and body loads of the transport elements.

        Only the nodes of the set that are part of the discrete system are loaded (see
        :meth:`~edelweissfe.numerics.dofnumbering.DofNumbering.locateFieldOnNodeSet`): every node of the set in a finite
        element model, those in the active domain in a meshfree one.

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
        locations
            The locations of the entities in the discrete system, from which the load finds where it acts.
        """

        # A point load on a field whose flux is a rate (e.g. a heat source in W on 'temperature') supplies nothing while
        # no time passes. In the zero increment at the beginning of a step, the transport elements balance the content
        # increment only (nothing flows); a point load acting there would be balanced against the capacity alone and
        # add the supply of one unit of time, at the beginning of every step.
        if timeStep.timeIncrement == 0.0 and self.field in rateFluxFields:
            return

        dofIndices, nodePositions = locations.numbering.locateFieldOnNodeSet(self.field, self.nodeSet)
        # one row of the load per node of the set: those of the nodes in the discrete system
        P[dofIndices] -= self.getCurrentLoad(timeStep)[nodePositions].flatten()
