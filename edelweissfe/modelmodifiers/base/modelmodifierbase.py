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

"""Abstract base class for all ModelModifier entities in EdelweissFE."""

from abc import ABC, abstractmethod

from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel


class ModelModifierBase(ABC):
    """Abstract base class for entities that dynamically mutate the FEModel topology,
    mesh, or state variables during analysis steps.
    """

    def __init__(self, name: str, model: FEModel, journal: Journal, **kwargs):
        self._name = name
        self._model = model
        self._journal = journal

    @property
    def name(self) -> str:
        """Name of the model modifier."""
        return self._name

    @abstractmethod
    def updateModel(self, model: FEModel, step, timeStep: float) -> bool:
        """Invoked by the solver at designated lifecycle hooks (e.g. start of increment).

        Parameters
        ----------
        model
            The FEModel object.
        step
            The current step.
        timeStep
            The current timeStep.

        Returns
        -------
        bool
            True if the model topology, element/node count, or DOF system changed,
            signaling the solver to rebuild equation system structures (DofManager,
            CSR matrices, solution vectors, and MPC transformations).
        """

    def onStepStart(self, model: FEModel, step):
        """Optional lifecycle hook called at the start of an analysis step."""

    def onIncrementEnd(self, model: FEModel, step, timeStep: float):
        """Optional lifecycle hook called after an increment converges."""
