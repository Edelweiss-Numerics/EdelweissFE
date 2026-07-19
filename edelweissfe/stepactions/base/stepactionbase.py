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

from abc import ABC

from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.fieldoutput import FieldOutputController


class StepActionBase(ABC):
    """This is the base class for all step actions.
    User defined step actions must implement the methods.

    Parameters
    ----------
    name
        The name of this step action.
    definition
        A dictionary containing the options for this step action.
    jobInfo
        A dictionary containing the information about the job.
    model
        The model tree.
    fieldOutputController
        The field output controlling object.
    journal
        The journal object for logging.
    """

    def __init__(
        self,
        name: str,
        definition: dict,
        jobInfo: dict,
        model: FEModel,
        fieldOutputController: FieldOutputController,
        journal: Journal,
    ):
        pass

    def updateStepAction(
        self,
        definition: dict,
        jobInfo: dict,
        model: FEModel,
        fieldOutputController: FieldOutputController,
        journal: Journal,
    ):
        """Is called when an updated definition is present for a new step.

        Parameters
        ----------
        definition
            A dictionary containing the options for this step action.
        jobInfo
            A dictionary containing the information about the job.
        model
            The model tree.
        fieldOutputController
            The field output controlling object.
        journal
            The journal object for logging.
        """

    def applyAtStepStart(self, model: FEModel):
        """Is called when a step starts.

        Parameters
        ----------
        model
            The current state of the model.
        """

    def applyAtStepEnd(self, model: FEModel):
        """Is called when a step successfully finished.

        Parameters
        ----------
        model
            The current state of the model.
        """

    def applyAtIncrementStart(self, model: FEModel, timeStep: TimeStep):
        """Is called when a step increment starts.

        Parameters
        ----------
        model
            The current state of the model.
        timeStep
            The definition of the time increment.
        """
