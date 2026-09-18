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
from edelweissfe.utils.schema import OptionSchemaProvider


class StepActionBase(OptionSchemaProvider, ABC):
    """This is the base class for all step actions.
    User defined step actions must implement the methods.

    Two construction paths
    -----------------------
    A step action is reached either from Python or from an ``.inp`` file, and the input file is a
    *serialization* of the Python path, not a second way of building the object. A step action with
    a typed constructor -- ``nSet`` is a node set, ``f_t`` is a callable, prescribed values are a
    ``dict`` -- overrides :meth:`fromStepActionDefinition` / :meth:`updateStepActionFromDefinition`
    to translate the parser's option mapping into a call to it. Everything string-shaped stays on
    that translation, which is the only thing the ``.inp`` front-end adds.

    A step action with an untyped constructor needs no such override: the two hooks below default
    to handing the raw ``definition`` dict straight to ``__init__``/``updateStepAction``. Which path
    a module takes is decided by whether it overrides the hooks -- ordinary polymorphism, not
    attribute probing, and no separate list of which modules use which convention to keep in sync.

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

    @classmethod
    def fromStepActionDefinition(
        cls,
        name: str,
        definition: dict,
        jobInfo: dict,
        model: FEModel,
        fieldOutputController: FieldOutputController,
        journal: Journal,
    ) -> "StepActionBase":
        """Create this step action from a parsed ``.inp`` step action definition.

        This is the one place a module's input-file shape (numbered component options, a ``f(t)``
        expression string, a node *set name*) is turned into the typed arguments its real
        constructor takes. Override it together with a typed ``__init__``; leave it alone and the
        default dict-consuming constructor is used unchanged.

        Parameters
        ----------
        name
            The name of the step action.
        definition
            The parsed option mapping for this step action.
        jobInfo
            A dictionary containing the information about the job.
        model
            The model tree.
        fieldOutputController
            The field output controlling object.
        journal
            The journal object for logging.

        Returns
        -------
        StepActionBase
            The constructed step action.
        """
        return cls(name, definition, jobInfo, model, fieldOutputController, journal)

    def updateStepActionFromDefinition(
        self,
        definition: dict,
        jobInfo: dict,
        model: FEModel,
        fieldOutputController: FieldOutputController,
        journal: Journal,
    ):
        """Update this step action from a parsed ``.inp`` step action definition.

        The update counterpart of :meth:`fromStepActionDefinition`, needed as its own hook because a
        later step re-declaring an action reaches the *instance*, not the class. Override both or
        neither: a module whose constructor is typed but whose update path still consumed a dict
        would work only until the first multi-step input.

        Parameters
        ----------
        definition
            The parsed option mapping for this step action.
        jobInfo
            A dictionary containing the information about the job.
        model
            The model tree.
        fieldOutputController
            The field output controlling object.
        journal
            The journal object for logging.
        """
        self.updateStepAction(definition, jobInfo, model, fieldOutputController, journal)

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

    def _checkSetChanged(self, theSet) -> bool:
        """Lazily detect whether ``theSet`` (a stable-identity
        :class:`~edelweissfe.sets.nodeset.NodeSet` or :class:`~edelweissfe.sets.elementset.ElementSet`)
        was mutated in-place (e.g. by AMR) since this step action last checked it.

        A step action that pre-sizes a derived array to ``len(theSet)`` (e.g. Dirichlet's
        ``delta``) calls this at its own per-increment entry point to recompute that array lazily,
        without registering as a
        :class:`~edelweissfe.models.meshdependent.MeshDependent`. A step action that
        merely iterates ``theSet`` needs no such check -- it sees new members automatically.

        Parameters
        ----------
        theSet
            The set whose version is being tracked.

        Returns
        -------
        bool
            True once per version bump of ``theSet`` since the last call for this same set.
        """
        setVersions = self.__dict__.setdefault("_setVersions", {})
        key = id(theSet)
        changed = setVersions.get(key, theSet._version) != theSet._version
        setVersions[key] = theSet._version
        return changed
