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
# Created on Thu May 12 18:35:44 2022

# @author: Matthias Neuner

from dataclasses import dataclass

import numpy as np

from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.numerics.dofnumbering import DofNumbering
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.stepactions.base.indirectdisplacementcontrolbase import (
    IndirectDisplacementControlBase,
)
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.caseinsensitivedict import CaseInsensitiveDict
from edelweissfe.utils.misc import withoutParserBookkeepingKeys
from edelweissfe.utils.schema import buildSchemaFromOptions, schemaField

"""
Indirect (displacement) controller for the NISTPArcLength solver
uses a ring to control the contraction, e.g., for tunneling simulations.

Currently 2D only!

The center is autotically computed from the bounding node coordinates.
"""


@dataclass(frozen=True)
class IndirectContractionControlSchema:
    """The scalar options of the ``indirectcontractioncontrol`` keyword, owned by this module and
    never mutated from outside it.

    ``name`` and ``contractionNSet`` are ``structuralOnly`` fields: ``contractionNSet`` names an
    existing model object, resolved by :meth:`fromStepActionDefinition` before the schema is even
    built, exactly like every other category's structural names, and ``name`` is popped even
    earlier, by ``helpers/inputfilehelpers.py``. Both are declared here purely so the rendered
    grammar surface documents them -- :func:`~edelweissfe.utils.schema.buildSchemaFromOptions`
    never actually sees either key; see :attr:`~edelweissfe.utils.schema.SchemaFieldMeta.structuralOnly`.
    """

    name: str | None = schemaField(
        description="Name of the step action.", dtype=str, default=None, required=True, structuralOnly=True
    )
    contractionNSet: str | None = schemaField(
        description="The node set defining the contraction ring",
        dtype=str,
        default=None,
        required=True,
        structuralOnly=True,
    )
    L: float | None = schemaField(
        description="Final distance (e.g. crack opening)", dtype=float, default=None, required=True
    )
    exportCVector: str | None = schemaField(description="File to export the computed c vector", dtype=str, default=None)
    absolute: bool | None = schemaField(description="Use absolute formulation", dtype=bool, default=True)


class StepAction(IndirectDisplacementControlBase):
    """Indirect (displacement) controller for the NISTPArcLength solver, controlling the average
    radial contraction of a ring of nodes, e.g. for tunneling simulations.

    The controlled quantity is the mean of the inward radial displacements of the nodes in
    ``contractionNSet``; the ring's center is computed from the bounding coordinates of that set.
    Currently 2D only. The target ``L`` and ``absolute`` mean what they mean for every indirect
    displacement control, see
    :class:`~edelweissfe.stepactions.base.indirectdisplacementcontrolbase.IndirectDisplacementControlBase`.

    The constructor is typed: it takes the node set itself rather than its name, so the c vector is
    derived from real coordinates. Nothing here parses an input file -- resolving
    ``contractionNSet=innerRing`` against the model and honouring ``exportCVector`` is the job of
    :meth:`fromStepActionDefinition` / :meth:`updateStepActionFromDefinition` below, which is the
    only part of this module the ``.inp`` front-end needs.

    Parameters
    ----------
    name
        The name of this step action.
    contractionNSet
        The node set defining the contraction ring.
    L
        The target of the step: with ``absolute``, the mean contraction at the end of the step,
        counted from the contraction of the model now; else its change over the step.
    model
        The model tree.
    journal
        The journal object for logging.
    absolute
        Whether ``L`` is the mean contraction at the end of the step, or its change over the step.
        Fixed at construction time: a later step re-declaring this action cannot change the
        formulation.
    """

    identification = "IndirectControl"

    #: Option schema for this step action, consumed by OptionSchemaProvider's registry.
    schema = IndirectContractionControlSchema

    def __init__(
        self,
        name: str,
        contractionNSet: NodeSet,
        L: float,
        model: FEModel,
        journal: Journal,
        absolute: bool = True,
    ):
        super().__init__(name, absolute)
        self.journal = journal
        self.model = model

        self.updateStepAction(contractionNSet, L)

    @classmethod
    def fromStepActionDefinition(cls, name, definition, jobInfo, model, fieldOutputController, journal):
        """Build this controller from a parsed ``>>indirectcontractioncontrol`` definition. See
        :class:`~edelweissfe.stepactions.base.stepactionbase.StepActionBase` for why this is separate from
        ``__init__``.

        The ``exportCVector`` dump is written here rather than in ``__init__`` because appending
        ``.csv`` to a user-supplied file name is input-file shaping, and a typed constructor should
        not write files as a side effect -- a programmatic caller that wants the dump can
        ``np.savetxt`` the public ``cVector`` itself.

        ``name`` and the parser's bookkeeping keys are stripped, and ``contractionNSet`` is
        structural (it names a model object), so both are popped before the remaining options are
        validated against :class:`IndirectContractionControlSchema`."""

        definition = CaseInsensitiveDict(withoutParserBookkeepingKeys(definition))
        definition.pop("name", None)
        contractionNSetName = definition.pop("contractionNSet")
        configuration = buildSchemaFromOptions(cls.schema, definition)

        stepAction = cls(
            name,
            model.nodeSets[contractionNSetName],
            configuration.L,
            model,
            journal,
            absolute=configuration.absolute,
        )

        if configuration.exportCVector is not None:
            np.savetxt(configuration.exportCVector + ".csv", stepAction.cVector)

        return stepAction

    def updateStepActionFromDefinition(self, definition, jobInfo, model, fieldOutputController, journal):
        """Update from a parsed ``>>indirectcontractioncontrol`` definition re-declared in a later
        step.

        Re-declaring the control target ``L`` per step is the normal way this action is used, so
        this path is not a corner case. Note that neither ``absolute`` nor ``exportCVector`` is
        re-read here: the formulation has always been fixed by the first declaration, and the c
        vector dump has always been written once, at creation. There is no ``update<keyword>``
        grammar for this module, so a re-declaration is always validated against the full
        ``indirectcontractioncontrol`` keyword and ``buildSchemaFromOptions`` is safe here too."""

        definition = CaseInsensitiveDict(withoutParserBookkeepingKeys(definition))
        definition.pop("name", None)
        contractionNSetName = definition.pop("contractionNSet")
        configuration = buildSchemaFromOptions(self.schema, definition)

        self.updateStepAction(model.nodeSets[contractionNSetName], configuration.L)

    def controlledDofIndices(self, numbering: DofNumbering) -> np.ndarray:
        """The indices of the in-plane displacements of the ring's nodes in the dof vector.

        Parameters
        ----------
        numbering
            The numbering of the degrees of freedom of the current discrete system.

        Returns
        -------
        np.ndarray
            The indices, node by node.
        """

        return np.hstack(
            [numbering.idcsOfFieldVariablesInDofVector[n.fields["displacement"]][:2] for n in self.contractionNSet]
        )

    def controlledValuesOfModel(self) -> np.ndarray:
        """The in-plane displacements of the ring's nodes in the model.

        Returns
        -------
        np.ndarray
            The displacements, node by node.
        """

        return np.hstack([n.fields["displacement"].values[:2] for n in self.contractionNSet])

    def finishIncrement(self, U, dU, dLambda, timeStep: TimeStep, numbering):
        """Called once an increment has converged; this controller keeps nothing from one increment to the next.

        Parameters
        ----------
        U
            The current solution vector.
        dU
            The current increment of the solution vector.
        dLambda
            The increment of the load parameter.
        timeStep
            The current time step.
        numbering
            The numbering of the degrees of freedom of the current discrete system.
        """

    def updateStepAction(self, contractionNSet: NodeSet, L: float):
        """Control a new contraction ring and target contraction.

        Parameters
        ----------
        contractionNSet
            The node set defining the contraction ring.
        L
            The target of the step, see the class docstring.
        """

        self.generateCVector(contractionNSet)

        self.declareTarget(L)

    def generateCVector(self, contractionNSet: NodeSet):
        """Derive the c vector of the inward radial unit vectors of the ring's nodes, scaled such
        that the controlled quantity is the ring's mean contraction.

        The ring's center is computed from the bounding coordinates of the node set.

        Parameters
        ----------
        contractionNSet
            The node set defining the contraction ring.
        """

        nNodes = len(contractionNSet)

        allCoordinates = np.array([n.coordinates for n in contractionNSet])

        x_min = np.min(allCoordinates[:, 0])
        x_max = np.max(allCoordinates[:, 0])
        y_min = np.min(allCoordinates[:, 1])
        y_max = np.max(allCoordinates[:, 1])

        x_center = 0.5 * (x_max + x_min)
        y_center = 0.5 * (y_max + y_min)

        cVector = []

        for n in contractionNSet:
            vec_n_to_center = np.array([x_center - n.coordinates[0], y_center - n.coordinates[1]])
            norm_vec_n_to_center = np.linalg.norm(vec_n_to_center)

            vec_n_to_center_normalized = vec_n_to_center / norm_vec_n_to_center

            cVector.append(vec_n_to_center_normalized)

        self.cVector = np.hstack(cVector)

        # dividing c vector to make 'average' contraction of ring:
        self.cVector *= 1.0 / nNodes

        self.contractionNSet = contractionNSet
