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
The locations of the entities of a discrete system: for every element and every constraint, the global indices of its
local degrees of freedom. In the vocabulary of the textbooks, the LM array: the ID array
(:class:`~edelweissfe.numerics.dofnumbering.DofNumbering`) applied to the connectivity of the entities.
"""

import copy
from collections import Counter
from itertools import chain

import numpy as np

from edelweissfe.config.phenomena import phenomena
from edelweissfe.nodecouplingentity.base.nodecouplingentity import (
    BaseNodeCouplingEntity,
)
from edelweissfe.numerics.dofnumbering import DofNumbering
from edelweissfe.numerics.dofvector import DofVector
from edelweissfe.numerics.parallelizationutilities import mapChunksInParallel


class DofLocations:
    """
    The locations of the entities of a discrete system: for every element and every constraint, the global indices of
    its local degrees of freedom, ``locations[entity]``, the *LM array* of the textbooks.

    Specifically, `DofLocations`:
     * Locates every entity in a :class:`~edelweissfe.numerics.dofnumbering.DofNumbering`, from the nodes it couples
       (its connectivity) and the fields it couples on them: the indices from which a system matrix assembly derives
       where each block goes, and which scatter and gather operations use.
     * Calculates accumulated elemental/constraint fluxes fieldwise for convergence tests.
     * Constructs the entity-aware :class:`DofVector` of its discrete system.

    The locations are immutable: they are determined by the numbering and the entities, and never change. A change of
    the connectivity that leaves the numbering as it is -- a contact search that couples other nodes -- makes new
    locations on the same numbering, see :meth:`withNewLocationsOf`.

    Every entity is located, or the construction raises: an entity that couples a field its node does not have has no
    location, which is an error of the model, see :meth:`_locateNodeCouplingEntities`.

    Parameters
    ----------
    numbering : DofNumbering
        The numbering of the degrees of freedom the entities are located in.
    elements : list of BaseNodeCouplingEntity, optional
        The list of Elements to locate.
    constraints : list of BaseNodeCouplingEntity, optional
        The list of Constraints to locate. Their scalar variables must be numbered.
    furtherEntities : list of BaseNodeCouplingEntity, optional
        Entities of other kinds, which a subclass knows the meaning of (e.g. the cells and particles of a meshfree
        model). They are located like elements, and computed like elements: they follow the elements in
        :attr:`elements`, and the constraints in the entities this class is a mapping of.
    """

    def __init__(
        self,
        numbering: DofNumbering,
        elements: list[BaseNodeCouplingEntity] = None,
        constraints: list[BaseNodeCouplingEntity] = None,
        furtherEntities: list[BaseNodeCouplingEntity] = None,
    ):
        if elements is None:
            elements = []
        if constraints is None:
            constraints = []
        if furtherEntities is None:
            furtherEntities = []

        #: The numbering of the degrees of freedom the entities are located in.
        self.numbering = numbering

        self._ofElements = self._locateNodeCouplingEntities(elements)
        self._ofConstraints = self._locateConstraints(constraints)
        self._ofFurtherEntities = self._locateNodeCouplingEntities(furtherEntities)

        self._mergeEntities()

    def _mergeEntities(self):
        """Merge what is known per kind of entity: the lists of the elements and of the constraints the locations hold,
        the merged mapping of every entity to its indices, and the accumulated nodal fluxes. The merged mapping is a new
        dict object on purpose: consumers cache plans keyed on its identity (the scatter template, the element loop's
        gather plan), and a mapping mutated in place would leave those plans silently stale."""

        #: Everything the solver computes like an element, in the order in which it computes it, that is the elements,
        #: followed by the further entities.
        self.elements = [*self._ofElements, *self._ofFurtherEntities]
        #: The constraints that take part in the discrete system, with their scalar variables. The solver evaluates
        #: exactly these, so that every constraint it evaluates has its rows in the system matrix, and every row of a
        #: constraint is filled.
        self.constraints = list(self._ofConstraints)
        #: Dictionary mapping every entity (elements, constraints, ...) to its indices in the DofVector. It is the one
        #: mapping the consumers that handle all entities at once (the system matrix assembly, the layout of the
        #: scatter vectors) read; a single location is read by ``locations[entity]``.
        self.ofEntity = self._ofElements | self._ofConstraints | self._ofFurtherEntities

        #: The accumulated nodal fluxes, computed at the first request, see :attr:`nAccumulatedNodalFluxesFieldwise`.
        self._nAccumulatedNodalFluxesFieldwise = None

    @property
    def nAccumulatedNodalFluxesFieldwise(self) -> dict:
        """For the Abaqus like convergence test, the number of degrees of freedom 'entity-wise' per field, that is
        Σ_(elements+constraints) Σ_nodes ( nDof (field) ).
        Computed from the entities of these locations at the first request, and kept: only the convergence criteria
        ask, so a solver without them does not pay.

        Returns
        -------
        dict
            The number of accumulated fluxes, for every field of the numbering.
        """

        if self._nAccumulatedNodalFluxesFieldwise is None:
            countsOfAllFields = self._accumulateNodalFluxesFieldwise([*self.elements, *self.constraints])
            self._nAccumulatedNodalFluxesFieldwise = {
                field: countsOfAllFields.get(field, 0) for field in self.numbering.fields
            }

        return self._nAccumulatedNodalFluxesFieldwise

    def __getitem__(self, entity: BaseNodeCouplingEntity) -> np.ndarray:
        """The location of an entity: the global indices of its local degrees of freedom, in the order of its local
        degrees of freedom.

        Parameters
        ----------
        entity
            An element, a constraint, or any other entity of these locations.

        Returns
        -------
        np.ndarray
            The global indices.

        Raises
        ------
        KeyError
            If the entity is not located in these locations.
        """

        return self.ofEntity[entity]

    def __contains__(self, entity: BaseNodeCouplingEntity) -> bool:
        """Whether the entity is located in these locations, i.e. whether it has rows in the discrete system.

        Parameters
        ----------
        entity
            An entity.

        Returns
        -------
        bool
            True if the entity is located.
        """

        return entity in self.ofEntity

    def withNewLocationsOf(self, constraints: list, movedEntities: list = None) -> "DofLocations":
        """The locations of the same entities in the same numbering, with the constraints, and the further entities
        that moved, located anew after their connectivity changed.

        A contact search re-assigns which nodes a constraint couples. That changes the constraints' locations and
        nothing else: no node, field, scalar variable or element is touched, so the numbering and the locations of
        every element stay valid, and only what is derived from the constraints is computed again -- by the same
        methods, in the same order, as the constructor computed it. That is a small fraction of what making
        locations for the same model costs.

        A change that moves the numbering -- a mesh refinement, a constraint gaining a scalar variable of its own -- is
        not a connectivity change, and needs a new :class:`~edelweissfe.numerics.dofnumbering.DofNumbering`.

        Parameters
        ----------
        constraints
            The constraints, with their current connectivity.
        movedEntities
            The further entities whose connectivity changed (see the constructor); the others keep the locations they
            were given. Only meaningful where the numbering stands, as for the constraints.

        Returns
        -------
        DofLocations
            The new locations; these locations are not changed.
        """

        relocated = copy.copy(self)
        relocated._ofConstraints = self._locateConstraints(constraints)
        if movedEntities:
            relocated._ofFurtherEntities = self._ofFurtherEntities | self._locateNodeCouplingEntities(movedEntities)
        relocated._mergeEntities()

        return relocated

    def _accumulateNodalFluxesFieldwise(self, entities: list) -> dict:
        """Count the nodal fluxes Σ_entities Σ_nodes ( nDof (field) ) for each field.

        A node shared by many entities contributes the same count to each of them, so the nodes are
        first counted, and each distinct node's fields are then looked up only once, weighted by
        that count. The result is an integer sum and hence identical to visiting every node of every
        entity.

        Parameters
        ----------
        entities
           The list of entities (elements or constraints), whose nodes are considered.

        Returns
        -------
        dict
            The number of accumulated fluxes per field, for every field in
            :data:`~edelweissfe.config.phenomena.phenomena`.
        """

        nAccumulatedFluxesFieldwise = dict.fromkeys(phenomena.keys(), 0)

        nodeMultiplicity = Counter(chain.from_iterable(e.nodes for e in entities))

        idcsOfFieldVariables = self.numbering.idcsOfFieldVariablesInDofVector
        for node, multiplicity in nodeMultiplicity.items():
            for field, fv in node.fields.items():
                indices = idcsOfFieldVariables.get(fv)
                if indices is not None:
                    nAccumulatedFluxesFieldwise[field] += multiplicity * len(indices)

        return nAccumulatedFluxesFieldwise

    def _locateNodeCouplingEntities(self, entities: list) -> dict:
        """Creates a dictionary containing the location (indices) of each entity (elements, ...)
        within the DofVector structure.

        The location of an entity is the indices of the field variables of its nodes, node after node and, on each
        node, field after field as the entity lists them (``fields``), reordered by its ``dofIndicesPermutation``
        unless that is None.

        An entity that cannot be located raises: one of its nodes does not have a field the entity couples there, or the
        field variable is not part of the numbering (see :meth:`_noLocationError`); so does an entity that lacks
        ``nodes``, ``fields`` or ``dofIndicesPermutation``, or lists fewer fields than it has nodes -- whether there are
        few entities or many.

        The entities are independent of each other, so they are located in chunks on the thread pool
        (:func:`~edelweissfe.numerics.parallelizationutilities.mapChunksInParallel`), unless there are so few of them
        that handing them to the pool would cost more than it saves: then the calling thread locates them. The
        dictionary holds the entities in the order they were given.

        Parameters
        ----------
        entities
            The list of entities to locate.

        Returns
        -------
        dict
            A dictionary containing the location mapping.

        Raises
        ------
        ValueError
            If an entity cannot be located.
        """
        if not entities:
            return {}

        entities = list(entities)

        # Looked up once: the worker below reads it for every field of every node.
        field_var_map = self.numbering.idcsOfFieldVariablesInDofVector

        def locateChunk(chunk):
            locatedInChunk = {}

            for entity in chunk:
                nodes, fieldsOnNodes, permutation = entity.nodes, entity.fields, entity.dofIndicesPermutation

                try:
                    indices = [
                        idx
                        for iNode, node in enumerate(nodes)
                        for nodeField in fieldsOnNodes[iNode]
                        for idx in field_var_map[node.fields[nodeField]]
                    ]
                except KeyError:
                    raise self._noLocationError(entity, fieldsOnNodes) from None

                destArr = np.array(indices, dtype=np.int32)
                locatedInChunk[entity] = destArr if permutation is None else destArr[permutation]

            return locatedInChunk

        # A few entities are not worth the hand-over to the thread pool.
        if len(entities) < 500:
            return locateChunk(entities)

        located = {}
        for locatedInChunk in mapChunksInParallel(entities, locateChunk):
            located.update(locatedInChunk)

        return located

    def _locateConstraints(self, constraints: list) -> dict:
        """Creates a dictionary containing the location (indices) of each constraint within the DofVector structure:
        the indices of the field variables of its nodes, node after node, followed by those of its scalar variables.

        Parameters
        ----------
        constraints
            The list of constraints to locate.

        Returns
        -------
        dict
            A dictionary containing the location mapping.

        Raises
        ------
        ValueError
            If a constraint cannot be located: see :meth:`_noLocationError`.
        """

        ofConstraints = {}
        field_var_map = self.numbering.idcsOfFieldVariablesInDofVector
        scalar_var_map = self.numbering.idcsOfScalarVariablesInDofVector

        for constraint in constraints:
            node_fields_gen = (
                field_var_map[node.fields[nodeField]]
                for iNode, node in enumerate(constraint.nodes)
                for nodeField in constraint.fieldsOnNodes[iNode]
            )
            scalar_vars_gen = ([scalar_var_map[v]] for v in constraint.scalarVariables)

            indices_gen = chain(chain.from_iterable(node_fields_gen), chain.from_iterable(scalar_vars_gen))

            try:
                ofConstraints[constraint] = np.fromiter(indices_gen, dtype=int)
            except KeyError:
                raise self._noLocationError(
                    constraint, constraint.fieldsOnNodes, constraint.scalarVariables, constraint.name
                ) from None

        return ofConstraints

    def _noLocationError(
        self, entity, fieldsOnNodes: list, scalarVariables: list = (), name: str | None = None
    ) -> ValueError:
        """The error for an entity that cannot be located: what is missing in the numbering, and which entity it is.
        The entity is named by its class (its module too, as many modules name their class alike), by its own name if it
        has one, and by the labels of its first nodes.

        Parameters
        ----------
        entity
            The entity that cannot be located.
        fieldsOnNodes
            The fields the entity couples on each of its nodes.
        scalarVariables
            The scalar variables of the entity.
        name
            The name of the entity, if it has one (a constraint has); None otherwise.

        Returns
        -------
        ValueError
            The error, to raise.
        """

        numberedFieldVariables = self.numbering.idcsOfFieldVariablesInDofVector

        nodeLabels = [node.label for node in entity.nodes]
        nodesNamed = f"{nodeLabels[:6]} ... ({len(nodeLabels)} nodes)" if len(nodeLabels) > 6 else f"{nodeLabels}"
        entityNamed = "" if name is None else f" {name!r}"
        whose = f"{type(entity).__module__}.{type(entity).__qualname__}{entityNamed} coupling the nodes {nodesNamed}"
        for node, fieldsOfNode in zip(entity.nodes, fieldsOnNodes):
            for field in fieldsOfNode:
                if field not in node.fields:
                    return ValueError(f"The {whose} cannot be located: its node {node.label} has no field '{field}'.")
                if node.fields[field] not in numberedFieldVariables:
                    return ValueError(
                        f"The {whose} cannot be located: the field '{field}' of its node {node.label} is not numbered."
                    )
        for scalarVariable in scalarVariables:
            if scalarVariable not in self.numbering.idcsOfScalarVariablesInDofVector:
                return ValueError(
                    f"The {whose} cannot be located: its scalar variable '{scalarVariable.name}' is not numbered."
                )

        return ValueError(f"The {whose} cannot be located.")

    def constructDofVector(
        self,
    ) -> DofVector:
        """Construct a vector with size=nDof and which has knowledge about
        the location of each entity.

        Returns
        -------
        DofVector
            A DofVector.
        """

        return DofVector(self.numbering.nDof, self.ofEntity)
