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


class _FluxCount:
    """The accumulated nodal fluxes of the entities whose connectivity is fixed, counted at the first request. All
    locations of a numbering share it, so that these entities are counted once, not after every change of the
    connectivity of others."""

    def __init__(self):
        #: The count per field, or None until it is asked for.
        self.perField = None


class DofLocations:
    """
    The locations of the entities of a discrete system: for every element and every constraint, the global indices of
    its local degrees of freedom, ``locations[entity]``, the *LM array* of the textbooks.

    Specifically, `DofLocations`:
     * Locates every entity in a :class:`~edelweissfe.numerics.dofnumbering.DofNumbering`, from the nodes it couples
       (its connectivity), the fields it couples on them, and the scalar variables it owns, all of them read from the
       one protocol of an entity
       (:class:`~edelweissfe.nodecouplingentity.base.nodecouplingentity.BaseNodeCouplingEntity`):
       the indices from which a system matrix assembly derives where each block goes, and which scatter and gather
       operations use.
     * Calculates accumulated elemental/constraint fluxes fieldwise for convergence tests.
     * Constructs the entity-aware :class:`DofVector` of its discrete system.

    The locations are immutable: they are determined by the numbering and the entities, and never change. A change of
    the connectivity that leaves the numbering as it is -- a contact search that couples other nodes -- makes new
    locations on the same numbering, see :meth:`withNewLocationsOf`.

    Every entity is located, or the construction raises: an entity that couples a field its node does not have has no
    location, which is an error of the model, see :meth:`_locate`.

    Parameters
    ----------
    numbering : DofNumbering
        The numbering of the degrees of freedom the entities are located in.
    elements : list of BaseNodeCouplingEntity, optional
        Everything the solver computes like an element, in the order in which it computes it: the finite elements that
        take part in the discrete system, see :meth:`~edelweissfe.models.femodel.FEModel.buildDiscreteSystem`, and, in
        a meshfree model, its cells and particles.
    constraints : list of BaseNodeCouplingEntity, optional
        The list of Constraints to locate. Their scalar variables must be numbered.
    """

    def __init__(
        self,
        numbering: DofNumbering,
        elements: list[BaseNodeCouplingEntity] = None,
        constraints: list[BaseNodeCouplingEntity] = None,
    ):
        #: The numbering of the degrees of freedom the entities are located in.
        self.numbering = numbering

        #: Everything the solver computes like an element, in the order in which it computes it. Every one of them has a
        #: location.
        self.elements = [] if elements is None else list(elements)
        #: The constraints that take part in the discrete system, with their scalar variables. The solver evaluates
        #: exactly these, so that every constraint it evaluates has its rows in the system matrix, and every row of a
        #: constraint is filled.
        self.constraints = [] if constraints is None else list(constraints)

        #: Dictionary mapping every entity (elements, then constraints) to its indices in the DofVector. It is the one
        #: mapping the consumers that handle all entities at once (the system matrix assembly, the layout of the
        #: scatter vectors) read; a single location is read by ``locations[entity]``. The locations are immutable, so
        #: whoever still holds the old locations holds a mapping that stays valid.
        self.ofEntity = self._locate([*self.elements, *self.constraints])

        #: The elements whose nodes stay as they are, and those whose nodes change with the motion
        #: (:attr:`~edelweissfe.nodecouplingentity.base.nodecouplingentity.BaseNodeCouplingEntity.connectivityChangesWithMotion`),
        #: which are the only ones that can be moved, see :meth:`withNewLocationsOf`.
        self._elementsWithFixedConnectivity = [e for e in self.elements if not e.connectivityChangesWithMotion]
        self._elementsWithMovingConnectivity = [e for e in self.elements if e.connectivityChangesWithMotion]
        #: The accumulated nodal fluxes of the elements with fixed connectivity, counted when first asked for (see
        #: :attr:`nAccumulatedNodalFluxesFieldwise`) and shared by all locations of the numbering, see
        #: :meth:`withNewLocationsOf`.
        self._fluxesOfFixedElements = _FluxCount()
        #: The accumulated nodal fluxes of all entities, computed at the first request.
        self._nAccumulatedNodalFluxesFieldwise = None

    @property
    def nAccumulatedNodalFluxesFieldwise(self) -> dict:
        """For the Abaqus like convergence test, the number of degrees of freedom 'entity-wise' per field, that is
        Σ_(elements+constraints) Σ_nodes ( nDof (field) ).
        Computed from the entities of these locations at the first request, and kept: only the convergence criteria
        ask, so a solver without them does not pay. The count of the elements whose connectivity is fixed is shared by
        all locations of a numbering, and made once: only the constraints and the elements whose connectivity changes
        with the motion (:attr:`~edelweissfe.nodecouplingentity.base.nodecouplingentity.BaseNodeCouplingEntity.connectivityChangesWithMotion`)
        are counted again for new locations.

        Returns
        -------
        dict
            The number of accumulated fluxes, for every field of the numbering.
        """

        if self._nAccumulatedNodalFluxesFieldwise is None:
            if self._fluxesOfFixedElements.perField is None:
                self._fluxesOfFixedElements.perField = self._accumulateNodalFluxesFieldwise(
                    self._elementsWithFixedConnectivity
                )
            ofMovingElements = self._accumulateNodalFluxesFieldwise(self._elementsWithMovingConnectivity)
            ofConstraints = self._accumulateNodalFluxesFieldwise(self.constraints)

            self._nAccumulatedNodalFluxesFieldwise = {
                field: self._fluxesOfFixedElements.perField.get(field, 0)
                + ofMovingElements.get(field, 0)
                + ofConstraints.get(field, 0)
                for field in self.numbering.fields
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

    def checkEveryDofIsLocated(self, indicesRelatedByMultiPointConstraints: np.ndarray | None = None):
        """Refuse a degree of freedom that no located entity touches. Such a degree of freedom has a number, but no row
        in the system matrix: a node that only an entity without kernels touches (e.g. a contact facet) is such a node.
        A solver cannot pin it with a Dirichlet boundary condition, nor integrate it, so the discrete system is not
        built.

        A multi-point constraint is no entity and owns no degrees of freedom, but a degree of freedom it relates, as
        slave or as master (e.g. of the reference point of a rigid body), has its row through the condensation.

        Parameters
        ----------
        indicesRelatedByMultiPointConstraints
            The indices of the degrees of freedom that multi-point constraints relate; None for none.

        Raises
        ------
        ValueError
            If a degree of freedom of the numbering is in no location; the error names the node and the field.
        """

        touched = np.zeros(self.numbering.nDof, dtype=bool)
        for indices in self.ofEntity.values():
            touched[indices] = True
        if indicesRelatedByMultiPointConstraints is not None:
            touched[indicesRelatedByMultiPointConstraints] = True

        untouched = np.flatnonzero(~touched)
        if untouched.size == 0:
            return

        index = int(untouched[0])
        for fieldName, fieldIndices in self.numbering.idcsOfFieldsInDofVector.items():
            if fieldIndices.start <= index < fieldIndices.stop:
                node = self.numbering.getNodeForIndexInDofVector(index)
                raise ValueError(
                    f"The field '{fieldName}' of node {node.label} is numbered, but no element and no constraint "
                    f"that takes part in the discrete system couples it ({untouched.size} degrees of freedom in all "
                    "are in no location): it would have no row in the system matrix. A node that only an entity "
                    "without kernels touches, e.g. a contact facet, has none."
                )
        raise ValueError(
            f"The degree of freedom {index} of a scalar variable is numbered, but no constraint couples it "
            f"({untouched.size} degrees of freedom in all are in no location)."
        )

    def withNewLocationsOf(self, constraints: list, movedEntities: list = None) -> "DofLocations":
        """The locations of the same entities in the same numbering, with the constraints, and the elements that
        moved, located anew after their connectivity changed.

        A contact search re-assigns which nodes a constraint couples. That changes the constraints' locations and
        nothing else: no node, field, scalar variable or element is touched, so the numbering and the locations of
        every element stay valid, and only what is derived from the constraints is computed again -- by the same
        method, in the same order, as the constructor computed it. That is a small fraction of what making
        locations for the same model costs.

        A change that moves the numbering -- a mesh refinement, a constraint gaining a scalar variable of its own -- is
        not a change of the connectivity, and needs a new :class:`~edelweissfe.numerics.dofnumbering.DofNumbering`:
        this method refuses it.

        Parameters
        ----------
        constraints
            The constraints, with their current connectivity.
        movedEntities
            The elements whose connectivity changed: those that declare that it changes with the motion
            (:attr:`~edelweissfe.nodecouplingentity.base.nodecouplingentity.BaseNodeCouplingEntity.connectivityChangesWithMotion`,
            e.g. the particles of a meshfree model whose kernel functions reach other nodes). The others keep the
            locations they were given. Only meaningful where the numbering stands, as for the constraints.

        Returns
        -------
        DofLocations
            The new locations; these locations are not changed.

        Raises
        ------
        ValueError
            If the scalar variables of the constraints are not exactly those the numbering numbers, in this order, or
            a moved entity is not an element located in these locations, or does not declare that its connectivity
            changes with the motion.
        """

        scalarVariables = [variable for constraint in constraints for variable in constraint.scalarVariables]
        if scalarVariables != list(self.numbering.idcsOfScalarVariablesInDofVector):
            raise ValueError(
                "The scalar variables {:} of the constraints are not those the numbering numbers, {:}: a change of "
                "the scalar variables is a change of the numbering, and needs a new numbering, not new "
                "locations.".format(
                    [variable.name for variable in scalarVariables],
                    [variable.name for variable in self.numbering.idcsOfScalarVariablesInDofVector],
                )
            )

        oldConstraints = set(self.constraints)
        notLocated = [
            entity
            for entity in movedEntities or ()
            if entity not in self.ofEntity or entity in oldConstraints or not entity.connectivityChangesWithMotion
        ]
        if notLocated:
            raise ValueError(
                "{:} moved entit{:} not an element located in these locations whose connectivity changes with the "
                "motion, which only locate anew what they hold: a change of the entities is a change of the "
                "numbering.".format(len(notLocated), "y is" if len(notLocated) == 1 else "ies are")
            )

        relocated = copy.copy(self)
        relocated.constraints = list(constraints)

        # The elements stay in their places, the constraints, located anew, follow them.
        ofEntity = dict(self.ofEntity)
        for constraint in self.constraints:
            del ofEntity[constraint]
        ofEntity.update(self._locate(movedEntities or []))
        ofEntity.update(self._locate(relocated.constraints))
        relocated.ofEntity = ofEntity

        relocated._nAccumulatedNodalFluxesFieldwise = None

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

    def _locate(self, entities: list) -> dict:
        """Creates a dictionary containing the location (indices) of each entity (elements, constraints, ...)
        within the DofVector structure.

        The location of an entity is the indices of the field variables of its nodes, node after node and, on each
        node, field after field as the entity lists them (``fields``), reordered by its ``dofIndicesPermutation``
        unless that is None; the indices of its scalar variables (``scalarVariables``) follow, after the permutation.

        An entity that cannot be located raises: one of its nodes does not have a field the entity couples there, or the
        field variable is not part of the numbering, or a scalar variable is not (see :meth:`_noLocationError`); so does
        an entity that lacks ``nodes``, ``fields``, ``dofIndicesPermutation`` or ``scalarVariables``, or lists fewer
        fields than it has nodes -- whether there are few entities or many.

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

        # Looked up once: the worker below reads them for every field of every node.
        fieldVariableIndices = self.numbering.idcsOfFieldVariablesInDofVector
        scalarVariableIndices = self.numbering.idcsOfScalarVariablesInDofVector

        def locateChunk(chunk):
            locatedInChunk = {}

            for entity in chunk:
                nodes, fields = entity.nodes, entity.fields
                permutation, scalarVariables = entity.dofIndicesPermutation, entity.scalarVariables

                try:
                    indices = [
                        idx
                        for iNode, node in enumerate(nodes)
                        for nodeField in fields[iNode]
                        for idx in fieldVariableIndices[node.fields[nodeField]]
                    ]
                    scalarIndices = [scalarVariableIndices[variable] for variable in scalarVariables]
                except KeyError:
                    raise self._noLocationError(entity) from None

                destArr = np.array(indices, dtype=np.int32)
                if permutation is not None:
                    destArr = destArr[permutation]
                if scalarIndices:
                    destArr = np.append(destArr, np.array(scalarIndices, dtype=np.int32))
                locatedInChunk[entity] = destArr

            return locatedInChunk

        # A few entities are not worth the hand-over to the thread pool.
        if len(entities) < 500:
            return locateChunk(entities)

        located = {}
        for locatedInChunk in mapChunksInParallel(entities, locateChunk):
            located.update(locatedInChunk)

        return located

    def _noLocationError(self, entity) -> ValueError:
        """The error for an entity that cannot be located: what is missing in the numbering, and which entity it is.
        The entity is named by its class, with its module, as many modules name their class alike, and by the labels of
        its first nodes.

        Parameters
        ----------
        entity
            The entity that cannot be located.

        Returns
        -------
        ValueError
            The error, to raise.
        """

        numberedFieldVariables = self.numbering.idcsOfFieldVariablesInDofVector

        nodeLabels = [node.label for node in entity.nodes]
        nodesNamed = f"{nodeLabels[:6]} ... ({len(nodeLabels)} nodes)" if len(nodeLabels) > 6 else f"{nodeLabels}"
        whose = f"{type(entity).__module__}.{type(entity).__qualname__} coupling the nodes {nodesNamed}"
        for node, fieldsOfNode in zip(entity.nodes, entity.fields):
            for field in fieldsOfNode:
                if field not in node.fields:
                    return ValueError(f"The {whose} cannot be located: its node {node.label} has no field '{field}'.")
                if node.fields[field] not in numberedFieldVariables:
                    return ValueError(
                        f"The {whose} cannot be located: the field '{field}' of its node {node.label} is not numbered."
                    )
        for scalarVariable in entity.scalarVariables:
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

        return DofVector(self.numbering.nDof, self)
