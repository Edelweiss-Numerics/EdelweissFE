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
This module contains important classes for describing the global equation system by means of a sparse system.
"""

import hashlib
from collections import Counter
from functools import cached_property
from itertools import chain

import numpy as np

from edelweissfe.config.phenomena import phenomena
from edelweissfe.fields.nodefield import NodeField
from edelweissfe.nodecouplingentity.base.nodecouplingentity import (
    BaseNodeCouplingEntity,
)
from edelweissfe.numerics.dofvector import DofVector
from edelweissfe.numerics.parallelizationutilities import mapChunksInParallel
from edelweissfe.points.node import Node
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.variables.scalarvariable import ScalarVariable


class DofManager:
    """
    The DofManager coordinates the global equation system structure.

    Specifically, `DofManager`:
     * Analyzes the simulation domain (nodes, scalar variables, elements, constraints, and node sets).
     * Collects and assigns global Degree-of-Freedom (DOF) indices for node fields and scalar variables.
     * Manages entity-to-index mappings for fast scatter/gather operations: the global DOF list of every element and
       constraint, from which a system matrix assembly derives where each block goes.
     * Calculates accumulated elemental/constraint fluxes fieldwise for convergence tests.
     * Constructs domain-aware `DofVector` and `ScatterDofVector` instances.

    Parameters
    ----------
    nodeFields : list of NodeField
        The list of NodeFields to be represented in the DofVector structure.
    scalarVariables : list of ScalarVariable, optional
        The list of ScalarVariables to be represented in the DofVector structure.
    elements : list of BaseNodeCouplingEntity, optional
        The list of Elements for which DOF index mappings will be created.
    constraints : list of BaseNodeCouplingEntity, optional
        The list of Constraints for which DOF index mappings will be created.
    nodeSets : list of NodeSet, optional
        The list of NodeSets for which DOF index mappings will be created.
    initializeAccumulatedNodalFluxesFieldwise : bool, optional
        Whether to compute accumulated nodal fluxes fieldwise during construction. Default is True.
    """

    def __init__(
        self,
        nodeFields: list[NodeField],
        scalarVariables: list[ScalarVariable] = None,
        elements: list[BaseNodeCouplingEntity] = None,
        constraints: list[BaseNodeCouplingEntity] = None,
        nodeSets: list[NodeSet] = None,
        initializeAccumulatedNodalFluxesFieldwise: bool = True,
    ):
        if scalarVariables is None:
            scalarVariables = []
        if elements is None:
            elements = []
        if constraints is None:
            constraints = []
        if nodeSets is None:
            nodeSets = []

        self.nDof = 0  #: The total number of degrees of freedom (and size of the DofVector)
        self.fields = list()  #: The list of field names in the DofVector
        self.idcsOfFieldsInDofVector = dict()  #: Dictionary mapping field names to all indices in the DofVector
        self.idcsOfScalarVariablesInDofVector = (
            dict()
        )  #: Dictionary mapping scalar variables to all indices in the DofVector

        self.idcsOfFieldVariablesInDofVector = (
            dict()
        )  #: Dictionary mapping nodal field variables to their indices in the DofVector
        self.idcsOfNodeFieldsInDofVector = (
            dict()
        )  #: Dictionary mapping a complete NodeField to all its indices in the DofVector
        #: Reverse dictionary mapping an index to its Host object (e.g., a Node).
        #:
        #: .. warning::
        #:    Not read within EdelweissFE (kept for external callers); building it costs a pass over
        #:    every degree of freedom.
        self.indexToHostObjectMapping = dict()
        self.accumulatedElementNDof = 0  #: Accumulated number of element DOFs (= sum of element vector sizes)
        self.largestNumberOfElNDof = 0  #: Size of the largest element DOF vector
        self.accumulatedConstraintNDof = 0  #: Accumulated number of constraint DOFs (= sum of constraint vector sizes)
        self.largestNumberOfConstraintNDof = 0  #: Size of the largest constraint DOF vector

        #: For each field, the node sets located so far, each with the indices of the field's degrees of freedom on its
        #: nodes and the positions of these nodes in the set; see :meth:`locateFieldOnNodeSet`.
        self._fieldsOnNodeSets = dict()
        self.idcsOfElementsInDofVector = dict()  #: Dictionary mapping an element to its indices in the DofVector
        self.idcsOfConstraintsInDofVector = dict()  #: Dictionary mapping a constraint to its indices in the DofVector

        # initialization:

        self._determineDofsAndTheirIndices(nodeFields, scalarVariables)

        self._gatherInformationAboutEntities(elements, constraints, nodeSets)

        self.idcsOfFieldsInDofVector = self.idcsOfNodeFieldsInDofVector
        self.fields = list(self.idcsOfFieldsInDofVector.keys())

        self.idcsOfBasicVariablesInDofVector = (
            self.idcsOfFieldVariablesInDofVector | self.idcsOfScalarVariablesInDofVector
        )
        self.idcsOfHigherOrderEntitiesInDofVector = self.idcsOfElementsInDofVector | self.idcsOfConstraintsInDofVector

        #: Which of the optional structures this manager carries, for :meth:`refreshConstraintIndices`.
        self._hasAccumulatedNodalFluxes = initializeAccumulatedNodalFluxesFieldwise

        if initializeAccumulatedNodalFluxesFieldwise:
            self.nAccumulatedNodalFluxesFieldwise = self._computeAccumulatedNodalFluxesFieldWise(self.fields)

        self.indexToHostObjectMapping |= self._determineIndexToNodeMap()

    def _determineDofsAndTheirIndices(self, nodeFields: list, scalarVariables: list):

        nodeFields = list(nodeFields)
        #: The NodeFields this DofManager numbers, by name: the only ones it reads from and writes to as a whole.
        self.numberedNodeFields = {nodeField.name: nodeField for nodeField in nodeFields}

        (
            self._nDofNodeFields,
            self.idcsOfFieldVariablesInDofVector,
            self.idcsOfNodeFieldsInDofVector,
        ) = self._reserveSpaceForNodeFields(self.nDof, nodeFields)

        self.nDof += self._nDofNodeFields

        (
            self._nDofScalarVariables,
            self.idcsOfScalarVariablesInDofVector,
        ) = self._reserveSpaceForScalarVariables(self.nDof, scalarVariables)

        self.nDof += self._nDofScalarVariables

    def _gatherInformationAboutEntities(self, elements, constraints, nodeSets):

        (
            self.accumulatedElementNDof,
            self._nAccumulatedNodalFluxesFieldwiseFromElements,
            self.largestNumberOfElNDof,
        ) = self._gatherElementsInformation(elements)

        (
            self.accumulatedConstraintNDof,
            self._nAccumulatedNodalFluxesFieldwiseFromConstraints,
            self.largestNumberOfConstraintNDof,
        ) = self._gatherConstraintsInformation(constraints)

        self._fieldsOnNodeSets = {
            field: {nodeSet: self._locateFieldOnNodeSet(field, nodeSet) for nodeSet in nodeSets}
            for field in self.idcsOfNodeFieldsInDofVector
        }
        self.idcsOfElementsInDofVector = self._locateNodeCouplingEntitiesInDofVector(elements)
        self.idcsOfConstraintsInDofVector = self._locateConstraintsInDofVector(constraints)

    def refreshConstraintIndices(self, constraints: list):
        """Re-locate the constraints' degrees of freedom after their connectivity changed.

        A contact search re-assigns which nodes a constraint couples. That changes the constraints'
        DOF footprints and nothing else: no node, field, scalar variable or element is touched, so
        the DOF numbering and every element's indices stay valid, and only the constraint-derived
        bookkeeping has to be recomputed -- by the same methods, in the same order, as the
        constructor computed it. That is a small fraction of what constructing a DofManager for the
        same model costs, and it is the whole reason this method exists rather than a second
        constructor call.

        This does NOT re-derive the degree-of-freedom layout itself: the node fields and the scalar
        variables keep the indices they were given. A change that moves those -- a mesh refinement,
        a constraint gaining a scalar variable of its own -- is not a connectivity change and needs
        a new DofManager.

        The merged entity mapping is rebuilt as a new dict object on purpose: consumers cache plans
        keyed on its identity (the scatter template, the element loop's gather plan), and a mapping
        mutated in place would leave those plans silently stale.

        Parameters
        ----------
        constraints
            The constraints, with their current connectivity.
        """

        (
            self.accumulatedConstraintNDof,
            self._nAccumulatedNodalFluxesFieldwiseFromConstraints,
            self.largestNumberOfConstraintNDof,
        ) = self._gatherConstraintsInformation(constraints)
        self.idcsOfConstraintsInDofVector = self._locateConstraintsInDofVector(constraints)
        self.idcsOfHigherOrderEntitiesInDofVector = self.idcsOfElementsInDofVector | self.idcsOfConstraintsInDofVector

        if self._hasAccumulatedNodalFluxes:
            self.nAccumulatedNodalFluxesFieldwise = self._computeAccumulatedNodalFluxesFieldWise(self.fields)

    @cached_property
    def dofNumberingDigest(self) -> str:
        """A digest, a hexadecimal string, of what decides the index of every degree of freedom: for every field, its
        name, its dimension, its slice in the DofVector and the labels of its nodes in the order they are numbered; and
        the names of the scalar variables, which follow, in the order they are numbered. Two equation systems with the
        same digest give every degree of freedom the same index, so a DofVector of one is a DofVector of the other --
        e.g., before and after a rebuild caused by a contact search, which changes the coupling of the degrees of
        freedom but not their numbering. The scalar variables are named, not counted: they are those of the
        constraints switched on, which a constraint may change within a step (switched from outside, or by itself), so
        that two systems with as many scalar variables may number different ones.

        The digest names the nodes by their labels, so it is the same in every run of a model: a restart checkpoint
        holds it together with the vectors it numbers. Computed at the first request; the numbering of a DofManager
        does not change."""

        digest = hashlib.blake2b(digest_size=16)
        for fieldName, fieldIndices in self.idcsOfFieldsInDofVector.items():
            nodeField = self.numberedNodeFields[fieldName]
            digest.update(repr((fieldName, nodeField.dimension, fieldIndices.start, fieldIndices.stop)).encode())
            labels = np.fromiter((node.label for node in nodeField.nodes), dtype=np.int64, count=len(nodeField.nodes))
            digest.update(labels.tobytes())
        digest.update(repr([scalarVariable.name for scalarVariable in self.idcsOfScalarVariablesInDofVector]).encode())

        return digest.hexdigest()

    def _reserveSpaceForNodeFields(
        self,
        idxStart: int,
        nodeFields: list[NodeField],
    ) -> tuple[int, dict, dict]:
        """Loop over all nodes to generate the global field-dof indices.

        Parameters
        ----------
        idxStart
            The starting index for the DOF numbering.
        nodeFields
            The list of NodeFields to process.

        Returns
        -------
        tuple
            output is a tuple of:
             * number of total DOFS
             * dictionary of field variables and indices
             * dictionary of fields and indices
        """
        idcsOfFieldsInDofVector = dict()
        currentIdxInDofVector = idxStart

        idcsOfNodeFieldVariablesInDofVector = dict()

        for nodeField in nodeFields:
            num_nodes = len(nodeField.nodes)
            dim = nodeField.dimension
            total_field_dofs = dim * num_nodes

            nextIdxInDofVector = currentIdxInDofVector + total_field_dofs
            idcsOfFieldsInDofVector[nodeField.name] = slice(currentIdxInDofVector, nextIdxInDofVector)

            all_indices = np.arange(currentIdxInDofVector, nextIdxInDofVector, dtype=int)
            indices_reshaped = all_indices.reshape(num_nodes, dim)

            for i, node in enumerate(nodeField.nodes):
                idcsOfNodeFieldVariablesInDofVector[node.fields[nodeField.name]] = indices_reshaped[i]

            currentIdxInDofVector = nextIdxInDofVector

        nDof = currentIdxInDofVector - idxStart

        return nDof, idcsOfNodeFieldVariablesInDofVector, idcsOfFieldsInDofVector

    def _reserveSpaceForScalarVariables(
        self, idxStart: int, scalarVariables: list
    ) -> tuple[int, dict[str, np.ndarray]]:
        """Loop over all ScalarVariables to generate their global indices.

        Returns
        -------
        tuple
            output is a tuple of:
             * number of total DOFS
             * dictionary of fields and indices:
                * field
                * indices
        """

        currentIdxInDofVector = idxStart

        idcsOfScalarVariablesInDofVector = {
            scalarVariable: currentIdxInDofVector + i for i, scalarVariable in enumerate(scalarVariables)
        }

        nDof = len(idcsOfScalarVariablesInDofVector)

        return (
            nDof,
            idcsOfScalarVariablesInDofVector,
        )

    def _determineIndexToNodeMap(
        self,
    ) -> dict[int, Node]:
        """Determine the map from each index (associated with a FieldVariable)
        in the DofVector to the corresponding attached Node oject.

        Returns
        -------
        dict[int, Node]
            The dictionary containing the map from index of a FieldVariable (component) in the DofVector to the respective Node instance.
        """
        indexToNodeMapping = dict()

        for (
            fieldVariable,
            fieldVariableIndices,
        ) in self.idcsOfFieldVariablesInDofVector.items():
            for index in fieldVariableIndices:
                indexToNodeMapping[index] = fieldVariable.node

        return indexToNodeMapping

    def _computeAccumulatedNodalFluxesFieldWise(self, fields: list) -> dict:
        """For the Abaqus like convergence test,
        the number of dofs 'entity-wise' is needed:
        = Σ_(elements+constraints) Σ_nodes ( nDof (field) ).

        Parameters
        ----------
        fields:
            The list of fields for which the accumulated nodal fluxes should be computed.

        Returns
        -------
        dict
            Number of accumulated fluxes per field:
                - Field
                - Number of accumulated fluxes
        """

        nAccumulatedNodalFluxesFieldwise = {}

        for field in fields:
            nAccumulatedNodalFluxesFieldwise[field] = self._nAccumulatedNodalFluxesFieldwiseFromElements.get(
                field, 0
            ) + self._nAccumulatedNodalFluxesFieldwiseFromConstraints.get(field, 0)

        return nAccumulatedNodalFluxesFieldwise

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

        idcsOfFieldVariables = self.idcsOfFieldVariablesInDofVector
        for node, multiplicity in nodeMultiplicity.items():
            for field, fv in node.fields.items():
                indices = idcsOfFieldVariables.get(fv)
                if indices is not None:
                    nAccumulatedFluxesFieldwise[field] += multiplicity * len(indices)

        return nAccumulatedFluxesFieldwise

    def _gatherElementsInformation(self, entities: list) -> tuple[int, dict, int]:
        """Generates some auxiliary information,
        which may be required by some modules of EdelweissFE.

        Parameters
        ----------
        entities
           The list of entities, for which the information is gathered.

        Returns
        -------
        tuple[int, dict, int]
            The tuple of
                - number of accumulated elemental degrees of freedom.
                - the number of  acummulated fluxes Σ_entities Σ_nodes ( nDof (field) ) for Abaqus-like convergence tests.
                - largest occuring number of dofs on any element.
        """
        accumulatedEntityNDof = 0
        largestNumberOfAnyEntitityDof = 0

        for e in entities:
            accumulatedEntityNDof += e.nDof
            largestNumberOfAnyEntitityDof = max(e.nDof, largestNumberOfAnyEntitityDof)

        nAccumulatedFluxesFieldwise = self._accumulateNodalFluxesFieldwise(entities)

        return (
            accumulatedEntityNDof,
            nAccumulatedFluxesFieldwise,
            largestNumberOfAnyEntitityDof,
        )

    def _gatherConstraintsInformation(self, entities: list) -> tuple[int, dict, int]:
        """Generates some auxiliary information,
        which may be required by some modules of EdelweissFE.

        Parameters
        ----------
        entities
           The list of entities, for which the information is gathered.

        Returns
        -------
        tuple[int, dict, int]
            The tuple of
                - number of accumulated elemental degrees of freedom.
                - the number of  acummulated fluxes Σ_entities Σ_nodes ( nDof (field) ) for Abaqus-like convergence tests.
                - largest occuring number of dofs on any element.
        """

        accumulatedEntityNDof = 0
        largestNumberOfAnyEntitityDof = 0

        for e in entities:
            accumulatedEntityNDof += e.nDof
            largestNumberOfAnyEntitityDof = max(e.nDof, largestNumberOfAnyEntitityDof)

        nAccumulatedFluxesFieldwise = self._accumulateNodalFluxesFieldwise(entities)

        return (
            accumulatedEntityNDof,
            nAccumulatedFluxesFieldwise,
            largestNumberOfAnyEntitityDof,
        )

    def _locateNodeCouplingEntitiesInDofVector(self, entities: list) -> dict:
        """Creates a dictionary containing the location (indices) of each entity (elements, ...)
        within the DofVector structure.

        The location of an entity is the indices of the field variables of its nodes, node after node and, on each
        node, field after field as the entity lists them (``fields``), reordered by its ``dofIndicesPermutation``
        unless that is None.

        An entity without a location is left out of the dictionary: one of its nodes does not have a field the entity
        couples there, or the field variable is not part of the DofVector. Any other fault of an entity raises -- it
        lacks ``nodes``, ``fields`` or ``dofIndicesPermutation``, or lists fewer fields than it has nodes -- whether
        there are few entities or many.

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
        """
        if not entities:
            return {}

        entities = list(entities)

        # Looked up once: the worker below reads it for every field of every node.
        field_var_map = self.idcsOfFieldVariablesInDofVector

        def locateChunk(chunk):
            locatedInChunk = {}

            for entity in chunk:
                # Read outside the try: an entity that cannot tell its nodes and fields is faulty, and must raise
                # instead of being taken for an entity without a location.
                nodes, fieldsOnNodes, permutation = entity.nodes, entity.fields, entity.dofIndicesPermutation

                try:
                    indices = [
                        idx
                        for iNode, node in enumerate(nodes)
                        for nodeField in fieldsOnNodes[iNode]
                        for idx in field_var_map[node.fields[nodeField]]
                    ]
                except (KeyError, AttributeError):
                    # a node without the field, or a field variable that is not part of the DofVector: no location
                    continue

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

    def _locateConstraintsInDofVector(self, constraints: list) -> dict:
        """Creates a dictionary containing the location (indices) of each entity (elements, constraints)
        within the DofVector structure.

        Returns
        -------
        dict
            A dictionary containing the location mapping.
        """

        idcsOfConstraintsInDofVector = {}
        field_var_map = self.idcsOfFieldVariablesInDofVector
        scalar_var_map = self.idcsOfScalarVariablesInDofVector

        for constraint in constraints:
            node_fields_gen = (
                field_var_map[node.fields[nodeField]]
                for iNode, node in enumerate(constraint.nodes)
                for nodeField in constraint.fieldsOnNodes[iNode]
            )
            scalar_vars_gen = ([scalar_var_map[v]] for v in constraint.scalarVariables)

            indices_gen = chain(chain.from_iterable(node_fields_gen), chain.from_iterable(scalar_vars_gen))

            idcsOfConstraintsInDofVector[constraint] = np.fromiter(indices_gen, dtype=int)

        return idcsOfConstraintsInDofVector

    def locateFieldOnNodeSet(self, field: str, nodeSet: NodeSet) -> tuple[np.ndarray, np.ndarray]:
        """Where the degrees of freedom of a field on the nodes of a node set lie in the DofVector, e.g. those a
        Dirichlet boundary condition prescribes, or those a node force acts on.

        Only the nodes whose field is part of this equation system count. In a finite element model, these are all
        nodes of the set that carry the field. A model whose equation system covers a part of its nodes only -- the
        active domain of a material point method model, the nodes in the support of an RKPM model's particles --
        leaves the other nodes of a set out. Values given per node of the set (the prescribed values of a boundary
        condition, the forces of a node force) are then selected by the positions of the remaining nodes in the set,
        which this method returns along with the indices.

        The node sets given at construction are located up front, any other one at its first request.

        Parameters
        ----------
        field
            The name of the field, one of this equation system.
        nodeSet
            The node set.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            - The indices of the degrees of freedom in the DofVector, node after node in the order of the set, on each
              node the components of the field.
            - The positions of these nodes in the set: the rows of an array that holds one row per node of the set.
        """

        locatedNodeSets = self._fieldsOnNodeSets[field]
        if nodeSet not in locatedNodeSets:
            locatedNodeSets[nodeSet] = self._locateFieldOnNodeSet(field, nodeSet)

        return locatedNodeSets[nodeSet]

    def _locateFieldOnNodeSet(self, field: str, nodeSet: NodeSet) -> tuple[np.ndarray, np.ndarray]:
        """Locate the degrees of freedom of a field on the nodes of a node set; see :meth:`locateFieldOnNodeSet`.

        Parameters
        ----------
        field
            The name of the field.
        nodeSet
            The node set.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The indices of the degrees of freedom, and the positions of their nodes in the set.
        """

        idcsOfFieldVariables = self.idcsOfFieldVariablesInDofVector

        # The indices of the field on the nodes of the set, skipping a node without the field or with a field variable
        # that is not part of this equation system (whose lookup gives None).
        indicesPerNode = [
            indices for node in nodeSet if (indices := idcsOfFieldVariables.get(node.fields.get(field))) is not None
        ]

        if len(indicesPerNode) == len(nodeSet):
            # every node of the set, as in a finite element model
            nodePositions = np.arange(len(nodeSet))
        else:
            nodePositions = np.array(
                [position for position, node in enumerate(nodeSet) if node.fields.get(field) in idcsOfFieldVariables],
                dtype=int,
            )

        # Concatenate the per-node index arrays as whole arrays, not element by element.
        dofIndices = (
            np.concatenate(indicesPerNode).astype(int, copy=False) if indicesPerNode else np.empty(0, dtype=int)
        )

        return dofIndices, nodePositions

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

        return DofVector(self.nDof, self.idcsOfHigherOrderEntitiesInDofVector)

    def getNodeForIndexInDofVector(self, index: int) -> Node:
        """Find the node for a given index in the equuation system.

        Parameters
        ----------
        index
            The index in the DofVector.

        Returns
        -------
        Node
            The attached Node.
        """

        return self.indexToHostObjectMapping[index]

    def _checkNodeFieldAccess(self, nodeField: NodeField, resultName: str, wholeField: bool):
        """Refuse an access to a node field entry this DofManager cannot serve: an entry no node field holds, a field
        it does not number, or -- for an access to the field as a whole, which addresses its rows by position -- a
        node field other than the one it numbers, even of the same name (e.g. the node field of the model, while
        this DofManager numbers that of an active domain).

        Parameters
        ----------
        nodeField
            The NodeField accessed.
        resultName
            The entry accessed.
        wholeField
            Whether the field is accessed as a whole, rather than on the nodes of a node set.

        Raises
        ------
        KeyError
            If the entry is not one of :attr:`NodeField.ENTRIES <edelweissfe.fields.nodefield.NodeField.ENTRIES>`.
        ValueError
            If this DofManager does not number the field, or, for an access as a whole, numbers another node field
            of that name.
        """

        if resultName not in NodeField.ENTRIES:
            raise KeyError(
                "a node field holds the entries {:}, not '{:}'".format(", ".join(NodeField.ENTRIES), resultName)
            )
        numberedNodeField = self.numberedNodeFields.get(nodeField.name)
        if numberedNodeField is None:
            raise ValueError("this equation system does not number the node field '{:}'".format(nodeField.name))
        if wholeField and nodeField is not numberedNodeField:
            raise ValueError(
                "this equation system numbers another node field '{:}' than the one given: a node field is read or "
                "written as a whole only by the DofManager that numbers it".format(nodeField.name)
            )

    def _locateNodeFieldOnNodeSet(self, nodeField: NodeField, nodeSet: NodeSet) -> tuple[np.ndarray, list]:
        """Where the values of a NodeField on the nodes of a node set lie, in the DofVector and in the NodeField: the
        nodes of the set whose field is part of this equation system (see :meth:`locateFieldOnNodeSet`).

        Parameters
        ----------
        nodeField
            The NodeField.
        nodeSet
            The NodeSet.

        Returns
        -------
        tuple[np.ndarray, list]
            - The indices of the degrees of freedom in the DofVector, node after node.
            - The rows of these nodes in the NodeField, in the same order.
        """

        indices, nodePositions = self.locateFieldOnNodeSet(nodeField.name, nodeSet)
        nodes = list(nodeSet)

        return indices, [nodeField.indexOfNode(nodes[position]) for position in nodePositions]

    def writeDofVectorToNodeField(
        self, dofVector: DofVector, nodeField: NodeField, resultName: str, nodeSet: NodeSet = None
    ) -> NodeField:
        """Write the values of a NodeField from the respective locations in a given DofVector; the counterpart of
        :meth:`writeNodeFieldToDofVector`.

        Parameters
        ----------
        dofVector
            The source DofVector.
        nodeField
            The NodeField to get the updated values. Written as a whole, it must be the one this DofManager numbers
            (see :attr:`numberedNodeFields`); written on a node set, any node field of a field this DofManager numbers,
            since its rows are then found by node.
        resultName
            The entry of the NodeField to write, one of
            :attr:`NodeField.ENTRIES <edelweissfe.fields.nodefield.NodeField.ENTRIES>`.
        nodeSet
            The NodeSet to write: its nodes whose field is part of this equation system (see
            :meth:`locateFieldOnNodeSet`); the other rows of the NodeField keep their values. If None, all nodes of the
            NodeField are written.

        Returns
        -------
        NodeField
            The updated NodeField.

        Raises
        ------
        KeyError
            If the entry is not one a node field holds.
        ValueError
            If this DofManager does not number the field, or, written as a whole, numbers another NodeField of that
            name.
        """

        self._checkNodeFieldAccess(nodeField, resultName, wholeField=nodeSet is None)

        if nodeSet is not None:
            indices, rowsInNodeField = self._locateNodeFieldOnNodeSet(nodeField, nodeSet)
            nodeField[resultName][rowsInNodeField] = dofVector[indices].reshape((-1, nodeField.dimension))

        else:
            indices = self.idcsOfNodeFieldsInDofVector[nodeField.name]
            data = dofVector[indices]
            nodeField[resultName][:] = data.reshape((-1, nodeField.dimension))

        return nodeField

    def writeNodeFieldToDofVector(
        self, dofVector: DofVector, nodeField: NodeField, resultName: str, nodeSet: NodeSet = None
    ):
        """Write the current values of an entire NodeField to the respective locations in a given DofVector.


        Parameters
        ----------
        dofVector
            The result DofVector.
        nodeField
            The NodeField holding the values. Read as a whole, it must be the one this DofManager numbers (see
            :attr:`numberedNodeFields`); read on a node set, any node field of a field this DofManager numbers, since
            its rows are then found by node.
        resultName
            The entry of the NodeField to read, one of
            :attr:`NodeField.ENTRIES <edelweissfe.fields.nodefield.NodeField.ENTRIES>`.
        NodeSet
            The NodeSet to consider: its nodes whose field is part of this equation system (see
            :meth:`locateFieldOnNodeSet`). If None, all nodes of the NodeField are considered.

        Returns
        -------
        DofVector
            The DofVector.

        Raises
        ------
        KeyError
            If the entry is not one a node field holds.
        ValueError
            If this DofManager does not number the field, or, read as a whole, numbers another NodeField of that name.
        """

        self._checkNodeFieldAccess(nodeField, resultName, wholeField=nodeSet is None)

        if nodeSet is not None:
            indices, rowsInNodeField = self._locateNodeFieldOnNodeSet(nodeField, nodeSet)
            dofVector[indices] = nodeField[resultName][rowsInNodeField].flatten()

        else:
            indices = self.idcsOfNodeFieldsInDofVector[nodeField.name]
            dofVector[indices] = nodeField[resultName].flatten()

        return dofVector

    def writeScalarVariablesToDofVector(self, dofVector: DofVector) -> DofVector:
        """Write the values of the scalar variables of this equation system to their locations in a given
        DofVector. Only the scalar variables this DofManager numbers are written: those of a constraint that is
        switched off are not part of the equation system.

        Parameters
        ----------
        dofVector
            The result DofVector.

        Returns
        -------
        DofVector
            The DofVector.
        """

        for scalarVariable, index in self.idcsOfScalarVariablesInDofVector.items():
            dofVector[index] = scalarVariable.value

        return dofVector

    def writeDofVectorToScalarVariables(self, dofVector: DofVector):
        """Set the values of the scalar variables of this equation system from their locations in a given
        DofVector; the counterpart of :meth:`writeScalarVariablesToDofVector`. A scalar variable this DofManager
        does not number keeps its value.

        Parameters
        ----------
        dofVector
            The source DofVector.
        """

        for scalarVariable, index in self.idcsOfScalarVariablesInDofVector.items():
            scalarVariable.value = dofVector[index]
