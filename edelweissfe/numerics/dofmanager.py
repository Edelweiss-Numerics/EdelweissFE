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

from concurrent.futures import ThreadPoolExecutor
from itertools import chain

import numpy as np

from edelweissfe.config.phenomena import phenomena
from edelweissfe.fields.nodefield import NodeField
from edelweissfe.nodecouplingentity.base.nodecouplingentity import (
    BaseNodeCouplingEntity,
)
from edelweissfe.numerics.dofvector import DofVector
from edelweissfe.numerics.parallelizationutilities import (
    getNumberOfThreads,
    isFreeThreadingSupported,
)
from edelweissfe.numerics.vijsystemmatrix import VIJSystemMatrix
from edelweissfe.points.node import Node
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.variables.scalarvariable import ScalarVariable


class DofManager:
    """
    The DofManager

     * analyzes the domain (nodes and constraints),
     * collects information about the necessary structure of the degrees of freedom
     * handles the active fields on each node
     * counts the accumulated number of associated elements on each dof (for the Abaqus like convergence test)
     * supplies the framework with DofVectors and VIJSystemMatrices

    Parameters
    ----------
    nodeFields
        The list of NodeFields which should be represented in the DofVector structure.
    scalarVariables
        The list of ScalarVariables which should be represented in the DofVector structure.
    elements
        The list of Elements for which a map to the respective indices should be created.
    constraints
        The list of Constraints for which a map to the respective indices should be created.
    nodeSets
        The list of NodeSets for which a map to the respective indices should be created.
    initializeVIJPattern
        Whether to initialize the VIJ pattern (I and J vectors) during construction. Can be set to False if the pattern will be initialized later or not needed.
    initializeAccumulatedNodalFluxesFieldwise
        Whether to compute the accumulated nodal fluxes fieldwise during construction. This is needed for the Abaqus like convergence test, but can be set to False if not needed.
    determiningIndexToHostObjectMappping
        Whether to determine the mapping from indices in the DofVector to their host objects (e.g., Nodes) during construction. This can be set to False if the mapping will be determined later or not needed.
    """

    def __init__(
        self,
        nodeFields: list[NodeField],
        scalarVariables: list[ScalarVariable] = None,
        elements: list[BaseNodeCouplingEntity] = None,
        constraints: list[BaseNodeCouplingEntity] = None,
        nodeSets: list[NodeSet] = None,
        initializeVIJPattern: bool = True,
        initializeAccumulatedNodalFluxesFieldwise: bool = True,
        determiningIndexToHostObjectMappping: bool = True,
    ):
        if scalarVariables is None:
            scalarVariables = []
        if elements is None:
            elements = []
        if constraints is None:
            constraints = []
        if nodeSets is None:
            nodeSets = []

        self.nDof = int()  #: The total number of degrees of freedom (and size of the DofVector)
        self.fields = list()  #: The list of fields which can be found in the Dofvector
        self.idcsOfFieldsInDofVector = dict()  #: The dictionary mapping the field names to all indices in the DofVector
        self.idcsOfScalarVariablesInDofVector = (
            dict()
        )  #: The dictionary mapping the scalar variables to all indices in the DofVector

        self.idcsOfFieldVariablesInDofVector = (
            dict()
        )  #: The dictionary mapping the nodal field variables to the indices in the DofVector
        self.idcsOfNodeFieldsInDofVector = (
            dict()
        )  #: The dictionary mapping the a complete NodeField to the all its indices in the DofVector
        self.indexToHostObjectMapping = (
            dict()
        )  #: The reverse dictionary mapping an index to the Host (e.g., a Node) holding the index's FieldvVariable
        self.accumulatedElementNDof = (
            int()
        )  #: The accumulated number of element dofs (= the sum of all element vector sizes)
        self.largestNumberOfElNDof = int()  #: The size of the largest of all element dof vectors
        self.accumulatedConstraintNDof = (
            int()
        )  #: The accumulated number of constraint dofs (= the sum of all constraint vector sizes)
        self.largestNumberOfConstraintNDof = int()  #: The size of the largest of all constraint dof vectors

        self.idcsOfFieldsOnNodeSetsInDofVector = (
            dict()
        )  #: The dictionary mapping for each field a NodeSet to the respective indices in the DofVector
        self.idcsOfElementsInDofVector = dict()  #: The dictionary mapping an element to it's indices in the DofVector
        self.idcsOfConstraintsInDofVector = (
            dict()
        )  #: The dictionary mapping a constraint to it's indices in the DofVector

        self._sizeVIJ = (
            int()
        )  #: The number of nonzero entries of the system matrix, resulting from the dense higher order entities contributions

        # initialization:

        self._determineDofsAndTheirIndices(nodeFields, scalarVariables)

        self._gatherInformationAboutEntities(elements, constraints, nodeSets)

        self.idcsOfFieldsInDofVector = self.idcsOfNodeFieldsInDofVector
        self.fields = self.idcsOfFieldsInDofVector.keys()

        self.idcsOfBasicVariablesInDofVector = (
            self.idcsOfFieldVariablesInDofVector | self.idcsOfScalarVariablesInDofVector
        )
        self.idcsOfHigherOrderEntitiesInDofVector = self.idcsOfElementsInDofVector | self.idcsOfConstraintsInDofVector

        if initializeAccumulatedNodalFluxesFieldwise:
            self.nAccumulatedNodalFluxesFieldwise = self._computeAccumulatedNodalFluxesFieldWise(self.fields)

        if initializeVIJPattern:
            self._sizeVIJ = self._accumulatedElementVIJSize + self._accumulatedConstraintVIJSize
            (self.I, self.J, self.idcsOfHigherOrderEntitiesInVIJ) = self._initializeVIJPattern()

        if determiningIndexToHostObjectMappping:
            self.indexToHostObjectMapping |= self._determineIndexToNodeMap()

    def _determineDofsAndTheirIndices(self, nodeFields: list, scalarVariables: list):

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
            self._accumulatedElementVIJSize,
            self._nAccumulatedNodalFluxesFieldwiseFromElements,
            self.largestNumberOfElNDof,
        ) = self._gatherElementsInformation(elements)

        (
            self.accumulatedConstraintNDof,
            self._accumulatedConstraintVIJSize,
            self._nAccumulatedNodalFluxesFieldwiseFromConstraints,
            self.largestNumberOfConstraintNDof,
        ) = self._gatherConstraintsInformation(constraints)

        self.idcsOfFieldsOnNodeSetsInDofVector = self._locateFieldsOnNodeSetsInDofVector(nodeSets)
        self.idcsOfElementsInDofVector = self._locateNodeCouplingEntitiesInDofVector(elements)
        self.idcsOfConstraintsInDofVector = self._locateConstraintsInDofVector(constraints)

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
        """For the VIJ (COO) system matrix and the Abaqus like convergence test,
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

    def _gatherElementsInformation(self, entities: list) -> tuple[int, int, int, int]:
        """Generates some auxiliary information,
        which may be required by some modules of EdelweissFE.

        Parameters
        ----------
        entities
           The list of entities, for which the information is gathered.

        Returns
        -------
        tuple[int,int]
            The tuple of
                - number of accumulated elemental degrees of freedom.
                - number of accumulated system matrix sizes.
                - the number of  acummulated fluxes Σ_entities Σ_nodes ( nDof (field) ) for Abaqus-like convergence tests.
                - largest occuring number of dofs on any element.
        """
        accumulatedEntityNDof = 0
        accumulatedEntityVIJSize = 0
        largestNumberOfAnyEntitityDof = 0

        nAccumulatedFluxesFieldwise = {k: 0 for k in phenomena.keys()}

        for e in entities:
            accumulatedEntityNDof += e.nDof
            accumulatedEntityVIJSize += e.nDof**2

            for node in e.nodes:
                for field, fv in node.fields.items():
                    indices = self.idcsOfFieldVariablesInDofVector.get(fv)
                    if indices is not None:
                        nAccumulatedFluxesFieldwise[field] += len(indices)

            largestNumberOfAnyEntitityDof = max(e.nDof, largestNumberOfAnyEntitityDof)

        return (
            accumulatedEntityNDof,
            accumulatedEntityVIJSize,
            nAccumulatedFluxesFieldwise,
            largestNumberOfAnyEntitityDof,
        )

    def _gatherConstraintsInformation(self, entities: list) -> tuple[int, int, int, int]:
        """Generates some auxiliary information,
        which may be required by some modules of EdelweissFE.

        Parameters
        ----------
        entities
           The list of entities, for which the information is gathered.

        Returns
        -------
        tuple[int,int]
            The tuple of
                - number of accumulated elemental degrees of freedom.
                - number of accumulated system matrix sizes.
                - the number of  acummulated fluxes Σ_entities Σ_nodes ( nDof (field) ) for Abaqus-like convergence tests.
                - largest occuring number of dofs on any element.
        """

        return self._gatherElementsInformation(entities)

    def _locateNodeCouplingEntitiesInDofVector(self, entities: list) -> dict:
        """Creates a dictionary containing the location (indices) of each entity (elements, ...)
        within the DofVector structure.

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

        # Ensure we have a list for chunking
        entities_list = list(entities)
        nEntities = len(entities_list)

        # Threading setup
        numThreads = getNumberOfThreads() if isFreeThreadingSupported() else 1

        # Heuristic: Don't spawn threads for tiny workloads
        if nEntities < 500 or numThreads == 1:
            return self._locate_serial_internal(entities_list)

        chunk_size = (nEntities + numThreads - 1) // numThreads
        chunks = [entities_list[i : i + chunk_size] for i in range(0, nEntities, chunk_size)]

        # 1. Localize the global map to a local variable
        # This prevents threads from accessing 'self' inside the hot loop
        global_fv_map = self.idcsOfFieldVariablesInDofVector

        def processEntityChunk(chunk):
            localMap = {}
            # 2. Bind the .get or dictionary access to a local name
            # This is a micro-optimization that avoids repeated attribute lookups
            fv_lookup = global_fv_map

            for ent in chunk:
                # 3. Cache entity attributes to local variables
                # This avoids the 'dot' overhead inside the nested loops
                nodes = ent.nodes
                fields_on_ent = ent.fields
                permutation = ent.dofIndicesPermutation

                # 4. Use a flat list comprehension or optimized loop
                # List comprehensions are generally faster than .append/.extend in Python
                try:
                    indices = [
                        idx
                        for iNode, node in enumerate(nodes)
                        for f_name in fields_on_ent[iNode]
                        for idx in fv_lookup[node.fields[f_name]]
                    ]
                except (KeyError, AttributeError):
                    continue

                # 5. Use np.int32 for better cache performance
                destArr = np.array(indices, dtype=np.int32)

                if permutation is not None:
                    localMap[ent] = destArr[permutation]
                else:
                    localMap[ent] = destArr
            return localMap

        idcsOfElementsInDofVector = {}
        # We use the executor context manager here as requested (no persistent executor)
        with ThreadPoolExecutor(max_workers=numThreads) as executor:
            # map() returns a generator; wrapping in list() or iterating merges results
            for partial_map in executor.map(processEntityChunk, chunks):
                idcsOfElementsInDofVector.update(partial_map)

        return idcsOfElementsInDofVector

    def _locate_serial_internal(self, entities: list) -> dict:
        """Fast serial fallback for small chunks to avoid thread overhead."""
        localMap = {}
        fv_lookup = self.idcsOfFieldVariablesInDofVector
        for ent in entities:
            try:
                indices = []
                for iNode, node in enumerate(ent.nodes):
                    for f_name in ent.fields[iNode]:
                        indices.extend(fv_lookup[node.fields[f_name]])
            except (KeyError, IndexError, AttributeError, TypeError):
                continue

            destArr = np.array(indices, dtype=np.int32)
            if ent.dofIndicesPermutation is not None:
                localMap[ent] = destArr[ent.dofIndicesPermutation]
            else:
                localMap[ent] = destArr
        return localMap

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

    def _locateFieldsOnNodeSetsInDofVector(self, nodeSets: list) -> dict:
        """Creates a dictionary containing the location (indices) of each entity (elements, constraints)
        within the DofVector structure.

        Parameters
        ----------
        nodeSets
                The list of NodeSets to consider.

        Returns
        -------
        dict
            A dictionary containing the location mapping.
        """

        nodeSetFieldsInDofVector = {}
        field_var_map = self.idcsOfFieldVariablesInDofVector

        for field in self.idcsOfNodeFieldsInDofVector:
            nodeSetFieldsInDofVector[field] = dict()
            for nSet in nodeSets:
                indices_gen = chain.from_iterable(
                    field_var_map[node.fields[field]] for node in nSet if field in node.fields
                )
                nodeSetFieldsInDofVector[field][nSet] = np.fromiter(indices_gen, dtype=int)

        return nodeSetFieldsInDofVector

    def _initializeVIJPattern(
        self,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Generate the IJ pattern for VIJ (COO) system matrices.

        Returns
        -------
        tuple
             - I vector
             - J vector
             - the entities to system matrix entry mapping.
        """

        entitiesInVIJ = {}

        sizeVIJ = self._sizeVIJ

        I = np.zeros(sizeVIJ, dtype=np.intc)  # noqa: E741
        J = np.zeros(sizeVIJ, dtype=np.intc)  # noqa: E741
        idxInVIJ = 0

        for (
            entity,
            entityIdcsInDofVector,
        ) in self.idcsOfHigherOrderEntitiesInDofVector.items():
            entitiesInVIJ[entity] = idxInVIJ

            nDofEntity = len(entityIdcsInDofVector)
            block_size = nDofEntity**2

            VIJLocations = np.tile(entityIdcsInDofVector, (nDofEntity, 1))
            I[idxInVIJ : idxInVIJ + block_size] = VIJLocations.flatten()
            J[idxInVIJ : idxInVIJ + block_size] = VIJLocations.flatten("F")
            idxInVIJ += block_size

        return I, J, entitiesInVIJ

    def constructVIJSystemMatrix(
        self,
    ) -> VIJSystemMatrix:
        """Construct a VIJ (COO) Sparse System matrix object, which also has knowledge about
        the location of each entity.

        Returns
        -------
        VIJSystemMatrix
            The system Matrix.
        """

        nDof = self.nDof
        I = self.I  # noqa: E741
        J = self.J

        return VIJSystemMatrix(nDof, I, J, self.idcsOfHigherOrderEntitiesInVIJ)

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

    def writeDofVectorToNodeField(self, dofVector, nodeField, resultName):
        """Write the current values of an entire NodeField from the respective locations in a given DofVector.


        Parameters
        ----------
        dofVector
            The source DofVector.
        nodeField
            The NodeField to get the updated values.
        resultname
            The name of the value entries held by the NodeField.

        Returns
        -------
        NodeField
            The updated NodeField.
        """

        if resultName not in nodeField:
            nodeField.createFieldValueEntry(resultName)

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
            The NodeField holding the values.
        resultname
            The name of the value entries held by the NodeField.
        NodeSet
            The NodeSet to consider. If None, all nodes of the NodeField are considered.

        Returns
        -------
        DofVector
            The DofVector.
        """

        if nodeSet is not None:
            if nodeSet not in self.idcsOfFieldsOnNodeSetsInDofVector.get(nodeField.name, {}):
                new_map = self._locateFieldsOnNodeSetsInDofVector([nodeSet])
                if nodeField.name not in self.idcsOfFieldsOnNodeSetsInDofVector:
                    self.idcsOfFieldsOnNodeSetsInDofVector[nodeField.name] = {}
                self.idcsOfFieldsOnNodeSetsInDofVector[nodeField.name].update(new_map[nodeField.name])

            indices = self.idcsOfFieldsOnNodeSetsInDofVector[nodeField.name][nodeSet]
            dofVector[indices] = nodeField.subset(nodeSet)[resultName].flatten()

        else:
            indices = self.idcsOfNodeFieldsInDofVector[nodeField.name]
            dofVector[indices] = nodeField[resultName].flatten()

        return dofVector
