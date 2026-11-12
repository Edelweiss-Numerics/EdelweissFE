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
The numbering of the degrees of freedom of a discrete system: which node fields and scalar variables have degrees of
freedom, and which global index each of them gets. In the vocabulary of the textbooks, the ID array.
"""

import hashlib
from functools import cached_property

import numpy as np

from edelweissfe.fields.nodefield import NodeField
from edelweissfe.numerics.dofvector import DofVector
from edelweissfe.points.node import Node
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.variables.scalarvariable import ScalarVariable


class DofNumbering:
    """
    The numbering of the degrees of freedom of a discrete system: the global index of every degree of freedom of every
    node field and of every scalar variable, the *ID array* of the textbooks. Unlike the ID array of a textbook, every
    degree of freedom is numbered, a prescribed one included.

    Specifically, `DofNumbering`:
     * Numbers the degrees of freedom of node fields, field after field and node after node, followed by those of the
       scalar variables.
     * Locates the degrees of freedom of a field on the nodes of a node set, e.g. those a Dirichlet boundary condition
       prescribes.
     * Passes the values between the node fields and the scalar variables, and a :class:`DofVector`.

    It knows nothing of the entities that couple the degrees of freedom: where an element or a constraint finds its
    degrees of freedom is the business of :class:`~edelweissfe.numerics.doflocations.DofLocations`, which is built on a
    numbering.

    A numbering is immutable: it is determined by its parameters, and never changes. A model whose degrees of freedom
    change -- a mesh refinement, a Lagrange multiplier entering the system -- is numbered anew, by a new
    `DofNumbering`.

    Parameters
    ----------
    nodeFields : list of NodeField
        The list of NodeFields to be represented in the DofVector structure.
    scalarVariables : list of ScalarVariable, optional
        The list of ScalarVariables to be represented in the DofVector structure.
    nodeSets : list of NodeSet, optional
        The list of NodeSets for which the degrees of freedom are located up front, see :meth:`locateFieldOnNodeSet`.
    """

    def __init__(
        self,
        nodeFields: list[NodeField],
        scalarVariables: list[ScalarVariable] = None,
        nodeSets: list[NodeSet] = None,
    ):
        if scalarVariables is None:
            scalarVariables = []
        if nodeSets is None:
            nodeSets = []

        nodeFields = list(nodeFields)

        #: The NodeFields this DofNumbering numbers, by name. They are the only ones it reads from and writes to as a
        #: whole.
        self.numberedNodeFields = {nodeField.name: nodeField for nodeField in nodeFields}

        nDofNodeFields, fieldVariableIndices, fieldIndices = self._reserveSpaceForNodeFields(0, nodeFields)
        nDofScalarVariables, scalarVariableIndices = self._reserveSpaceForScalarVariables(
            nDofNodeFields, scalarVariables
        )

        #: Dictionary mapping nodal field variables to their indices in the DofVector
        self.idcsOfFieldVariablesInDofVector = fieldVariableIndices
        #: Dictionary mapping field names to all indices (a slice) in the DofVector
        self.idcsOfFieldsInDofVector = fieldIndices
        #: Dictionary mapping scalar variables to their index in the DofVector
        self.idcsOfScalarVariablesInDofVector = scalarVariableIndices
        #: The total number of degrees of freedom (and size of the DofVector)
        self.nDof = nDofNodeFields + nDofScalarVariables
        #: The list of field names in the DofVector
        self.fields = list(self.idcsOfFieldsInDofVector.keys())

        #: For each field, the node sets located so far, each with the indices of the field's degrees of freedom on its
        #: nodes and the positions of these nodes in the set; see :meth:`locateFieldOnNodeSet`.
        self._fieldsOnNodeSets = {
            field: {nodeSet: self._locateFieldOnNodeSet(field, nodeSet) for nodeSet in nodeSets}
            for field in self.idcsOfFieldsInDofVector
        }

    @cached_property
    def digest(self) -> str:
        """A digest of what decides the index of every degree of freedom, a hexadecimal string over, for every field,
        its name, its dimension, its slice in the DofVector and the labels of its nodes in the order they are numbered,
        and over the names of the scalar variables, which follow, in the order they are numbered. Two numberings with
        the same digest give every degree of freedom the same index, so a DofVector of one is a DofVector of the other
        -- e.g., the vectors of a restart checkpoint and the numbering of the resumed model, which is built anew. The
        scalar variables are named, not counted: they are those of the constraints that take part, which a constraint
        may change by itself, by coming within reach or leaving it, so that two numberings with as many scalar
        variables may number different ones.

        The digest names the nodes by their labels, so it is the same in every run of a model: a restart checkpoint
        holds it together with the vectors it numbers. Computed at the first request; a DofNumbering
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

    @cached_property
    def indexToHostObjectMapping(self) -> dict[int, Node]:
        """The node of every index of a field variable in the DofVector, built on first use. Only the report of the
        largest residual of an implicit solver reads it (through :meth:`getNodeForIndexInDofVector`), while building it
        costs a pass over every degree of freedom, which a numbering that is never asked -- one of the many an explicit
        solver builds over a run, say -- does not pay."""

        return self._determineIndexToNodeMap()

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

    def locateFieldOnNodeSet(self, field: str, nodeSet: NodeSet) -> tuple[np.ndarray, np.ndarray]:
        """Where the degrees of freedom of a field on the nodes of a node set lie in the DofVector, e.g. those a
        Dirichlet boundary condition prescribes, or those a node force acts on.

        Only the nodes whose field is part of this discrete system count. In a finite element model, these are all
        nodes of the set that carry the field. A model whose discrete system covers a part of its nodes only -- the
        active domain of a material point method model, the nodes in the support of an RKPM model's particles --
        leaves the other nodes of a set out. Values given per node of the set (the prescribed values of a boundary
        condition, the forces of a node force) are then selected by the positions of the remaining nodes in the set,
        which this method returns along with the indices.

        The node sets given at construction are located up front, any other one at its first request.

        Parameters
        ----------
        field
            The name of the field, one of this discrete system.
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
        # that is not part of this discrete system (whose lookup gives None).
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

    def getNodeForIndexInDofVector(self, index: int) -> Node:
        """Find the node for a given index in the discrete system.

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
        """Refuse an access to a node field entry this DofNumbering cannot serve: an entry no node field holds, a field
        it does not number, or -- for an access to the field as a whole, which addresses its rows by position -- a
        node field other than the one it numbers, even of the same name (e.g. the node field of the model, while
        this DofNumbering numbers that of an active domain).

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
            If this DofNumbering does not number the field, or, for an access as a whole, numbers another node field
            of that name.
        """

        if resultName not in NodeField.ENTRIES:
            raise KeyError(
                "a node field holds the entries {:}, not '{:}'".format(", ".join(NodeField.ENTRIES), resultName)
            )
        numberedNodeField = self.numberedNodeFields.get(nodeField.name)
        if numberedNodeField is None:
            raise ValueError("this discrete system does not number the node field '{:}'".format(nodeField.name))
        if wholeField and nodeField is not numberedNodeField:
            raise ValueError(
                "this discrete system numbers another node field '{:}' than the one given: a node field is read or "
                "written as a whole only by the DofNumbering that numbers it".format(nodeField.name)
            )

    def _locateNodeFieldOnNodeSet(self, nodeField: NodeField, nodeSet: NodeSet) -> tuple[np.ndarray, list]:
        """Where the values of a NodeField on the nodes of a node set lie, in the DofVector and in the NodeField: the
        nodes of the set whose field is part of this discrete system (see :meth:`locateFieldOnNodeSet`).

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
            The NodeField to get the updated values. Written as a whole, it must be the one this DofNumbering numbers
            (see :attr:`numberedNodeFields`); written on a node set, any node field of a field this DofNumbering
            numbers, since its rows are then found by node.
        resultName
            The entry of the NodeField to write, one of
            :attr:`NodeField.ENTRIES <edelweissfe.fields.nodefield.NodeField.ENTRIES>`.
        nodeSet
            The NodeSet to write: its nodes whose field is part of this discrete system (see
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
            If this DofNumbering does not number the field, or, written as a whole, numbers another NodeField of that
            name.
        """

        self._checkNodeFieldAccess(nodeField, resultName, wholeField=nodeSet is None)

        if nodeSet is not None:
            indices, rowsInNodeField = self._locateNodeFieldOnNodeSet(nodeField, nodeSet)
            nodeField[resultName][rowsInNodeField] = dofVector[indices].reshape((-1, nodeField.dimension))

        else:
            indices = self.idcsOfFieldsInDofVector[nodeField.name]
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
            The NodeField holding the values. Read as a whole, it must be the one this DofNumbering numbers (see
            :attr:`numberedNodeFields`); read on a node set, any node field of a field this DofNumbering numbers, since
            its rows are then found by node.
        resultName
            The entry of the NodeField to read, one of
            :attr:`NodeField.ENTRIES <edelweissfe.fields.nodefield.NodeField.ENTRIES>`.
        NodeSet
            The NodeSet to consider: its nodes whose field is part of this discrete system (see
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
            If this DofNumbering does not number the field, or, read as a whole, numbers another NodeField of that name.
        """

        self._checkNodeFieldAccess(nodeField, resultName, wholeField=nodeSet is None)

        if nodeSet is not None:
            indices, rowsInNodeField = self._locateNodeFieldOnNodeSet(nodeField, nodeSet)
            dofVector[indices] = nodeField[resultName][rowsInNodeField].flatten()

        else:
            indices = self.idcsOfFieldsInDofVector[nodeField.name]
            dofVector[indices] = nodeField[resultName].flatten()

        return dofVector

    def writeScalarVariablesToDofVector(self, dofVector: DofVector) -> DofVector:
        """Write the values of the scalar variables of this discrete system to their locations in a given
        DofVector. Only the scalar variables this DofNumbering numbers are written: those of a constraint that is
        switched off are not part of the discrete system.

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
        """Set the values of the scalar variables of this discrete system from their locations in a given
        DofVector; the counterpart of :meth:`writeScalarVariablesToDofVector`. A scalar variable this DofNumbering
        does not number keeps its value.

        Parameters
        ----------
        dofVector
            The source DofVector.
        """

        for scalarVariable, index in self.idcsOfScalarVariablesInDofVector.items():
            scalarVariable.value = dofVector[index]
