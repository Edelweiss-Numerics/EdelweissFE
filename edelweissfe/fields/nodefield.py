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
# Created on Tue Dec 18 09:18:25 2018

# @author: Matthias Neuner

import numpy as np

from edelweissfe.points.node import Node
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.sets.nodeset import NodeSet


class NodeFieldSubset:
    pass


class NodeField:
    """
    This class represents a node field.
    A node field associates every node with the entries of a field variable listed in :attr:`ENTRIES`:
    the solution ``U``, its increment ``dU``, the nodal force ``P``, and the rates ``V`` and ``A``. Every
    node field holds all of them, from its construction on; their meaning is defined once, in
    :doc:`/documentation/nodefieldresults`.
    Furthermore, for convencience, it allows to get fast access to values for individual nodes or node sets.

    The purpose is to store field data in an efficient, contiguos manner rather than distributing it across
    all individual nodes.

    .. code-block:: console

        Example:

        NodeField 'Displacement'
           values: {'U' : [[0,0,0],  # Node (1)
                           [1,0,0],  # Node (2)
                           [0,1,0]]} # Node (3)

        Spatial representation:

        (1) --------- (2)
         | *             *
         |  *+---------+   *+---------+
         |   | [0,0,0] |    | [1,0,0] |
        (3)  +---------+    +---------+
          *
           * +---------+
             | [0,1,0] |
             +---------+

    Parameters
    ----------
    fieldName
        The name of the field.
    dimension
        The dimension of the field.
    nodes
        The associated nodes. Only nodes with active fields are considered.
    """

    #: The entries every node field holds, see :doc:`/documentation/nodefieldresults`: the solution ``U``,
    #: the increment ``dU`` of the last accepted increment, the nodal force ``P``, and the rate ``V`` and
    #: second rate ``A`` of the solution. There are no others.
    ENTRIES = ("U", "dU", "P", "V", "A")

    def __init__(self, fieldName: str, dimension: int, nodeSet: NodeSet):
        self.name = fieldName
        self.associatedSet = nodeSet
        self.nodes = [n for n in nodeSet if fieldName in n.fields]
        self.dimension = dimension
        self._indicesOfNodesInArray = {n: i for i, n in enumerate(self.nodes)}
        self._subsetCache = dict()
        self._values = {entry: np.zeros((len(self.nodes), dimension), dtype=float) for entry in self.ENTRIES}
        self._version = 0  #: bumped on every in-place mutation (resize); stable identity for AMR

    def resize(self, nodes) -> None:
        """Resize this NodeField in-place for a new list of associated nodes, preserving values
        for every node identity present both before and after the resize (e.g. AMR-retained
        nodes keep their converged state; new nodes get a zeroed entry until a caller fills it
        in, e.g. with an interpolated warm start).

        This object's identity is preserved -- consumers holding a reference to this NodeField
        (or a :class:`NodeFieldSubset` of it) see the new size without re-fetching from the
        model.

        Parameters
        ----------
        nodes
            The new (possibly larger or smaller) list of nodes this NodeField is associated with.
            Only nodes with this field active are retained, exactly as at construction.
        """
        oldIndicesOfNodesInArray = self._indicesOfNodesInArray
        oldValues = self._values

        self.nodes = [n for n in nodes if self.name in n.fields]
        self._indicesOfNodesInArray = {n: i for i, n in enumerate(self.nodes)}

        # nodes retained across the resize (e.g. AMR-untouched nodes); computed once and reused for
        # every value entry below, instead of a per-node dict lookup repeated for each entry
        commonNodes = oldIndicesOfNodesInArray.keys() & self._indicesOfNodesInArray.keys()
        oldIdx = np.fromiter((oldIndicesOfNodesInArray[n] for n in commonNodes), dtype=np.intp, count=len(commonNodes))
        newIdx = np.fromiter(
            (self._indicesOfNodesInArray[n] for n in commonNodes), dtype=np.intp, count=len(commonNodes)
        )

        newValues = dict()
        for entry, oldArray in oldValues.items():
            newArray = np.zeros((len(self.nodes), self.dimension), dtype=float)
            if len(oldIdx):
                newArray[newIdx] = oldArray[oldIdx]
            newValues[entry] = newArray
        self._values = newValues

        self._version += 1

    def _getNodeFieldSubsetClass(
        self,
    ):
        return NodeFieldSubset

    def __getitem__(self, key):
        return self._values[key]

    def __contains__(self, key):
        return key in self._values

    def indexOfNode(self, node) -> int:
        """
        The row index of a node's entries within this field's value arrays.

        Parameters
        ----------
        node
            The node.

        Returns
        -------
        int
            The row index.
        """
        return self._indicesOfNodesInArray[node]

    def subset(self, subset) -> NodeFieldSubset:
        """
        Get a view on a subset of the field.

        Parameters
        ----------
        subset
            The subset, e.g., a single :class:`Node, or a :class:`NodeSet or :class:`ElementSet.

        Returns
        -------
        NodeFieldSubset
            The subset of the present NodeField.
        """
        return self._getSubsetFromCache(subset)

    def _getSubsetFromCache(self, subset) -> NodeFieldSubset:
        """
        Exploit a cache to reuse already constructed NodeFieldSubsets.
        If the subset does not exist, it will be created here.

        Parameters
        ----------
        subset
            The subset, e.g., a single Node, or a NodeSet or ElementSet.

        Returns
        -------
        NodeFieldSubset
            The subset of the present NodeField.
        """

        if subset in self._subsetCache:
            return self._subsetCache[subset]
        else:
            self._subsetCache[subset] = self._getNodeFieldSubsetClass()(self, subset)
            return self._subsetCache[subset]

    def copyEntriesFromOther(self, other, fieldValueEntries: list[str] = None):
        """
        Copy values from another NodeField.
        Of the nodes, the intersection is considered; the other nodes of this field get zero.

        Parameters
        ----------
        other
            The NodeField to copy from.
        fieldValueEntries
            The list of entries which should be copied. Default: all entries are copied.
        """

        if not fieldValueEntries:
            fieldValueEntries = self.ENTRIES

        commonNodes = self._indicesOfNodesInArray.keys() & other._indicesOfNodesInArray.keys()

        for fieldValueEntry in fieldValueEntries:
            self[fieldValueEntry][:] = 0.0
            idcsHere = [self._indicesOfNodesInArray[n] for n in commonNodes]
            idcsOther = [other._indicesOfNodesInArray[n] for n in commonNodes]
            self[fieldValueEntry][idcsHere] = other[fieldValueEntry][idcsOther]

    def addEntriesFromOther(self, other, fieldValueEntries: list[str] | dict[str, str] = None):
        """
        Add values from another NodeField.
        Of the nodes, the intersection is considered.

        Parameters
        ----------
        other
            The NodeField to add from.
        fieldValueEntries
            The entries which should be added. Default: all entries are added.
            May be a list of entry names shared by both fields, or a dict mapping an
            entry name on ``other`` to the (possibly differently named) entry name on
            ``self`` it should be accumulated into.
        """

        if not fieldValueEntries:
            fieldValueEntries = self.ENTRIES
        if not isinstance(fieldValueEntries, dict):
            fieldValueEntries = {entry: entry for entry in fieldValueEntries}

        commonNodes = self._indicesOfNodesInArray.keys() & other._indicesOfNodesInArray.keys()
        idcsHere = [self._indicesOfNodesInArray[n] for n in commonNodes]
        idcsOther = [other._indicesOfNodesInArray[n] for n in commonNodes]

        for otherEntry, selfEntry in fieldValueEntries.items():
            self[selfEntry][idcsHere] += other[otherEntry][idcsOther]


class NodeFieldSubset(NodeField):
    def __init__(self, parentNodeField, subset):
        self.parentNodeField = parentNodeField
        self.name = parentNodeField.name
        self.associatedSet = subset
        self._seenParentVersion = parentNodeField._version
        self.nodes = self._getSubsetNodes(subset)
        self._indicesOfNodesInParentArray = np.array([parentNodeField._indicesOfNodesInArray[n] for n in self.nodes])

    def _healIfParentResized(self):
        """Rebuild the cached parent-index array if the parent NodeField was resized (e.g. by AMR)
        since this subset was created or last accessed. Since the parent NodeField (and, for a
        NodeSet/ElementSet-based subset, ``associatedSet`` itself) has stable identity, this makes
        the subset transparent to a mesh mutation -- no observer registration needed.

        This keys off the *parent's* version only, not ``associatedSet``'s own version -- correct
        as long as every mutator that changes a NodeSet/ElementSet's membership also resizes every
        NodeField of that model in the same call (true today: hadaptivity._materialize always goes
        through FEModel._resizeNodeFieldsForNodes, which resizes every NodeField, whenever any set
        changes). A future mutator that changes set membership without touching NodeFields would
        need this subset to key off ``associatedSet._version`` as well, or this heal would silently
        miss it."""
        if self.parentNodeField._version != self._seenParentVersion:
            self.nodes = self._getSubsetNodes(self.associatedSet)
            self._indicesOfNodesInParentArray = np.array(
                [self.parentNodeField._indicesOfNodesInArray[n] for n in self.nodes]
            )
            self._seenParentVersion = self.parentNodeField._version

    def __getitem__(self, key):
        self._healIfParentResized()
        return self.parentNodeField[key][self._indicesOfNodesInParentArray]

    def __contains__(self, key):
        self._healIfParentResized()
        return key in self.nodes

    def subset(self, subset):
        raise Exception("Subsets of subsets are not yet implemented!")

    def _getSubsetNodes(self, subset) -> list[Node]:
        """
        Get the nodes associated with a subset.
        Only nodes with the active field are considered.

        Parameters
        ----------
        subset
            The subset, e.g., a single Node, a NodeSet or ElementSet.

        Returns
        -------
        list[Node]
            The list of subset nodes.
        """
        if isinstance(subset, Node):
            nodeCandidates = [
                subset,
            ]
        elif isinstance(subset, ElementSet):
            nodeCandidates = subset.extractNodeSet()
        elif isinstance(subset, NodeSet):
            nodeCandidates = subset
        else:
            raise Exception("Invalid subset")

        return [n for n in nodeCandidates if n in self.parentNodeField._indicesOfNodesInArray]
