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
# Created on Fri Jan 27 19:53:45 2017

# @author: Matthias Neuner

import textwrap
from operator import attrgetter

import h5py
import numpy as np

from edelweissfe.config.phenomena import getFieldSize, phenomena
from edelweissfe.fields.nodefield import NodeField
from edelweissfe.journal.journal import Journal
from edelweissfe.models.modelchange import TopologyRecord
from edelweissfe.models.topologypipeline import TopologyPipeline
from edelweissfe.numerics.parallelizationutilities import (
    chunked_iterable,
    getNumberOfThreads,
    getThreadPool,
    isFreeThreadingSupported,
)
from edelweissfe.utils.checkpoint import readRestartDataInto, writeRestartDataOf
from edelweissfe.utils.exceptions import RestartError, TopologyError
from edelweissfe.variables.fieldvariable import FieldVariable
from edelweissfe.variables.scalarvariable import ScalarVariable


class FEModel:
    """This is is a standard finite element model tree.
    It takes care of the correct number of variables,
    for nodes and scalar degrees of freedem, and it manages the fields.


    Parameters
    ----------
    dimension
        The dimension of the model.
    """

    identification = "FEModel"

    def __init__(self, dimension: int):
        self.time = 0.0  #: Current time of the model.
        self.nodes = {}  #: Nodes in the model.
        self.elements = {}  #: Elements in the model.
        self.nodeSets = {}  #: NodeSets in the model.
        self.nodeFields = {}  #: NodeFields in the model.
        self.elementSets = {}  #: ElementSets in the model.
        self.sections = {}  #: Sections in the model.
        self.surfaces = {}  #: Surface definitions in the model.
        self.constraints = {}  #: Constraints in the model.
        self.constraintSets = {}  #: ConstraintsSets in the model.
        self.multiPointConstraints = {}  #: Multi-point (DOF-elimination) constraints in the model.
        self.modelModifiers = {}  #: Model modifiers (dynamic topology / mesh mutation entities) in the model.
        self.contactFacetRecipes = {}  #: facet elSet name -> (surfaceName, prefix, triangulation, nodalWeights).
        self.materials = {}  #: Materials in the model.
        self.analyticalFields = {}  #: AnalyticalFields in the model.
        self.scalarVariables = {}  #: ScalarVariables in the model.
        self.additionalParameters = {}  #: Additional information.
        self.rigidBodies = {}  #: RigidBodies in the model.
        self.elementProperties = []  #: Element properties.
        self.domainSize = dimension  #: Spatial dimension of the model
        self.fieldOutputController = None  #: Set once by the driver; lets in-model entities (e.g. AMR markers) look up a named *fieldOutput by value, not just by declaration.
        #: How the mesh may change during a run; see :class:`~edelweissfe.models.topologypipeline.TopologyPipeline`.
        self.topology = TopologyPipeline(self)

    def __copy__(self):
        """A shallow copy of the model, with a topology pipeline that acts on the copy.

        A shallow copy shares every attribute with the original, which would include the pipeline --
        and a pipeline changes the model it was created for. The copy therefore gets its own pipeline
        bound to it (see :meth:`~edelweissfe.models.topologypipeline.TopologyPipeline.boundTo`), which
        is what a reduced model needs, e.g. EdelweissMeshfree's active sub-model.
        """

        clone = object.__new__(type(self))
        clone.__dict__.update(self.__dict__)
        clone.topology = self.topology.boundTo(clone)
        return clone

    def createNode(self, node):
        """Add a freshly created node to the model.

        Parameters
        ----------
        node
            The node, already carrying a label obtained from :meth:`~edelweissfe.models.topologypipeline.TopologyPipeline.reserveNodeNumbers`.
        """

        if not self.topology.isOpen:
            raise TopologyError(
                "node {:} was created outside a topology change: only model modifiers may create "
                "or delete nodes, inside TopologyPipeline.changes()".format(node.label)
            )
        if node.label in self.nodes:
            raise TopologyError(
                "node label {:} is already taken -- node labels are reserved via "
                "TopologyPipeline.reserveNodeNumbers() and never recycled".format(node.label)
            )

        self.nodes[node.label] = node

    def createElement(self, element):
        """Add a freshly created element to the model.

        Parameters
        ----------
        element
            The element, already carrying a number obtained from :meth:`~edelweissfe.models.topologypipeline.TopologyPipeline.reserveElementNumbers`.
        """

        if not self.topology.isOpen:
            raise TopologyError(
                "element {:} was created outside a topology change: only model modifiers may create "
                "or delete elements, inside TopologyPipeline.changes()".format(element.elNumber)
            )
        if element.elNumber in self.elements:
            raise TopologyError(
                "element number {:} is already taken -- element numbers are reserved via "
                "TopologyPipeline.reserveElementNumbers() and never recycled".format(element.elNumber)
            )

        self.elements[element.elNumber] = element

    def removeElement(self, elNumber: int):
        """Remove an element from the model. Its number is retired, never reissued.

        Parameters
        ----------
        elNumber
            The number of the element to remove.
        """

        if not self.topology.isOpen:
            raise TopologyError(
                "element {:} was deleted outside a topology change: only model modifiers may create "
                "or delete elements, inside TopologyPipeline.changes()".format(elNumber)
            )

        del self.elements[elNumber]

    def _populateNodeFieldVariablesFromElements(
        self,
    ):
        """Creates FieldVariables on Nodes depending on the all
        elements.
        """
        for element in self.elements.values():
            for node, elementNodeFields in zip(element.nodes, element.fields):
                for field in elementNodeFields:
                    if field not in node.fields:
                        node.fields[field] = FieldVariable(node, field)

    def _populateNodeFieldVariablesFromConstraints(
        self,
    ):
        """Creates FieldVariables on Nodes depending on the all
        constraints.
        """

        for constraint in self.constraints.values():
            for node, nodeFields in zip(constraint.nodes, constraint.fieldsOnNodes):
                for field in nodeFields:
                    if field not in node.fields:
                        node.fields[field] = FieldVariable(node, field)

    def _createNodeFieldsFromNodes(self, nodes: list, nodeSets: list) -> dict[str, NodeField]:
        """Bundle nodal FieldVariables together in contiguous NodeFields.

        Parameters
        ----------
        nodes
            The list of Nodes from which the NodeFields should be created.
        nodeSets
            The list of NodeSets, which should be considered in the index map of the NodeFields.

        Returns
        -------
        dict[str,NodeField]
            The dictionary containing the NodeField instances for every active field."""

        domainSize = self.domainSize

        theNodeFields = dict()
        for field in phenomena.keys():
            theNodeField = NodeField(field, getFieldSize(field, domainSize), nodes)

            if theNodeField.nodes:
                theNodeFields[field] = theNodeField

        return theNodeFields

    def _linkFieldVariableObjects(self, nodes):
        """Link NodeFields to individual FieldVariable objects.

        Parameters
        ----------
        nodes
            Nodes to be linked

        Inside a deferring :meth:`~edelweissfe.models.topologypipeline.TopologyPipeline.changes` window this only notes the request; the link is
        made once, for these nodes as they are then, when the window closes.
        """

        if self.topology.isDeferringFieldBookkeeping:
            self.topology.deferFieldVariableLink(nodes)
            return

        for node in nodes:
            for field, fieldVariable in node.fields.items():
                nodeField = self.nodeFields[field]
                idx = nodeField._indicesOfNodesInArray[node]
                fieldVariable.values = self.nodeFields[field]["U"][idx, :]

        return

    def _requestAdditionalScalarVariable(self, name: str):
        """Create a new scalar variables

        Parameters
        ----------
        name
            The name of the variable.

        Returns
        -------
        ScalarVariable
            The instance of the variable.
        """
        if name in self.scalarVariables:
            raise Exception("ScalarVariable with name {:} already exists!".format(name))

        self.scalarVariables[name] = ScalarVariable()
        return self.scalarVariables[name]

    def _createAndAssignScalarVariableForConstraints(self, journal: Journal):
        """Create ScalarVariables for constraints.

        Parameters
        ----------
        journal
            The journal intance.
        """
        # we may have additional scalar degrees of freedom, not associated with any node (e.g, lagrangian multipliers of constraints)

        for constraintName, constraint in self.constraints.items():
            nAdditionalScalarVariables = constraint.getNumberOfAdditionalNeededScalarVariables()
            if nAdditionalScalarVariables > 0:
                journal.message(
                    "Constraint {:} requests {:} additional ScalarVariables".format(
                        constraintName, nAdditionalScalarVariables
                    ),
                    self.identification,
                    2,
                )

                scalarVariables = [
                    self._requestAdditionalScalarVariable("{:}_{:}".format(constraintName, i))
                    for i in range(nAdditionalScalarVariables)
                ]

                constraint.assignAdditionalScalarVariables(scalarVariables)

    def _prepareVariablesAndFields(self, journal):
        """Prepare all variables and fields for a simulation.

        Parameters
        ----------
        journal
            The journal instance.
        """
        journal.message(
            "Activating fields on nodes from Elements and Constraints",
            self.identification,
        )
        self._populateNodeFieldVariablesFromElements()
        self._populateNodeFieldVariablesFromConstraints()

        journal.message("Bundling fields on nodes to NodeFields", self.identification)
        self.nodeFields = self._createNodeFieldsFromNodes(self.nodeSets["all"], self.nodeFields.values())

        journal.message("Assembling ScalarVariables", self.identification)
        self.scalarVariables = dict()
        self._createAndAssignScalarVariableForConstraints(journal)

    def _resizeNodeFieldsForNodes(self, journal: Journal):
        """Resize the existing NodeFields in place for the current ``self.nodeSets["all"]``,
        instead of rebuilding :attr:`nodeFields` from scratch as :meth:`_prepareVariablesAndFields`
        does. Used by mesh mutators (e.g. AMR's ``hadaptivity._materialize``) so that NodeField
        identity -- and hence any :class:`~edelweissfe.fields.nodefield.NodeFieldSubset` or
        reference a consumer cached -- survives a topology change. ScalarVariables (e.g. Lagrange
        multipliers of constraints) are rebuilt, but values are preserved by name for constraints
        that still exist, so their converged state survives the refinement too.

        Everything here is recomputed from the current elements, constraints and ``nodeSets["all"]``
        -- the node order of the resized fields is the model's node creation order, which is why one
        call after several mutations yields the same layout as one call per mutation. Inside a
        deferring :meth:`~edelweissfe.models.topologypipeline.TopologyPipeline.changes` window the call is therefore only noted, and made once when
        the window closes.

        Parameters
        ----------
        journal
            The journal instance.
        """
        if self.topology.isDeferringFieldBookkeeping:
            self.topology.deferNodeFieldResize(journal)
            return

        journal.message(
            "Activating fields on nodes from Elements and Constraints",
            self.identification,
        )
        self._populateNodeFieldVariablesFromElements()
        self._populateNodeFieldVariablesFromConstraints()

        journal.message("Resizing NodeFields", self.identification)
        nodes = self.nodeSets["all"]
        for nodeField in self.nodeFields.values():
            nodeField.resize(nodes)

        # a phenomenon activated for the first time (e.g. by a newly materialized constraint) has
        # no NodeField yet -- create it exactly as _prepareVariablesAndFields would
        for field in phenomena.keys():
            if field not in self.nodeFields:
                newNodeField = NodeField(field, getFieldSize(field, self.domainSize), nodes)
                if newNodeField.nodes:
                    self.nodeFields[field] = newNodeField

        journal.message("Assembling ScalarVariables", self.identification)
        # rebuilding scalarVariables cold-starts every value (ScalarVariable() defaults to 0.0), which
        # would silently discard converged Lagrange-multiplier values on every AMR refinement, even
        # though node fields are warm-started. Snapshot by name and restore the overlapping subset so
        # unchanged constraints keep their converged multiplier and only genuinely new ones cold-start.
        previousScalarVariableValues = {name: v.value for name, v in self.scalarVariables.items()}
        self.scalarVariables = dict()
        self._createAndAssignScalarVariableForConstraints(journal)
        for name, v in self.scalarVariables.items():
            if name in previousScalarVariableValues:
                v.value = previousScalarVariableValues[name]

    def _prepareElements(self, journal: Journal):
        """Prepare elements for a simulation.
        In detail, sections are assigned.


        Parameters
        ----------
        journal
            The journal instance.
        """
        for section in self.sections.values():
            section.assignSectionPropertiesToModel(self)

        for elementProperty in self.elementProperties:
            elementProperty.assignElementPropertiesToModel(self)

        # check if all elements are assigned a material
        materialAssigned = np.fromiter(map(attrgetter("hasMaterial"), self.elements.values()), dtype=bool)
        if not materialAssigned.all():
            elementIds = np.array([str(elId) for elId in self.elements.keys()])[np.logical_not(materialAssigned)]
            raise Exception(f"No material was assigned to element(s) with id(s) {', '.join(elementIds)}.")

    def prepareYourself(self, journal: Journal):
        """Prepare the model for a simulation.
        Creates the variables, bundles the fields,
        and initializes elements.


        Parameters
        ----------
        journal
            The journal instance.
        """

        self.topology.adoptSetupElementNumbers()
        self.topology.ensureSurfaceFacetModifier(journal)
        self.topology.checkModelModifierDomains()
        self._prepareVariablesAndFields(journal)
        self._prepareElements(journal)

    def advanceToTime(self, time: float):
        """Accept the current state of the model and sub instances, and
        set the new time.

        Parameters
        ----------
        time
            The new time.
        """

        self.time = time

        self._acceptElementStates()

        # Left serial deliberately. Measured on the 337 471-dof explicit anchor pry-out model, where
        # the element loop below costs 19.09 ms per call: these two together cost 0.015 ms, i.e. less
        # than a tenth of a percent of it. There is nothing here to parallelize.
        for constraint in self.constraints.values():
            constraint.acceptLastState()

        for mpc in self.multiPointConstraints.values():
            mpc.acceptLastState()

    def _acceptElementStates(self):
        """Let every element accept its computed state, across the available threads.

        An element's ``acceptLastState`` touches only that element's own state buffers -- for a
        Marmot element it is the single ``self._stateVars[:] = self._stateVarsTemp`` copy -- so the
        elements are independent of one another and this loop parallelizes without any coordination.

        It is worth parallelizing because of how the explicit solver uses it, not because of how the
        implicit ones do. An implicit analysis calls this once per *converged* increment, amortised
        over a Newton loop and a linear solve, where 19 ms disappears into the noise. Explicit
        dynamics calls it on every one of millions of increments: on the anchor pry-out model it was
        15.6 % of the whole step, the single largest cost outside the element kernels themselves, and
        every millisecond of it was serial Python holding 31 of 32 threads idle.
        """

        elements = self.elements
        numThreads = getNumberOfThreads() if isFreeThreadingSupported() else 1

        if numThreads == 1 or len(elements) < numThreads:
            for element in elements.values():
                element.acceptLastState()
            return

        def acceptChunk(chunk):
            for element in chunk:
                element.acceptLastState()

        # The same chunk granularity the parallel element computation uses: four chunks per thread,
        # enough to even out the spread between cheap facet elements and expensive solid ones
        # without paying dispatch overhead per element.
        chunkSize = max(1, len(elements) // (numThreads * 4))
        chunks = chunked_iterable(elements.values(), chunkSize)

        # list(), not a bare call: Executor.map is lazy, and leaving the iterator unconsumed would
        # return here with workers still writing element state.
        list(getThreadPool(numThreads).map(acceptChunk, chunks))

    def writeRestart(self, restartFile: h5py.File):
        """Write the current (converged) state of the model to a restart checkpoint.

        Does not serialize model topology (nodes/elements/sets/sections/materials) -- only state
        that a rebuild from the original ``.inp`` file cannot reproduce: node field values, element
        quadrature-point history, scalar variables (e.g. Lagrange multipliers), and stateful
        constraints' internal history (e.g. frictional contact).

        Parameters
        ----------
        restartFile
            An open, writable :class:`h5py.File` (or group) to write the checkpoint into.
        """

        f = restartFile

        f.attrs["time"] = self.time

        nodeFieldsGroup = f.create_group("nodeFields")
        for nf in self.nodeFields.values():
            nodeFieldGroup = nodeFieldsGroup.create_group(nf.name)
            for entryName, entryValues in nf._values.items():
                nodeFieldGroup.create_dataset(entryName, data=entryValues)

        scalarVariablesGroup = f.create_group("scalarVariables")
        for name, scalarVariable in self.scalarVariables.items():
            scalarVariablesGroup.attrs[name] = scalarVariable.value

        writeRestartDataOf(f.create_group("modelModifiers"), self.modelModifiers)

        elementsGroup = f.create_group("elements")
        for elNumber, element in self.elements.items():
            elementsGroup.create_dataset(str(elNumber), data=element.getStateVars())

        writeRestartDataOf(f.create_group("constraints"), self.constraints)

        # The ordered record of every applied model-modifier decision (e.g. AMR refinements). A
        # resumed run replays these through the modifiers' own apply() -- see readRestart and
        # TopologyPipeline.replayHistory -- rather than serializing the resulting topology directly, so there
        # is exactly one code path that mutates topology, live or replayed.
        historyGroup = f.create_group("topologyHistory")
        historyGroup.attrs["setupFingerprint"] = self.topology.setupFingerprint()
        historyGroup.attrs["count"] = len(self.topology.history)
        for index, record in enumerate(self.topology.history):
            recordGroup = historyGroup.create_group("{:06d}".format(index))
            recordGroup.attrs["modifier"] = record.modifier
            recordGroup.attrs["roundNumber"] = record.roundNumber
            recordGroup.attrs["time"] = record.time
            recordGroup.attrs["fingerprint"] = record.fingerprint
            for entryName, entryValues in record.plan.items():
                recordGroup.create_dataset(entryName, data=entryValues)

    def readRestart(self, restartFile: h5py.File, journal: Journal = None):
        """Read the state of the model from a restart checkpoint written by :meth:`writeRestart`.

        The model must already have been rebuilt from the original ``.inp`` file (same topology)
        and :meth:`prepareYourself` called, before this is called.

        Parameters
        ----------
        restartFile
            An open, readable :class:`h5py.File` (or group) to read the checkpoint from.
        journal
            Optional Journal for the progress messages of the topology replay.
        """

        f = restartFile

        self.time = f.attrs["time"]

        # Must run before every restore below: replaying the topology history can materialize
        # elements/nodes a plain rebuild from the .inp file cannot reproduce (e.g. AMR-refined
        # children) -- the node-field and element-statevar restores that follow address elements by
        # label and would silently miss anything not already in self.elements/self.nodeFields yet.
        historyGroup = f["topologyHistory"]
        if self.topology.setupFingerprint() != historyGroup.attrs["setupFingerprint"]:
            raise RestartError(
                "the model rebuilt from the input file is not the one the checkpoint was written from: "
                "its mesh (nodes, elements, coordinates, numbering) differs"
            )
        records = []
        for index in range(int(historyGroup.attrs["count"])):
            recordGroup = historyGroup["{:06d}".format(index)]
            records.append(
                TopologyRecord(
                    modifier=str(recordGroup.attrs["modifier"]),
                    roundNumber=int(recordGroup.attrs["roundNumber"]),
                    time=float(recordGroup.attrs["time"]),
                    plan={entryName: values[:] for entryName, values in recordGroup.items()},
                    fingerprint=str(recordGroup.attrs["fingerprint"]),
                )
            )
        self.topology.replayHistory(records, journal)
        readRestartDataInto(f["modelModifiers"], self.modelModifiers)
        # Bring the mesh-dependent consumers (ties, contact surfaces) up to date with the replayed
        # mesh here, as part of restoring the model: the uninterrupted run's consumers had caught up
        # with every change before the checkpoint was written, so a resumed run must start the same.
        self.topology.refreshMeshDependents()

        for nf in self.nodeFields.values():
            storedField = f["nodeFields"].get(nf.name)
            if storedField is None:
                raise RestartError("the checkpoint holds no node field {:}".format(nf.name))

            # Iterate the checkpoint's entries, not the model's -- entries created only later by a
            # solver (e.g. the explicit solver's 'V') would otherwise never be restored.
            for entryName, storedValues in storedField.items():
                if entryName not in nf:
                    nf.createFieldValueEntry(entryName)
                storedValues.read_direct(nf[entryName])

        for name, scalarVariable in self.scalarVariables.items():
            scalarVariable.value = f["scalarVariables"].attrs[name]

        # One uniform loop, by element number, with no skip set and nothing swallowed -- sound only
        # because the replay above reproduces the original numbering exactly, verified against the
        # recorded fingerprint. A missing element here means the replayed model does not match the
        # one checkpointed, which must be reported, not silently skipped.
        for elementKey, stateVars in f["elements"].items():
            elNumber = int(elementKey)
            element = self.elements.get(elNumber)
            if element is None:
                raise RestartError(
                    "the checkpoint holds state for element {:}, which does not exist after the "
                    "topology replay -- the replayed model does not match the one checkpointed".format(elNumber)
                )
            element.setStateVars(stateVars[:])

        # A rigid body moves its surface nodes into the current configuration, a pure function of
        # its reference point's restored DOFs: put them where the uninterrupted run had them.
        for rigidBody in self.rigidBodies.values():
            rigidBody.updateKinematics()

        readRestartDataInto(f["constraints"], self.constraints)


def printPrettyModelSummary(model: FEModel, journal: Journal):
    identification = "PrettyModelSummary"

    def wrapList(theList):
        for line in textwrap.wrap("[" + ", ".join(theList) + "]"):
            journal.message(
                "  {:<20} ".format(line),
                identification,
                0,
            )

    journal.message(
        "Finite element model with spatial dimension {:} has".format(model.domainSize),
        identification,
        0,
    )
    journal.message(
        " {:<20}{:<15} ".format("nodes:", len(model.nodes)),
        identification,
        0,
    )
    journal.message(
        " {:<20}{:<15} ".format("node sets:", len(model.nodeSets)),
        identification,
        0,
    )
    wrapList(model.nodeSets.keys())
    journal.message(
        " {:<20}{:<15} ".format("node fields:", len(model.nodeFields)),
        identification,
        0,
    )
    wrapList(["{:} ({:})".format(k, len(v.nodes)) for k, v in model.nodeFields.items()])
    journal.message(
        " {:<20}{:<15}".format("elements: ", len(model.elements)),
        identification,
        0,
    )
    journal.message(
        " {:<20}{:<15}".format("element sets: ", len(model.elementSets)),
        identification,
        0,
    )
    wrapList(model.elementSets.keys())
    if model.constraints:
        journal.message(
            " {:<20}{:<15}".format("constraints: ", len(model.constraints)),
            identification,
            0,
        )
    if model.multiPointConstraints:
        journal.message(
            " {:<20}{:<15}".format("multi-point constraints: ", len(model.multiPointConstraints)),
            identification,
            0,
        )
    if model.scalarVariables:
        journal.message(
            " {:<20}{:<15}".format("scalar variables: ", len(model.scalarVariables)),
            identification,
            0,
        )
    journal.message(
        " {:<20}{:<15}".format("materials: ", len(model.materials)),
        identification,
        0,
    )
    wrapList(model.materials.keys())

    if model.analyticalFields:
        journal.message(
            " {:<20}{:<15}".format("analytical fields: ", len(model.analyticalFields)),
            identification,
            0,
        )
        wrapList(model.analyticalFields.keys())
