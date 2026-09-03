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

import hashlib
import textwrap
from contextlib import contextmanager
from operator import attrgetter

import h5py
import numpy as np

from edelweissfe.config.phenomena import getFieldSize, phenomena
from edelweissfe.fields.nodefield import NodeField
from edelweissfe.journal.journal import Journal
from edelweissfe.models.modelchange import ModelChange, TopologyRecord, coalesce
from edelweissfe.utils.exceptions import RestartError, TopologyError
from edelweissfe.utils.performancetiming import timeit
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
        self.meshDependents = []  #: Consumers that cache mesh-derived state; see :meth:`refreshMeshDependents`.
        self.topologyVersion = 0  #: Bumped on every structural mutation; drives pull-based reconcile.
        self._changeLog = []  #: Recorded :class:`ModelChange` per mutation, newest last.
        self.contactFacetRecipes = {}  #: facet elSet name -> (surfaceName, prefix, triangulation).
        self.materials = {}  #: Materials in the model.
        self.analyticalFields = {}  #: AnalyticalFields in the model.
        self.scalarVariables = {}  #: ScalarVariables in the model.
        self.additionalParameters = {}  #: Additional information.
        self.rigidBodies = {}  #: RigidBodies in the model.
        self.elementProperties = []  #: Element properties.
        self.domainSize = dimension  #: Spatial dimension of the model
        self.fieldOutputController = None  #: Set once by the driver; lets in-model entities (e.g. AMR markers) look up a named *fieldOutput by value, not just by declaration.
        #: High-water mark of the element number allocator; see :meth:`reserveElementNumbers`.
        self._nextElementNumber = 1
        #: High-water mark of the node number allocator; see :meth:`reserveNodeNumbers`.
        self._nextNodeNumber = 1
        self._topologyOpen = False  #: True only inside :meth:`topologyChanges`; see there.
        #: Guard against a model modifier that keeps planning in response to its own output.
        self.maxTopologyRounds = 16
        #: Ordered record of every applied model-modifier decision; see :meth:`updateTopology`. This
        #: IS the restart history -- a resumed run replays it rather than re-deciding.
        self.topologyHistory = []
        #: Compare each replayed round's fingerprint against the recorded one. On by default: it is
        #: the difference between "the resumed run diverged" and "it diverged HERE".
        self.verifyTopologyFingerprints = True

    @contextmanager
    def topologyChanges(self):
        """The only scope in which elements may be created or deleted.

        Opened once around model setup, and once per increment around the model modifiers. Outside
        it, :meth:`createElement` and :meth:`removeElement` raise -- which is what makes "only model
        modifiers mutate the topology" an enforced property rather than a convention, and what lets
        :meth:`reserveElementNumbers` guarantee that element numbering is a pure function of the
        ordered creation sequence.

        Nesting is permitted and is a no-op for the inner scope: setup-time helpers may open a
        window without knowing whether their caller already did.
        """

        wasOpen = self._topologyOpen
        self._topologyOpen = True
        try:
            yield
        finally:
            self._topologyOpen = wasOpen

    def reserveElementNumbers(self, count: int = 1) -> range:
        """Reserve ``count`` fresh element numbers.

        The allocator is **monotonic**: numbers are never recycled, and are never derived from
        ``max(self.elements)``. Both properties matter beyond tidiness.

        Deriving the next number from ``max(self.elements)`` makes numbering a function of the
        deletion history as well as the creation history -- a contact facet set that is deleted and
        rebuilt (the common case between two refinements) hands the freed numbers straight back out
        -- so a restart replay would have to reproduce creations, deletions *and* their interleaving
        to renumber identically. With one monotonic counter it only has to reproduce the ordered
        creation sequence, which is exactly what the recorded topology history holds.

        Never recycling additionally means a number refers to one element for the model's entire
        lifetime, so :meth:`~edelweissfe.models.modelchange.ModelChange.mergedWith`'s documented
        no-reuse assumption holds, and a reference cached by number cannot silently alias a
        different element.

        Parameters
        ----------
        count
            How many consecutive numbers to reserve.

        Returns
        -------
        range
            The reserved numbers, in ascending order.
        """

        if not self._topologyOpen:
            raise TopologyError(
                "element numbers may only be reserved during a topology change -- see FEModel.topologyChanges()"
            )
        if count < 1:
            raise ValueError("cannot reserve {:} element numbers".format(count))

        first = self._nextElementNumber
        self._nextElementNumber += count
        return range(first, self._nextElementNumber)

    def adoptSetupElementNumbers(self):
        """Raise the allocator above every element number setup has already handed out.

        Called once, at the end of model setup. The base mesh (input file and every mesh generator)
        numbers its elements as a pure function of the input file, is re-run identically by a
        resumed run before the checkpoint is read, and is never renumbered afterwards -- so those
        numbers need no allocator. This just makes sure nothing minted later can collide with them.
        """

        self._nextElementNumber = max(self._nextElementNumber, max(self.elements.keys(), default=0) + 1)

    def reserveNodeNumbers(self, count: int = 1) -> range:
        """Reserve ``count`` fresh node labels. The node-side counterpart of
        :meth:`reserveElementNumbers`, monotonic for the same reasons.

        The pattern this replaces is ``len(model.nodes) + 1``, which is a positional guess, not an
        allocator: once anything is deleted the dict has gaps, and the "next" label lands on a live
        node and silently overwrites it.

        Parameters
        ----------
        count
            How many consecutive labels to reserve.

        Returns
        -------
        range
            The reserved labels, in ascending order.
        """

        if not self._topologyOpen:
            raise TopologyError(
                "node labels may only be reserved during a topology change -- see FEModel.topologyChanges()"
            )
        if count < 1:
            raise ValueError("cannot reserve {:} node labels".format(count))

        first = self._nextNodeNumber
        self._nextNodeNumber += count
        return range(first, self._nextNodeNumber)

    def adoptSetupNodeNumbers(self):
        """Raise the node allocator above every label setup has already handed out; the node-side
        counterpart of :meth:`adoptSetupElementNumbers`, called alongside it.
        """

        self._nextNodeNumber = max(self._nextNodeNumber, max(self.nodes.keys(), default=0) + 1)

    def createNode(self, node):
        """Add a freshly created node to the model.

        Parameters
        ----------
        node
            The node, already carrying a label obtained from :meth:`reserveNodeNumbers`.
        """

        if not self._topologyOpen:
            raise TopologyError(
                "node {:} was created outside a topology change: only model modifiers may create "
                "or delete nodes, inside FEModel.topologyChanges()".format(node.label)
            )
        if node.label in self.nodes:
            raise TopologyError(
                "node label {:} is already taken -- node labels are reserved via "
                "FEModel.reserveNodeNumbers() and never recycled".format(node.label)
            )

        self.nodes[node.label] = node

    def createElement(self, element):
        """Add a freshly created element to the model.

        Parameters
        ----------
        element
            The element, already carrying a number obtained from :meth:`reserveElementNumbers`.
        """

        if not self._topologyOpen:
            raise TopologyError(
                "element {:} was created outside a topology change: only model modifiers may create "
                "or delete elements, inside FEModel.topologyChanges()".format(element.elNumber)
            )
        if element.elNumber in self.elements:
            raise TopologyError(
                "element number {:} is already taken -- element numbers are reserved via "
                "FEModel.reserveElementNumbers() and never recycled".format(element.elNumber)
            )

        self.elements[element.elNumber] = element

    def removeElement(self, elNumber: int):
        """Remove an element from the model. Its number is retired, never reissued.

        Parameters
        ----------
        elNumber
            The number of the element to remove.
        """

        if not self._topologyOpen:
            raise TopologyError(
                "element {:} was deleted outside a topology change: only model modifiers may create "
                "or delete elements, inside FEModel.topologyChanges()".format(elNumber)
            )

        del self.elements[elNumber]

    def ensureSurfaceFacetModifier(self, journal: Journal):
        """Create the implicit facet-regeneration modifier, if any facet recipe was declared.

        Retiling a contact/tie surface creates and deletes elements, so it is a topology change and
        belongs in the topology-update phase -- not in a consumer's refresh, which is where it used
        to live and which is what made every tie a mutating consumer. Users never declare this
        modifier: they already declared the ``*surface`` recipe it acts on.

        Ordered **last**, so that within a round it reacts to whatever the primary modifiers (a
        refinement, a deposition) just did.
        """

        if not self.contactFacetRecipes or "surfaceFacets" in self.modelModifiers:
            return

        from edelweissfe.modelmodifiers.surfacefacets.surfacefacets import (
            ModelModifier as SurfaceFacetsModifier,
        )

        self.modelModifiers["surfaceFacets"] = SurfaceFacetsModifier("surfaceFacets", self, journal)

    def checkModelModifierDomains(self):
        """Refuse a model in which two modifiers claim the same element.

        Run once, at the end of setup. Each modifier declares what it owns via
        :meth:`~edelweissfe.modelmodifiers.base.modelmodifierbase.ModelModifierBase.declaredDomain`;
        an overlap means both will mutate the same element and each will end up holding stale
        references to the other's work. Failing here costs a clear message at startup; failing later
        costs a corrupted element set or a node that is both Dirichlet-prescribed and an MPC slave,
        discovered mid-solve.
        """

        claimed = list(self.modelModifiers.items())
        for index, (name, modifier) in enumerate(claimed):
            domain = modifier.declaredDomain(self)
            if not domain:
                continue
            for otherName, otherModifier in claimed[index + 1 :]:
                overlap = domain & otherModifier.declaredDomain(self)
                if overlap:
                    raise TopologyError(
                        "model modifiers {!r} and {!r} both claim {:} of the same element(s) "
                        "(e.g. {:}). Two modifiers cannot own one element: each mutates it directly, "
                        "so the other is left holding a stale reference. Restrict their element sets "
                        "so they do not overlap, or combine them into a single modifier.".format(
                            name, otherName, len(overlap), sorted(overlap)[0]
                        )
                    )

    def updateTopology(self, step=None, timeStep: float = None) -> bool:
        """Run every model modifier to a fixed point, inside one topology window.

        Modifiers depend on each other -- refinement invalidates a tied surface's facets, a
        deposition modifier creates elements refinement may then want to split, and a 2:1 balance
        may need to refine what another modifier just activated. Rather than asking the user to
        declare a dependency order (which cannot express mutual dependence anyway), each **round**
        offers every modifier the net change since that modifier last planned. A round in which
        nobody plans anything is the fixed point.

        Determinism comes from the round structure, not from luck: within a round, modifiers run in
        ``self.modelModifiers`` order, which is input-file order.

        Returns
        -------
        bool
            True if the topology changed, so the solver rebuilds its equation system.

        Raises
        ------
        TopologyError
            If the rounds do not settle within :attr:`maxTopologyRounds`, which means some modifier
            keeps planning in response to its own output. The message names the offenders.
        """

        changed = False
        with self.topologyChanges():
            # Seeded with the version at the START of this update, not None: a modifier must see
            # what earlier modifiers did in the SAME round. Seeding with None meant a purely
            # reactive modifier (one that only acts on someone else's change) was handed None in
            # round 1 -- after the change it needed to see had already happened -- and then had its
            # version stamped, so round 2 showed nothing new either. It never reacted at all.
            lastPlannedVersion = {name: self.topologyVersion for name in self.modelModifiers}
            # Which modifier touched which element, over the WHOLE update rather than one round.
            # Two modifiers mutating one element is a conflict even when their declared domains are
            # disjoint -- e.g. one deleting what the other just created -- and the result depends on
            # their order, which is exactly the kind of thing that must not decide a simulation
            # quietly. Spanning all rounds matters because the rounds exist precisely so that a
            # modifier can react to another's output, which is when the collision is most likely.
            touchedBy = {}  # element number -> (modifier name, round it was touched in)
            roundNumber = 0
            while True:
                roundNumber += 1
                plannedThisRound = []
                for name, modifier in self.modelModifiers.items():
                    change = self.changesSince(lastPlannedVersion[name])
                    lastPlannedVersion[name] = self.topologyVersion
                    plan = modifier.plan(self, change, step, timeStep)
                    if plan is None:
                        continue
                    modelChange = modifier.apply(self, plan)
                    # A modifier may plan and then find nothing left to do. Recording that would
                    # rebuild the equation system for nothing, pay a topology fingerprint for
                    # nothing, and let the no-op modifier burn through maxTopologyRounds and be
                    # named as the one that would not settle.
                    if modelChange is not None and modelChange.isEmpty:
                        continue
                    if modelChange is not None:
                        for elNumber in modelChange.addedElements | modelChange.removedElements:
                            previousName, previousRound = touchedBy.setdefault(elNumber, (name, roundNumber))
                            if previousName != name:
                                raise TopologyError(
                                    "model modifiers {!r} (round {:}) and {!r} (round {:}) both changed "
                                    "element {:} in one topology update. Whichever ran second silently "
                                    "won; make their domains disjoint, or have one react to the other's "
                                    "change without mutating the same element.".format(
                                        previousName, previousRound, name, roundNumber, elNumber
                                    )
                                )
                    self.recordTopologyChange(roundNumber, name, modifier, plan, modelChange)
                    plannedThisRound.append(name)
                    changed = True
                if not plannedThisRound:
                    break
                if roundNumber >= self.maxTopologyRounds:
                    raise TopologyError(
                        "model modifiers did not settle within {:} rounds; still planning in the "
                        "last round: {:}. A modifier must return None from plan() once the change "
                        "since its own last plan no longer touches its domain.".format(
                            self.maxTopologyRounds, ", ".join(plannedThisRound)
                        )
                    )
        return changed

    def topologyFingerprint(self) -> str:
        """A short digest of the model's topology *and its numbering*, for verifying that a restart
        replay reproduced the original run.

        Covers exactly what a replay must get right and nothing else: every element's number, type
        and connectivity (in order -- a rotated connectivity is a real difference), and every node's
        label and coordinates. Deliberately excludes solution state, so a mismatch means the *mesh*
        diverged, not that the solver took a different path.

        Recorded per round in the topology history, this turns "the resumed run diverged somewhere"
        into "increment 471, round 2, modifier amr" -- a divergence you can bisect rather than hunt.

        Not cheap: it walks the whole mesh, measured at 0.188 s on 64k elements / 69k nodes, and
        :meth:`recordTopologyChange` pays it unconditionally for every *applied* modifier decision
        (``verifyTopologyFingerprints`` gates only the replay-time comparison, not this).

        Uses blake2b rather than :func:`hash`, whose string hashing is randomised per process and
        would make the digest differ between two runs of the *same* code.

        Returns
        -------
        str
            A 32-character hex digest.
        """

        digest = hashlib.blake2b(digest_size=16)
        for elNumber in sorted(self.elements):
            element = self.elements[elNumber]
            digest.update(b"E|%d|%s|" % (elNumber, element.elType.encode()))
            digest.update(b",".join(b"%d" % node.label for node in element.nodes))
        for label in sorted(self.nodes):
            digest.update(b"N|%d|" % label)
            digest.update(np.asarray(self.nodes[label].coordinates, dtype=float).tobytes())
        return digest.hexdigest()

    def recordTopologyChange(
        self, roundNumber: int, name: str, modifier, plan, modelChange, time: float = None
    ) -> TopologyRecord:
        """Register everything an applied decision produced: the replay record and the changeset.

        Two things are recorded here, deliberately in one place:

        * an entry in :attr:`topologyHistory`, holding the plan in the modifier's own serializable
          form plus the resulting topology fingerprint -- which is what lets a resumed run be
          checked round by round instead of only at the end;
        * the ``modelChange`` itself, via :meth:`notifyModelChanged`, so :meth:`changesSince` and
          :meth:`refreshMeshDependents` can see it.

        A model modifier therefore reports what it did in exactly one way: by returning a
        :class:`~edelweissfe.models.modelchange.ModelChange` from
        :meth:`~edelweissfe.modelmodifiers.base.modelmodifierbase.ModelModifierBase.apply`. It must
        not call :meth:`notifyModelChanged` itself -- there is no second channel to keep in sync,
        and so no way to update one and forget the other.

        Both callers of ``apply`` (the live :meth:`updateTopology` loop and
        :meth:`replayTopologyHistory`) route through here, so a replayed run records the same
        changesets in the same order as the run it replays.

        Cost is one :meth:`topologyFingerprint` per *applied* decision -- not per iteration, but not
        free either: 0.188 s measured on 64k elements / 69k nodes.

        Parameters
        ----------
        time
            Model time to stamp the record with. Defaults to the current :attr:`time`; a replay
            passes the recorded time instead, so a resumed run's history carries the times the
            decisions were originally made rather than the time it was resumed at.
        """

        if modelChange is not None:
            self.notifyModelChanged(modelChange.kind, modelChange)

        record = TopologyRecord(
            modifier=name,
            roundNumber=roundNumber,
            time=float(self.time if time is None else time),
            plan=modifier.encodePlan(plan),
            fingerprint=self.topologyFingerprint(),
            nElementsAdded=len(modelChange.addedElements) if modelChange is not None else 0,
            nElementsRemoved=len(modelChange.removedElements) if modelChange is not None else 0,
            nNodesAdded=len(modelChange.addedNodes) if modelChange is not None else 0,
        )
        self.topologyHistory.append(record)
        return record

    def replayTopologyHistory(self, records, journal: Journal = None):
        """Reconstruct the topology by re-applying recorded decisions, in order.

        This is the whole point of the plan/apply split: the modifier's :meth:`apply` runs here
        exactly as it did live, fed a decoded plan instead of a freshly evaluated one. There is no
        replay-specific code path to drift from the live one -- which is what the previous design
        had, and why a resumed run silently renumbered its elements.

        Parameters
        ----------
        records
            The :class:`~edelweissfe.models.modelchange.TopologyRecord` sequence to replay.
        journal
            Optional Journal for progress messages.

        Raises
        ------
        TopologyError
            If a replayed round's fingerprint differs from the recorded one (when
            :attr:`verifyTopologyFingerprints`), naming the exact record -- so a divergence is
            located rather than merely detected.
        """

        with self.topologyChanges():
            for index, record in enumerate(records):
                modifier = self.modelModifiers.get(record.modifier)
                if modifier is None:
                    raise TopologyError(
                        "the checkpoint records a decision by model modifier {!r}, which this model "
                        "does not define -- the input file must declare the same modifiers as the run "
                        "being resumed".format(record.modifier)
                    )
                plan = modifier.decodePlan(record.plan)
                modelChange = modifier.apply(self, plan)
                replayed = self.recordTopologyChange(
                    record.roundNumber, record.modifier, modifier, plan, modelChange, time=record.time
                )
                if self.verifyTopologyFingerprints and record.fingerprint:
                    if replayed.fingerprint != record.fingerprint:
                        raise TopologyError(
                            "restart replay diverged at record {:} of {:}: modifier {!r}, round {:}, "
                            "time {:}. The replayed topology does not match the recorded one, so this "
                            "modifier's apply() is not a pure function of (model, plan).".format(
                                index, len(records), record.modifier, record.roundNumber, record.time
                            )
                        )
        for name, modifier in self.modelModifiers.items():
            modifier.restoreDecisionState([r for r in records if r.modifier == name])
        if journal is not None:
            journal.message(
                "Replayed {:} recorded topology change(s); {:} elements, {:} nodes".format(
                    len(records), len(self.elements), len(self.nodes)
                ),
                self.identification,
                0,
            )

    def registerMeshDependent(self, consumer):
        """Register a :class:`~edelweissfe.models.meshdependent.MeshDependent` to be refreshed after
        every topology update.

        Registration is the freshness guarantee: a consumer that is not in this list is never told
        the mesh changed. That matters most for the consumers the solver does not otherwise tick --
        multi-point constraints live in ``model.multiPointConstraints``, which no per-increment sweep
        iterates, so a tie could only ever learn about refinement this way.
        """

        if not any(consumer is registered for registered in self.meshDependents):
            self.meshDependents.append(consumer)

    @timeit("refresh mesh dependents")
    def refreshMeshDependents(self) -> bool:
        """Let every registered mesh-dependent consumer catch up, once, after the topology update.

        Phase 2 of the increment (see :meth:`updateTopology` for phase 1). Consumers are pure
        readers here -- the topology window is closed -- so **their order does not matter** and no
        fixed-point iteration is needed: none of them can invalidate another's work.

        Each consumer sees the *net* change across every round of the topology update, which is why
        this is pull and not push: a push fires per mutation, i.e. at moments that are by
        construction mid-pipeline, handing a consumer a state that no longer exists by the time the
        solve begins.

        Returns
        -------
        bool
            True if any consumer reported that its DOF footprint changed.
        """

        # materialise the list: any() would short-circuit and leave later consumers unrefreshed
        return any([consumer.refreshIfMeshChanged(self) for consumer in self.meshDependents])

    def notifyModelChanged(self, changeType, change: ModelChange = None):
        """Record a model mutation: bump :attr:`topologyVersion` and append the changeset, so that
        every :class:`~edelweissfe.models.meshdependent.MeshDependent` can catch up from
        :meth:`changesSince` at the end of the topology update.

        **Model modifiers must not call this.** :meth:`recordTopologyChange` calls it for them, from
        the :class:`~edelweissfe.models.modelchange.ModelChange` their ``apply`` returns; see there.
        It remains available for code that mutates the model outside the modifier pipeline.

        Recording only -- there is no synchronous callback. Consumers are refreshed once, by
        :meth:`refreshMeshDependents`, after the modifiers have settled; see there for why.

        Parameters
        ----------
        changeType
            The :class:`~edelweissfe.models.modelchangeobserver.ModelChangeType` of the mutation.
        change
            The structured :class:`ModelChange` describing what changed. If omitted, an empty one
            (bare ``changeType`` marker only, e.g. for a modifier that hasn't adopted the changeset
            yet) is recorded instead.
        """
        self.topologyVersion += 1
        if change is None:
            change = ModelChange(kind=changeType)
        change.version = self.topologyVersion
        self._changeLog.append(change)

    def changesSince(self, version: int) -> ModelChange:
        """The :class:`ModelChange` coalesced across every mutation recorded after ``version``, or
        ``None`` if the model hasn't changed since. A pull-based consumer compares its own
        last-seen version against :attr:`topologyVersion` and, on a mismatch, reconciles from this,
        then adopts the new :attr:`topologyVersion` as its own last-seen version."""
        if version >= self.topologyVersion:
            return None
        return coalesce([c for c in self._changeLog if c.version > version])

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
        """

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

        self.adoptSetupElementNumbers()
        self.ensureSurfaceFacetModifier(journal)
        self.checkModelModifierDomains()
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

        for el in self.elements.values():
            el.acceptLastState()

        for constraint in self.constraints.values():
            constraint.acceptLastState()

        for mpc in self.multiPointConstraints.values():
            mpc.acceptLastState()

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

        elementsGroup = f.create_group("elements")
        for elNumber, element in self.elements.items():
            try:
                stateVars = element.getStateVars()
            except NotImplementedError:
                continue
            elementsGroup.create_dataset(str(elNumber), data=stateVars)

        constraintsGroup = f.create_group("constraints")
        for name, constraint in self.constraints.items():
            restartData = constraint.getRestartData()
            if restartData is None:
                continue
            constraintGroup = constraintsGroup.create_group(name)
            for entryName, entryValues in restartData.items():
                constraintGroup.create_dataset(entryName, data=entryValues)

        # The ordered record of every applied model-modifier decision (e.g. AMR refinements). A
        # resumed run replays these through the modifiers' own apply() -- see readRestart and
        # replayTopologyHistory -- rather than serializing the resulting topology directly, so there
        # is exactly one code path that mutates topology, live or replayed.
        historyGroup = f.create_group("topologyHistory")
        historyGroup.attrs["count"] = len(self.topologyHistory)
        for index, record in enumerate(self.topologyHistory):
            recordGroup = historyGroup.create_group("{:06d}".format(index))
            recordGroup.attrs["modifier"] = record.modifier
            recordGroup.attrs["roundNumber"] = record.roundNumber
            recordGroup.attrs["time"] = record.time
            recordGroup.attrs["fingerprint"] = record.fingerprint
            for entryName, entryValues in record.plan.items():
                recordGroup.create_dataset(entryName, data=entryValues)

    def readRestart(self, restartFile: h5py.File):
        """Read the state of the model from a restart checkpoint written by :meth:`writeRestart`.

        The model must already have been rebuilt from the original ``.inp`` file (same topology)
        and :meth:`prepareYourself` called, before this is called.

        Parameters
        ----------
        restartFile
            An open, readable :class:`h5py.File` (or group) to read the checkpoint from.
        """

        f = restartFile

        self.time = f.attrs["time"]

        # Must run before every restore below: replaying the topology history can materialize
        # elements/nodes a plain rebuild from the .inp file cannot reproduce (e.g. AMR-refined
        # children) -- the node-field and element-statevar restores that follow address elements by
        # label and would silently miss anything not already in self.elements/self.nodeFields yet.
        records = []
        historyGroup = f["topologyHistory"]
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
        self.replayTopologyHistory(records)

        for nf in self.nodeFields.values():
            storedField = f["nodeFields"].get(nf.name)
            if storedField is None:
                continue

            # Iterate what the CHECKPOINT holds, not what this model has created so far. The two
            # differ: writeRestart stores every entry a node field has at write time, while at read
            # time the driver has created only 'U' and 'P'. The explicit solver's velocity entry 'V'
            # is created later, the first time the solver writes it -- so keying this loop on the
            # model silently dropped 'V' from every resume, leaving the run to integrate from zero
            # velocity with no error anywhere. Entries the model has not created yet are created
            # here, so they are populated before any solver asks for them.
            for entryName, storedValues in storedField.items():
                if entryName not in nf:
                    nf.createFieldValueEntry(entryName)
                storedValues.read_direct(nf[entryName])

        for name, scalarVariable in self.scalarVariables.items():
            scalarVariable.value = f["scalarVariables"].attrs[name]

        # One uniform loop, by element number, with no skip set and nothing swallowed -- sound only
        # because the replay above reproduces the original numbering exactly, verified round by
        # round via each record's own fingerprint. A missing element here means the replayed model
        # does not match the one checkpointed, which must be reported, not silently skipped.
        for elementKey, stateVars in f["elements"].items():
            elNumber = int(elementKey)
            element = self.elements.get(elNumber)
            if element is None:
                raise RestartError(
                    "the checkpoint holds state for element {:}, which does not exist after the "
                    "topology replay -- the replayed model does not match the one checkpointed".format(elNumber)
                )
            element.setStateVars(stateVars[:])

        for name, constraint in self.constraints.items():
            if name not in f["constraints"]:
                continue
            restartData = {entryName: values[:] for entryName, values in f["constraints"][name].items()}
            constraint.setRestartData(restartData)


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
