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
"""The topology pipeline: how, and only how, the mesh of a model may change during a run.

See :class:`TopologyPipeline`.
"""

import copy
import hashlib
from contextlib import contextmanager

import numpy as np

from edelweissfe.journal.journal import Journal
from edelweissfe.models.modelchange import ModelChange, TopologyRecord, coalesce
from edelweissfe.utils.exceptions import TopologyError
from edelweissfe.utils.performancetiming import timeit


class TopologyPipeline:
    """Everything about changing the mesh of a :class:`~edelweissfe.models.femodel.FEModel`
    during a run, held by the model as ``model.topology``.

    The model itself is the mesh and its data. This class is the machinery around changing it:

    - the **topology window** (:meth:`changes`), the only scope in which elements and nodes
      may be created or removed;
    - the **number allocators** (:meth:`reserveElementNumbers`, :meth:`reserveNodeNumbers`), which
      make numbering a pure function of the creation order;
    - the **fixed-point rounds** (:meth:`update`), in which every model modifier (e.g. AMR)
      plans and applies until none has anything left to do;
    - the **change log** (:attr:`version`, :meth:`changesSince`) and the registry of
      **mesh dependents** (:meth:`refreshMeshDependents`), from which ties and contact catch up;
    - the **history** of every applied decision, its **fingerprint**, and the **replay** of that
      history on restart (:meth:`replayHistory`).

    Parameters
    ----------
    model
        The model whose mesh this pipeline changes.
    """

    identification = "TopologyPipeline"

    def __init__(self, model):
        self._model = model
        #: Consumers that cache mesh-derived state; see :meth:`refreshMeshDependents`.
        self.meshDependents = []
        #: Bumped on every structural mutation; drives pull-based reconcile.
        self.version = 0
        #: Recorded :class:`ModelChange` per mutation, newest last.
        self._changeLog = []
        #: High-water mark of the element number allocator; see :meth:`reserveElementNumbers`.
        self._nextElementNumber = 1
        #: High-water mark of the node number allocator; see :meth:`reserveNodeNumbers`.
        self._nextNodeNumber = 1
        #: True only inside :meth:`changes`; see there.
        self.isOpen = False
        #: Guard against a model modifier that keeps planning in response to its own output.
        self.maxRounds = 16
        #: Ordered record of every applied model-modifier decision; see :meth:`update`. This
        #: IS the restart history -- a resumed run replays it rather than re-deciding.
        self.history = []
        #: Compare the replayed topology's fingerprint against the recorded one, once, after the whole
        #: history has been replayed. On by default: it is the difference between "the resumed run
        #: diverged" and "it did not". See :meth:`replayHistory`.
        self.verifyFingerprints = True
        #: Additionally compute and compare a fingerprint after *every* replayed record, which turns
        #: "the resumed run diverged" into "it diverged HERE". Off by default: it costs one whole-mesh
        #: walk per record, so a long history replays in O(records x mesh) rather than O(mesh).
        #: Switch it on to locate a divergence the final check reported.
        self.verifyFingerprintsPerRecord = False
        #: The fingerprint of the mesh as the input file built it, before any model modifier acted;
        #: see :meth:`setupFingerprint`.
        self._setupFingerprint = None
        #: True inside a :meth:`changes` window that defers the node-field bookkeeping.
        self.isDeferringFieldBookkeeping = False
        self._deferredNodeFieldResizeJournal = None
        self._deferredFieldVariableLinkNodes = None

    def boundTo(self, model) -> "TopologyPipeline":
        """A shallow copy of this pipeline that acts on ``model``, for a shallow copy of the model.

        As for the model itself, the containers (history, change log, mesh dependents) are shared with
        this pipeline, and the counters, the version and the window state are the copy's own.

        Parameters
        ----------
        model
            The model the copy acts on.
        """

        pipeline = copy.copy(self)
        pipeline._model = model
        return pipeline

    def deferNodeFieldResize(self, journal: Journal):
        """Note a node-field resize requested inside a deferring window; it runs when the window closes."""

        self._deferredNodeFieldResizeJournal = journal

    def deferFieldVariableLink(self, nodes):
        """Note a field-variable relink requested inside a deferring window; it runs when the window closes."""

        self._deferredFieldVariableLinkNodes = nodes

    @contextmanager
    def changes(self, deferFieldBookkeeping: bool = False):
        """The only scope in which elements may be created or deleted.

        Opened once around model setup, and once per increment around the model modifiers. Outside
        it, :meth:`~edelweissfe.models.femodel.FEModel.createElement` and :meth:`~edelweissfe.models.femodel.FEModel.removeElement` raise -- which is what makes "only model
        modifiers mutate the topology" an enforced property rather than a convention, and what lets
        :meth:`reserveElementNumbers` guarantee that element numbering is a pure function of the
        ordered creation sequence.

        Nesting is permitted and is a no-op for the inner scope: setup-time helpers may open a
        window without knowing whether their caller already did.

        Parameters
        ----------
        deferFieldBookkeeping
            Postpone the node-field bookkeeping a mesh mutator requests through
            :meth:`~edelweissfe.models.femodel.FEModel._resizeNodeFieldsForNodes` and :meth:`~edelweissfe.models.femodel.FEModel._linkFieldVariableObjects` until this
            window closes, and run it exactly once then. Both are idempotent recomputations from the
            model's *current* nodes and elements -- neither one touches numbering, connectivity or
            anything the :meth:`fingerprint` covers -- so the state after one flush at the
            end equals the state after a flush per mutation; only the intermediate, immediately
            overwritten field layouts are skipped. That is what :meth:`replayHistory` wants:
            no increment is solved between two replayed records, so nothing consumes those layouts,
            and a flush per record makes a long replay O(records x mesh) instead of O(mesh). The
            live per-increment window does *not* defer -- the solver runs on the fields right after
            each update. Honoured by the outermost window only, like the window itself.
        """

        wasOpen = self.isOpen
        self.isOpen = True
        deferring = deferFieldBookkeeping and not wasOpen
        if deferring:
            self.isDeferringFieldBookkeeping = True
        try:
            yield
            if deferring:
                self.isDeferringFieldBookkeeping = False
                self._flushDeferredFieldBookkeeping()
        finally:
            self.isOpen = wasOpen
            if deferring:
                self.isDeferringFieldBookkeeping = False
                self._deferredNodeFieldResizeJournal = None
                self._deferredFieldVariableLinkNodes = None

    def _flushDeferredFieldBookkeeping(self):
        """Run the node-field bookkeeping postponed inside a deferring :meth:`changes`
        window, in the order a mutator issues it: resize first, then relink -- relinking against a
        NodeField that does not yet hold every node's field variable raises."""

        journal = self._deferredNodeFieldResizeJournal
        nodes = self._deferredFieldVariableLinkNodes
        self._deferredNodeFieldResizeJournal = None
        self._deferredFieldVariableLinkNodes = None
        if journal is not None:
            self._model._resizeNodeFieldsForNodes(journal)
        if nodes is not None:
            self._model._linkFieldVariableObjects(nodes)

    def reserveElementNumbers(self, count: int = 1) -> range:
        """Reserve ``count`` fresh element numbers.

        The allocator is **monotonic**: numbers are never recycled, and are never derived from
        ``max(self._model.elements)``. Both properties matter beyond tidiness.

        Deriving the next number from ``max(self._model.elements)`` makes numbering a function of the
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

        if not self.isOpen:
            raise TopologyError(
                "element numbers may only be reserved during a topology change -- see model.topology.changes()"
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

        self._nextElementNumber = max(self._nextElementNumber, max(self._model.elements.keys(), default=0) + 1)

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

        if not self.isOpen:
            raise TopologyError(
                "node labels may only be reserved during a topology change -- see model.topology.changes()"
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

        self._nextNodeNumber = max(self._nextNodeNumber, max(self._model.nodes.keys(), default=0) + 1)

    def ensureSurfaceFacetModifier(self, journal: Journal):
        """Create the implicit facet-regeneration modifier, if any facet recipe was declared.

        Retiling a contact/tie surface creates and deletes elements, so it is a topology change and
        belongs in the topology-update phase -- not in a consumer's refresh, which is where it used
        to live and which is what made every tie a mutating consumer. Users never declare this
        modifier: they already declared the ``*surface`` recipe it acts on.

        Ordered **last**, so that within a round it reacts to whatever the primary modifiers (a
        refinement, a deposition) just did.
        """

        if not self._model.contactFacetRecipes or "surfaceFacets" in self._model.modelModifiers:
            return

        from edelweissfe.modelmodifiers.surfacefacets.surfacefacets import (
            ModelModifier as SurfaceFacetsModifier,
        )

        self._model.modelModifiers["surfaceFacets"] = SurfaceFacetsModifier("surfaceFacets", self._model, journal)

    def checkModelModifierDomains(self):
        """Refuse a model in which two modifiers claim the same element.

        Run once, at the end of setup. Each modifier declares what it owns via
        :meth:`~edelweissfe.modelmodifiers.base.modelmodifierbase.ModelModifierBase.declaredDomain`;
        an overlap means both will mutate the same element and each will end up holding stale
        references to the other's work. Failing here costs a clear message at startup; failing later
        costs a corrupted element set or a node that is both Dirichlet-prescribed and an MPC slave,
        discovered mid-solve.
        """

        claimed = list(self._model.modelModifiers.items())
        for index, (name, modifier) in enumerate(claimed):
            domain = modifier.declaredDomain(self._model)
            if not domain:
                continue
            for otherName, otherModifier in claimed[index + 1 :]:
                overlap = domain & otherModifier.declaredDomain(self._model)
                if overlap:
                    raise TopologyError(
                        "model modifiers {!r} and {!r} both claim {:} of the same element(s) "
                        "(e.g. {:}). Two modifiers cannot own one element: each mutates it directly, "
                        "so the other is left holding a stale reference. Restrict their element sets "
                        "so they do not overlap, or combine them into a single modifier.".format(
                            name, otherName, len(overlap), sorted(overlap)[0]
                        )
                    )

    def update(self, step=None) -> bool:
        """Run every model modifier to a fixed point, inside one topology window.

        Modifiers depend on each other -- refinement invalidates a tied surface's facets, a
        deposition modifier creates elements refinement may then want to split, and a 2:1 balance
        may need to refine what another modifier just activated. Rather than asking the user to
        declare a dependency order (which cannot express mutual dependence anyway), each **round**
        offers every modifier the net change since that modifier last planned. A round in which
        nobody plans anything is the fixed point.

        Determinism comes from the round structure, not from luck: within a round, modifiers run in
        ``self._model.modelModifiers`` order, which is input-file order.

        Returns
        -------
        bool
            True if the topology changed, so the solver rebuilds its equation system.

        Raises
        ------
        TopologyError
            If the rounds do not settle within :attr:`maxRounds`, which means some modifier
            keeps planning in response to its own output. The message names the offenders.
        """

        changed = False
        with self.changes():
            # Seeded with the version at the START of this update, not None: a modifier must see
            # what earlier modifiers did in the SAME round. Seeding with None meant a purely
            # reactive modifier (one that only acts on someone else's change) was handed None in
            # round 1 -- after the change it needed to see had already happened -- and then had its
            # version stamped, so round 2 showed nothing new either. It never reacted at all.
            lastPlannedVersion = {name: self.version for name in self._model.modelModifiers}
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
                for name, modifier in self._model.modelModifiers.items():
                    change = self.changesSince(lastPlannedVersion[name])
                    lastPlannedVersion[name] = self.version
                    plan = modifier.plan(self._model, change, step)
                    if plan is None:
                        continue
                    modelChange = modifier.apply(self._model, plan)
                    # A modifier may plan and then find nothing left to do. Recording that would
                    # rebuild the equation system for nothing, pay a topology fingerprint for
                    # nothing, and let the no-op modifier burn through maxRounds and be
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
                    self.recordChange(roundNumber, name, modifier, plan, modelChange)
                    plannedThisRound.append(name)
                    changed = True
                if not plannedThisRound:
                    break
                if roundNumber >= self.maxRounds:
                    raise TopologyError(
                        "model modifiers did not settle within {:} rounds; still planning in the "
                        "last round: {:}. A modifier must return None from plan() once the change "
                        "since its own last plan no longer touches its domain.".format(
                            self.maxRounds, ", ".join(plannedThisRound)
                        )
                    )
        return changed

    def recordSetupFingerprint(self):
        """Record the :meth:`fingerprint` of the mesh as the input file built it, once the model is
        set up and before any model modifier acts; see :meth:`setupFingerprint`."""

        self._setupFingerprint = self.fingerprint()

    def setupFingerprint(self) -> str:
        """The :meth:`fingerprint` of the mesh as the input file built it. A checkpoint carries it, so
        that a resumed run can verify that it was rebuilt from the same mesh -- the recorded
        topology changes are verified by the replay, but they say nothing about a mesh no modifier
        ever touched.

        Returns
        -------
        str
            The digest.

        Raises
        ------
        RuntimeError
            If the model was never set up (:meth:`~edelweissfe.models.femodel.FEModel.prepareYourself`).
        """

        if self._setupFingerprint is None:
            raise RuntimeError("the model was not set up (prepareYourself), so its setup mesh is unknown")
        return self._setupFingerprint

    def fingerprint(self) -> str:
        """A short digest of the model's topology *and its numbering*, for verifying that a restart
        replay reproduced the original run.

        Covers exactly what a replay must get right and nothing else: every element's number, type
        and connectivity (in order -- a rotated connectivity is a real difference), and every node's
        label and reference coordinates. The surface nodes of a discrete rigid body hold its current
        position instead; the body reports their reference coordinates
        (:meth:`~edelweissfe.rigidbodies.rigidbody.RigidBody.referenceCoordinatesOfMovedNodes`), so that
        the motion of a rigid body is not mistaken for a change of the mesh. A resumed run replays the
        topology before it restores the displacement, i.e. with the body still at its reference position. Deliberately excludes solution state, so a mismatch means the *mesh*
        diverged, not that the solver took a different path.

        Recorded per round in the topology history, this turns "the resumed run diverged somewhere"
        into "increment 471, round 2, modifier amr" -- a divergence you can bisect rather than hunt.

        Not cheap: it walks the whole mesh, measured at 0.188 s on 64k elements / 69k nodes. A live
        run pays it once per *applied* modifier decision, in :meth:`recordChange`. A replay
        pays it once for the whole history (:meth:`replayHistory` carries the recorded
        digests forward and checks the final one), unless ``verifyFingerprintsPerRecord``
        asks for the per-record walk to locate a divergence.

        Uses blake2b rather than :func:`hash`, whose string hashing is randomised per process and
        would make the digest differ between two runs of the *same* code.

        Returns
        -------
        str
            A 32-character hex digest.
        """

        referenceCoordinatesOfMovedNodes = {}
        for rigidBody in self._model.rigidBodies.values():
            referenceCoordinatesOfMovedNodes.update(rigidBody.referenceCoordinatesOfMovedNodes())

        digest = hashlib.blake2b(digest_size=16)
        for elNumber in sorted(self._model.elements):
            element = self._model.elements[elNumber]
            digest.update(b"E|%d|%s|" % (elNumber, element.elType.encode()))
            digest.update(b",".join(b"%d" % node.label for node in element.nodes))
        for label in sorted(self._model.nodes):
            digest.update(b"N|%d|" % label)
            referenceCoordinates = referenceCoordinatesOfMovedNodes.get(label, self._model.nodes[label].coordinates)
            digest.update(np.asarray(referenceCoordinates, dtype=float).tobytes())
        return digest.hexdigest()

    def recordChange(
        self, roundNumber: int, name: str, modifier, plan, modelChange, time: float = None, fingerprint: str = None
    ) -> TopologyRecord:
        """Register everything an applied decision produced: the replay record and the changeset.

        Two things are recorded here, deliberately in one place:

        * an entry in :attr:`history`, holding the plan in the modifier's own serializable
          form plus the resulting topology fingerprint -- which is what lets a resumed run be
          checked round by round instead of only at the end;
        * the ``modelChange`` itself, stamped with the next :attr:`version`, so :meth:`changesSince`
          and :meth:`refreshMeshDependents` can see it.

        A model modifier therefore reports what it did in exactly one way: by returning a
        :class:`~edelweissfe.models.modelchange.ModelChange` from
        :meth:`~edelweissfe.modelmodifiers.base.modelmodifierbase.ModelModifierBase.apply` -- there
        is no second channel to keep in sync, and so no way to update one and forget the other.

        Both callers of ``apply`` (the live :meth:`update` loop and
        :meth:`replayHistory`) route through here, so a replayed run records the same
        changesets in the same order as the run it replays.

        Cost is one :meth:`fingerprint` per *applied* decision -- not per iteration, but not
        free either: 0.188 s measured on 64k elements / 69k nodes -- unless the caller supplies the
        fingerprint, which a replay does (see :meth:`replayHistory`).

        Parameters
        ----------
        time
            Model time to stamp the record with. Defaults to the current :attr:`time`; a replay
            passes the recorded time instead, so a resumed run's history carries the times the
            decisions were originally made rather than the time it was resumed at.
        fingerprint
            The digest to record. Defaults to :meth:`fingerprint` of the model as it is
            now, which is what a live run wants. A replay passes the digest the original run
            recorded, so the replayed history carries the digests that were verified rather than
            fresh ones that would launder a divergence into the next checkpoint.
        """

        if modelChange is not None:
            self.version += 1
            modelChange.version = self.version
            self._changeLog.append(modelChange)

        record = TopologyRecord(
            modifier=name,
            roundNumber=roundNumber,
            time=float(self._model.time if time is None else time),
            plan=modifier.encodePlan(plan),
            fingerprint=self.fingerprint() if fingerprint is None else fingerprint,
            nElementsAdded=len(modelChange.addedElements) if modelChange is not None else 0,
            nElementsRemoved=len(modelChange.removedElements) if modelChange is not None else 0,
            nNodesAdded=len(modelChange.addedNodes) if modelChange is not None else 0,
        )
        self.history.append(record)
        return record

    def replayHistory(self, records, journal: Journal = None):
        """Reconstruct the topology by re-applying recorded decisions, in order.

        This is the whole point of the plan/apply split: the modifier's :meth:`apply` runs here
        exactly as it did live, fed a decoded plan instead of a freshly evaluated one. There is no
        replay-specific code path to drift from the live one -- which is what the previous design
        had, and why a resumed run silently renumbered its elements.

        What a replay does *not* repeat per record is the whole-mesh work around ``apply`` whose
        result is a pure function of the final mesh: the :meth:`fingerprint` walk (the
        recorded digests are carried forward and the final one is checked) and the node-field
        bookkeeping (deferred to the end of the window, see :meth:`changes`). A history of
        a few hundred refinements on a mesh of tens of thousands of elements replayed in minutes
        otherwise, all of it spent re-deriving state that the next record, or :meth:`~edelweissfe.models.femodel.FEModel.readRestart`,
        overwrote right away.

        Parameters
        ----------
        records
            The :class:`~edelweissfe.models.modelchange.TopologyRecord` sequence to replay.
        journal
            Optional Journal for progress messages.

        Raises
        ------
        TopologyError
            If the replayed topology's fingerprint differs from the one recorded with the last
            record (when :attr:`verifyFingerprints`). With
            :attr:`verifyFingerprintsPerRecord` every record is checked as it is replayed
            and the error names the first diverging one -- so a divergence is located rather than
            merely detected.
        """

        perRecord = self.verifyFingerprints and self.verifyFingerprintsPerRecord
        with self.changes(deferFieldBookkeeping=True):
            for index, record in enumerate(records):
                modifier = self._model.modelModifiers.get(record.modifier)
                if modifier is None:
                    raise TopologyError(
                        "the checkpoint records a decision by model modifier {!r}, which this model "
                        "does not define -- the input file must declare the same modifiers as the run "
                        "being resumed".format(record.modifier)
                    )
                plan = modifier.decodePlan(record.plan)
                modelChange = modifier.apply(self._model, plan)
                # Carry the recorded digest forward instead of recomputing it -- unless every record
                # is to be verified, which needs the replayed one.
                replayed = self.recordChange(
                    record.roundNumber,
                    record.modifier,
                    modifier,
                    plan,
                    modelChange,
                    time=record.time,
                    fingerprint=None if perRecord else record.fingerprint,
                )
                if perRecord and replayed.fingerprint != record.fingerprint:
                    raise TopologyError(
                        "restart replay diverged at record {:} of {:}: modifier {!r}, round {:}, "
                        "time {:}. The replayed topology does not match the recorded one, so this "
                        "modifier's apply() is not a pure function of (model, plan).".format(
                            index, len(records), record.modifier, record.roundNumber, record.time
                        )
                    )
        if self.verifyFingerprints and not perRecord and records and records[-1].fingerprint:
            if self.fingerprint() != records[-1].fingerprint:
                last = records[-1]
                raise TopologyError(
                    "restart replay diverged: after replaying all {:} record(s) the topology does not "
                    "match the fingerprint recorded with the last one (modifier {!r}, round {:}, time "
                    "{:}). Some modifier's apply() is not a pure function of (model, plan); set "
                    "model.topology.verifyFingerprintsPerRecord=True to locate the first diverging "
                    "record.".format(len(records), last.modifier, last.roundNumber, last.time)
                )
        if journal is not None:
            journal.message(
                "Replayed {:} recorded topology change(s); {:} elements, {:} nodes".format(
                    len(records), len(self._model.elements), len(self._model.nodes)
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

        Phase 2 of the increment (see :meth:`update` for phase 1). Consumers are pure
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
        return any([consumer.refreshIfMeshChanged(self._model) for consumer in self.meshDependents])

    def changesSince(self, version: int) -> ModelChange:
        """The :class:`ModelChange` coalesced across every mutation recorded after ``version``, or
        ``None`` if the model hasn't changed since. A pull-based consumer compares its own
        last-seen version against :attr:`version` and, on a mismatch, reconciles from this,
        then adopts the new :attr:`version` as its own last-seen version."""
        if version >= self.version:
            return None
        return coalesce([c for c in self._changeLog if c.version > version])
