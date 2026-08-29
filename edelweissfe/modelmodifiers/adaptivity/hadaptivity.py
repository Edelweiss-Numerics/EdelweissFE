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

"""Dynamic h-adaptivity model modifier for HEX20 hanging-node AMR."""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from edelweissfe.adaptivity.hex20topology import Hex20Topology
from edelweissfe.adaptivity.refinement import AdaptiveMesh
from edelweissfe.adaptivity.statetransfer.perstatevar import PerStateVarStateTransfer
from edelweissfe.config.elementlibrary import getElementClass
from edelweissfe.config.markerlibrary import getMarkerClass
from edelweissfe.config.registry import RegistryLookupError
from edelweissfe.config.statetransferstrategies import getStateTransferStrategyClass
from edelweissfe.constraints.hangingnode import Constraint as HangingNodeConstraint
from edelweissfe.journal.journal import Journal
from edelweissfe.modelmodifiers.base.modelmodifierbase import ModelModifierBase
from edelweissfe.models.femodel import FEModel
from edelweissfe.models.modelchange import ModelChange, coalesce
from edelweissfe.models.modelchangeobserver import ModelChangeType
from edelweissfe.points.node import Node
from edelweissfe.utils.exceptions import TopologyError
from edelweissfe.utils.performancetiming import timeit
from edelweissfe.utils.schema import (
    buildSchemaFromOptions,
    schemaField,
    subKeywordField,
)


@dataclass(frozen=True)
class HAdaptivityMarkerSchema:
    """The grammar common to every ``>>marker`` block.

    A ``>>marker`` block is polymorphic on ``type``: the remaining options depend on which marker
    that selects, and are owned/validated by that marker's own schema (a
    :class:`~edelweissfe.adaptivity.marking.MarkerOptionsBase` subclass, e.g.
    :class:`~edelweissfe.adaptivity.marking.RecoveryErrorMarkerSchema`) rather than being flattened
    into one union here. This schema therefore declares only the two options every marker shares --
    ``type`` (the dispatch key) and ``initialOnly`` -- with the type-specific options documented on
    each marker in :mod:`edelweissfe.adaptivity.marking` and reachable through the ``marker``
    registry category (:mod:`edelweissfe.config.markerlibrary`).
    """

    type: str | None = schemaField(
        description=(
            "Marker type, resolved through the 'marker' registry: fieldOutput, elementSet, nodeSet, "
            "surface, recoveryError. The type-specific options are defined by the selected marker's "
            "own schema."
        ),
        dtype=str,
        default=None,
        required=True,
    )
    initialOnly: bool = schemaField(description="Evaluate only once at simulation start", dtype=bool, default=False)


@dataclass(frozen=True)
class HAdaptivitySchema:
    """The options this model modifier accepts, owned by this module and never mutated from
    outside it.

    ``marker`` is declared optional even though at least one is required in practice -- that
    invariant is enforced in :meth:`ModelModifier.__init__`, not by the grammar (a schema field
    cannot express "at least one of a repeatable sub-keyword").
    """

    moduleOptions: dict = schemaField(description="Internal", dtype=dict, default_factory=dict)
    elSet: str | None = schemaField(
        description=(
            "Fallback for 'refineElSet' if that is not given. Does not restrict marking itself -- "
            "each '>>marker' scopes its own eligible elements (a fieldOutput's associated set, an "
            "elementSet/nodeSet/surface's members)."
        ),
        dtype=str,
        default=None,
    )
    refineElSet: str | None = schemaField(
        description=(
            "Restrict the AMR octree mirror itself to this element set, e.g. the solid elements in "
            "a mesh that also contains contact-facet elements. Elements outside this set never "
            "become octree roots and are left untouched by refinement. Defaults to 'elSet' if given, "
            "otherwise to every 20-node (HEX20-family) element in the model."
        ),
        dtype=str,
        default=None,
    )
    maxLevel: int = schemaField(description="Maximum refinement level.", dtype=int, default=1)
    minMarkedElements: int = schemaField(
        description=(
            "Minimum number of eligible elements that must be marked before a refinement pass is "
            "triggered. Marked elements persist (accumulate) across increments -- across calls where "
            "fewer than this many are marked, no refinement happens and no equation system rebuild is "
            "triggered -- until the accumulated count reaches this threshold, at which point all of "
            "them are refined together in a single pass. Note that individual markers may cap their "
            "own marks per pass (e.g. 'maxRefinedFraction') or expand them (e.g. 'halo') before "
            "accumulating here. Default 1 refines as soon as any element is marked (previous behavior)."
        ),
        dtype=int,
        default=1,
    )
    splitFactor: int = schemaField(
        description=(
            "Number of equal parts per axis a marked element is split into (2 = octree bisection "
            "into 8 children; 3 = 3x3x3 = 27 children, etc.). The hanging-node coupling stays exact "
            "for any factor."
        ),
        dtype=int,
        default=2,
    )
    elementType: str | None = schemaField(
        description="Element type to instantiate for children (default: like parents).", dtype=str, default=None
    )
    elementProvider: str = schemaField(description="Element provider.", dtype=str, default="marmot")
    stateTransfer: str = schemaField(
        description="Quadrature-point state-transfer strategy for the whole state block: nearestQp|projection|virgin.",
        dtype=str,
        default="nearestQp",
    )
    stateTransferOverrides: str | None = schemaField(
        description=(
            "Per-state-variable overrides routing named variables to a different strategy, e.g. "
            "'strain:projection, stress:virgin'. Comma-separated 'name:strategy' pairs."
        ),
        dtype=str,
        default=None,
    )
    marker: tuple = subKeywordField(
        description="AMR marker definition. At least one is required.", schema=HAdaptivityMarkerSchema
    )


@dataclass(frozen=True)
class RefinementPlan:
    """One refinement decision, as octree element ids.

    Eids rather than element numbers: an eid is this modifier's own identifier for a cell, minted by
    its private octree counter and reproduced exactly by replaying the same decisions. Element
    numbers are assigned by the model's allocator in an order that also depends on what else minted,
    so they are not a decision this modifier can record and re-apply.
    """

    eids: tuple

    def __init__(self, eids):
        object.__setattr__(self, "eids", tuple(int(eid) for eid in eids))


def _buildStateTransferStrategy(defaultName, overridesSpec):
    """Construct the state-transfer strategy from the input arguments. With no per-variable
    overrides this is just the named default strategy; otherwise a
    :class:`~edelweissfe.adaptivity.statetransfer.perstatevar.PerStateVarStateTransfer` wrapping the
    default with the named overrides."""
    default = getStateTransferStrategyClass(defaultName)()
    if not overridesSpec:
        return default
    overrides = {}
    for entry in overridesSpec.split(","):
        entry = entry.strip()
        if not entry:
            continue
        name, strategyName = entry.rsplit(":", 1)
        overrides[name.strip()] = getStateTransferStrategyClass(strategyName.strip())()
    return PerStateVarStateTransfer(default, overrides) if overrides else default


def _connectedComponents(elements: list) -> dict:
    """Partition elements into connected bodies via union-find over shared node labels.

    Two elements belong to the same body if they share at least one node label. The resulting
    component id namespaces the refinement node registry and confines hanging-node classification
    to one body, so two bodies meeting at a flush interface -- a tied surface pair (``adjust`` moves
    the slave nodes exactly onto the master surface), a zero-gap contact pair, a duplicated-node
    crack plane -- are neither collapsed onto shared node labels nor welded together by refinement.

    Parameters
    ----------
    elements
        The refineable elements, in the order in which they become octree roots.

    Returns
    -------
    dict
        element -> component id, densely numbered from 0 in order of first appearance.
    """
    parentOf = list(range(len(elements)))

    def find(i):
        while parentOf[i] != i:
            parentOf[i] = parentOf[parentOf[i]]  # path halving
            i = parentOf[i]
        return i

    def union(i, j):
        rootI, rootJ = find(i), find(j)
        if rootI != rootJ:
            parentOf[max(rootI, rootJ)] = min(rootI, rootJ)

    firstElementAtNode = {}
    for i, element in enumerate(elements):
        for node in element.nodes:
            union(i, firstElementAtNode.setdefault(node.label, i))

    componentOfElement = {}
    denseIds = {}
    for i, element in enumerate(elements):
        componentOfElement[element] = denseIds.setdefault(find(i), len(denseIds))
    return componentOfElement


#: The node-field value entries carried across a refinement by isoparametric interpolation from the
#: parent element. ``"U"`` is the solution and is always present. ``"V"`` is the velocity, which only
#: exists when an explicit dynamic solver put it there -- and it *has* to be carried, because unlike
#: an implicit solver, which reconstructs everything it needs from ``"U"``, a central-difference
#: scheme holds kinetic state that nothing else can reproduce: a new node whose velocity defaulted to
#: zero would silently lose it. Interpolating it with the same operator as ``"U"`` is also what keeps
#: the two consistent -- the shape functions are a partition of unity, so a uniform velocity field is
#: reproduced exactly and the patch's momentum is conserved exactly in that case (the general case
#: differs at second order in the velocity gradient across the parent, which is discretisation error,
#: not a defect).
#:
#: An entry absent from a given node field is skipped, so this list is safe to extend.
WARM_STARTED_NODE_FIELD_ENTRIES = ("U", "V")


class ModelModifier(ModelModifierBase):
    #: Option schema for this model modifier, per OptionSchemaProvider. Documentation-only
    #: (see HAdaptivitySchema's own docstring) -- construction still goes through the
    #: Module-based mechanism below.
    schema = HAdaptivitySchema

    def __init__(self, name: str, model: FEModel, journal: Journal, *args, **kwargs):
        super().__init__(name, model, journal, *args, **kwargs)
        options = buildSchemaFromOptions(HAdaptivitySchema, kwargs)

        self._name = name
        self._model = model
        self._journal = journal

        # Markers are resolved by 'type' through the L3 marker registry and each builds itself from
        # its own >>marker options via fromOptions (validated against that marker's own schema), so
        # this loop is marker-agnostic: adding a marker means registering it, not editing an if/elif
        # here. The 'type' key is the dispatch key, not a marker option, so it is stripped before the
        # marker validates the rest.
        self.markers = []
        for m_opt in options.moduleOptions.get("marker", []):
            m_type = m_opt.get("type")
            if not m_type:
                raise ValueError(
                    f"hAdaptivity modifier {name!r}: a '>>marker' block is missing its required "
                    "'type' (e.g. 'type=fieldOutput', 'type=recoveryError', 'type=nodeSet')."
                )
            try:
                markerClass = getMarkerClass(m_type)
            except RegistryLookupError as e:
                raise ValueError(f"hAdaptivity modifier {name!r}: {e}") from e
            # 'type' is the dispatch key (already consumed above); 'inputFile' is parser bookkeeping
            # stamped onto every module keyword's options. Everything else is a real marker option,
            # validated against the marker's own schema inside fromOptions.
            markerOptions = {key: value for key, value in m_opt.items() if key.casefold() not in ("type", "inputfile")}
            self.markers.append(markerClass.fromOptions(markerOptions))
        if not self.markers:
            raise ValueError(
                f"hAdaptivity modifier {name!r} defines no '>>marker' block. At least one is required, "
                "e.g. '>>marker, type=fieldOutput, fieldOutput=stress, expression=\"abs(x) > 0.1\"' "
                "(referencing an already-declared 'perElement' *fieldOutput)."
            )

        self.maxLevel = options.maxLevel
        self.minMarkedElements = max(1, options.minMarkedElements)
        self._pendingMarkedElements = set()  # elements marked but not yet refined (below minMarkedElements)
        # Diagnostics only, for the journal and for tests. The authoritative record of what this
        # modifier did -- the one a restart replays -- is model.topologyHistory.
        self._committedOccasions = []
        self.splitFactor = options.splitFactor
        self._stateTransfer = _buildStateTransferStrategy(options.stateTransfer, options.stateTransferOverrides)
        self._provider = options.elementProvider
        # element -> its section, so children inherit the parent's material (multi-material meshes)
        self._sectionOf = {}
        for section in model.sections.values():
            for elementSet in section.elSets:
                for element in elementSet:
                    self._sectionOf[element] = section

        # restrict the octree mirror to the refineable solid elements: a model that also contains
        # e.g. contact-facet elements (2/3 nodes) must not have those become octree roots. Prefer an
        # explicit restriction; otherwise fall back to the 20-node (HEX20-family) elements, which is
        # the only family this modifier supports anyway.
        refineSetName = options.refineElSet or options.elSet
        if refineSetName is not None:
            refineElements = list(model.elementSets[refineSetName])
        else:
            refineElements = [el for el in model.elements.values() if len(el.nodes) == 20]
        if not refineElements:
            raise ValueError(
                "hAdaptivity found no refineable (20-node) elements in the model; specify "
                "'refineElSet' (or 'elSet') to select the solid element set explicitly."
            )

        # Which elements this instance owns. Checked pairwise against every other modifier by
        # FEModel.checkModelModifierDomains at the end of setup -- two hAdaptivity instances cannot
        # independently own overlapping elements, since each maintains its own AdaptiveMesh mirror
        # and materializes/deletes elements directly in the model.
        self._refineElementNumbers = {el.elNumber for el in refineElements}

        # element type: infer from a refineable element if not given
        anyEl = refineElements[0]
        self._elementType = options.elementType or anyEl.elType
        self._elementClass = getElementClass(self._elementType, self._provider)

        # bodies of the refineable mesh: node labels are namespaced per body, so coincident nodes of
        # two bodies (a tied interface -- 'adjust' makes it flush by default --, a zero-gap contact
        # pair, a duplicated-node crack plane) are never deduplicated into one label
        componentOfElement = _connectedComponents(refineElements)

        # build the AdaptiveMesh mirror, sharing node labels with the live model. Only the nodes of
        # the refineable elements are seeded: a node the octree does not own must not be able to
        # claim a coordinate key, and only an octree-owned node can be seeded with a body.
        self._topology = Hex20Topology()
        # The mirror mints its new node labels from the model's own allocator, so octree and
        # model share one monotonic node counter instead of each keeping their own.
        self._mesh = AdaptiveMesh(
            splitFactor=self.splitFactor, topology=self._topology, reserve_labels=model.reserveNodeNumbers
        )
        self._eidToEl = {}  # mesh element id -> live element
        #: Diagnostics only, parallel to _committedOccasions; see there.
        self._committedOccasionEids = []
        for el in refineElements:
            componentId = componentOfElement[el]
            for n in el.nodes:
                self._mesh.registry.seed(n.label, n.coordinates, componentId)
            coords = np.array([n.coordinates for n in el.nodes])
            eid = self._mesh.add_root(coords, componentId)
            self._eidToEl[eid] = el
        # nodes outside the refineable mesh are not seeded, but their labels are taken:
        # keep the registry's high-water mark above them so new nodes never collide with them
        self._mesh.registry.reserve_labels_up_to(max(model.nodes.keys(), default=0))

        # all-encompassing sets (contain every node, e.g. 'all', 'ALLNODES') are not boundary BCs --
        # they just gain every new node; rebuild them wholesale, don't guard/track them
        allLabels = set(model.nodes.keys())
        self._allLikeSets = {name for name, ns in model.nodeSets.items() if {n.label for n in ns.nodes} == allLabels}
        # track the remaining (boundary) node sets so real BCs gain new boundary nodes on refinement
        for setName, nodeSet in model.nodeSets.items():
            if setName not in self._allLikeSets:
                self._mesh.define_node_set(setName, [n.label for n in nodeSet])

        # track element sets so user element sets propagate child elements on refinement
        elToEid = {el: eid for eid, el in self._eidToEl.items()}
        # passengers of a tracked set: members the octree mirror does not know (non-refineable
        # elements, e.g. HEX8, interface elements, contact facets). A mixed set would lose them on
        # the first refinement, since _materialize rebuilds the set from mesh element ids only
        self._untrackedOfElementSet = {}  # element set name -> list of non-mirrored members
        for setName, elementSet in model.elementSets.items():
            eids = [elToEid[el] for el in elementSet if el in elToEid]
            # a set with no refineable member (e.g. a contact-facet-only set) is left untracked, so
            # _materialize never overwrites it with an emptied-out ElementSet
            if eids:
                self._mesh.define_element_set(setName, eids)
                self._untrackedOfElementSet[setName] = [el for el in elementSet if el not in elToEid]

        # track element-based surfaces so surface loads stay consistent under refinement
        for surfaceName, surface in model.surfaces.items():
            pairs = [
                (elToEid[el], faceID) for faceID, elementSet in surface.items() for el in elementSet if el in elToEid
            ]
            if pairs:
                self._mesh.define_surface(surfaceName, pairs)

        # companion hanging-node MPC (records set in memory), registered as a multi-point constraint
        self._hanging = HangingNodeConstraint(name + "_hanging", model)
        model.multiPointConstraints[name + "_hanging"] = self._hanging
        self._converged = False  # set True once an increment has converged
        self._lastRefinedTime = None  # model.time of the last refinement (guards re-refine on cutback)
        self._isFirstCall = True
        # parent-parametric coords of each child's nodes (used for warm-start interpolation)
        self._octantParams = self._topology.subdivision_children_param(self.splitFactor)

    def declaredDomain(self, model: FEModel) -> set:
        """The refineable roots this instance owns; see
        :meth:`~edelweissfe.modelmodifiers.base.modelmodifierbase.ModelModifierBase.declaredDomain`.
        Two hAdaptivity blocks over the same elements must be combined into one with several
        ``>>marker`` lines instead -- including ``initialOnly`` ones."""

        return self._refineElementNumbers

    @property
    def actsOnlyAtSimulationStart(self) -> bool:
        """True exactly when every marker is an ``initialOnly`` one.

        Not an approximation: :meth:`plan` evaluates *only* the ``initialOnly`` markers on its first
        call and *only* the others on every later one, so a modifier whose markers are all
        ``initialOnly`` provably plans nothing after that first call. See
        :attr:`~edelweissfe.modelmodifiers.base.modelmodifierbase.ModelModifierBase.actsOnlyAtSimulationStart`.

        Returns
        -------
        bool
            Whether this modifier is fully served by a single topology update at the start.
        """

        return all(marker.initialOnly for marker in self.markers)

    @timeit("AMR")
    def plan(self, model: FEModel, change, step, timeStep: float) -> "RefinementPlan | None":
        """Evaluate the markers and decide which octree cells to refine. See
        :meth:`~edelweissfe.modelmodifiers.base.modelmodifierbase.ModelModifierBase.plan`.

        The decision is returned as octree eids rather than element numbers: eids are this
        modifier's own stable identifiers, reproducible by its replay, whereas element numbers are
        assigned by the model's allocator in an order that depends on what else minted.
        """

        # Nothing this modifier cares about changed since it last planned in this topology update
        # -- another modifier's mutation. Returning None here is what lets the pipeline settle.
        if change is not None and not (change.addedElements or change.removedElements):
            return None

        # Do not re-refine if the solver is re-trying the exact same time state after a cutback
        if self._lastRefinedTime is not None and abs(model.time - self._lastRefinedTime) < 1e-12:
            return None

        elForEid = {v: k for k, v in self._eidToEl.items()}
        marked_elements = set()

        if self._isFirstCall:
            initial_markers = [m for m in self.markers if m.initialOnly]
            for m in initial_markers:
                elements = m.mark(model, self._eidToEl.values(), self._mesh)
                marked_elements.update(elements)

        # dynamic markers (not initialOnly) evaluate the converged solution, so they need at least
        # one increment to have actually converged -- on the very first call, the topology update
        # runs before increment 1 is solved and model fields still hold the pre-solve initial
        # condition, which is meaningless to mark on regardless of which field a given marker
        # evaluates. Gating on displacement magnitude instead would wrongly skip markers that
        # evaluate other fields (stress, strain, ...) whenever displacement itself stays tiny.
        if not self._isFirstCall:
            dynamic_markers = [m for m in self.markers if not m.initialOnly]
            for m in dynamic_markers:
                marked_elements.update(m.mark(model, self._eidToEl.values(), self._mesh))

        self._isFirstCall = False

        # freshly marked elements accumulate onto any still-pending ones from earlier increments; a
        # stale pending element that another path already refined/removed is dropped by the
        # elForEid/maxLevel filter below, same as a freshly marked one would be.
        self._pendingMarkedElements.update(marked_elements)

        if not self._pendingMarkedElements:
            return None

        # keep only active elements below maxLevel
        with timeit("marking filter"):
            eligible = [
                el
                for el in sorted(self._pendingMarkedElements, key=lambda e: e.elNumber)
                if el in elForEid and self._mesh.elements[elForEid[el]]["level"] < self.maxLevel
            ]
        self._pendingMarkedElements = set(eligible)

        if len(eligible) < self.minMarkedElements:
            if eligible:
                self._journal.message(
                    "AMR ModelModifier: {:} element(s) marked, deferring refinement until {:} accumulate".format(
                        len(eligible), self.minMarkedElements
                    ),
                    "hadaptivity",
                    1,
                )
            return None

        self._pendingMarkedElements = set()

        # Stamped here, not in apply(): it guards the *next* planning pass against re-refining after
        # a cutback, and apply() must not read solution state (model.time included).
        self._lastRefinedTime = float(model.time)

        return RefinementPlan(eids=[elForEid[el] for el in eligible])

    @timeit("AMR")
    def apply(self, model: FEModel, plan: "RefinementPlan"):
        """Refine exactly the cells named by ``plan`` and materialize the resulting children: the
        octree split, 2:1 balance, hanging-node MPCs, element/node/set bookkeeping, and the
        :class:`ModelChange` notification.

        Pure octree/topology mechanics with no dependence on solution history, which is what lets a
        live run and a restart replay share it: given the same plan they produce byte-identical
        topology, element numbers included. See
        :meth:`~edelweissfe.modelmodifiers.base.modelmodifierbase.ModelModifierBase.apply`.

        Parameters
        ----------
        model
            The FEModel object, mutated in place.
        plan
            The refinement decision, as octree eids.

        Returns
        -------
        ModelChange
            The changeset this refinement produced.
        """

        markedEids = list(plan.eids)
        # Captured now, not at the end: the refined parents are popped from _eidToEl during
        # materialisation, so afterwards their element numbers are no longer resolvable here.
        markedElementNumbers = [self._eidToEl[eid].elNumber for eid in markedEids if eid in self._eidToEl]

        # refine + 2:1 balance in the mirror
        nBefore = len(self._mesh.active())
        with timeit("refine & balance"):
            for eid in markedEids:
                if self._mesh.elements[eid]["active"]:
                    self._mesh.refine(eid)
            self._mesh.balance_2to1()

        with timeit("hanging nodes"):
            records = self._mesh.hanging_mpc_records()  # computed once (expensive), reused below

        with timeit("materialize"):
            change = self._materialize(model, records)

        self._hanging.setRecords(records)
        # The change is not announced here: it is returned below, and the pipeline records it (see
        # FEModel.recordTopologyChange). Consumers re-index later, once, in refreshMeshDependents.
        self._journal.message(
            "AMR ModelModifier: marked {:}, refined -> active elements {:} -> {:}, {:} hanging nodes".format(
                len(markedEids), nBefore, len(self._mesh.active()), len(records)
            ),
            "hadaptivity",
            0,
        )
        self._committedOccasions.append(markedElementNumbers)
        self._committedOccasionEids.append(list(markedEids))
        return change

    def _materialize(self, model: FEModel, records: dict):
        mesh = self._mesh
        reg = mesh.registry

        # Element numbers come from the model's single monotonic allocator
        # (FEModel.reserveElementNumbers). This modifier deliberately keeps no counter of its own:
        # the one it used to keep had to be resynced against max(model.elements) on every call,
        # because a tied surface's facets -- rebuilt via the observer/MeshDependent escape hatches
        # fired at the end of THIS very call -- claim labels in between, and a private counter would
        # collide with (and silently overwrite) them, after which they were deleted as "stale",
        # orphaning the solid elements that had taken their labels.

        # snapshot the converged nodal values BEFORE the mesh mutates, for the warm start
        oldValues = {}
        # Runs on the replay path too. It is dead work there -- readRestart overwrites every node
        # field right afterwards -- but apply() is ONE code path, and a "skip this on replay" branch
        # is exactly the kind of live/replay divergence that made a resumed run rebuild a different
        # mesh. If this ever costs measurably, the flag belongs in the recorded plan, not in an
        # ambient replay mode.
        for fieldName, nodeField in model.nodeFields.items():
            for entryName in WARM_STARTED_NODE_FIELD_ENTRIES:
                if entryName in nodeField:
                    entryValues = np.asarray(nodeField[entryName])
                    oldValues[(fieldName, entryName)] = {
                        node: entryValues[nodeField._indicesOfNodesInArray[node]].copy() for node in nodeField.nodes
                    }

        # new nodes
        newNodes = {}
        for label, coord in reg.coordinates.items():
            if label not in model.nodes:
                node = Node(label, np.asarray(coord, dtype=float))
                model.createNode(node)
                newNodes[label] = node

        active = set(mesh.active())
        newValues = {key: {} for key in oldValues}  # interpolated values for new nodes, per (field, entry)

        # Every octree cell that must become a model element but is not one yet. Usually that is
        # exactly the children of the cells refined in this call. It is not always: 2:1 balancing
        # refines until the mesh is graded, and can therefore split a cell it created earlier in
        # the same call, leaving an active leaf whose parent is itself brand new. Walking each such
        # leaf up to its nearest materialised ancestor collects those intermediate cells as well;
        # they are created below and removed again with the other refined parents, so however many
        # levels a cascade went, every one of them is handled by the same "split a materialised
        # parent into its children" code.
        pending = set()
        for eid in active - set(self._eidToEl):
            ancestor = eid
            while ancestor is not None and ancestor not in self._eidToEl and ancestor not in pending:
                pending.add(ancestor)
                ancestor = mesh.elements[ancestor]["parent"]

        # One changeset per materialised level, coalesced at the end: the merge is what resolves an
        # intermediate's create-then-remove into the direct parent -> grandchild relation a consumer
        # needs, rather than leaving a phantom element in both the added and the removed set (see
        # ModelChange.mergedWith).
        levelChanges = []
        newChildEids = set()
        while pending:
            # Sorted, with the whole level's numbers reserved up front: which octree child gets
            # which element number is then a pure function of this sorted list of eids -- not of the
            # order an unordered set happened to iterate in, and not of what else claimed a number
            # partway through the loop.
            levelEids = sorted(eid for eid in pending if mesh.elements[eid]["parent"] in self._eidToEl)
            if not levelEids:
                raise TopologyError(
                    "AMR: {:} active octree cell(s) (e.g. {:}) have no materialised ancestor, so "
                    "they cannot be turned into elements. The octree mirror and the model would "
                    "disagree about which elements exist.".format(len(pending), sorted(pending)[0])
                )
            change = ModelChange(kind=ModelChangeType.REFINEMENT)
            childNumbers = model.reserveElementNumbers(len(levelEids))
            with timeit("elements & state transfer"):
                for eid, elNumber in zip(levelEids, childNumbers):
                    e = mesh.elements[eid]
                    parentEid = e["parent"]
                    parentEl = self._eidToEl[parentEid]
                    child = self._elementClass(self._elementType, elNumber)
                    child.setNodes([model.nodes[label] for label in e["conn"]])
                    self._sectionOf[parentEl].assignSectionPropertiesToElement(child)
                    # Runs on replay too, identically: apply() is one code path, and element state
                    # is restored by number afterwards either way.
                    self._stateTransfer.transferState(parentEl, [child], self._topology)

                    # warm start: interpolate each NEW node's field values from the parent via the
                    # HEX20 isoparametric map, so the increment restarts from a consistent state,
                    # not zero
                    octant = mesh.elements[parentEid]["children"].index(eid)
                    childParams = self._octantParams[octant]
                    for i, label in enumerate(e["conn"]):
                        node = model.nodes[label]
                        if label in newNodes and any(node not in newValues[f] for f in oldValues):
                            N = self._topology.shape_functions(*childParams[i])
                            for key, vals in oldValues.items():
                                # An intermediate parent's own nodes are new, so they are not in the
                                # pre-mutation snapshot -- the level above interpolated them, and the
                                # next level down interpolates from that in turn.
                                interpolated = newValues[key]
                                parentVals = [vals[pn] if pn in vals else interpolated.get(pn) for pn in parentEl.nodes]
                                if all(v is not None for v in parentVals):
                                    newValues[key][node] = N @ np.array(parentVals)

                    model.createElement(child)
                    self._eidToEl[eid] = child
                    self._sectionOf[child] = self._sectionOf[parentEl]

                    change.addedElements.add(child.elNumber)
                    change.parentToChildren.setdefault(parentEl.elNumber, []).append(child.elNumber)

            # per-face parent -> child tiling (the faceMap), while parents are still materialized
            for parentEid in {mesh.elements[eid]["parent"] for eid in levelEids}:
                parentLabel = self._eidToEl[parentEid].elNumber
                childEids = mesh.elements[parentEid]["children"]
                for faceID, faceIndex in self._topology.faceid_to_face.items():
                    childLabels = [
                        self._eidToEl[childEids[j]].elNumber
                        for j in self._topology.face_child_indices(faceIndex, self.splitFactor)
                    ]
                    change.faceMap[(parentLabel, faceID)] = [(label, faceID) for label in childLabels]

            levelChanges.append(change)
            newChildEids |= set(levelEids)
            pending -= set(levelEids)

        # The new nodes and the removals ride on the LAST level's changeset: nothing created there
        # is transient (only intermediates are, and they always have a level below them), so the
        # coalesce below cannot drop them.
        change = levelChanges[-1] if levelChanges else ModelChange(kind=ModelChangeType.REFINEMENT)
        change.addedNodes |= set(newNodes.keys())

        # remove refined parents, transient intermediates included (sorted, so the changeset is
        # built in a reproducible order)
        for eid in sorted(set(self._eidToEl) - active):
            el = self._eidToEl.pop(eid)
            model.removeElement(el.elNumber)
            change.removedElements.add(el.elNumber)

        if len(levelChanges) > 1:
            change = coalesce(levelChanges)

        # The octree mirror decides which elements exist; if the model no longer agrees, every
        # consumer downstream is reading a mesh that is not the one being refined. Cheap next to
        # everything else in here, and it turns a silent desync into a located failure.
        if set(self._eidToEl) != active:
            raise TopologyError(
                "AMR: the octree mirror and the model disagree after materialisation -- {:} active "
                "cell(s) without an element, {:} element(s) without an active cell".format(
                    len(active - set(self._eidToEl)), len(set(self._eidToEl) - active)
                )
            )

        # keep model.surfaces in sync: parent (eid,faceID) -> child faces
        for surfaceName, pairs in mesh.surfaces.items():
            if surfaceName in model.surfaces:
                if any(meid in newChildEids for meid, _ in pairs):
                    change.changedSurfaces.add(surfaceName)
                byFace = defaultdict(list)
                # Sorted: this fixes the member order of the rebuilt surface, and a contact/tie
                # facet generator hands out facet element labels in exactly that order.
                for meid, faceID in sorted(pairs):
                    if meid in self._eidToEl:
                        byFace[faceID].append(self._eidToEl[meid])
                model.surfaces[surfaceName].replaceData({f: els for f, els in byFace.items()})

        with timeit("sets & fields sync"):
            # Tracked (non-all) node sets that gain nodes are rebuilt with the new members (excluding
            # hanging slave nodes, whose motion is set by the MPC).
            slaves = set(records.keys())
            for setName, labels in mesh.nodeSets.items():
                present = {n.label for n in model.nodeSets[setName].nodes}
                if any(label not in present and label not in slaves for label in labels):
                    members = [model.nodes[label] for label in sorted(labels) if label not in slaves]
                    model.nodeSets[setName].replaceMembers(members)
                    change.changedNodeSets.add(setName)

            # sync all element sets (user sets like 'concrete' and all-encompassing sets)
            allNodes = list(model.nodes.values())
            if newNodes:
                for setName in self._allLikeSets | {"all"}:
                    model.nodeSets[setName].replaceMembers(allNodes)
                    change.changedNodeSets.add(setName)
            for setName, eids in mesh.elementSets.items():
                if setName in model.elementSets:
                    if eids & newChildEids:
                        change.changedElementSets.add(setName)
                    # sorted, for the same reason as the surface sync above: this order becomes the
                    # element set's member order, which downstream generators number entities by
                    elements = [self._eidToEl[eid] for eid in sorted(eids) if eid in self._eidToEl]
                    # carry the non-mirrored members along: the octree only knows refineable elements,
                    # so a mixed set would silently drop them here. Members deleted from the model in
                    # the meantime are filtered out by their label
                    elements += [el for el in self._untrackedOfElementSet[setName] if el.elNumber in model.elements]
                    model.elementSets[setName].replaceMembers(elements)
            model.elementSets["all"].replaceMembers(list(model.elements.values()))
            change.changedElementSets.add("all")

        with timeit("fields resize & restore"):
            # resize node fields in place to include the new nodes, then restore the warm start:
            # converged values on the retained nodes and interpolated values on the new nodes.
            # Both U (current) and P (previous converged) get the same warm-start value, so the first
            # Newton iteration after refinement sees a normal residual rather than a spurious dU = U - P
            # = U - 0 cold-restart spike on every retained/new node (P-field warm-start fix).
            model._resizeNodeFieldsForNodes(self._journal)
            for fieldName, nodeField in model.nodeFields.items():
                if "U" not in nodeField:
                    nodeField.createFieldValueEntry("U")
                if "P" not in nodeField:
                    nodeField.createFieldValueEntry("P")
                U = nodeField["U"]
                P = nodeField["P"]
                old = oldValues.get((fieldName, "U"), {})
                new = newValues.get((fieldName, "U"), {})
                for node in nodeField.nodes:
                    idx = nodeField._indicesOfNodesInArray[node]
                    if node in old:
                        U[idx] = old[node]
                        P[idx] = old[node]
                    elif node in new:
                        U[idx] = new[node]
                        P[idx] = new[node]

                # Every other warm-started entry gets the interpolation and nothing else -- in
                # particular NOT the "P := U" trick above, which exists only so an implicit solver
                # sees a sane first residual. An entry that is not present here (the usual case for
                # "V", which only an explicit solver creates) is simply skipped.
                for entryName in WARM_STARTED_NODE_FIELD_ENTRIES:
                    if entryName == "U" or entryName not in nodeField:
                        continue
                    entryValues = nodeField[entryName]
                    oldEntry = oldValues.get((fieldName, entryName), {})
                    newEntry = newValues.get((fieldName, entryName), {})
                    for node in nodeField.nodes:
                        idx = nodeField._indicesOfNodesInArray[node]
                        if node in oldEntry:
                            entryValues[idx] = oldEntry[node]
                        elif node in newEntry:
                            entryValues[idx] = newEntry[node]

        # Separately timed: this relinks EVERY node's field variables, so its cost scales with the
        # whole mesh rather than with what this refinement actually changed.
        with timeit("relink field variables"):
            model._linkFieldVariableObjects(model.nodeSets["all"])
        return change

    def encodePlan(self, plan: "RefinementPlan") -> dict:
        """Serialize a :class:`RefinementPlan` -- just the octree eids it names."""

        return {"eids": np.array(plan.eids, dtype=int)}

    def decodePlan(self, data: dict) -> "RefinementPlan":
        """Inverse of :meth:`encodePlan`."""

        return RefinementPlan(eids=[int(eid) for eid in data["eids"]])

    def restoreDecisionState(self, records) -> None:
        """Re-establish what the *next* decision needs, after a restart replay.

        Two things, neither of which touches the mesh:

        - the cutback guard, so the first post-resume call does not re-refine at a time this
          modifier already refined at;
        - the initial-marker latch, since a checkpoint only exists after an increment converged, so
          a resumed run is never truly making its first call.

        Notably absent: the pending marks. Those are re-derived by the next :meth:`plan` from the
        restored solution state -- which is exactly what the live run would have done -- so they need
        no checkpointing at all.
        """

        if records:
            self._lastRefinedTime = float(records[-1].time)
        self._isFirstCall = False
