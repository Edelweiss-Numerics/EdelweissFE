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
from edelweissfe.models.modelchange import ModelChange
from edelweissfe.models.modelchangeobserver import ModelChangeType
from edelweissfe.points.node import Node
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

        # two hAdaptivity instances cannot independently own overlapping elements: each maintains
        # its own AdaptiveMesh mirror and materializes/deletes elements directly in the model, so a
        # second instance refining/removing an element the first still tracks leaves the first with
        # a stale reference (an Element object no longer in model.elements) -- which later corrupts
        # element-set membership (a "deleted" element gets carried back into e.g. 'fixed_all') and
        # can surface as a node simultaneously Dirichlet-prescribed and a hanging-node MPC slave.
        # Fail loud at construction time instead of silently corrupting state deep in the solve loop.
        refineElementNumbers = {el.elNumber for el in refineElements}
        for otherName, otherModifier in model.modelModifiers.items():
            if isinstance(otherModifier, ModelModifier):
                overlap = refineElementNumbers & otherModifier._refineElementNumbers
                if overlap:
                    raise ValueError(
                        f"hAdaptivity modifier {name!r} and existing modifier {otherName!r} both "
                        f"claim {len(overlap)} of the same element(s) (e.g. label "
                        f"{sorted(overlap)[0]}) as refineable roots via overlapping 'refineElSet'/"
                        "'elSet' (or no restriction at all). Combine all markers -- including "
                        "'initialOnly' ones -- into a single hAdaptivity block via multiple "
                        "'>>marker' lines instead of stacking separate modifiers over the same "
                        "elements."
                    )
        self._refineElementNumbers = refineElementNumbers

        # element type: infer from a refineable element if not given
        anyEl = refineElements[0]
        self._elementType = options.elementType or anyEl.elType
        self._elementClass = getElementClass(self._elementType, self._provider)
        self._nextElLabel = max(model.elements.keys()) + 1

        # bodies of the refineable mesh: node labels are namespaced per body, so coincident nodes of
        # two bodies (a tied interface -- 'adjust' makes it flush by default --, a zero-gap contact
        # pair, a duplicated-node crack plane) are never deduplicated into one label
        componentOfElement = _connectedComponents(refineElements)

        # build the AdaptiveMesh mirror, sharing node labels with the live model. Only the nodes of
        # the refineable elements are seeded: a node the octree does not own must not be able to
        # claim a coordinate key, and only an octree-owned node can be seeded with a body.
        self._topology = Hex20Topology()
        self._mesh = AdaptiveMesh(splitFactor=self.splitFactor, topology=self._topology)
        self._eidToEl = {}  # mesh element id -> live element
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

    @timeit("AMR")
    def updateModel(self, model: FEModel, step, timeStep: float) -> bool:
        # Do not re-refine if the solver is re-trying the exact same time state after a cutback
        if self._lastRefinedTime is not None and abs(model.time - self._lastRefinedTime) < 1e-12:
            return False

        elForEid = {v: k for k, v in self._eidToEl.items()}
        marked_elements = set()

        if self._isFirstCall:
            initial_markers = [m for m in self.markers if m.initialOnly]
            for m in initial_markers:
                elements = m.mark(model, self._eidToEl.values(), self._mesh)
                marked_elements.update(elements)

        # dynamic markers (not initialOnly) evaluate the converged solution, so they need at least
        # one increment to have actually converged -- on the very first call, updateModel() runs
        # before increment 1 is solved and model fields still hold the pre-solve initial condition,
        # which is meaningless to mark on regardless of which field a given marker evaluates.
        # Gating on displacement magnitude instead would wrongly skip markers that evaluate other
        # fields (stress, strain, ...) whenever displacement itself happens to stay tiny.
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
            return False

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
            return False

        markedEids = [elForEid[el] for el in eligible]
        self._pendingMarkedElements = set()

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
        # notify observers (e.g. Dirichlet BCs, Ensight output manager) so they re-index against the mutated mesh
        with timeit("notify observers"):
            model.notifyModelChanged(ModelChangeType.REFINEMENT, change)
        self._journal.message(
            "AMR ModelModifier: marked {:}, refined -> active elements {:} -> {:}, {:} hanging nodes".format(
                len(markedEids), nBefore, len(self._mesh.active()), len(records)
            ),
            "hadaptivity",
            0,
        )
        self._lastRefinedTime = float(model.time)
        return True

    def _materialize(self, model: FEModel, records: dict):
        mesh = self._mesh
        reg = mesh.registry

        # Resync against the model's current label range before claiming any new ones. Other
        # components can legitimately claim element labels between two refinements -- notably a
        # tied surface's facets, rebuilt via the observer/MeshDependent escape hatches fired at the
        # end of THIS very call (see below), which pick their labels fresh from max(model.elements).
        # self._nextElLabel is otherwise a plain running counter that would stay oblivious to that
        # and, on the next call, collide with (and silently overwrite) those facets -- which then
        # get erroneously deleted as "stale" the next time they are rebuilt, orphaning the solid
        # elements (and their nodes) that stole their labels. Only ever advance the counter.
        self._nextElLabel = max(self._nextElLabel, max(model.elements.keys(), default=0) + 1)

        # snapshot the converged nodal values BEFORE the mesh mutates, for the warm start
        oldValues = {}
        for fieldName, nodeField in model.nodeFields.items():
            if "U" in nodeField:
                U = np.asarray(nodeField["U"])
                oldValues[fieldName] = {
                    node: U[nodeField._indicesOfNodesInArray[node]].copy() for node in nodeField.nodes
                }

        # new nodes
        newNodes = {}
        for label, coord in reg.coordinates.items():
            if label not in model.nodes:
                node = Node(label, np.asarray(coord, dtype=float))
                model.nodes[label] = node
                newNodes[label] = node

        active = set(mesh.active())
        materialized = set(self._eidToEl.keys())
        newValues = {fieldName: {} for fieldName in oldValues}  # interpolated values for new nodes
        newChildEids = active - materialized

        # the changeset this call produces (its faceMap/*Sets entries reflect the tracked element
        # sets and surfaces above)
        change = ModelChange(kind=ModelChangeType.REFINEMENT, addedNodes=set(newNodes.keys()))

        # new child elements (single level of new refinement per call -> parents are materialized)
        # Iterate SORTED: element labels are handed out in this order, so an unordered set here would
        # make which child gets which label depend on set iteration order rather than on the mesh.
        with timeit("elements & state transfer"):
            for eid in sorted(newChildEids):
                e = mesh.elements[eid]
                parentEid = e["parent"]
                parentEl = self._eidToEl[parentEid]
                child = self._elementClass(self._elementType, self._nextElLabel)
                self._nextElLabel += 1
                child.setNodes([model.nodes[label] for label in e["conn"]])
                self._sectionOf[parentEl].assignSectionPropertiesToElement(child)
                self._stateTransfer.transferState(parentEl, [child], self._topology)  # state transfer

                # warm start: interpolate each NEW node's field values from the parent via the
                # HEX20 isoparametric map, so the increment restarts from a consistent state, not zero
                octant = mesh.elements[parentEid]["children"].index(eid)
                childParams = self._octantParams[octant]
                for i, label in enumerate(e["conn"]):
                    node = model.nodes[label]
                    if label in newNodes and any(node not in newValues[f] for f in oldValues):
                        N = self._topology.shape_functions(*childParams[i])
                        for fieldName, vals in oldValues.items():
                            if all(pn in vals for pn in parentEl.nodes):
                                parentVals = np.array([vals[pn] for pn in parentEl.nodes])
                                newValues[fieldName][node] = N @ parentVals

                model.elements[child.elNumber] = child
                self._eidToEl[eid] = child
                self._sectionOf[child] = self._sectionOf[parentEl]

                change.addedElements.add(child.elNumber)
                change.parentToChildren.setdefault(parentEl.elNumber, []).append(child.elNumber)

        # per-face parent -> child tiling (the faceMap), while parents are still materialized
        newlyRefinedParentEids = {mesh.elements[eid]["parent"] for eid in newChildEids}
        for parentEid in newlyRefinedParentEids:
            parentLabel = self._eidToEl[parentEid].elNumber
            childEids = mesh.elements[parentEid]["children"]
            for faceID, faceIndex in self._topology.faceid_to_face.items():
                childLabels = [
                    self._eidToEl[childEids[j]].elNumber
                    for j in self._topology.face_child_indices(faceIndex, self.splitFactor)
                ]
                change.faceMap[(parentLabel, faceID)] = [(label, faceID) for label in childLabels]

        # remove refined parents
        for eid in materialized - active:
            el = self._eidToEl.pop(eid)
            del model.elements[el.elNumber]
            change.removedElements.add(el.elNumber)

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
                old = oldValues.get(fieldName, {})
                new = newValues.get(fieldName, {})
                for node in nodeField.nodes:
                    idx = nodeField._indicesOfNodesInArray[node]
                    if node in old:
                        U[idx] = old[node]
                        P[idx] = old[node]
                    elif node in new:
                        U[idx] = new[node]
                        P[idx] = new[node]

            model._linkFieldVariableObjects(model.nodeSets["all"])
        return change
