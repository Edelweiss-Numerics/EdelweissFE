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
from edelweissfe.adaptivity.marking import RefineableElements
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
            "Minimum number of eligible elements one evaluation of the markers must mark for a "
            "refinement pass to happen; with fewer, nothing is refined and no equation system is "
            "rebuilt. Marks are not carried over to later evaluations: they are a function of the "
            "current state only. Individual markers may cap their own marks (e.g. "
            "'maxRefinedFraction') or expand them (e.g. 'halo') before this count is taken. Default 1 "
            "refines as soon as any element is marked."
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
#: parent element: the solution ``"U"``, and the dynamic state ``"V"`` and ``"A"``, which nothing else can
#: reproduce. The shape functions are a partition of unity, so a uniform velocity field (and its momentum)
#: is carried exactly. The implicit dynamic solver re-solves ``"A"`` from equilibrium on the next increment
#: (see :mod:`~edelweissfe.solvers.nonlinearimplicitdynamic`). Entries absent from a node field are skipped.
WARM_STARTED_NODE_FIELD_ENTRIES = ("U", "V", "A")


class ModelModifier(ModelModifierBase):
    #: Option schema for this model modifier, per OptionSchemaProvider.
    schema = HAdaptivitySchema

    #: Whether the step-start markers are still to be evaluated: they act on the first decision only.
    checkpointedState = {"_isFirstCall": bool}

    def __init__(self, name: str, model: FEModel, journal: Journal, *args, **kwargs):
        super().__init__(name, model, journal, *args, **kwargs)
        options = buildSchemaFromOptions(HAdaptivitySchema, kwargs)

        self._name = name
        self._model = model
        self._journal = journal

        # Markers are resolved by 'type' through the marker registry and built from their own >>marker
        # options, validated against that marker's own schema.
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
            # 'type' is the dispatch key and 'inputFile' is parser bookkeeping; neither is a marker option.
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
        # Diagnostics only; the record a restart replays is model.topology.history.
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

        # element -> its named properties (e.g. bulk viscosity), so children inherit them too: they are
        # assigned when the model is prepared, which a child created here never goes through.
        self._elementPropertiesOf = {}
        for elementProperty in model.elementProperties:
            for element in model.elementSets[elementProperty.elSetName]:
                self._elementPropertiesOf.setdefault(element, []).append(elementProperty)

        # restrict the octree mirror to the refineable solid elements (not e.g. contact-facet elements):
        # the given set, or else all 20-node (HEX20-family) elements.
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

        # Which elements this instance owns; TopologyPipeline.checkModelModifierDomains rejects overlaps
        # with other modifiers, since each maintains its own mirror and edits the model directly.
        self._refineElementNumbers = {el.elNumber for el in refineElements}

        # element type of the children: the one given, or else the parent's type (multi-material meshes)
        self._elementType = options.elementType
        self._elementClasses = {}

        # bodies of the refineable mesh: node labels are namespaced per body, so coincident nodes of
        # two bodies (tied interface, zero-gap contact, duplicated-node crack) are never merged
        componentOfElement = _connectedComponents(refineElements)

        # build the AdaptiveMesh mirror: the roots are the refineable elements with the model's own
        # connectivity and node labels; new node labels come from the model's own allocator.
        self._topology = Hex20Topology()
        self._mesh = AdaptiveMesh(
            splitFactor=self.splitFactor, topology=self._topology, reserve_labels=model.topology.reserveNodeNumbers
        )
        self._eidToEl = {}  # mesh element id -> live element
        #: Diagnostics only, parallel to _committedOccasions.
        self._committedOccasionEids = []
        for el in refineElements:
            componentId = componentOfElement[el]
            for n in el.nodes:
                self._mesh.registry.seed(n.label, n.coordinates, componentId)
            coords = np.array([n.coordinates for n in el.nodes])
            eid = self._mesh.add_root(coords, [n.label for n in el.nodes], componentId)
            self._eidToEl[eid] = el
        # what the markers see: a live view of the active elements, whose node adjacency (for a
        # marker's halo) is kept across planning passes until _materialize changes the mesh
        self._refineableElements = RefineableElements(self._eidToEl.values())
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
        # members of a tracked set the octree mirror does not know (e.g. HEX8, contact facets), which
        # _materialize would otherwise drop when it rebuilds the set from mesh element ids
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

        # Companion hanging-node MPC, registered FIRST: contested slave DOFs are resolved by model order,
        # and a hanging node on a tie's slave surface must be kept on its parent edge by this constraint
        # (its coarse parents are tie slaves themselves, so the tie still acts through them).
        self._hanging = HangingNodeConstraint(name + "_hanging", model)
        model.multiPointConstraints = {name + "_hanging": self._hanging, **model.multiPointConstraints}
        self._isFirstCall = True
        # parent-parametric coords of each child's nodes (used for warm-start interpolation)
        self._octantParams = self._topology.subdivision_children_param(self.splitFactor)

    def declaredDomain(self, model: FEModel) -> set:
        """The refineable roots this instance owns; see
        :meth:`~edelweissfe.modelmodifiers.base.modelmodifierbase.ModelModifierBase.declaredDomain`.
        Two hAdaptivity blocks over the same elements must be combined into one with several
        ``>>marker`` lines instead -- including ``initialOnly`` ones."""

        return self._refineElementNumbers

    def syncNodeCoordinates(self, coordinatesByLabel: dict) -> None:
        """Update this instance's own :class:`~edelweissfe.adaptivity.refinement.AdaptiveMesh`
        mirror's cached node coordinates after an EXTERNAL mutation of ``model.nodes`` -- e.g. a
        geometry-snap modifier (``edelweissfe.modelmodifiers.geometry.surfacesnap``) reprojecting
        AMR-created boundary nodes onto an analytical surface.

        Without this, the mirror silently keeps subdividing from its own pre-mutation coordinates:
        ``AdaptiveMesh.refine`` calls ``Hex20Topology.subdivide(e["coords"], ...)``, where
        ``e["coords"]`` was captured once, at the moment that element's own parent was refined --
        never re-read from ``model.nodes`` afterward. A LATER refinement of an already-corrected
        element would therefore generate its children from the geometry as it stood before the
        correction, silently discarding it. This is a reactive fix, called from the correcting
        modifier's own ``apply()`` (it has no way to reach into this mirror on its own); it does
        nothing for a label this mirror does not track.

        Parameters
        ----------
        coordinatesByLabel
            Mapping of node label to its new coordinate. Only labels this mirror actually knows
            about (its own registry, or the connectivity of one of its active elements) are
            touched.
        """

        reg = self._mesh.registry
        touched = set()
        for label, coord in coordinatesByLabel.items():
            if label in reg.coordinates:
                reg.coordinates[label] = np.asarray(coord, dtype=float)
                touched.add(label)
        if not touched:
            return
        # `AdaptiveMesh.refine` subdivides from the per-element `e["coords"]`, so resync those too.
        for eid in self._mesh.active():
            e = self._mesh.elements[eid]
            if touched.isdisjoint(e["conn"]):
                continue
            e["coords"] = np.array([reg.coordinates[label] for label in e["conn"]])

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
    def plan(self, model: FEModel, change, step) -> "RefinementPlan | None":
        """Evaluate the markers and decide which octree cells to refine. See
        :meth:`~edelweissfe.modelmodifiers.base.modelmodifierbase.ModelModifierBase.plan`.

        The decision is returned as octree eids rather than element numbers: eids are this
        modifier's own stable identifiers, reproducible by its replay, whereas element numbers are
        assigned by the model's allocator in an order that depends on what else minted.
        """

        # The markers are evaluated once per topology update, in its first round (change is None);
        # returning None in later rounds lets the pipeline settle.
        if change is not None:
            return None

        elForEid = {v: k for k, v in self._eidToEl.items()}
        marked_elements = set()

        if self._isFirstCall:
            initial_markers = [m for m in self.markers if m.initialOnly]
            for m in initial_markers:
                elements = m.mark(model, self._refineableElements, self._mesh)
                marked_elements.update(elements)

        # dynamic markers evaluate the converged solution, so they are skipped on the first call, which
        # runs before increment 1 is solved.
        if not self._isFirstCall:
            dynamic_markers = [m for m in self.markers if not m.initialOnly]
            for m in dynamic_markers:
                marked_elements.update(m.mark(model, self._refineableElements, self._mesh))

        self._isFirstCall = False

        if not marked_elements:
            return None

        # keep only active elements below maxLevel
        with timeit("marking filter"):
            eligible = [
                el
                for el in sorted(marked_elements, key=lambda e: e.elNumber)
                if el in elForEid and self._mesh.elements[elForEid[el]]["level"] < self.maxLevel
            ]

        if len(eligible) < self.minMarkedElements:
            if eligible:
                self._journal.message(
                    "AMR ModelModifier: {:} element(s) marked, fewer than minMarkedElements={:}; not refining".format(
                        len(eligible), self.minMarkedElements
                    ),
                    "hadaptivity",
                    1,
                )
            return None

        return RefinementPlan(eids=[elForEid[el] for el in eligible])

    @timeit("AMR")
    def apply(self, model: FEModel, plan: "RefinementPlan"):
        """Refine exactly the cells named by ``plan`` and materialize the resulting children: the
        octree split, 2:1 balance, hanging-node MPCs, element/node/set bookkeeping, and the
        :class:`ModelChange` notification.

        The topology is a pure function of the plan, which is what lets a live run and a restart
        replay share this method: given the same plan they produce byte-identical topology, element
        numbers included. It also transfers the solution state onto the new elements and nodes; in
        a replay that transfer acts on the initial state and is then overwritten by the checkpoint's
        state, so it costs time but cannot change the result. See
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
        # Captured now: the refined parents are popped from _eidToEl during materialisation.
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

        with timeit("conformity check"):
            # Raises if any node on an active element's boundary is neither one of its nodes nor a
            # hanging-node slave.
            self._mesh.check_conformity(records)

        with timeit("materialize"):
            change = self._materialize(model, records)

        self._hanging.setRecords(records)
        # The change is not announced here: the pipeline records it (TopologyPipeline.recordChange).
        self._journal.message(
            "AMR ModelModifier: marked {:}, refined -> active elements {:} -> {:}, {:} hanging nodes".format(
                len(markedEids), nBefore, len(self._mesh.active()), len(records)
            ),
            "hadaptivity",
            1,
        )
        self._committedOccasions.append(markedElementNumbers)
        self._committedOccasionEids.append(list(markedEids))
        return change

    def _makeElement(self, elementType, elNumber):
        """Instantiate a child element of the given type, resolving (and caching) its class once per type."""
        if elementType not in self._elementClasses:
            self._elementClasses[elementType] = getElementClass(elementType, self._provider)
        return self._elementClasses[elementType](elementType, elNumber)

    def _materialize(self, model: FEModel, records: dict):
        """Turn the refined octree into model elements, nodes, sets and fields.

        The steps, in order:

        1. snapshot the converged nodal values, for the warm start;
        2. create the new nodes;
        3. create the child elements, level by level, with state transfer and interpolated nodal
           values;
        4. remove the refined parents, and check that octree and model agree;
        5. update the surfaces and the node and element sets;
        6. resize the node fields and write the warm-start values.

        Everything here runs identically live and on restart replay: ``apply()`` is one code path.
        Element numbers come from the model's single allocator (:meth:`TopologyPipeline.reserveElementNumbers`);
        this modifier keeps no counter of its own, so labels claimed by other consumers in between
        can never collide with it.

        Parameters
        ----------
        model
            The model tree.
        records
            The hanging-node constraint records, by slave node label.

        Returns
        -------
        ModelChange
            The net change of this refinement.
        """

        mesh = self._mesh
        # the active elements are about to change; so does the adjacency a marker's halo grows over
        self._refineableElements.invalidateNodeAdjacency()
        oldValues = self._snapshotNodalValues(model)
        newNodes = self._createNewNodes(model)

        active = set(mesh.active())
        newValues = {key: {} for key in oldValues}  # interpolated values for new nodes, per (field, entry)
        levelChanges, newChildEids = self._createChildElements(model, active, newNodes, oldValues, newValues)

        # The new nodes and the removals ride on the LAST level's changeset, which holds no transient
        # intermediates, so the coalesce below cannot drop them.
        change = levelChanges[-1] if levelChanges else ModelChange(kind=ModelChangeType.REFINEMENT)
        change.addedNodes |= set(newNodes.keys())
        self._removeRefinedParents(model, active, change)
        if len(levelChanges) > 1:
            change = coalesce(levelChanges)
        self._checkOctreeMatchesModel(active)

        self._updateSurfaces(model, newChildEids, change)
        with timeit("sets & fields sync"):
            self._updateSets(model, records, newNodes, newChildEids, change)

        with timeit("fields resize & restore"):
            self._resizeNodeFieldsAndWarmStart(model, oldValues, newValues)

        # Separately timed: this relinks EVERY node's field variables.
        with timeit("relink field variables"):
            model._linkFieldVariableObjects(model.nodeSets["all"])
        return change

    def _snapshotNodalValues(self, model: FEModel) -> dict:
        """Copy the converged values of every warm-started node field entry, before the mesh changes.

        On the replay path this is dead work -- readRestart overwrites every node field right
        afterwards -- but a "skip this on replay" branch is exactly the kind of live/replay
        divergence that made a resumed run rebuild a different mesh.

        Returns
        -------
        dict
            ``{(fieldName, entryName): {node: value}}``.
        """

        oldValues = {}
        for fieldName, nodeField in model.nodeFields.items():
            for entryName in WARM_STARTED_NODE_FIELD_ENTRIES:
                if entryName in nodeField:
                    entryValues = np.asarray(nodeField[entryName])
                    oldValues[(fieldName, entryName)] = {
                        node: entryValues[nodeField._indicesOfNodesInArray[node]].copy() for node in nodeField.nodes
                    }
        return oldValues

    def _createNewNodes(self, model: FEModel) -> dict:
        """Create a model node for every octree node the model does not have yet.

        Returns
        -------
        dict
            The new nodes, by label.
        """

        newNodes = {}
        for label, coord in self._mesh.registry.coordinates.items():
            if label not in model.nodes:
                node = Node(label, np.asarray(coord, dtype=float))
                model.createNode(node)
                newNodes[label] = node
        return newNodes

    def _cellsToCreate(self, active: set) -> set:
        """Every octree cell that must become a model element but is not one yet.

        Usually that is exactly the children of the cells refined in this call. It is not always:
        2:1 balancing can split a cell it created earlier in the same call, leaving an active leaf
        whose parent is itself brand new. Walking each such leaf up to its nearest materialized
        ancestor collects those intermediate cells as well; they are created and removed again with
        the other refined parents, so however many levels a cascade went, every one of them is
        handled by the same "split a materialized parent into its children" code.
        """

        mesh = self._mesh
        pending = set()
        for eid in active - set(self._eidToEl):
            ancestor = eid
            while ancestor is not None and ancestor not in self._eidToEl and ancestor not in pending:
                pending.add(ancestor)
                ancestor = mesh.elements[ancestor]["parent"]
        return pending

    def _createChildElements(
        self, model: FEModel, active: set, newNodes: dict, oldValues: dict, newValues: dict
    ) -> tuple[list, set]:
        """Create the child elements of every refined parent, one octree level at a time.

        One changeset per level, coalesced by the caller: the merge is what resolves an
        intermediate's create-then-remove into the direct parent -> grandchild relation a consumer
        needs, rather than leaving a phantom element in both the added and the removed set (see
        :meth:`ModelChange.mergedWith`).

        Parameters
        ----------
        model
            The model tree.
        active
            The active octree cells.
        newNodes
            The nodes created in this refinement, by label.
        oldValues
            The snapshot of the converged nodal values, see :meth:`_snapshotNodalValues`.
        newValues
            Filled with the values interpolated at the new nodes, keyed like ``oldValues``.

        Returns
        -------
        tuple[list, set]
            The changeset of every level, and the octree ids of all cells created.
        """

        mesh = self._mesh
        pending = self._cellsToCreate(active)
        levelChanges = []
        newChildEids = set()
        while pending:
            # Sorted, with the whole level's numbers reserved up front, so element numbering is a pure
            # function of the eids (reproducible on restart replay).
            levelEids = sorted(eid for eid in pending if mesh.elements[eid]["parent"] in self._eidToEl)
            if not levelEids:
                raise TopologyError(
                    "AMR: {:} active octree cell(s) (e.g. {:}) have no materialised ancestor, so "
                    "they cannot be turned into elements. The octree mirror and the model would "
                    "disagree about which elements exist.".format(len(pending), sorted(pending)[0])
                )
            change = ModelChange(kind=ModelChangeType.REFINEMENT)
            childNumbers = model.topology.reserveElementNumbers(len(levelEids))
            with timeit("elements & state transfer"):
                for eid, elNumber in zip(levelEids, childNumbers):
                    self._createChildElement(model, eid, elNumber, newNodes, oldValues, newValues, change)
            self._recordFaceMap(levelEids, change)

            levelChanges.append(change)
            newChildEids |= set(levelEids)
            pending -= set(levelEids)
        return levelChanges, newChildEids

    def _createChildElement(
        self,
        model: FEModel,
        eid: int,
        elNumber: int,
        newNodes: dict,
        oldValues: dict,
        newValues: dict,
        change: ModelChange,
    ):
        """Create the model element of one octree child cell, inheriting section, properties and
        state from its parent, and interpolate the nodal values at its new nodes."""

        mesh = self._mesh
        e = mesh.elements[eid]
        parentEid = e["parent"]
        parentEl = self._eidToEl[parentEid]
        child = self._makeElement(self._elementType or parentEl.elType, elNumber)
        child.setNodes([model.nodes[label] for label in e["conn"]])
        self._sectionOf[parentEl].assignSectionToElement(child, model)
        for elementProperty in self._elementPropertiesOf.get(parentEl, ()):
            child.assignProperty(elementProperty.propertyName, elementProperty.values)
        self._stateTransfer.transferState(parentEl, [child], self._topology)

        # warm start: interpolate each NEW node's field values from the parent via the HEX20
        # isoparametric map, so the increment restarts from a consistent state, not zero
        octant = mesh.elements[parentEid]["children"].index(eid)
        childParams = self._octantParams[octant]
        for i, label in enumerate(e["conn"]):
            node = model.nodes[label]
            if label in newNodes and any(node not in newValues[f] for f in oldValues):
                N = self._topology.shape_functions(*childParams[i])
                for key, vals in oldValues.items():
                    # An intermediate parent's new nodes are not in the snapshot; the level above
                    # interpolated them.
                    interpolated = newValues[key]
                    parentVals = [vals[pn] if pn in vals else interpolated.get(pn) for pn in parentEl.nodes]
                    if all(v is not None for v in parentVals):
                        newValues[key][node] = N @ np.array(parentVals)

        model.createElement(child)
        self._eidToEl[eid] = child
        self._sectionOf[child] = self._sectionOf[parentEl]
        if parentEl in self._elementPropertiesOf:
            self._elementPropertiesOf[child] = self._elementPropertiesOf[parentEl]

        change.addedElements.add(child.elNumber)
        change.parentToChildren.setdefault(parentEl.elNumber, []).append(child.elNumber)

    def _recordFaceMap(self, levelEids: list, change: ModelChange):
        """Record which child faces tile each parent face, while the parents still exist."""

        mesh = self._mesh
        for parentEid in {mesh.elements[eid]["parent"] for eid in levelEids}:
            parentLabel = self._eidToEl[parentEid].elNumber
            childEids = mesh.elements[parentEid]["children"]
            for faceID, faceIndex in self._topology.faceid_to_face.items():
                childLabels = [
                    self._eidToEl[childEids[j]].elNumber
                    for j in self._topology.face_child_indices(faceIndex, self.splitFactor)
                ]
                change.faceMap[(parentLabel, faceID)] = [(label, faceID) for label in childLabels]

    def _removeRefinedParents(self, model: FEModel, active: set, change: ModelChange):
        """Remove every element whose octree cell is no longer active: the refined parents,
        transient intermediates included (sorted, so the changeset is built in a reproducible
        order)."""

        for eid in sorted(set(self._eidToEl) - active):
            el = self._eidToEl.pop(eid)
            model.removeElement(el.elNumber)
            change.removedElements.add(el.elNumber)

    def _checkOctreeMatchesModel(self, active: set):
        """Raise unless every active octree cell has exactly one model element, and vice versa.

        The octree mirror decides which elements exist; if the model no longer agrees, every
        consumer downstream is reading a mesh that is not the one being refined.
        """

        if set(self._eidToEl) != active:
            raise TopologyError(
                "AMR: the octree mirror and the model disagree after materialisation -- {:} active "
                "cell(s) without an element, {:} element(s) without an active cell".format(
                    len(active - set(self._eidToEl)), len(set(self._eidToEl) - active)
                )
            )

    def _updateSurfaces(self, model: FEModel, newChildEids: set, change: ModelChange):
        """Replace each refined parent face in ``model.surfaces`` by its child faces."""

        for surfaceName, pairs in self._mesh.surfaces.items():
            if surfaceName in model.surfaces:
                if any(meid in newChildEids for meid, _ in pairs):
                    change.changedSurfaces.add(surfaceName)
                byFace = defaultdict(list)
                # Sorted: contact/tie facet generators number facet elements in this member order.
                for meid, faceID in sorted(pairs):
                    if meid in self._eidToEl:
                        byFace[faceID].append(self._eidToEl[meid])
                model.surfaces[surfaceName].replaceData({f: els for f, els in byFace.items()})

    def _updateSets(self, model: FEModel, records: dict, newNodes: dict, newChildEids: set, change: ModelChange):
        """Add the new nodes and elements to the node and element sets they belong to."""

        mesh = self._mesh
        # Tracked (non-all) node sets that gain nodes are rebuilt with the new members (excluding
        # hanging slave nodes, whose motion is set by the MPC).
        slaves = set(records.keys())
        for setName, labels in mesh.nodeSets.items():
            present = {n.label for n in model.nodeSets[setName].nodes}
            if any(label not in present and label not in slaves for label in labels):
                members = [model.nodes[label] for label in sorted(labels) if label not in slaves]
                model.nodeSets[setName].replaceMembers(members)
                change.changedNodeSets.add(setName)

        allNodes = list(model.nodes.values())
        if newNodes:
            for setName in self._allLikeSets | {"all"}:
                model.nodeSets[setName].replaceMembers(allNodes)
                change.changedNodeSets.add(setName)
        for setName, eids in mesh.elementSets.items():
            if setName in model.elementSets:
                if eids & newChildEids:
                    change.changedElementSets.add(setName)
                # sorted, as for the surfaces: downstream generators number entities by member order
                elements = [self._eidToEl[eid] for eid in sorted(eids) if eid in self._eidToEl]
                # carry the non-mirrored members along, minus those deleted from the model meanwhile
                elements += [el for el in self._untrackedOfElementSet[setName] if el.elNumber in model.elements]
                model.elementSets[setName].replaceMembers(elements)
        model.elementSets["all"].replaceMembers(list(model.elements.values()))
        change.changedElementSets.add("all")

    def _resizeNodeFieldsAndWarmStart(self, model: FEModel, oldValues: dict, newValues: dict):
        """Resize the node fields to the new nodes, then write the warm-start values: the
        converged value on a retained node, the interpolated value on a new one.

        ``U`` is also written to ``P`` (the previous converged solution), so the first Newton
        iteration after refinement sees a normal residual rather than a spurious ``dU = U - P = U``
        on every node. The other warm-started entries (``V``, ``A``, present only for the dynamic
        solvers) get just their own value.

        On the replay path the model postpones the resize to the end of the replay window -- see
        :meth:`TopologyPipeline.changes` -- so the writes below are then dead work, overwritten by
        readRestart. The decision lives in the model: this method issues the same calls either way.
        """

        model._resizeNodeFieldsForNodes(self._journal)
        for fieldName, nodeField in model.nodeFields.items():
            if "U" not in nodeField:
                nodeField.createFieldValueEntry("U")
            if "P" not in nodeField:
                nodeField.createFieldValueEntry("P")
            for entryName in WARM_STARTED_NODE_FIELD_ENTRIES:
                if entryName not in nodeField:
                    continue
                targets = [nodeField["U"], nodeField["P"]] if entryName == "U" else [nodeField[entryName]]
                old = oldValues.get((fieldName, entryName), {})
                new = newValues.get((fieldName, entryName), {})
                for node in nodeField.nodes:
                    value = old[node] if node in old else new.get(node)
                    if value is None:
                        continue
                    idx = nodeField._indicesOfNodesInArray[node]
                    for target in targets:
                        target[idx] = value

    def encodePlan(self, plan: "RefinementPlan") -> dict:
        """Serialize a :class:`RefinementPlan` -- just the octree eids it names."""

        return {"eids": np.array(plan.eids, dtype=int)}

    def decodePlan(self, data: dict) -> "RefinementPlan":
        """Inverse of :meth:`encodePlan`."""

        return RefinementPlan(eids=[int(eid) for eid in data["eids"]])
