#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  ---------------------------------------------------------------------
#  EdelweissFE  -- adaptivity manager (WS-H)
#  Unit of Strength of Materials and Structural Analysis, University of Innsbruck
#  Matthias Neuner matthias.neuner@uibk.ac.at
#  ---------------------------------------------------------------------

"""Dynamic h-adaptivity manager for HEX20 hanging-node AMR.

Realized as a zero-DOF constraint so it plugs into the existing per-increment ``updateConnectivity``
hook: at the start of every increment it (WS-G) marks elements from the converged state, (WS-B/C)
refines them with 2:1 balance in an :class:`AdaptiveMesh` mirror, mutates the live ``FEModel`` (new
Marmot child elements + nodes, parents removed), (WS-F) transfers quadrature-point state to the
children, and (MPC) regenerates the flattened hanging-node records on the companion ``hangingnode``
multi-point constraint. Returning ``True`` makes the solver rebuild the DOF system and the MPC
condensation.
"""

from collections import defaultdict

import numpy as np

from edelweissfe.adaptivity.hex20topology import hex20_shape, octant_children_param
from edelweissfe.adaptivity.marking import markElements
from edelweissfe.adaptivity.refinement import AdaptiveMesh
from edelweissfe.adaptivity.statetransfer import transferStateNearestQp
from edelweissfe.config.elementlibrary import getElementClass
from edelweissfe.constraints.base.constraintbase import ConstraintBase
from edelweissfe.constraints.hangingnode import Constraint as HangingNodeConstraint
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.models.modelchangeobserver import ModelChangeType
from edelweissfe.points.node import Node
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.surfaces.entitybasedsurface import EntityBasedSurface
from edelweissfe.utils.caseinsensitivedict import CaseInsensitiveDict
from edelweissfe.utils.inputlanguage import InputLanguage, Module
from edelweissfe.utils.misc import (
    caseInsensitiveKwargsChecker,
    castKwargsValuesAndAddDefaults,
)

module = Module("adaptivity", "Dynamic hanging-node h-adaptivity manager for HEX20 elements.")
inputLanguage = InputLanguage()
keyword = "constraint"
if keyword in inputLanguage:
    inputLanguage[keyword].addModule(module)
module.addRequiredArg("result", "Quadrature-point result to mark on (e.g. 'stress', 'nonlocal damage').", str)
module.addRequiredArg("expression", "Boolean expression in x (the reduced per-element scalar), e.g. 'x > 0.1'.", str)
module.addOptionalArg("reducer", "Reduction over components/QPs: absmax|max|min|mean.", str, "absmax")
module.addOptionalArg(
    "elSet",
    "Restrict marking to this element set (e.g. the elements that carry the "
    "marked result). Others are never marked/refined.",
    str,
    None,
)
module.addOptionalArg("maxLevel", "Maximum refinement level.", int, 1)
module.addOptionalArg("elementType", "Element type to instantiate for children (default: like parents).", str, None)
module.addOptionalArg("elementProvider", "Element provider.", str, "marmot")
documentation = [module]


class Constraint(ConstraintBase):
    @caseInsensitiveKwargsChecker([kw.name for kw in module.requiredArgs], [kw.name for kw in module.optionalArgs])
    @castKwargsValuesAndAddDefaults(module)
    def __init__(self, name: str, model: FEModel, *args, **kwargs):
        super().__init__(name, model, *args, **kwargs)
        kwargs = CaseInsensitiveDict(kwargs)

        self._name = name
        self._model = model
        self._journal = Journal()
        self.result = kwargs["result"]
        self.expression = kwargs["expression"]
        self.reducer = kwargs["reducer"]
        self.maxLevel = kwargs["maxLevel"]
        # restrict marking to a given element set (its labels); others are never refined
        self._markLabels = {el.elNumber for el in model.elementSets[kwargs["elSet"]]} if kwargs["elSet"] else None
        self._provider = kwargs["elementProvider"]
        # element -> its section, so children inherit the parent's material (multi-material meshes)
        self._sectionOf = {}
        for section in model.sections.values():
            for elementSet in section.elSets:
                for element in elementSet:
                    self._sectionOf[element] = section

        # element type: infer from an existing element if not given
        anyEl = next(iter(model.elements.values()))
        self._elementType = kwargs["elementType"] or anyEl.elType
        self._elementClass = getElementClass(self._elementType, self._provider)
        self._nextElLabel = max(model.elements.keys()) + 1

        # build the AdaptiveMesh mirror, sharing node labels with the live model
        self._mesh = AdaptiveMesh()
        for label, node in model.nodes.items():
            self._mesh.registry.seed(label, node.coordinates)
        self._eidToEl = {}  # mesh element id -> live element
        for label, el in model.elements.items():
            coords = np.array([n.coordinates for n in el.nodes])
            eid = self._mesh.add_root(coords)
            self._eidToEl[eid] = el

        # all-encompassing sets (contain every node, e.g. 'all', 'ALLNODES') are not boundary BCs --
        # they just gain every new node; rebuild them wholesale, don't guard/track them
        allLabels = set(model.nodes.keys())
        self._allLikeSets = {name for name, ns in model.nodeSets.items() if {n.label for n in ns.nodes} == allLabels}
        # track the remaining (boundary) node sets so real BCs gain new boundary nodes on refinement
        for setName, nodeSet in model.nodeSets.items():
            if setName not in self._allLikeSets:
                self._mesh.define_node_set(setName, [n.label for n in nodeSet])

        # track element sets so user element sets propagate child elements on refinement (Finding 1)
        elToEid = {el: eid for eid, el in self._eidToEl.items()}
        for setName, elementSet in model.elementSets.items():
            eids = [elToEid[el] for el in elementSet if el in elToEid]
            self._mesh.define_element_set(setName, eids)

        # track element-based surfaces so surface loads stay consistent under refinement (Finding 2)
        for surfaceName, surface in model.surfaces.items():
            pairs = [
                (elToEid[el], faceID) for faceID, elementSet in surface.items() for el in elementSet if el in elToEid
            ]
            if pairs:
                self._mesh.define_surface(surfaceName, pairs)

        # companion hanging-node MPC (records set in memory), registered as a multi-point constraint
        self._hanging = HangingNodeConstraint(name + "_hanging", model)
        model.multiPointConstraints[name + "_hanging"] = self._hanging
        self._converged = False  # set True once an increment has converged (via acceptLastState)
        self._octantParams = octant_children_param()  # parent-parametric coords of each child's nodes

    # ---- zero-DOF ConstraintBase interface ----
    @property
    def nodes(self):
        return []

    @property
    def fieldsOnNodes(self):
        return []

    @property
    def nDof(self):
        return 0

    def getNumberOfAdditionalNeededScalarVariables(self):
        return 0

    def applyConstraint(self, U_np, dU, PExt, K, timeStep):
        pass

    def acceptLastState(self):
        self._converged = True

    # ---- the adaptation ----
    def updateConnectivity(self, model: FEModel) -> bool:
        # only adapt once a non-trivial solution has developed; the hook is called on increments
        # whose converged state is not yet written back (all-zero), and marking on a zero state is
        # meaningless. This also makes AMR follow the *developed* field, as intended.
        stateMagnitude = max(
            (float(np.abs(np.asarray(nf["U"])).max()) for nf in model.nodeFields.values() if "U" in nf),
            default=0.0,
        )
        if stateMagnitude < 1e-12:
            return False

        elForEid = {v: k for k, v in self._eidToEl.items()}

        # (WS-G) mark, keeping only active elements below maxLevel
        marked = markElements(model, self.result, self.expression, self.reducer, elementLabels=self._markLabels)
        markedEids = [
            elForEid[model.elements[label]]
            for label in marked
            if label in model.elements
            and model.elements[label] in elForEid
            and self._mesh.elements[elForEid[model.elements[label]]]["level"] < self.maxLevel
        ]
        if not markedEids:
            return False

        # (WS-B/C) refine + 2:1 balance in the mirror
        nBefore = len(self._mesh.active())
        for eid in markedEids:
            if self._mesh.elements[eid]["active"]:
                self._mesh.refine(eid)
        self._mesh.balance_2to1()

        records = self._mesh.hanging_mpc_records()  # computed once (expensive), reused below
        self._materialize(model, records)
        self._hanging.setRecords(records)
        # notify observers (e.g. Dirichlet BCs) so they re-index against the mutated mesh (Finding 5)
        model.notifyModelChanged(ModelChangeType.REFINEMENT)
        self._journal.message(
            "AMR: marked {:}, refined -> active elements {:} -> {:}, {:} hanging nodes".format(
                len(markedEids), nBefore, len(self._mesh.active()), len(records)
            ),
            "adaptivity",
            0,
        )
        return True

    def _materialize(self, model: FEModel, records: dict):
        mesh = self._mesh
        reg = mesh.registry

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

        # new child elements (single level of new refinement per call -> parents are materialized)
        for eid in active - materialized:
            e = mesh.elements[eid]
            parentEid = e["parent"]
            parentEl = self._eidToEl[parentEid]
            child = self._elementClass(self._elementType, self._nextElLabel)
            self._nextElLabel += 1
            child.setNodes([model.nodes[label] for label in e["conn"]])
            self._sectionOf[parentEl].assignSectionPropertiesToElement(child)  # init + material (inherit parent's)
            transferStateNearestQp(parentEl, [child])  # WS-F (state)

            # warm start (WS-H): interpolate each NEW node's field values from the parent via the
            # HEX20 isoparametric map, so the increment restarts from a consistent state, not zero
            octant = mesh.elements[parentEid]["children"].index(eid)
            childParams = self._octantParams[octant]
            for i, label in enumerate(e["conn"]):
                node = model.nodes[label]
                if label in newNodes and any(node not in newValues[f] for f in oldValues):
                    N = hex20_shape(*childParams[i])
                    for fieldName, vals in oldValues.items():
                        if all(pn in vals for pn in parentEl.nodes):
                            parentVals = np.array([vals[pn] for pn in parentEl.nodes])
                            newValues[fieldName][node] = N @ parentVals

            model.elements[child.elNumber] = child
            self._eidToEl[eid] = child
            self._sectionOf[child] = self._sectionOf[parentEl]

        # remove refined parents
        for eid in materialized - active:
            el = self._eidToEl.pop(eid)
            del model.elements[el.elNumber]

        # keep model.surfaces in sync (Finding 2): parent (eid,faceID) -> child faces
        for surfaceName, pairs in mesh.surfaces.items():
            if surfaceName in model.surfaces:
                byFace = defaultdict(list)
                for meid, faceID in pairs:
                    if meid in self._eidToEl:
                        byFace[faceID].append(self._eidToEl[meid])
                model.surfaces[surfaceName] = EntityBasedSurface(surfaceName, {f: els for f, els in byFace.items()})

        # Tracked (non-all) node sets that gain nodes are rebuilt with the new members (excluding
        # hanging slave nodes, whose motion is set by the MPC). This is safe for any set NOT cached
        # by a step action; a Dirichlet-referenced set only grows if refinement reaches that boundary
        # face, in which case its step action would need re-pointing (not yet supported -- but
        # interior / free-surface refinement, e.g. a reentrant corner, never touches a Dirichlet BC).
        slaves = set(records.keys())
        for setName, labels in mesh.nodeSets.items():
            present = {n.label for n in model.nodeSets[setName].nodes}
            if any(label not in present and label not in slaves for label in labels):
                members = [model.nodes[label] for label in sorted(labels) if label not in slaves]
                model.nodeSets[setName] = NodeSet(setName, members)

        # sync all element sets (user sets like 'concrete' and all-encompassing sets) -- Finding 1
        from edelweissfe.sets.elementset import ElementSet

        allNodes = list(model.nodes.values())
        for setName in self._allLikeSets | {"all"}:
            model.nodeSets[setName] = NodeSet(setName, allNodes)
        for setName, eids in mesh.elementSets.items():
            if setName in model.elementSets:
                elements = [self._eidToEl[eid] for eid in eids if eid in self._eidToEl]
                model.elementSets[setName] = ElementSet(setName, elements)
        model.elementSets["all"] = ElementSet("all", list(model.elements.values()))

        # rebuild node fields to include the new nodes, then restore the warm start: converged values
        # on the retained nodes and interpolated values on the new nodes (Finding 1)
        model._prepareVariablesAndFields(self._journal)
        for fieldName, nodeField in model.nodeFields.items():
            if "U" not in nodeField:
                nodeField.createFieldValueEntry("U")
            U = nodeField["U"]
            old = oldValues.get(fieldName, {})
            new = newValues.get(fieldName, {})
            for node in nodeField.nodes:
                idx = nodeField._indicesOfNodesInArray[node]
                if node in old:
                    U[idx] = old[node]
                elif node in new:
                    U[idx] = new[node]
