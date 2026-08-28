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

"""The element number allocator and the topology mutation window.

These pin the two properties the restart replay is built on: element numbering is a pure
function of the ordered *creation* sequence -- never of the deletion history, never of
``max(model.elements)`` -- and entities may only be created or deleted inside a topology change.
"""

from pathlib import Path as _Path

import numpy as np
import pytest

from edelweissfe.adaptivity.refinement import NodeRegistry
from edelweissfe.modelmodifiers.surfacefacets.surfacefacets import (
    ModelModifier as _SurfaceFacetsModifier,
)
from edelweissfe.models.femodel import FEModel
from edelweissfe.models.meshdependent import MeshDependent
from edelweissfe.models.modelchange import ModelChange
from edelweissfe.models.modelchangeobserver import ModelChangeType as _MCT
from edelweissfe.solvers.base.nonlinearsolverbase import NonlinearSolverBase
from edelweissfe.utils.exceptions import TopologyError

_REPO_ROOT = _Path(__file__).resolve().parents[1]


class _StubElement:
    """A bare element stand-in: the allocator, the window and the fingerprint only look at
    ``elNumber``, ``elType`` and ``nodes``."""

    elType = "STUB"
    nodes = ()

    def __init__(self, elNumber: int):
        self.elNumber = elNumber


def _modelWithSetupElements(*labels: int) -> FEModel:
    """A model carrying elements as the base mesh generators leave them -- placed directly, with
    the allocator not yet raised above them."""

    model = FEModel(3)
    for label in labels:
        model.elements[label] = _StubElement(label)
    return model


def test_reserved_numbers_are_consecutive_and_monotonic():
    model = _modelWithSetupElements()
    with model.topologyChanges():
        first = model.reserveElementNumbers(3)
        second = model.reserveElementNumbers(2)

    assert list(first) == [1, 2, 3]
    assert list(second) == [4, 5]


def test_numbers_are_never_recycled_after_removal():
    """The heart of the allocator: a rebuilt contact facet set must not be handed the numbers of the
    facets it just replaced, or numbering would depend on the deletion history and a replay would
    have to reproduce that too."""

    model = _modelWithSetupElements()
    with model.topologyChanges():
        (number,) = model.reserveElementNumbers(1)
        model.createElement(_StubElement(number))
        model.removeElement(number)

        (afterRemoval,) = model.reserveElementNumbers(1)

    assert afterRemoval != number
    assert afterRemoval == number + 1


def test_allocator_ignores_the_current_maximum():
    """max(model.elements) drops when the highest-numbered elements are deleted; the allocator must
    not follow it down."""

    model = _modelWithSetupElements()
    with model.topologyChanges():
        numbers = model.reserveElementNumbers(4)
        for number in numbers:
            model.createElement(_StubElement(number))
        for number in numbers:
            model.removeElement(number)

        assert not model.elements  # max(model.elements, default=0) + 1 would restart at 1 here
        (afterEmptying,) = model.reserveElementNumbers(1)

    assert afterEmptying == numbers[-1] + 1


def test_adopt_setup_element_numbers_clears_the_base_mesh():
    model = _modelWithSetupElements(1, 2, 17)
    model.adoptSetupElementNumbers()

    with model.topologyChanges():
        (number,) = model.reserveElementNumbers(1)

    assert number == 18


def test_adopt_setup_element_numbers_never_lowers_the_mark():
    model = _modelWithSetupElements(1, 2)
    with model.topologyChanges():
        model.reserveElementNumbers(50)

    model.adoptSetupElementNumbers()

    with model.topologyChanges():
        (number,) = model.reserveElementNumbers(1)

    assert number == 51


def test_reserve_outside_a_topology_change_raises():
    model = _modelWithSetupElements()
    with pytest.raises(TopologyError, match="topology change"):
        model.reserveElementNumbers(1)


def test_create_outside_a_topology_change_raises():
    model = _modelWithSetupElements()
    with pytest.raises(TopologyError, match="outside a topology change"):
        model.createElement(_StubElement(1))


def test_remove_outside_a_topology_change_raises():
    model = _modelWithSetupElements(1)
    with pytest.raises(TopologyError, match="outside a topology change"):
        model.removeElement(1)


def test_creating_a_taken_number_raises():
    model = _modelWithSetupElements(1)
    with model.topologyChanges():
        with pytest.raises(TopologyError, match="already taken"):
            model.createElement(_StubElement(1))


def test_windows_nest_without_closing_early():
    """A setup helper may open a window without knowing whether its caller already did; the inner
    scope must not close the outer one."""

    model = _modelWithSetupElements()
    with model.topologyChanges():
        with model.topologyChanges():
            model.reserveElementNumbers(1)
        model.reserveElementNumbers(1)  # outer window still open

    with pytest.raises(TopologyError):
        model.reserveElementNumbers(1)


def test_window_closes_on_exception():
    model = _modelWithSetupElements()
    with pytest.raises(RuntimeError):
        with model.topologyChanges():
            raise RuntimeError("modifier blew up")

    with pytest.raises(TopologyError):
        model.reserveElementNumbers(1)


def test_parsed_element_set_keeps_its_declaration_order():
    """An ``*elSet`` must come out in the order it was declared.

    Element sets are OrderedSets, and their member order reaches element *numbering*: an
    element-based ``*surface`` is built from such a set, and ``surfaceElementGenerator`` walks it
    handing out sequential facet element labels. Elements are hashed by identity, so building the
    set through a raw ``set()`` fixed that order from object *addresses* -- reproducible only for a
    bit-identical allocation history, which a resumed run does not have (its ``.inp`` carries
    ``*restart``, and it has an extra open file). Verified to fail before the fix, with this exact
    fixture producing [1, 4, 9, 3, 8, 12, 2, 7, 11, 6, 5, 10].

    See the determinism contract in the module docstring.
    """

    from collections import defaultdict

    from edelweissfe.generators.abqmodelconstructor import AbqModelConstructor
    from edelweissfe.journal.journal import Journal

    nElements = 12
    inputFile = defaultdict(list)
    inputFile["node"] = [
        {"nSet": None, "datalines": ["{:}, {:}, 0.0".format(i, float(i)) for i in range(1, 2 * nElements + 3)]}
    ]
    inputFile["element"] = [
        {
            "type": "CPE4",
            "provider": "edelweiss",
            "elSet": None,
            "elset": None,
            "datalines": [
                "{:}, {:}, {:}, {:}, {:}".format(e, e, e + 1, e + nElements + 2, e + nElements + 1)
                for e in range(1, nElements + 1)
            ],
        }
    ]
    inputFile["elSet"] = [{"elSet": "declared", "datalines": [", ".join(str(e) for e in range(1, nElements + 1))]}]

    model = AbqModelConstructor(Journal(verbose=False)).createGeometryFromInputFile(FEModel(2), inputFile)

    assert [el.elNumber for el in model.elementSets["declared"]] == list(range(1, nElements + 1))


class _StubModifier:
    """A modifier that plans a fixed number of times, then settles."""

    initiatesTopologyChanges = True

    def __init__(self, name, plansLeft, log, reactsToOthers=False):
        self.name = name
        self._plansLeft = plansLeft
        self._log = log
        self._reactsToOthers = reactsToOthers

    def plan(self, model, change, step, timeStep):
        # react to another modifier's mutation once, then settle -- the contract that makes the
        # pipeline converge (see ModelModifierBase.plan)
        if change is not None and not self._reactsToOthers:
            return None
        if self._plansLeft <= 0:
            return None
        self._plansLeft -= 1
        return {"who": self.name}

    def encodePlan(self, plan):
        return {"who": plan["who"]}

    def decodePlan(self, data):
        return {"who": str(data["who"])}

    def restoreDecisionState(self, records):
        pass

    def apply(self, model, plan):
        # A modifier reports what it did in exactly one way: by returning the change. Notifying here
        # as well would record it twice; see FEModel.recordTopologyChange.
        self._log.append(plan["who"])
        (number,) = model.reserveElementNumbers(1)
        model.createElement(_StubElement(number))
        change = ModelChange(kind=_MCT.REFINEMENT)
        change.addedElements.add(number)
        return change


def _modelWithModifiers(**modifiers) -> FEModel:
    model = FEModel(3)
    model.modelModifiers.update(modifiers)
    return model


def test_a_single_round_suffices_when_nobody_reacts():
    log = []
    model = _modelWithModifiers(
        amr=_StubModifier("amr", 1, log),
        printer=_StubModifier("printer", 1, log),
    )
    assert model.updateTopology(step=None, timeStep=0.0) is True
    # both planned in round 1; in round 2 each sees only the other's change and settles
    assert log == ["amr", "printer"]


def test_modifiers_run_in_declaration_order_every_round():
    log = []
    model = _modelWithModifiers(
        amr=_StubModifier("amr", 2, log, reactsToOthers=True),
        facets=_StubModifier("facets", 2, log, reactsToOthers=True),
    )
    model.updateTopology(step=None, timeStep=0.0)
    assert log == ["amr", "facets", "amr", "facets"]


def test_no_change_means_no_topology_update():
    log = []
    model = _modelWithModifiers(amr=_StubModifier("amr", 0, log))
    assert model.updateTopology(step=None, timeStep=0.0) is False
    assert log == []


def test_non_convergence_raises_naming_the_offender():
    log = []
    model = _modelWithModifiers(runaway=_StubModifier("runaway", 10**6, log, reactsToOthers=True))
    model.maxTopologyRounds = 4
    with pytest.raises(TopologyError, match="did not settle within 4 rounds.*runaway"):
        model.updateTopology(step=None, timeStep=0.0)


def test_the_window_is_closed_again_after_the_update():
    model = _modelWithModifiers(amr=_StubModifier("amr", 1, []))
    model.updateTopology(step=None, timeStep=0.0)
    with pytest.raises(TopologyError):
        model.reserveElementNumbers(1)


class _StubMeshDependent:
    """A consumer that records how many times it was refreshed, and with what."""

    _lastSeenTopologyVersion = 0

    def __init__(self, relevant=True):
        self.refreshes = []
        self._relevant = relevant

    def refresh(self, model, change):
        self.refreshes.append(change)
        return self._relevant

    refreshIfMeshChanged = MeshDependent.refreshIfMeshChanged


def test_the_pipeline_records_exactly_the_change_a_modifier_returns():
    """The whole contract in one test: a modifier only ever returns its change, and the pipeline is
    what records it. If that recording is dropped, changesSince() sees nothing and every
    mesh-dependent silently misses the update; if it happens twice, the version count is wrong."""

    model = _modelWithModifiers(amr=_OwningModifier("amr", [], owns={1}, touches={99}))
    consumer = _StubMeshDependent()
    model.registerMeshDependent(consumer)
    versionBefore = model.topologyVersion

    model.updateTopology(step=None, timeStep=0.0)

    assert model.topologyVersion == versionBefore + 1, "one apply, recorded exactly once"
    assert model.changesSince(versionBefore).addedElements == {99}
    assert model.refreshMeshDependents() is True
    assert [change.addedElements for change in consumer.refreshes] == [{99}]


def test_consumers_refresh_once_on_the_net_change_of_all_rounds():
    """The point of pull: a consumer sees the settled model, not each intermediate round. Under the
    old push observer it would have been called once per mutation."""

    log = []
    model = _modelWithModifiers(
        amr=_StubModifier("amr", 2, log, reactsToOthers=True),
        facets=_StubModifier("facets", 2, log, reactsToOthers=True),
    )
    consumer = _StubMeshDependent()
    model.registerMeshDependent(consumer)

    model.updateTopology(step=None, timeStep=0.0)
    assert len(log) == 4, "four mutations across two rounds"
    assert consumer.refreshes == [], "consumers must not be refreshed during the topology update"

    assert model.refreshMeshDependents() is True
    assert len(consumer.refreshes) == 1, "one refresh, on the net change"


def test_refresh_reports_whether_any_consumer_changed_its_footprint():
    model = _modelWithModifiers(amr=_StubModifier("amr", 1, []))
    indifferent = _StubMeshDependent(relevant=False)
    model.registerMeshDependent(indifferent)
    model.updateTopology(step=None, timeStep=0.0)
    assert model.refreshMeshDependents() is False
    assert len(indifferent.refreshes) == 1


def test_registration_is_idempotent_and_a_quiet_model_refreshes_nobody():
    model = FEModel(3)
    consumer = _StubMeshDependent()
    model.registerMeshDependent(consumer)
    model.registerMeshDependent(consumer)
    assert len(model.meshDependents) == 1
    assert model.refreshMeshDependents() is False
    assert consumer.refreshes == []


def _tinyMeshModel(elementNumbers=(1, 2), shiftCoordinate=0.0):
    """Two CPE4s sharing an edge, numbered as asked -- enough to exercise numbering, connectivity
    and coordinates without a solver."""

    from edelweissfe.config.elementlibrary import getElementClass
    from edelweissfe.points.node import Node

    model = FEModel(2)
    coords = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (0.0, 1.0), (1.0, 1.0), (2.0, 1.0)]
    for label, (x, y) in enumerate(coords, start=1):
        model.nodes[label] = Node(label, np.array([x + shiftCoordinate, y]))

    ElementClass = getElementClass("CPE4", "edelweiss")
    for elNumber, conn in zip(elementNumbers, [(1, 2, 5, 4), (2, 3, 6, 5)]):
        element = ElementClass("CPE4", elNumber)
        element.setNodes([model.nodes[label] for label in conn])
        model.elements[elNumber] = element
    return model


def test_fingerprint_is_stable_across_processes():
    """blake2b, not hash(): Python randomises string hashing per process, so a hash()-based digest
    would differ between two runs of the same code and render the whole check useless."""

    import os
    import subprocess
    import sys

    script = (
        "import numpy as np\n"
        "from edelweissfe.models.femodel import FEModel\n"
        "from edelweissfe.points.node import Node\n"
        "from edelweissfe.config.elementlibrary import getElementClass\n"
        "m = FEModel(2)\n"
        "for label, (x, y) in enumerate([(0.,0.),(1.,0.),(1.,1.),(0.,1.)], start=1):\n"
        "    m.nodes[label] = Node(label, np.array([x, y]))\n"
        "e = getElementClass('CPE4', 'edelweiss')('CPE4', 1)\n"
        "e.setNodes([m.nodes[i] for i in (1, 2, 3, 4)])\n"
        "m.elements[1] = e\n"
        "print(m.topologyFingerprint())\n"
    )
    digests = set()
    for seed in ("0", "1", "random"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, cwd=str(_REPO_ROOT), env=env
        )
        assert result.returncode == 0, result.stderr
        digests.add(result.stdout.strip())

    assert len(digests) == 1, "digest varies with PYTHONHASHSEED: {:}".format(digests)
    assert digests != {""}


def test_fingerprint_detects_renumbering():
    """The failure this whole plan exists to catch: same mesh, different element numbers."""

    assert _tinyMeshModel(elementNumbers=(1, 2)).topologyFingerprint() != (
        _tinyMeshModel(elementNumbers=(7, 8)).topologyFingerprint()
    )


def test_fingerprint_detects_moved_nodes():
    assert _tinyMeshModel().topologyFingerprint() != _tinyMeshModel(shiftCoordinate=1e-12).topologyFingerprint()


def test_fingerprint_ignores_solution_state():
    """A mismatch must mean the mesh diverged, not that the solver took a different path."""

    model = _tinyMeshModel()
    before = model.topologyFingerprint()
    model.time = 17.0
    model.scalarVariables["lambda"] = object()
    assert model.topologyFingerprint() == before


def test_fingerprint_is_insensitive_to_dict_insertion_order():
    """Two models built in different orders are the same mesh and must agree -- otherwise the check
    would fire on differences that do not matter."""

    forward = _tinyMeshModel(elementNumbers=(1, 2))
    backward = _tinyMeshModel(elementNumbers=(1, 2))
    backward.elements = dict(reversed(list(backward.elements.items())))
    backward.nodes = dict(reversed(list(backward.nodes.items())))
    assert forward.topologyFingerprint() == backward.topologyFingerprint()


class _OwningModifier(_StubModifier):
    """A modifier that claims a fixed set of elements, and can be told to mutate an arbitrary one."""

    def __init__(self, name, log, owns=(), touches=(), reactsToOthers=False):
        super().__init__(name, plansLeft=1, log=log, reactsToOthers=reactsToOthers)
        self._owns = set(owns)
        self._touches = set(touches)

    def declaredDomain(self, model):
        return self._owns

    def apply(self, model, plan):
        self._log.append(plan["who"])
        change = ModelChange(kind=_MCT.REFINEMENT)
        change.addedElements |= self._touches
        return change


def test_overlapping_modifier_domains_are_refused_at_setup():
    """Two modifiers owning one element each end up holding stale references to the other's work.
    Cheap to catch at startup, expensive to discover mid-solve."""

    model = _modelWithModifiers(
        amr_left=_OwningModifier("amr_left", [], owns={1, 2, 3}),
        amr_right=_OwningModifier("amr_right", [], owns={3, 4}),
    )
    with pytest.raises(TopologyError, match=r"both claim 1 of the same element\(s\).*3"):
        model.checkModelModifierDomains()


def test_disjoint_modifier_domains_are_accepted():
    model = _modelWithModifiers(
        amr_left=_OwningModifier("amr_left", [], owns={1, 2}),
        amr_right=_OwningModifier("amr_right", [], owns={3, 4}),
    )
    model.checkModelModifierDomains()  # must not raise


def test_two_modifiers_changing_one_element_in_a_round_is_refused():
    """Disjoint *declared* domains are not enough: one modifier deleting what another just created,
    within the same round, is order-dependent and must not decide a simulation quietly."""

    log = []
    model = _modelWithModifiers(
        first=_OwningModifier("first", log, owns={1}, touches={99}),
        # reactsToOthers, so it still plans after seeing the first modifier's change in the same
        # round -- which is the only way two modifiers can collide within one round at all
        second=_OwningModifier("second", log, owns={2}, touches={99}, reactsToOthers=True),
    )
    with pytest.raises(TopologyError, match=r"'first' \(round 1\) and 'second' \(round 1\).*element 99"):
        model.updateTopology(step=None, timeStep=0.0)


class _LateModifier(_OwningModifier):
    """An owning modifier that holds off until the given round, so its mutation lands in a different
    round than the one it collides with."""

    def __init__(self, name, log, owns, touches, fromRound):
        super().__init__(name, log, owns=owns, touches=touches, reactsToOthers=True)
        self._fromRound = fromRound
        self._roundsSeen = 0

    def plan(self, model, change, step, timeStep):
        self._roundsSeen += 1
        if self._roundsSeen < self._fromRound:
            return None
        return super().plan(model, change, step, timeStep)


def test_two_modifiers_changing_one_element_in_different_rounds_is_refused():
    """The rounds exist so a modifier can react to another's output, which is exactly when two of
    them are most likely to reach for the same element -- so the guard has to span the whole update,
    not just one round, and has to say which rounds collided."""

    log = []
    model = _modelWithModifiers(
        first=_OwningModifier("first", log, owns={1}, touches={99}),
        second=_LateModifier("second", log, owns={2}, touches={99}, fromRound=2),
    )
    with pytest.raises(TopologyError, match=r"'first' \(round 1\) and 'second' \(round 2\).*element 99"):
        model.updateTopology(step=None, timeStep=0.0)


def test_the_same_modifier_may_touch_an_element_in_successive_rounds():
    """The guard is about two modifiers colliding, not about one modifier revisiting its own work."""

    log = []
    model = _modelWithModifiers(solo=_OwningModifier("solo", log, owns={1}, touches={99}))
    model.updateTopology(step=None, timeStep=0.0)
    assert log == ["solo"]


def test_an_empty_changeset_is_not_a_change():
    """A modifier may plan and then find nothing left to do. Counting that as a change would rebuild
    the equation system for nothing, pay a topology fingerprint for nothing, and record a decision a
    restart would then faithfully replay to no effect."""

    class _NoOpModifier(_StubModifier):
        def apply(self, model, plan):
            self._log.append(plan["who"])
            return ModelChange(kind=_MCT.REFINEMENT)  # planned, then found nothing to do

    log = []
    model = _modelWithModifiers(noop=_NoOpModifier("noop", plansLeft=1, log=log))
    assert model.updateTopology(step=None, timeStep=0.0) is False
    assert log == ["noop"], "it did plan and apply -- what must not follow is being counted"
    assert model.topologyHistory == []
    assert model.topologyVersion == 0


@pytest.mark.parametrize(
    "populate",
    [
        lambda c: c.addedNodes.add(1),
        lambda c: c.removedNodes.add(1),
        lambda c: c.addedElements.add(1),
        lambda c: c.removedElements.add(1),
        lambda c: c.parentToChildren.update({1: [2]}),
        lambda c: c.faceMap.update({(1, 2): [(3, 2)]}),
        lambda c: c.changedNodeSets.add("nodes"),
        lambda c: c.changedElementSets.add("elements"),
        lambda c: c.changedSurfaces.add("surface"),
    ],
)
def test_a_changeset_is_empty_only_if_every_collection_is(populate):
    """``kind`` alone is not a change, but any one populated collection is -- a modifier that only
    re-tiled a surface, or only changed an element set's membership, did change the model."""

    change = ModelChange(kind=_MCT.REFINEMENT)
    assert change.isEmpty
    populate(change)
    assert not change.isEmpty


def test_a_consumer_cannot_mutate_the_topology():
    """Phase 2 runs with the window closed, so a mesh-dependent that tries to create an element
    raises instead of quietly mutating behind the pipeline's back -- which is what moving facet
    regeneration out of constraint refresh and into its own modifier bought."""

    class _MutatingConsumer(_StubMeshDependent):
        def refresh(self, model, change):
            (number,) = model.reserveElementNumbers(1)  # must raise: window is closed
            model.createElement(_StubElement(number))
            return True

    log = []
    model = _modelWithModifiers(amr=_StubModifier("amr", 1, log))
    consumer = _MutatingConsumer()
    model.registerMeshDependent(consumer)

    model.updateTopology(step=None, timeStep=0.0)
    with pytest.raises(TopologyError, match="only be reserved during a topology change"):
        model.refreshMeshDependents()


def test_a_purely_reactive_modifier_sees_the_first_round_change():
    """The bug this pins cost two marmot regressions: a modifier that only ever reacts to someone
    else's change was handed None in round 1 -- after the change it needed had already happened --
    and then had its version stamped, so round 2 showed nothing new either. It never reacted at all,
    and contact ran against facets still tiling the unrefined surface.

    A reactive modifier must see, in round 1, what an earlier modifier did in round 1.
    """

    seen = []

    class _Reactive(_StubModifier):
        def plan(self, model, change, step, timeStep):
            seen.append(change)
            return None

    log = []
    model = _modelWithModifiers(
        acts=_OwningModifier("acts", log, owns={1}, touches={7}),
        reacts=_Reactive("reacts", plansLeft=0, log=log),
    )
    model.updateTopology(step=None, timeStep=0.0)

    assert seen, "the reactive modifier was never asked to plan"
    assert seen[0] is not None, "round 1 handed it None despite an earlier modifier having changed the model"
    assert 7 in seen[0].addedElements


class _NonAdaptingSolver:
    """A solver that never runs the topology update -- the explicit ones, which do support ties."""

    identification = "nonAdapting"
    supportsMPC = True
    supportsModelModifiers = False
    validateModelCapabilities = NonlinearSolverBase.validateModelCapabilities


def test_the_implicit_facet_retiling_does_not_disqualify_a_non_adapting_solver():
    """Every ``*surface`` recipe brings the facet modifier along, whether or not anything adapts the
    mesh. It is purely reactive, so a solver that never runs the topology update loses nothing by
    not polling it -- and refusing the model would rule out plain contact and tie analyses on the
    explicit solvers, which do support them."""

    model = _modelWithModifiers(surfaceFacets=_SurfaceFacetsModifier("surfaceFacets", FEModel(3), None))
    _NonAdaptingSolver().validateModelCapabilities(model)  # must not raise


def test_a_self_starting_modifier_is_refused_by_a_non_adapting_solver_and_named():
    """The guard still has to fire for a modifier that would act on its own: silently never
    adapting a model the user asked to adapt is the failure it exists to prevent."""

    model = _modelWithModifiers(
        amr=_StubModifier("amr", 1, []),
        surfaceFacets=_SurfaceFacetsModifier("surfaceFacets", FEModel(3), None),
    )
    with pytest.raises(NotImplementedError, match=r"does not run the topology update.*\bamr\b"):
        _NonAdaptingSolver().validateModelCapabilities(model)


# ---------------------------------------------------------------------------------------------
# The node allocator, as seen through the adaptive refinement registry
# ---------------------------------------------------------------------------------------------


class _StubNode:
    """The registry only ever reads a node's label and coordinates."""

    def __init__(self, label: int, coordinates):
        self.label = label
        self.coordinates = np.asarray(coordinates, dtype=float)


def _modelWithSetupNodes(*labels: int) -> FEModel:
    """A model carrying nodes as the base mesh generators leave them, with the allocator raised
    above them exactly as the end of model setup does."""

    model = FEModel(3)
    for label in labels:
        model.nodes[label] = _StubNode(label, [float(label), 0.0, 0.0])
    model.adoptSetupNodeNumbers()
    return model


def test_a_standalone_registry_still_mints_on_its_own():
    """The octree layer is model-agnostic and must stay usable without one."""

    registry = NodeRegistry()
    registry.seed(7, [0.0, 0.0, 0.0])

    assert registry.label([1.0, 0.0, 0.0]) == 8


def test_a_registry_label_cannot_collide_with_one_the_model_hands_out():
    """The reason the registry takes an allocator at all: minting max+1 by itself made the octree a
    second, independent source of node labels, free to drift into the model's."""

    model = _modelWithSetupNodes(1, 2, 3)
    registry = NodeRegistry(reserve_labels=model.reserveNodeNumbers)
    for node in model.nodes.values():
        registry.seed(node.label, node.coordinates)

    with model.topologyChanges():
        minted = [registry.label([0.0, float(i), 0.0]) for i in range(3)]
        fromModel = list(model.reserveNodeNumbers(3))
        mintedAfterwards = [registry.label([0.0, 0.0, float(i) + 1.0]) for i in range(3)]

    assert not set(minted) & set(model.nodes)
    assert not set(minted) & set(fromModel)
    assert not set(mintedAfterwards) & set(fromModel)
    assert sorted(minted + fromModel + mintedAfterwards) == list(range(4, 13))


def test_an_already_known_coordinate_consumes_no_label():
    """Most label() calls are lookups of a node shared by several elements; only the minting branch
    may draw a number, or refinement would burn labels at a wild rate."""

    model = _modelWithSetupNodes(1)
    registry = NodeRegistry(reserve_labels=model.reserveNodeNumbers)
    registry.seed(1, [0.0, 0.0, 0.0])

    with model.topologyChanges():
        assert registry.label([0.0, 0.0, 0.0]) == 1
        (afterwards,) = model.reserveNodeNumbers(1)

    assert afterwards == 2


def test_minting_outside_a_topology_change_raises():
    """A registry wired to a model inherits the model's mutation window: new nodes appear only where
    the pipeline can see them."""

    model = _modelWithSetupNodes(1)
    registry = NodeRegistry(reserve_labels=model.reserveNodeNumbers)

    with pytest.raises(TopologyError, match="topology change"):
        registry.label([5.0, 0.0, 0.0])
