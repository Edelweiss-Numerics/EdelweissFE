The topology pipeline: how a model changes itself
==================================================

An adaptive analysis mutates its own model while it runs. Refinement splits elements, a deposition
module adds them, contact and tie surfaces regenerate their facets. Every one of those creates and
destroys elements and nodes -- and every one of them must produce *identical* results when the
analysis is resumed from a checkpoint, or restored material state lands on the wrong element.

This page describes how EdelweissFE arranges that. It is the page to read before writing a new
model modifier, or before writing anything that caches data derived from the mesh.

The short version, and the mnemonic worth remembering:

    **Modifiers plan and apply; mesh-dependents refresh; then we solve.**


One increment, three phases
---------------------------

.. code-block:: text

    ╔══════════════════════════════════════════════════════════[ window OPEN ]══╗
    ║ PHASE 1 — TOPOLOGY UPDATE   the only place elements/nodes are born or die ║
    ║                                                                           ║
    ║   amr ─────────▶ printer ─────────▶ facets ─────▶ any plan applied?       ║
    ║   plan(model)    plan(model)        plan(model)          │   │            ║
    ║      ↓              ↓                  ↓                 │   │            ║
    ║   apply(…)       apply(…)           apply(…)             │   │            ║
    ║        modifiers run in declared order                   │   │            ║
    ║   ┌──────────────── yes: run another round ◀─────────────┘   │            ║
    ║   └──▶ (back to amr)                      no: fixed point ───┼──┐          ║
    ╚══════════════════════════════════════════════════════════════╪══╪═════════╝
                                                                   │  │
    ╔══════════════════════════════════════════════════════════════▼══▼═════════╗
    ║ PHASE 2 — REFRESH  [ window CLOSED ]  pure readers: nothing born or dies  ║
    ║   each mesh-dependent asks: has topologyVersion moved since I last looked? ║
    ║        yes → refresh once, from the NET change across all rounds           ║
    ║   order is irrelevant — none of them can invalidate another's work         ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
                                          │
    ╔══════════════════════════════════════▼════════════════════════════════════╗
    ║ PHASE 3 — SOLVE   assemble the equation system and iterate to convergence  ║
    ╚═══════════════════════════════════════════════════════════════════════════╝

Phase 1 is :meth:`~edelweissfe.models.femodel.FEModel.updateTopology`, phase 2 is
:meth:`~edelweissfe.models.femodel.FEModel.refreshMeshDependents`. The solver calls both, in that
order, before every solve.


Why rounds
~~~~~~~~~~

Modifiers depend on each other. Refinement invalidates a tied surface's facets; a deposition
modifier creates elements refinement may then want to split; a 2:1 balance may need to refine what
another modifier just activated. A declared dependency order cannot express mutual dependence, and
asking users to get one right is a poor trade.

So each **round** offers every modifier the net change since *that modifier* last planned. A round
in which nobody plans anything is the fixed point, and it ends the phase. Determinism comes from the
structure, not from luck: within a round, modifiers run in the order they were declared in the input
file.

A modifier that keeps planning in response to its own output would never settle. That is a bug in
the modifier, and :attr:`~edelweissfe.models.femodel.FEModel.maxTopologyRounds` turns it into a loud
error naming the offender rather than a hang.


Writing a model modifier: ``plan`` and ``apply``
------------------------------------------------

A modifier implements two halves, and the split is the single most important thing on this page.

:meth:`~edelweissfe.modelmodifiers.base.modelmodifierbase.ModelModifierBase.plan`
    Decide what to do, **without doing it**. May read anything: markers, node fields, the current
    time, the step. Returns a *serializable* description of the decision, or ``None`` if there is
    nothing to do.

:meth:`~edelweissfe.modelmodifiers.base.modelmodifierbase.ModelModifierBase.apply`
    Carry the decision out, mutating the model. **Must not read solution state** -- it is a pure
    function of ``(model, plan)``. Everything the decision depended on belongs in the plan.

.. code-block:: python

    def plan(self, model, change, step, timeStep):
        if change is not None and not change.touchesElementSet(self.myElSet):
            return None                      # not my business -- let the pipeline settle
        marked = self.evaluateMarkers(model) # reads solution state: allowed here
        return None if not marked else MyPlan(ids=sorted(marked))

    def apply(self, model, plan):
        numbers = model.reserveElementNumbers(len(plan.ids))
        for identifier, elNumber in zip(plan.ids, numbers):
            model.createElement(self.makeElement(elNumber, identifier))
        return change                        # a ModelChange describing what happened

Two rules follow from the signatures and are worth stating explicitly:

**Return ``None`` from** ``plan`` **when the change does not touch your domain.** That is what lets
the pipeline reach a fixed point instead of looping. ``change`` is ``None`` on the first round of an
update, meaning "evaluate freshly".

**Never write** ``model.elements`` **directly.** Use
:meth:`~edelweissfe.models.femodel.FEModel.reserveElementNumbers`,
:meth:`~edelweissfe.models.femodel.FEModel.createElement` and
:meth:`~edelweissfe.models.femodel.FEModel.removeElement`.

**Say so if your modifier is purely reactive.** A modifier that can only act in response to another
one's mutation -- the implicit surface-facet retiling is the only current example -- sets
``initiatesTopologyChanges = False``. A solver that never runs the topology update (the explicit
ones) then still accepts a model carrying it, while still refusing one whose modifiers would
silently never run.


Why the split exists: restart
------------------------------

A resumed run must rebuild exactly the mesh it resumed -- same elements, same *numbers*, same
connectivity -- because element state is restored by number.

The obvious way to arrange that is for each modifier to serialize its own history and implement its
own replay. EdelweissFE did exactly that once, and it produced a real bug: a resumed analysis
rebuilt a differently-numbered mesh, so heavily damaged elements were restored virgin, and the
solve collapsed into a cutback spiral several increments later.

The root cause was not the numbering. It was that **replay ran through different code than the live
run**, and two implementations of one mutation always drift apart. The drift showed up as a handful
of individually-reasonable patches -- "skip the warm start on replay", "this is never really the
first call", "materialise level-wise instead of occasion-wise" -- which together changed the result.

The ``plan``/``apply`` split removes the second implementation:

.. code-block:: text

        LIVE RUN                                      RESTART REPLAY
    modifier.plan(model, …)                        read recorded plan
    reads markers, state, time                     no markers re-evaluated
         │        │                                     │            │
         │        └── records ──▶ topologyHistory ── restores ───────┘
         │                                                           │
         └──────────── applies ──▶┌───────────────────────┐◀─ applies┘
                                  │ modifier.apply(model, │
                                  │               plan)   │
                                  │ ONE code path         │
                                  │ reads no solution state│
                                  └───────────────────────┘
                                              │
                        identical element numbers · identical topology

:attr:`~edelweissfe.models.femodel.FEModel.topologyHistory` records every applied decision -- the
modifier's own serialized plan, via
:meth:`~edelweissfe.modelmodifiers.base.modelmodifierbase.ModelModifierBase.encodePlan`, plus the
resulting :meth:`~edelweissfe.models.femodel.FEModel.topologyFingerprint`. A restart replays it
through :meth:`~edelweissfe.models.femodel.FEModel.replayTopologyHistory`, which calls the same
``apply``. ``plan`` is never called during a replay.

The fingerprint is not decoration. Recorded per decision, it turns *"the resumed run diverged
somewhere"* into *"it diverged at record 12, modifier* ``amr`` *, round 2"* -- a divergence you can
bisect rather than hunt.

**What is not checkpointed:** decision-side state, such as a marker's buffer of pending marks. The
next ``plan`` re-derives it from the restored solution state -- exactly as the live run would have.
If a modifier genuinely needs something back, it overrides the optional
:meth:`~edelweissfe.modelmodifiers.base.modelmodifierbase.ModelModifierBase.restoreDecisionState`.
That is deliberately separate from ``apply``: it affects how the *next* decision is made, so getting
it wrong cannot corrupt the mesh.


Element numbers
---------------

Element numbers come from one monotonic allocator,
:meth:`~edelweissfe.models.femodel.FEModel.reserveElementNumbers`. They are **never recycled**, and
never derived from ``max(model.elements)``.

That is not tidiness. Deriving the next number from the current maximum makes numbering a function
of the *deletion* history as well as the creation history -- a contact facet set that is deleted and
rebuilt hands the freed numbers straight back out -- so a replay would have to reproduce creations,
deletions *and* their interleaving. With one monotonic counter it only has to reproduce the ordered
creation sequence, which is exactly what the recorded history holds.

Never recycling also means a number refers to one element for the model's entire lifetime, so a
reference cached by number cannot silently come to mean something else.

Numbers may only be reserved inside a **topology window**
(:meth:`~edelweissfe.models.femodel.FEModel.topologyChanges`), which the pipeline opens around
phase 1. Outside it, creating or deleting an element raises
:class:`~edelweissfe.utils.exceptions.TopologyError`. This is what makes "only model modifiers
change the topology" an enforced property rather than a convention.


Reacting to a change: three tiers, deliberately
------------------------------------------------

Anything that caches data derived from the mesh has to notice when the mesh changes. There are three
ways, in increasing order of power -- **use the weakest one that works**.

**Tier 0 -- do nothing.** Node sets, element sets, surfaces and node fields have stable identity:
they are mutated in place, never replaced. A component that caches ``model.nodeSets["mySet"]`` and
iterates it later sees any refinement automatically. Most code needs nothing else, and that is by
design.

**Tier 1 -- a set-changed self-check.** A component that pre-sizes a *derived* array to a
container's size needs to notice a resize, since the array does not grow on its own. It compares the
container's version counter at its own point of use (``ConstraintBase._checkSetChanged``,
``StepActionBase._checkSetChanged``). This has **no timing dependency at all** -- it checks when it
uses the data -- which makes it correct on a replay for free.

**Tier 2 --** :class:`~edelweissfe.models.meshdependent.MeshDependent`. For cached *geometry*:
contact facets, tie records, projections. Register once with
:meth:`~edelweissfe.models.femodel.FEModel.registerMeshDependent`, implement
:meth:`~edelweissfe.models.meshdependent.MeshDependent.refresh`, and phase 2 calls it once per
increment with the net change.

.. code-block:: python

    class MyConstraint(ConstraintBase, MeshDependent):
        def __init__(self, name, model, ...):
            model.registerMeshDependent(self)

        def refresh(self, model, change):
            if not change.touchesSurface(self.mySurface):
                return False                 # not my business
            self.rebuildCachedGeometry(model)
            return True                      # my DOF footprint changed

Registration is the freshness guarantee, not ceremony: a consumer that is not registered is never
told. That matters most for consumers the solver does not otherwise tick -- multi-point constraints
live in ``model.multiPointConstraints``, which no per-increment sweep iterates, so a tie can learn
about refinement *only* this way.

``refresh`` **must not create or delete entities.** Topology is a modifier's business, and the
window is closed by the time phase 2 runs, so an attempt raises.


Why pull, and not a push notification
--------------------------------------

EdelweissFE once notified consumers synchronously, at the instant of each mutation. That mechanism
is gone, and the reason is worth understanding, because it is a general one.

With modifiers running to a fixed point, a per-mutation callback necessarily fires **mid-pipeline**.
Round 3 rebuilds what round 2 invalidated, so a consumer notified in round 1 is handed a state that
no longer exists by the time the solve begins. And a consumer that *mutates* in response -- every
tie and contact does, since they mint facet elements -- would do so re-entrantly, inside the
modifier's own loop.

Push is not merely riskier here; it is at the wrong granularity. It reports transients, and
consumers want the settled model. So phase 2 pulls: each consumer diffs against its own last-seen
:attr:`~edelweissfe.models.femodel.FEModel.topologyVersion` and reconciles once, from the change
coalesced across every round.


A checklist for a new modifier
-------------------------------

#. Implement ``plan`` (may read state) and ``apply`` (may not).
#. Return ``None`` from ``plan`` when the incoming change does not touch your domain.
#. Reserve element numbers from the model; never write ``model.elements``.
#. Implement ``encodePlan``/``decodePlan`` so your decision survives a checkpoint.
#. Override ``restoreDecisionState`` only if ``plan`` needs history back -- not to rebuild the mesh.
#. Verify with a restart round-trip and compare
   :meth:`~edelweissfe.models.femodel.FEModel.topologyFingerprint`, not just element counts.
