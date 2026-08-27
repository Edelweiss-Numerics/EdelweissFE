Model modifiers
===============

.. automodule:: edelweissfe.config.modelmodifiers
    :members: __doc__

Unlike constraints, step actions or output managers -- which act on a *fixed* mesh -- a model
modifier may change the mesh topology itself during an analysis: adding or removing nodes and
elements, re-partitioning element/node sets and surfaces, and reallocating the solution fields.
A modifier is declared with the ``*modelModifier`` keyword. At the start of every increment the
solver runs **all** modifiers to a fixed point via
:meth:`~edelweissfe.models.femodel.FEModel.updateTopology`, then lets mesh-dependent consumers catch
up, then solves; when the topology changed, the equation system (DOF manager, sparsity pattern,
solution vectors and any multi-point-constraint transformation) is rebuilt first. A modifier itself
is written as two halves -- :meth:`~edelweissfe.modelmodifiers.base.modelmodifierbase.ModelModifierBase.plan`,
which decides and may read solution state, and
:meth:`~edelweissfe.modelmodifiers.base.modelmodifierbase.ModelModifierBase.apply`, which carries the
decision out and may not. **See** :doc:`topologypipeline` **for the full contract, why it is split
that way, and what a new modifier must implement**; this page covers the individual modifiers.

**Topological containers have stable identity.** :class:`~edelweissfe.sets.nodeset.NodeSet`,
:class:`~edelweissfe.sets.elementset.ElementSet`, :class:`~edelweissfe.surfaces.entitybasedsurface.
EntityBasedSurface` and :class:`~edelweissfe.fields.nodefield.NodeField` are created once and
mutated in place (``replaceMembers``/``replaceData``/``resize``) by a mesh-mutating modifier --
they are never replaced with a new object under the same name. A component that simply caches one
of these, e.g. ``self._nodes = model.nodeSets["mySet"]`` in ``__init__`` and later iterates
``self._nodes``, sees any later refinement automatically: no observer, no ``MeshDependent``, no
re-fetch. This is deliberate -- it lets a student write a new constraint, step action or output
without ever thinking about AMR.

A component that additionally pre-sizes a *derived* array or object to a container's current size
(e.g. a Dirichlet BC's ``delta`` tiled to ``len(nSet)``, or a field output's result collector
pinning a snapshot element list) does need to notice a resize, since the array does not grow with
the container on its own. Two mechanisms remain, narrowed to exactly these cases:

* **Lazy version check** -- each mutable container carries a ``_version`` counter, bumped on every
  in-place mutation. A component compares its own last-seen version against the container's current
  one at its own per-increment entry point (e.g. ``StepActionBase._checkSetChanged``,
  ``ConstraintBase._checkSetChanged``) and recomputes the derived array only on a mismatch --
  see :mod:`~edelweissfe.stepactions.dirichlet`, :mod:`~edelweissfe.stepactions.nodeforces` and
  :class:`~edelweissfe.utils.fieldoutput.ElementFieldOutput` for examples. This needs no
  registration and therefore has no observer lifecycle to leak.
* **Registered mesh dependent** -- for derived *geometry* that must be regenerated before the next
  equation-system rebuild (facet-based contact and tie; see below), a component registers itself via
  :meth:`~edelweissfe.models.femodel.FEModel.registerMeshDependent` and implements
  :meth:`~edelweissfe.models.meshdependent.MeshDependent.refresh`. Once per increment, after every
  modifier has settled, :meth:`~edelweissfe.models.femodel.FEModel.refreshMeshDependents` hands it
  the *net* change since it last looked -- added/removed nodes and elements, the parent -> children
  map, the per-face child tiling, and which node/element sets or surfaces were touched (with
  ``touchesSurface``/``touchesNodeSet``/``touchesElementSet`` early-outs so a consumer can skip a
  change that doesn't concern it).

  The synchronous push notification that used to exist alongside this has been removed: with
  modifiers running to a fixed point, a per-mutation callback fires mid-pipeline and hands the
  consumer a state that no longer exists by the time the solve begins. See :doc:`topologypipeline`.

``hAdaptivity`` - Hanging-node h-adaptivity for HEX20
-----------------------------------------------------

Module ``edelweissfe.modelmodifiers.adaptivity.hadaptivity``

.. automodule:: edelweissfe.modelmodifiers.adaptivity.hadaptivity
    :members: __doc__

Dynamic adaptive :math:`h`-refinement of 20-node serendipity hexahedra (``GC3D20`` / ``GC3D20R``)
in the small-strain, multifield (displacement + nonlocal damage) regime. Each increment the
modifier evaluates one or more **markers** (see below), subdivides every marked
element into ``splitFactor**3`` children (default ``splitFactor=2``, i.e. eight octree children;
``splitFactor=3`` gives a 3x3x3 split into 27, and so on -- honouring curved edges via the parent
isoparametric map), enforces a one-level face-balance, transfers the converged nodal values (parent isoparametric
interpolation) and quadrature-point history (via a pluggable state-transfer strategy, see
``edelweissfe.adaptivity.statetransfer``) to the children, and couples the resulting
non-conforming interface with an exact hanging-node multi-point constraint. Element/node sets, sections and element-based surfaces are propagated to
the children so material assignment and surface loads stay consistent.

The octree mirror only ever tracks the refineable 20-node elements: a model that also contains
elements of a different kind (e.g. lower-order contact-facet elements bonded to the mesh) is left
untouched by construction, since anything without exactly 20 nodes is skipped automatically. To
restrict refinement explicitly (rather than relying on the node-count heuristic), or to refine only
part of a purely 20-node mesh, set ``refineElSet`` (falls back to ``elSet`` if given) to the element
set that should become the octree mirror. Marking (see below) is likewise restricted to the octree
mirror's own elements by default -- there is nothing to gain from evaluating the marking expression
on an element that can never be refined anyway, and most non-solid element types don't expose most
quadrature-point results in the first place.

The non-conforming 2:1 interface is coupled kinematically rather than by mortar: octree refinement
is non-conforming but *nested*, the QUAD8 face-trace (and 3-node quadratic edge) spaces are
invariant under the axis-aligned affine sub-maps of a uniform subdivision (of *any* factor, not only
bisection), so pinning each hanging
(slave) node to the coarse serendipity trace, :math:`u_s = \sum_a N_a(\xi_s)\, u_{m_a}`, is exact.
The same field-independent weights apply to every field on the node (equal-order serendipity), so a
single record per hanging node covers displacement and nonlocal damage alike. The constraint itself
(``*constraint, type=hangingnode``) is documented under :doc:`constraints`.

**Refinement markers.** Which elements are refined each increment is decided by one or more
``>>marker`` sub-keywords; the modifier refines the *union* of their marked sets. The available
types are ``elementSet`` / ``nodeSet`` / ``surface`` (geometric, typically ``initialOnly`` for a
fixed pre-refinement), ``fieldOutput`` (a boolean threshold on a ``perElement`` field output), and
``recoveryError`` (a Zienkiewicz--Zhu recovered-gradient error estimator on a nodal field, with
Dörfler bulk marking -- for gradient-enhanced damage). The theory of error-estimator marking, the
``averaging`` vs ``spr`` recovery, and why a *reactive* ``recoveryError`` marker is best paired with
a *predictive* ``fieldOutput`` marker ahead of a propagating front are all covered under
:doc:`adaptivitytheory`.

.. pprint:: modelmodifier:hadaptivity
    :caption: Options:

.. literalinclude:: ../../../testfiles/marmot/AMR_DynamicRefinement/test.inp
    :language: edelweiss
    :caption: Example (dynamic refinement of a two-field GC3D20R cantilever):
              ``testfiles/marmot/AMR_DynamicRefinement/test.inp``

Batching refinement across increments
--------------------------------------

Every refinement pass forces the solver to rebuild the equation system (DOF manager, sparsity
pattern, solution vectors and any multi-point-constraint transformation), which is expensive on a
large model. Left unchecked, a marker whose criterion is crossed by elements one at a time --
rather than in a single burst -- triggers that rebuild on every increment a lone element newly
qualifies. ``minMarkedElements`` (default ``1``, i.e. the previous behaviour: refine as soon as
anything is marked) raises the bar: newly marked elements accumulate across increments, and the
modifier defers refining until the accumulated count reaches ``minMarkedElements``, at which point
all of them are refined together in a single pass. This trades refinement latency (a marked element
may sit unrefined, still on the coarse mesh, for a few extra increments) for fewer, larger equation-
system rebuilds.

.. literalinclude:: ../../../testfiles/marmot/AMR_MinMarkedElements/test.inp
    :language: edelweiss
    :caption: Example (refinement deferred indefinitely because only one of the two elements ever
              crosses the marker threshold, so the accumulated count never reaches
              ``minMarkedElements=2``): ``testfiles/marmot/AMR_MinMarkedElements/test.inp``

State-variable transfer strategies
----------------------------------

.. automodule:: edelweissfe.adaptivity.statetransfer
    :members: __doc__

The strategy for handing a refined parent element's quadrature-point history to its children is
selectable with the ``stateTransfer`` argument (default ``nearestQp``):

* ``nearestQp`` -- :class:`~edelweissfe.adaptivity.statetransfer.nearestquadraturepoint.NearestQuadraturePointCopy`:
  each child quadrature point copies, verbatim, the nearest parent quadrature point (matched in the
  parent reference cube). Admissible by construction -- the recommended default.
* ``projection`` -- :class:`~edelweissfe.adaptivity.statetransfer.projection.PolynomialProjection`:
  a tensor-product polynomial is fitted to the parent quadrature-point values by least squares and
  resampled at the children. Smooth across octants, but may produce an inadmissible internal state.
* ``virgin`` -- :class:`~edelweissfe.adaptivity.statetransfer.virgin.VirginState`: children keep
  their freshly-initialised state; history is discarded (sound only when refining ahead of the
  process zone).

Different state variables generally require different treatment, and **which** ones may be projected,
copied, or reset is entirely a property of the constitutive model -- there is no universal choice.
(For a hypoelastic model whose stress update depends only on the *strain increment*, for instance, a
stored total strain may be irrelevant while the true history variables -- stresses, back-stresses,
plastic strains, damage, hardening -- are what matter.) ``stateTransferOverrides`` therefore leaves
the policy to the user: it routes named state variables to their own strategy while the
``stateTransfer`` default handles all the rest. The names are the material's / element's own
state-variable names, located within a quadrature-point block via the element's ``getStateVarSlice``
hook and dispatched by
:class:`~edelweissfe.adaptivity.statetransfer.perstatevar.PerStateVarStateTransfer`. For example,
``stateTransfer=nearestQp`` with ``stateTransferOverrides='<var1>:projection, <var2>:virgin'`` copies
everything except ``<var1>`` (projected) and ``<var2>`` (reset to its initial value).

.. literalinclude:: ../../../testfiles/marmot/AMR_DynamicRefinementProjection/test.inp
    :language: edelweiss
    :caption: Example (per-state-variable transfer device; the routed variable is arbitrary, chosen
              only to exercise the mechanism): ``testfiles/marmot/AMR_DynamicRefinementProjection/test.inp``

Predictor after a refinement
----------------------------

On the increment in which the mesh is mutated the solver already rebuilds the equation system and
starts from a zero predictor. The *following* increment, however, extrapolates from that increment's
solution increment, which conflates the load advance with the one-off warm-start/remesh settling
transient -- a questionable predictor when refining into a softening zone. The ``NISTSolver`` option
``extrapolateAfterModelChange`` (default ``True``, i.e. previous behaviour) can be set ``False`` to
also start the increment after a refinement from a zero predictor::

    >>options, category=NISTSolver, extrapolation=linear, extrapolateAfterModelChange=False

This changes only the Newton starting guess, not the converged solution.

Re-equilibration after a refinement
-----------------------------------

By default the increment in which the mesh is refined advances the load *and* settles the
warm-started refined mesh in a single solve. Near a softening process zone -- exactly where
refinement is triggered -- coupling the load advance with the one-off warm-start settling transient
can prevent recovery. The ``NISTSolver`` option ``equilibrateAfterModelChange`` (default ``False``)
inserts, immediately after a refinement, one constant-load, zero-time re-equilibration increment
(no load advance, zero Dirichlet increment) that settles the refined mesh to equilibrium at the last
converged load *before* the load is advanced::

    >>options, category=NISTSolver, equilibrateAfterModelChange=True

On a path-independent material this is non-destructive (it changes only the increment sequence, not
the converged root); for a history-dependent material it yields a physically distinct, relaxed path,
which is the intended effect. The equilibration solve integrates materials with ``dT = 0``, so it
suits rate-independent models; rate-dependent materials see no time advance during it by design.

Compatibility with facet-based contact and tie
-------------------------------------------------

Refining a solid whose surface feeds a facet-based contact or :mod:`~edelweissfe.constraints.tie`
constraint works out of the box: the modifier keeps the relevant ``*surface`` definition in sync
with the refined child faces, and the constraint -- a :class:`~edelweissfe.models.meshdependent.
MeshDependent` -- regenerates its facets from it. :mod:`~edelweissfe.constraints.
nodetodeformablesurfacepenalty` notices via :meth:`~edelweissfe.models.femodel.FEModel.changesSince`
at its own next connectivity update (a pull, since that tick already runs before the equation
system is rebuilt); a tie has no such early tick of its own (its only hook is called *from inside*
that rebuild, too late to safely swap in new facet elements), so it reconciles via the model's push
notification instead. Either way, no separate wiring is needed. A model with both solid elements to
be refined and pre-existing contact-facet elements should restrict the octree mirror explicitly with
``refineElSet`` (see above), since a facet element is never itself 20-node but need not be excluded
by name::

    *modelModifier, type=hAdaptivity, name=amr
    result=stress
    expression='x > 300.0'
    reducer=absmax
    maxLevel=1
    refineElSet=lower_all

.. literalinclude:: ../../../testfiles/marmot/AMR_ContactRefinePatch/test.inp
    :language: edelweiss
    :caption: Example (the master surface's solid block is refined mid-run while contact is already
              engaged): ``testfiles/marmot/AMR_ContactRefinePatch/test.inp``

.. literalinclude:: ../../../testfiles/marmot/AMR_TieRefine/test.inp
    :language: edelweiss
    :caption: Example (the master surface's solid block is refined mid-run while already tied):
              ``testfiles/marmot/AMR_TieRefine/test.inp``

The rigid-body contact constraints (:mod:`~edelweissfe.constraints.nodetorigidsurfacepenalty`,
:mod:`~edelweissfe.constraints.nodetodiscreterigidbodypenalty`) are likewise ``MeshDependent``, but
lighter still: their master geometry is rigid (an analytic plane, or a triangulated rigid body), so
refinement only ever grows their watched slave ``nSet`` -- no facet regeneration, no per-slave
history, just a refreshed node list at the next :meth:`updateConnectivity` tick.

.. literalinclude:: ../../../testfiles/marmot/AMR_RigidContactRefine/test.inp
    :language: edelweiss
    :caption: Example (the block is refined mid-run while its face already rests against an
              analytic rigid wall): ``testfiles/marmot/AMR_RigidContactRefine/test.inp``

Transparency for ordinary (AMR-unaware) consumers
--------------------------------------------------

A constraint, step action or output that only caches a node/element set or a surface -- e.g.
``self._nodes = model.nodeSets["mySet"]`` at construction -- needs none of the above: it sees a
later refinement automatically, because that container is mutated in place rather than replaced
(see the note at the top of this page). ``testfiles/marmot/AMR_TransparencyProbe`` is the
acceptance test for this: a minimal ``ConstraintBase`` (``edelweissfe.constraints.
amrtransparencyprobe``) that does exactly this, registers no observer and implements no
``MeshDependent``, yet raises if its cached node set is ever found not to have grown across a
refinement it should have seen -- guarding against a regression that reintroduces replacing a set
instead of mutating it.

Restart / checkpointing
------------------------

A model modifier that mutates topology, like ``hAdaptivity``, cannot rely on the plain
reconstruct-then-overwrite scheme every other checkpointed component uses (see ``*restart``): a
refined mesh's new elements and nodes aren't in the ``.inp`` file to rebuild from. Instead, the
checkpoint records only the *decisions* that drove each past change -- and the resumed run replays
them through the modifier's own
:meth:`~edelweissfe.modelmodifiers.base.modelmodifierbase.ModelModifierBase.apply`, the very same
code the live run executed. The marker evaluation that produced a decision is never re-run.

**A modifier does not implement its own restart.** It implements
:meth:`~edelweissfe.modelmodifiers.base.modelmodifierbase.ModelModifierBase.encodePlan` and
:meth:`~edelweissfe.modelmodifiers.base.modelmodifierbase.ModelModifierBase.decodePlan` so that its
decision survives a checkpoint; :class:`~edelweissfe.models.femodel.FEModel` records every applied
decision in :attr:`~edelweissfe.models.femodel.FEModel.topologyHistory` and replays it. An earlier
design had each modifier serializing its own history and implementing its own replay, which is
precisely how a resumed run came to rebuild a differently-numbered mesh -- two implementations of
one mutation always drift. See :doc:`topologypipeline`.

Implementing your own model modifiers
-------------------------------------

Subclass from the model-modifier base class in module
``edelweissfe.modelmodifiers.base.modelmodifierbase``

.. automodule:: edelweissfe.modelmodifiers.base.modelmodifierbase
    :members:
