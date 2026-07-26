Model modifiers
===============

.. automodule:: edelweissfe.config.modelmodifiers
    :members: __doc__

Unlike constraints, step actions or output managers -- which act on a *fixed* mesh -- a model
modifier may change the mesh topology itself during an analysis: adding or removing nodes and
elements, re-partitioning element/node sets and surfaces, and reallocating the solution fields.
A modifier is declared with the ``*modelModifier`` keyword and is invoked by the solver at the
start of every increment via :meth:`~edelweissfe.modelmodifiers.base.modelmodifierbase.ModelModifierBase.updateModel`;
when it reports a change, the solver rebuilds the equation system (DOF manager, sparsity pattern,
solution vectors and any multi-point-constraint transformation) before continuing.

Because a mutation invalidates references cached by other subsystems (step actions caching node
sets, output managers caching the mesh, field outputs caching result collectors), modifiers
broadcast a :class:`~edelweissfe.models.modelchangeobserver.ModelChangeType` event, together with a
structured :class:`~edelweissfe.models.modelchange.ModelChange` describing exactly what changed,
through the model's observer mechanism (:meth:`~edelweissfe.models.femodel.FEModel.notifyModelChanged`);
push-style subsystems re-bind themselves in their ``onModelChanged`` callbacks.

A subsystem with its own per-increment tick (most do) can instead reconcile lazily, by *pulling*:
compare its own last-seen value against :attr:`~edelweissfe.models.femodel.FEModel.topologyVersion`
(bumped on every mutation) and, on a mismatch, fetch the net change since then via
:meth:`~edelweissfe.models.femodel.FEModel.changesSince`, which coalesces every mutation missed into
a single :class:`~edelweissfe.models.modelchange.ModelChange` -- added/removed nodes and elements,
the parent -> children map, the per-face child tiling, and which node/element sets or surfaces were
touched (with ``touchesSurface``/``touchesNodeSet``/``touchesElementSet`` early-outs so a consumer
can skip a change that doesn't concern it). Pull needs no registration and therefore has no observer
lifecycle to leak, and a consumer always reconciles at its own point of use, after the mutation is
complete.

``hAdaptivity`` - Hanging-node h-adaptivity for HEX20
-----------------------------------------------------

Module ``edelweissfe.modelmodifiers.adaptivity.hadaptivity``

.. automodule:: edelweissfe.modelmodifiers.adaptivity.hadaptivity
    :members: __doc__

Dynamic adaptive :math:`h`-refinement of 20-node serendipity hexahedra (``GC3D20`` / ``GC3D20R``)
in the small-strain, multifield (displacement + nonlocal damage) regime. Each increment the
modifier evaluates a user marking expression on a quadrature-point result, subdivides every marked
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

.. pprint:: edelweissfe.modelmodifiers.adaptivity.hadaptivity.documentation
    :caption: Options:

.. literalinclude:: ../../../testfiles/marmot/AMR_DynamicRefinement/test.inp
    :language: edelweiss
    :caption: Example (dynamic refinement of a two-field GC3D20R cantilever):
              ``testfiles/marmot/AMR_DynamicRefinement/test.inp``

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
history, just a refreshed node list at the next :meth:`updateConnectivity` tick::

.. literalinclude:: ../../../testfiles/marmot/AMR_RigidContactRefine/test.inp
    :language: edelweiss
    :caption: Example (the block is refined mid-run while its face already rests against an
              analytic rigid wall): ``testfiles/marmot/AMR_RigidContactRefine/test.inp``

Implementing your own model modifiers
-------------------------------------

Subclass from the model-modifier base class in module
``edelweissfe.modelmodifiers.base.modelmodifierbase``

.. automodule:: edelweissfe.modelmodifiers.base.modelmodifierbase
    :members:
