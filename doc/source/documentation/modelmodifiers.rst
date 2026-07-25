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
broadcast a :class:`~edelweissfe.models.modelchangeobserver.ModelChangeType` event through the
model's observer mechanism (:meth:`~edelweissfe.models.femodel.FEModel.notifyModelChanged`), and
those subsystems re-bind themselves in their ``onModelChanged`` callbacks.

``hAdaptivity`` - Hanging-node h-adaptivity for HEX20
-----------------------------------------------------

Module ``edelweissfe.modelmodifiers.adaptivity.hadaptivity``

.. automodule:: edelweissfe.modelmodifiers.adaptivity.hadaptivity
    :members: __doc__

Dynamic adaptive :math:`h`-refinement of 20-node serendipity hexahedra (``GC3D20`` / ``GC3D20R``)
in the small-strain, multifield (displacement + nonlocal damage) regime. Each increment the
modifier evaluates a user marking expression on a quadrature-point result, subdivides every marked
element into eight octree children (honouring curved edges via the parent isoparametric map),
enforces a 2:1 face-balance, transfers the converged nodal values (parent isoparametric
interpolation) and quadrature-point history (via a pluggable state-transfer strategy, see
``edelweissfe.adaptivity.statetransfer``) to the children, and couples the resulting
non-conforming interface with an exact hanging-node multi-point constraint. Element/node sets, sections and element-based surfaces are propagated to
the children so material assignment and surface loads stay consistent.

The non-conforming 2:1 interface is coupled kinematically rather than by mortar: octree refinement
is non-conforming but *nested*, the QUAD8 face-trace (and 3-node quadratic edge) spaces are
invariant under the axis-aligned affine sub-maps of octree bisection, so pinning each hanging
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

Implementing your own model modifiers
-------------------------------------

Subclass from the model-modifier base class in module
``edelweissfe.modelmodifiers.base.modelmodifierbase``

.. automodule:: edelweissfe.modelmodifiers.base.modelmodifierbase
    :members:
