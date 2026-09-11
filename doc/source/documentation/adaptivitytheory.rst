Adaptive refinement: error estimation and marking
==================================================

This page documents the theory behind *marking* for the dynamic hanging-node
:math:`h`-adaptivity modifier (:mod:`~edelweissfe.modelmodifiers.adaptivity.hadaptivity`):
how EdelweissFE decides **which** elements to refine at a topology check, with particular attention to
gradient-enhanced (implicit-gradient) damage models. The mechanics of *how* an element is then
subdivided -- octree children, curved-edge honouring, state transfer, and the exact hanging-node
multi-point constraint -- are documented under :doc:`modelmodifiers`; this page is only about the
refinement *criterion*.

Every formula stated here is implemented literally in
:mod:`~edelweissfe.adaptivity.marking`; the recovered-gradient error indicator has been verified
against a reference implementation and against analytic fields (exactly-linear field :math:`\to`
zero indicator; symmetric mesh and field :math:`\to` equal indicators on mirror elements).


The adaptive loop and where markers sit
---------------------------------------

*When* the topology update runs is the solver's decision, and the two solver families differ: the
implicit solvers run :meth:`~edelweissfe.models.femodel.FEModel.updateTopology` at the start of
every increment, while the explicit dynamic solver runs it at the *end* of a finalized increment
and only every ``topology-check-frequency`` increments -- see
:ref:`adaptivity-explicit-dynamics` below. Either way, the update asks the modifier to
:meth:`~edelweissfe.modelmodifiers.base.modelmodifierbase.ModelModifierBase.plan` and then to
:meth:`~edelweissfe.modelmodifiers.base.modelmodifierbase.ModelModifierBase.apply` its decision.
That update

#. evaluates each configured **marker**, obtaining from each a *set* of elements to refine,
#. **unions** those sets (see :ref:`multiple-markers`),
#. keeps only elements below ``maxLevel``,
#. subdivides them, rebalances, transfers state, and rebuilds the equation system.

A marker is a small object with a single responsibility -- given the current model, return the set
of elements it wants refined. The base class is
:class:`~edelweissfe.adaptivity.marking.MarkerBase`; a marker is declared with a ``>>marker``
sub-keyword of ``*modelModifier, type=hAdaptivity``. The available types are:

``elementSet`` / ``nodeSet`` / ``surface``
    Purely geometric: mark a named set of elements, every element touching a named node set, or
    every element carrying a named surface. Typically used ``initialOnly`` for a fixed
    pre-refinement.

``fieldOutput``
    Mark by thresholding an already-declared ``perElement`` ``*fieldOutput``: ``operator`` (one of
    ``>``, ``>=``, ``<``, ``<=``, ``==``, ``!=``) against ``threshold``, applied element-wise, marking
    the element if any entry passes. The *quantity* being thresholded is shaped by that fieldOutput's
    own ``f(x)`` -- e.g. ``f(x)='np.abs(x)'`` for a magnitude, or
    ``f(x)='eigVal(x.reshape(-1,6))[:,0]...'`` for a principal stress -- so the reduction lives in the
    (reusable) fieldOutput and the marker carries only the refinement policy. See
    :class:`~edelweissfe.adaptivity.marking.FieldOutputMarker`.

``recoveryError``
    A recovery-based *a posteriori error estimator* on the gradient of a nodal field, with Dörfler
    bulk marking -- the subject of most of this page. See
    :class:`~edelweissfe.adaptivity.marking.RecoveryErrorMarker`.

.. _adaptivity-explicit-dynamics:

Adaptivity under explicit dynamics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The explicit dynamic solver ``NED`` (:doc:`solvers`) also runs modifiers, but neither every
increment nor at the same point in one, for reasons that are worth stating because they constrain
how a marker may be configured.

**Cadence.** An explicit increment is thousands of times cheaper than an implicit one and there
are correspondingly more of them -- millions in a routine analysis. Evaluating a marker in each is
not affordable, so the check runs every ``topology-check-frequency`` increments, and
``topology-check-frequency`` must be a **multiple of** ``output-frequency``: a marker reads the
last *finalized* field output, so a check on an increment whose output was never finalized would
decide on stale results. Setting it to ``0`` runs the topology update exactly once, before the
increment loop, which is what an ``initialOnly`` pre-refinement needs; a modifier that acts later
than simulation start is then refused rather than silently ignored (see
:attr:`~edelweissfe.modelmodifiers.base.modelmodifierbase.ModelModifierBase.actsOnlyAtSimulationStart`).

**Placement.** The check sits at the *end* of a finalized increment, not the start, for three
independent reasons: the marker refines on the last finalized field output, so anywhere earlier it
would decide on stale results; the velocity of a central-difference scheme is staggered half an
increment behind the displacement, so the pairing of the two is unambiguous only *between*
increments; and it keeps a topology change disjoint from the failure path, whose restore vectors
are sized by the old equation system. Increment 0 is excluded deliberately -- the time stepper
yields a zero-length increment first, before anything has been solved, and a marker evaluated
there would refine on the initial condition.

**What a refinement changes for an explicit run.** Refining lowers the stable time increment, so
the critical time step is recomputed from the new mesh (unlike a mere contact-connectivity change,
which leaves the mesh alone and therefore reuses it). The lumped mass is rebuilt; because that
mass *is* the operator an explicit increment divides by, the solver reports the total mass, the
per-component linear momentum and the kinetic energy across every topology change, and accumulates
the relative mass drift over all of them so that many individually-tolerable changes cannot
silently add up to a meaningful one. The velocity field is carried across refinement by the same
warm-start interpolation as the displacement -- omitting it would restart the new nodes from rest.


Motivation: gradient-enhanced damage
------------------------------------

In an implicit-gradient (Peerlings/Geers) damage model the *local* driving field
:math:`e_\mathrm{loc}` (an equivalent strain, a plastic hardening variable, an energy release rate)
is regularised into a *nonlocal* field :math:`\bar e` by a screened-Poisson (Helmholtz) equation
with internal length :math:`\ell`,

.. math::
   :label: helmholtz

   \bar e - \ell^2\,\nabla^2 \bar e = e_\mathrm{loc}
   \qquad\text{in }\Omega,\qquad
   \nabla\bar e\cdot\mathbf n = 0 \text{ on }\partial\Omega,

and damage :math:`\omega` evolves from the history :math:`\bar\kappa = \max_\tau \bar e(\tau)`. The
nonlocal field :math:`\bar e` is *smooth* (the Helmholtz operator is smoothing), but across the
process zone it develops a steep gradient over a band of width :math:`\sim\ell`. Resolving that band
-- not the value of :math:`\bar e`, but its **gradient** -- is what controls mesh objectivity of the
dissipated energy.

Equation :eq:`helmholtz` is, term for term, a stationary heat-conduction problem with a reaction
term: the "flux" conjugate to :math:`\bar e` is :math:`\mathbf q = \ell^2\nabla\bar e`. Every
error estimator developed for the temperature field of a thermal analysis therefore transfers
directly by recovering :math:`\nabla\bar e` in place of the heat flux -- which is exactly what the
``recoveryError`` marker does. In EdelweissFE the nonlocal field is an ordinary nodal field named
``nonlocal damage``, carried (for ``GC3D20``/``GC3D20R``) by all 20 serendipity nodes at equal
order with the displacement field.


Recovery-based a posteriori error estimation
--------------------------------------------

For a finite-element field :math:`\bar e^h` the discrete gradient :math:`\nabla\bar e^h` is
*discontinuous* across element boundaries and, for a smooth exact solution, converges one order more
slowly than :math:`\bar e^h` itself. The Zienkiewicz--Zhu idea is to construct a **recovered**
gradient :math:`\nabla\bar e^{*}` that is continuous and a *better* (super-convergent) approximation
of the true gradient, and to use the discrepancy between the two as a local error indicator:

.. math::
   :label: indicator

   \eta_K = \bigl\lVert\, \nabla\bar e^{*} - \nabla\bar e^{h}\, \bigr\rVert_{L^2(K)}
          = \left( \int_K \lvert \nabla\bar e^{*} - \nabla\bar e^{h}\rvert^2 \,\mathrm dV \right)^{1/2},
   \qquad
   \eta^2 = \sum_K \eta_K^2 .

Because :math:`\nabla\bar e^{*}` is closer to the exact gradient than :math:`\nabla\bar e^{h}`, the
element quantity :math:`\eta_K` estimates the local gradient (energy-norm) error. The intuition is
purely geometric:

* where the mesh **resolves** the field, :math:`\nabla\bar e^{h}` is already nearly continuous and
  close to :math:`\nabla\bar e^{*}`, so :math:`\eta_K` is small;
* where a steep band sits on a **too-coarse** mesh, :math:`\nabla\bar e^{h}` jumps sharply from one
  element to the next, the recovered (smoothed) field departs strongly from it, and :math:`\eta_K`
  is large.

The constant factor :math:`\ell^2` relating :math:`\nabla\bar e` to the physical flux is omitted: it
scales every :math:`\eta_K` equally and so cannot change the *ranking* that marking is based on.

The integral in :eq:`indicator` is evaluated at the element's :math:`2\times2\times2` Gauss points,
and both fields are sampled there: :math:`\nabla\bar e^{h}` directly from the shape-function
derivatives, and :math:`\nabla\bar e^{*}` by interpolating the recovered *nodal* gradients with the
element shape functions.


Gradient recovery
~~~~~~~~~~~~~~~~~

Two recovery schemes are available, selected by the ``recovery`` option.

**Nodal averaging** (``recovery='averaging'``, Zienkiewicz--Zhu 1987). The recovered gradient at a
node is the volume-weighted mean of the FE gradient sampled at that node by every element sharing
it,

.. math::

   \nabla\bar e^{*}(\mathbf x_a)
   = \frac{\sum_{K \ni a} V_K\, \nabla\bar e^{h}\big|_K(\mathbf x_a)}
          {\sum_{K \ni a} V_K},

and :math:`\nabla\bar e^{*}` elsewhere is the shape-function interpolation of these nodal values.
Cheap and robust; the default. See :func:`~edelweissfe.adaptivity.marking._recover_averaging`.

**Superconvergent patch recovery** (``recovery='spr'``, Zienkiewicz--Zhu 1992). For each corner
(vertex) node :math:`a` a *patch* :math:`\omega_a` is formed from the elements sharing it, and a
complete polynomial of order :math:`p` (matching the element order) is fitted to the FE gradient
sampled at the **super-convergent points** of the patch -- the element Gauss points -- by least
squares:

.. math::
   :label: spr

   \nabla\bar e^{h}(\mathbf x) \approx \mathbf p(\mathbf x)^\mathsf T\,\mathbf a,
   \qquad
   \mathbf a = \Bigl(\sum_{i\in\omega_a} \mathbf p_i\,\mathbf p_i^\mathsf T\Bigr)^{-1}
               \sum_{i\in\omega_a} \mathbf p_i\, \nabla\bar e^{h}_i,
   \qquad
   \mathbf p = [\,1,\,x,\,y,\,z,\,x^2,\,y^2,\,z^2,\,xy,\,yz,\,zx\,]^\mathsf T .

The recovered value at the node is the fit evaluated there, :math:`\nabla\bar e^{*}(\mathbf x_a) =
\mathbf p(\mathbf x_a)^\mathsf T\mathbf a` (with the fit centred on the node, simply the constant
term :math:`\mathbf a_0`). Edge-**midside** nodes, which are not patch centres, are recovered by
averaging the fits of the two corner patches at each end of the edge, evaluated at the midside
location. Boundary patches with too few sampling points for a full quadratic fall back to a linear
fit, and, failing that, to the plain mean. See
:func:`~edelweissfe.adaptivity.marking._recover_spr`.

SPR is the sharper choice for the serendipity hexahedron and typically has near-unit *effectivity*
(estimated / true error :math:`\approx 1`), at the cost of a small least-squares solve per patch.

.. note::

   **Why the Gauss points.** For the 20-node serendipity hexahedron the gradient is
   *super-convergent* at the :math:`2\times2\times2` reduced-integration (Barlow) points -- exactly
   the points at which ``GC3D20R`` integrates. Sampling the FE gradient there, rather than at the
   nodes, is what makes the recovered field higher-order. Both recovery schemes and the error
   integral use these points.


Marking: which of the ranked elements to refine
------------------------------------------------

The indicators :math:`\{\eta_K\}` are turned into a marked set by **Dörfler (bulk) marking**: sort
the elements by :math:`\eta_K` in descending order and mark the smallest set :math:`\mathcal M`
whose squared indicators first reach a fraction :math:`\theta` of the total,

.. math::
   :label: doerfler

   \sum_{K\in\mathcal M}\eta_K^2 \;\ge\; \theta \sum_{K}\eta_K^2,
   \qquad \theta = \texttt{markFraction}\in(0,1].

This concentrates refinement on the largest error contributors. The marked set is additionally
**hard-capped** at a fraction ``maxRefinedFraction`` of the eligible elements.

**Why a plain ranking suffices here.** Classical adaptivity chooses both *where* and *how deeply* to
refine, iterating until an error target is met. The hanging-node modifier is normally run at a
**fixed single level** (``maxLevel=1``) and a fixed subdivision factor (``splitFactor``): each
marked element is split once, into :math:`\texttt{splitFactor}^3` children. With the refinement
*depth* fixed, only the *location* is adaptive -- so marking collapses to "rank the elements and
refine the worst ones", which is precisely :eq:`doerfler`. A single-level split with
``splitFactor=3`` (a :math:`3\times3\times3 = 27`-child subdivision, exactly hanging-node coupled)
delivers a three-fold local resolution increase in one step, often all that is affordable.

**Why the budget cap matters with a direct solver.** EdelweissFE assembles a single sparse system
and factorises it with a direct solver (PARDISO). Each refinement changes the sparsity pattern,
forcing a fresh symbolic factorisation, and direct-solve cost grows super-linearly in the number of
unknowns. ``maxRefinedFraction`` bounds how many elements a single pass may add, keeping the
per-increment factorisation cost predictable.


.. _multiple-markers:

Reactive and predictive marking: why more than one marker
---------------------------------------------------------

Markers are **composable**: one topology update refines the *union* of all configured markers'
sets. For a *stationary* problem one ``recoveryError`` marker is enough. For a **propagating**
localization -- the usual case in damage -- a single marker is not, and the recommended
configuration uses **two markers together**, for a reason worth stating carefully.

The recovery estimator is **reactive**: :math:`\eta_K` only becomes large once an element *already*
contains a steep, under-resolved gradient. But the process zone **moves**. By the time
:math:`\eta_K` flags an element, the damage band has already entered it *on the coarse mesh* -- and a
coarse discretization of a localizing band biases the dissipated energy, so the very mesh objectivity
that gradient enhancement is meant to restore is compromised for that increment, and the band can
"lock" onto the coarse discretization.

The remedy is a **predictive** marker that refines *ahead* of the front, keyed on a quantity that
becomes active *slightly before* the nonlocal gradient sharpens -- the local driving field
:math:`e_\mathrm{loc}`, a plastic/damage hardening variable (e.g. ``alphaP``), or a stress/strain
criterion -- expressed as a ``fieldOutput`` threshold marker. The two markers answer complementary
questions:

.. list-table::
   :header-rows: 1
   :widths: 22 30 48

   * - Marker
     - Question it answers
     - Type
   * - Reactive
     - *Where is the solution under-resolved now?*
     - ``recoveryError`` (this page)
   * - Predictive
     - *Where is the solution about to develop a feature?*
     - ``fieldOutput`` threshold on the driving field

Neither alone suffices. **Reactive only** is always one step behind the front: the band localizes on
a coarse element before refinement catches up, and, with a direct solver, catching up costs extra
corrector re-solves (each a fresh factorisation). **Predictive only** is a crude heuristic -- no
error control, it over-refines wherever damage merely initiates, cannot adapt to the actual
discretization error, and is blind to any feature not tied to that one state variable. Their union
gives both: the predictive term keeps the mesh fine *before* the band arrives (preserving mesh
objectivity and avoiding extra factorisations), and the reactive estimator supplies principled,
error-controlled refinement of the active zone and anywhere else the discretization is poor. This is
the **predictor--corrector** strategy of phase-field fracture adaptivity, expressed as two
composable markers.


Usage
-----

A ``recoveryError`` marker names the nodal field whose recovered-gradient error drives refinement,
and optionally the recovery method, bulk fraction and budget cap::

    *modelModifier, type=hAdaptivity, name=amr
    ** reactive: ZZ recovered-gradient error of the nonlocal field localizes on the process zone
    >>marker, type=recoveryError, nodeField='nonlocal damage', markFraction=0.5, maxRefinedFraction=0.1, recovery=spr
    ** predictive: refine ahead of the front where the local driving field starts to grow
    >>marker, type=fieldOutput, fieldOutput=alphaPForAMR, operator='>', threshold=1e-4
    elSet=concrete
    maxLevel=1
    splitFactor=3

The predictive marker's ``fieldOutput`` (here ``alphaPForAMR``) is an ordinary ``perElement``
output of the local driving variable, covering the quadrature points of interest with no ``f(x)``
reduction. The full option set of the ``recoveryError`` sub-keyword:

.. pprint:: modelmodifier:hadaptivity
    :caption: ``hAdaptivity`` options (the ``>>marker`` sub-keyword is rendered as its own table):

A minimal, self-contained regression case (a two-element ``GC3D20R`` cantilever whose nonlocal-damage
gradient concentrates at the fixed end, so the marker refines exactly one element) is:

.. literalinclude:: ../../../testfiles/marmot/AMR_RecoveryError/test.inp
    :language: edelweiss
    :caption: ``testfiles/marmot/AMR_RecoveryError/test.inp`` (``recovery='averaging'``; the
              ``AMR_RecoveryErrorSPR`` sibling is identical but for ``recovery=spr``).


Implementation
--------------

.. automodule:: edelweissfe.adaptivity.marking
    :members: RecoveryErrorMarker

Per increment the marker gathers, for every eligible 20-node element, its node coordinates and
nonlocal-field values plus a compact global node index, then evaluates the indicator
:eq:`indicator` for the whole element set at once in
:func:`~edelweissfe.adaptivity.marking._recovery_indicators`: all per-element :math:`3\times3`
Jacobians, gradients and the nodal-averaging accumulation are batched into single ``numpy`` calls
(``einsum`` and a stacked ``linalg.inv``), so there is no Python-level per-element loop for the
default path. The recovery step is delegated to
:func:`~edelweissfe.adaptivity.marking._recover_averaging` or
:func:`~edelweissfe.adaptivity.marking._recover_spr`, and the Dörfler selection :eq:`doerfler` and
budget cap are applied to the resulting ranking.


Limitations and extensions
---------------------------

* The recovered field is resolved at the fixed single refinement level; the estimator ranks *where*
  to refine but does not itself choose refinement *depth* (that is ``maxLevel`` / ``splitFactor``).
* SPR uses a complete quadratic patch basis, reduced to linear (or a mean) on patches too small for
  it; an anisotropic (Hessian-metric) recovered field, which would allow band-aligned stretched
  refinement, is not implemented -- the modifier refines isotropically.
* Refinement is monotone: there is no de-refinement (coarsening) of the wake behind a passed front.
* The internal length :math:`\ell` sets the resolution the band actually needs (roughly
  :math:`h \lesssim \ell/2` for the quadratic serendipity element); the marker chooses *where* to
  reach that resolution, but the initial mesh must at least marginally resolve :math:`\ell` for the
  estimator to fire on a band that is otherwise never captured.


References
----------

* O. C. Zienkiewicz, J. Z. Zhu, *A simple error estimator and adaptive procedure for practical
  engineering analysis*, Int. J. Numer. Methods Eng. **24** (1987) 337--357.
* O. C. Zienkiewicz, J. Z. Zhu, *The superconvergent patch recovery and a posteriori error
  estimates. Part 1: The recovery technique*, Int. J. Numer. Methods Eng. **33** (1992) 1331--1364.
* W. Dörfler, *A convergent adaptive algorithm for Poisson's equation*, SIAM J. Numer. Anal. **33**
  (1996) 1106--1124.
* R. H. J. Peerlings, R. de Borst, W. A. M. Brekelmans, M. G. D. Geers, *Gradient-enhanced damage
  modelling of concrete fracture*, Mech. Cohes.-Frict. Mater. **3** (1998) 323--342.
* T. Heister, M. F. Wheeler, T. Wick, *A primal-dual active set method and predictor-corrector mesh
  adaptivity for computing fracture propagation using a phase-field approach*, Comput. Methods Appl.
  Mech. Eng. **290** (2015) 466--495.
