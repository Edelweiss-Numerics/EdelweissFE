Constraints
===========

.. automodule:: edelweissfe.config.constraints
    :members: __doc__

``equalvaluelagrangian`` - Constrain nodal values to equal values
-----------------------------------------------------------------

Module ``edelweissfe.constraints.equalvaluelagrangian``

.. automodule:: edelweissfe.constraints.equalvaluelagrangian
    :members: __doc__

.. pprint:: constraint:equalvaluelagrangian
    :caption: Options:

.. literalinclude:: ../../../testfiles/marmot/EqualValueLagrangianConstraint/test.inp
    :language: edelweiss
    :caption: Example: ``testfiles/marmot/EqualValueLagrangianConstraint/test.inp``


``equalvaluepenalty`` - Constrain nodal values to equal values
--------------------------------------------------------------

Module ``edelweissfe.constraints.equalvaluepenalty``

.. automodule:: edelweissfe.constraints.equalvaluepenalty
    :members: __doc__

.. pprint:: constraint:equalvaluepenalty
    :caption: Options:

.. literalinclude:: ../../../testfiles/marmot/EqualValuePenaltyConstraint/test.inp
    :language: edelweiss
    :caption: Example: ``testfiles/marmot/EqualValuePenaltyConstraint/test.inp``


``linearizedrigidbody`` - Linearized rigid body constraints
-----------------------------------------------------------

Module ``edelweissfe.constraints.linearizedrigidbody``

.. automodule:: edelweissfe.constraints.linearizedrigidbody
    :members: __doc__

.. pprint:: constraint:linearizedrigidbody
    :caption: Options:

.. literalinclude:: ../../../testfiles/marmot/LinearizedRigidBodyConstraint/test.inp
    :language: edelweiss
    :caption: Example 2D: ``testfiles/marmot/LinearizedRigidBodyConstraint/test.inp``

.. literalinclude:: ../../../testfiles/marmot/LinearizedRigidBodyConstraint2D/test.inp
    :language: edelweiss
    :caption: Example 2D: ``testfiles/marmot/LinearizedRigidBodyConstraint2D/test.inp``

.. literalinclude:: ../../../testfiles/marmot/LinearizedRigidBodyConstraint3D/test.inp
    :language: edelweiss
    :caption: Example 3D: ``testfiles/marmot/LinearizedRigidBodyConstraint3D/test.inp``


``rigidbody`` - Geometrically exact rigid body constraints in 3D
---------------------------------------------------------------------------

Module ``edelweissfe.constraints.rigidbody``

.. automodule:: edelweissfe.constraints.rigidbody
    :members: __doc__

.. pprint:: constraint:rigidbody
    :caption: Options

.. literalinclude:: ../../../testfiles/marmot/RigidBodyConstraintLargeDeformations3D/test.inp
    :language: edelweiss
    :caption: Example: ``testfiles/marmot/RigidBodyConstraintLargeDeformations3D/test.inp``

``penaltyindirectcontrol`` - Penalty based indirect control
-----------------------------------------------------------

Module ``edelweissfe.constraints.penaltyindirectcontrol``

.. automodule:: edelweissfe.constraints.penaltyindirectcontrol
    :members: __doc__

.. pprint:: constraint:penaltyindirectcontrol
    :caption: Options

.. literalinclude:: ../../../testfiles/marmot/PenaltyBasedIndirectControl/test.inp
    :language: edelweiss
    :caption: Example: ``testfiles/marmot/PenaltyBasedIndirectControl/test.inp``


``directionalspringpenalty`` - Assigning a stiffness to specific degrees of freedom
---------------------------------------------------------------------------------------

Module ``edelweissfe.constraints.directionalspringpenalty``

.. automodule:: edelweissfe.constraints.directionalspringpenalty
    :members: __doc__

.. pprint:: constraint:directionalspringpenalty
    :caption: Options:

.. literalinclude:: ../../../testfiles/marmot/DirectionalSpringPenaltyConstraint/test.inp
    :language: edelweiss
    :caption: Example: ``testfiles/marmot/DirectionalSpringPenaltyConstraint/test.inp``

``nodetorigidsurfacepenalty`` - Preventing nodes from penetrating a defined rigid boundary
-----------------------------------------------------------------------------------------------

Module ``edelweissfe.constraints.nodetorigidsurfacepenalty``

.. automodule:: edelweissfe.constraints.nodetorigidsurfacepenalty
    :members: __doc__

.. pprint:: constraint:nodetorigidsurfacepenalty
    :caption: Options:

.. literalinclude:: ../../../testfiles/marmot/NodeToRigidSurfacePenaltyConstraintLinear/test.inp
    :language: edelweiss
    :caption: Example: ``testfiles/marmot/NodeToRigidSurfacePenaltyConstraintLinear/test.inp``

``nodetodiscreterigidbodypenalty`` - Contact against a discrete rigid body
-----------------------------------------------------------------------------------------------

Module ``edelweissfe.constraints.nodetodiscreterigidbodypenalty``

.. automodule:: edelweissfe.constraints.nodetodiscreterigidbodypenalty
    :members: __doc__

This is a frictionless, penalty-based, unilateral (contact-only) constraint that prevents the nodes
of a slave node set from penetrating the moving surface of a
:doc:`discrete rigid body <rigidbodies>`. It is the discrete-mesh counterpart of
``nodetorigidsurfacepenalty`` (which uses an analytical boundary) and reuses the rigid body's
own closest-point surface query for the geometry evaluation. Currently 3D only, and implicit /
explicit-static only (see :doc:`solvers`).

**Theory**

Let a slave node :math:`s` have reference position :math:`\mathbf{X}_s` and displacement DOF
:math:`\mathbf{u}_s`, so its current position is :math:`\mathbf{x}_s = \mathbf{X}_s + \mathbf{u}_s`.
The rigid body carries a reference point (RP) with displacement DOF :math:`\mathbf{u}_{RP}` and a
rotation pseudo-vector DOF :math:`\boldsymbol{\theta}_{RP}`; its current position is
:math:`\mathbf{x}_{RP} = \mathbf{X}_{RP} + \mathbf{u}_{RP}` and its orientation is
:math:`\mathbf{R} = \exp(\mathrm{skew}(\boldsymbol{\theta}_{RP}))` (see
:doc:`rigidbodies` for the rigid-body kinematics).

For each slave node the rigid body's surface query returns the signed distance :math:`d_s`
(negative when the node is *inside* the body) and the outward unit normal :math:`\mathbf{n}_s` of
the closest surface point, evaluated for the body's *current* pose. Contact is active whenever
:math:`d_s < 0`, with gap (penetration)

.. math::
    g_s = -d_s \; > 0 .

The penalty normal force is, for the two available ``type`` formulations,

.. math::
    f_n = k\, g_s \quad(\texttt{linear},\ \text{constant tangent } k),
    \qquad
    f_n = \tfrac{1}{2}\, k\, g_s^2 \quad(\texttt{quadratic},\ \text{tangent } k\, g_s),

with :math:`k` the user-supplied ``penalty``.

With the contact-point moment arm :math:`\mathbf{r}_s = \mathbf{x}_s - \mathbf{x}_{RP}` about the
current RP, the (negative) gap gradient with respect to the involved DOFs
:math:`(\mathbf{u}_s,\, \mathbf{u}_{RP},\, \boldsymbol{\theta}_{RP})` defines the generalized
direction vectors

.. math::
    \mathbf{w}_s = -\mathbf{n}_s , \qquad
    \mathbf{w}_{RP} =
    \begin{bmatrix}
        \mathbf{n}_s \\[2pt]
        \left(\mathbf{R}\, \mathbf{J}_r(\boldsymbol{\theta}_{RP})\right)^{\!\top}
        (\mathbf{r}_s \times \mathbf{n}_s)
    \end{bmatrix},

and the residual (external force) and stiffness contributions are assembled exactly as in the
node-to-rigid-surface penalty constraint,

.. math::
    \mathbf{P}_{ext} \mathrel{-{=}} f_n\, \mathbf{w} , \qquad
    \mathbf{K} \mathrel{+{=}} k\, (\mathbf{w} \otimes \mathbf{w}) + f_n\, \nabla \mathbf{w} ,

per slave node, where :math:`\mathbf{w}` collects the slave and RP blocks.

The factor :math:`\mathbf{J}_r(\boldsymbol{\theta}_{RP})` is the **right Jacobian of the SO(3)
exponential map**,

.. math::
    \mathbf{J}_r(\boldsymbol{\theta}) = \mathbf{I}
    - \frac{1-\cos\varphi}{\varphi^2}\,\mathrm{skew}(\boldsymbol{\theta})
    + \frac{\varphi-\sin\varphi}{\varphi^3}\,\mathrm{skew}(\boldsymbol{\theta})^2 ,
    \qquad \varphi = \lVert \boldsymbol{\theta} \rVert .

It is required because the stored rotation DOF :math:`\boldsymbol{\theta}_{RP}` is the *raw total*
pseudo-vector (fed through the exponential map every iteration), and the exponential map does not
compose additively: incrementing :math:`\boldsymbol{\theta}_{RP}` by
:math:`\delta\boldsymbol{\theta}` does not produce a physical spin of
:math:`\delta\boldsymbol{\theta}` unless the body is currently unrotated. The map
:math:`\mathbf{R}\,\mathbf{J}_r` converts a perturbation of the stored DOF into the physical
spatial rotation it actually produces; it reduces to the identity at
:math:`\boldsymbol{\theta}_{RP} = \mathbf{0}`, where :math:`\mathbf{w}_{RP}` collapses to the
naive :math:`\mathbf{r}_s \times \mathbf{n}_s`. Omitting it makes the RP-moment term a few percent
off at :math:`\sim 0.1` rad accumulated rotation and :math:`\sim 15\%` off at :math:`\sim 0.4` rad.

Both :math:`\mathbf{n}_s` and :math:`\mathbf{r}_s` are recomputed from the *current, total*
solution every Newton iteration (no per-increment caching), so the coupling between the RP DOFs and
the contact residual is exact. Because the rigid surface is a triangulated (piecewise-planar) mesh,
:math:`\mathbf{n}_s` is exactly constant within a facet — the intra-facet curvature term of the
tangent vanishes identically — but it still rotates rigidly with the RP. The geometric-stiffness
term :math:`f_n\, \nabla \mathbf{w}` accounts for that rigid rotation of the normal and for the
:math:`\mathbf{J}_r` dependence, giving the exact linearization of :math:`\mathbf{w}` in the
implemented blocks (verified against finite differences for arbitrary accumulated rotation). The
:math:`\boldsymbol{\theta}_{RP}\boldsymbol{\theta}_{RP}` self-block, which would require the second
derivative of the exponential map, is omitted; this affects only the convergence rate, not the
correctness of the converged solution. The per-block breakdown of the tangent is given in the inline
comments of the ``applyConstraint`` method in the module source.

.. pprint:: constraint:nodetodiscreterigidbodypenalty
    :caption: Options:

.. literalinclude:: ../../../testfiles/edelweiss-only/NodeToDiscreteRigidBodyContact/test.inp
    :language: edelweiss
    :caption: Example: ``testfiles/edelweiss-only/NodeToDiscreteRigidBodyContact/test.inp``

``nodetodeformablesurfacepenalty`` - Node-to-deformable-surface contact
-------------------------------------------------------------------------

The complete theory of this constraint -- surface discretization and triangulation, tributary
areas, finite- and small-sliding gap kinematics, penalty and augmented-Lagrange normal laws,
Coulomb friction with consistent tangents, solver integration, and the verification methodology
-- is documented in :doc:`contacttheory`.

Module ``edelweissfe.constraints.nodetodeformablesurfacepenalty``

.. automodule:: edelweissfe.constraints.nodetodeformablesurfacepenalty
    :members: __doc__

.. pprint:: constraint:nodetodeformablesurfacepenalty
    :caption: Options:

.. literalinclude:: ../../../testfiles/edelweiss-only/NodeToDeformableSurfaceContact/test.inp
    :language: edelweiss
    :caption: Example: ``testfiles/edelweiss-only/NodeToDeformableSurfaceContact/test.inp``

.. literalinclude:: ../../../testfiles/edelweiss-only/NodeToDeformableSurfaceContactFrictionHexa20/test.inp
    :language: edelweiss
    :caption: Example (small sliding, Coulomb friction, hexa20 midside triangulation):
              ``testfiles/edelweiss-only/NodeToDeformableSurfaceContactFrictionHexa20/test.inp``

``tie`` - Surface-to-surface tie (DOF elimination)
--------------------------------------------------

An Abaqus-style tie constraint bonding a slave surface rigidly to a deformable master surface.
Unlike the penalty- and Lagrange-multiplier-based constraints above, the tie is enforced by
master-slave DOF elimination (multi-point constraint condensation): each slave node's
displacement is constrained to the frozen facet-linear interpolation of its closest master facet,
:math:`u_s = \sum_a N_a \, u_{m_a}`, and the slave DOFs are condensed out of the equation system
-- exact enforcement, no penalty parameter, no additional DOFs, zero added stiffness. The same
mechanism serves the implicit solvers (system matrix transformation
:math:`\tilde{K} = T^T K \, T + C`, see
:class:`~edelweissfe.numerics.mpctransformation.MultiPointConstraintTransformation`) and explicit
dynamics (mass-conserving row-sum folding of slave masses and forces onto the masters, direct
kinematic slaving -- the critical time step is untouched).

The implicit solver's ``useAmgclMPCCondensation`` option (:doc:`solvers`, ``NIST``) offers an
alternative way to compute :math:`T^T K T + C` through AMGCL's own OpenMP-threaded sparse
matrix-matrix product/sum instead of SciPy's single-threaded CSR routines (correctness-equivalent
to floating-point precision; measured ~2.4--2.6x faster offline and ~7.3% faster end-to-end live on
a large-scale contact+AMR+tie model). It is off by default, pending a live gate at more thread
counts. An earlier cached-pattern alternative was tried first and removed: it restricted the
SpGEMMs to the eliminated DOFs, but their cost still tracked the full system size, not the
(typically much smaller) number of eliminated DOFs, making it measurably *slower* than the direct
expression -- now superseded by the AMGCL-threaded option on every axis measured.

On matching interface meshes, tying is exactly equivalent to merging the interface nodes: the
patch test passes to machine precision, and both the implicit and the explicit (central
difference) solutions reproduce the monolithic mesh identically. On non-matching meshes the
constraint is still enforced exactly, but the classical node-to-surface limitation applies: the
slave surface's consistent nodal force pattern, redistributed by the facet-linear weights, does
not match the master's own consistent pattern, so a uniform stress state exhibits a bounded,
refinement-convergent interface perturbation (quantified in
``testfiles/edelweiss-only/TieHexa20Patch/test_mismatched.inp``) -- the same obstruction that
motivates mortar methods.

For quadratic (hexa20/quad8) faces, generate both facet surfaces with ``triangulation=midside``:
the corner triangulation excludes the midside nodes from the facet node lists entirely, leaving
slave midside nodes untied.

Module ``edelweissfe.constraints.tie``

.. automodule:: edelweissfe.constraints.tie
    :members: __doc__

.. pprint:: constraint:tie
    :caption: Options:

.. literalinclude:: ../../../testfiles/edelweiss-only/TieHexa20Patch/test.inp
    :language: edelweiss
    :caption: Example (hexa20, midside triangulation): ``testfiles/edelweiss-only/TieHexa20Patch/test.inp``

.. literalinclude:: ../../../testfiles/edelweiss-only/TieNED/test.inp
    :language: edelweiss
    :caption: Example (explicit dynamics): ``testfiles/edelweiss-only/TieNED/test.inp``

Contested slave degrees of freedom
----------------------------------

A degree of freedom can be condensed out only once, so when several multi-point constraints ask for
the same slave DOF, exactly one may keep it. This arises routinely under :math:`h`-adaptivity: a
hanging node created on a refined tie interface is a ``hangingnode`` slave, and the tie would like to
constrain it too.

The decision is made **centrally**, where every constraint's records are collected, and not inside
any constraint: the first constraint in model order keeps the DOF, and later claims on it are
dropped and reported. Resolving it centrally is what makes the result independent of the order in
which constraints are *refreshed* -- a constraint that inspected its peers mid-refresh would read
some of them in their pre-refinement state.

**Model order therefore carries meaning.** For the hanging-node case the precedence is not
arbitrary, and it is expressed by ``hAdaptivity`` registering its hanging-node constraint at the
front of the model's multi-point constraints:

* If the **hanging-node constraint** keeps the node, the refined and unrefined meshes stay exactly
  conforming, and the node is still tied -- indirectly but almost exactly, because its coarse parent
  nodes are themselves tie slaves and it is their interpolation.
* If the **tie** keeps it instead, the node is bonded exactly, but nothing holds it on its coarse
  parent edge any more, and the refined and unrefined meshes come apart there.

One constraint has a stand-in for its condition and the other does not, which is what settles it.
Measured on the coarse anchor pry-out, the two precedences differ by 5.3e-02 relative displacement
and eventually by the refined mesh itself.

.. note::

   This used to be resolved inside the tie, which built its records against its peers'
   ``claimedSlaveNodes()`` while the refresh sweep was still running -- so peers that had not yet
   refreshed answered with their *pre-refinement* claims. A node that had been a hanging node before
   the latest refinement, and was not one after it, was reported as still claimed, and the tie
   dropped it, leaving it constrained by nothing at all.

Multi-point constraints and Dirichlet boundary conditions
----------------------------------------------------------

A degree of freedom eliminated by a multi-point constraint -- a ``tie`` slave, a ``hangingnode``
slave -- cannot also be prescribed by a Dirichlet boundary condition: the constraint already
determines it from its masters, and a prescribed value would be a second, independent equation for
the same unknown. Where a boundary condition's node set overlaps a constraint's slave nodes, the
solvers resolve the overlap automatically: the conflicting constraint equation is dropped,
for that component only, so that the boundary condition takes precedence. This is how Abaqus
resolves the same overlap, and it needs no configuration.

The alternative -- rejecting such a model and requiring the user to subtract the constraint's slave
nodes from the boundary condition's node set -- is worth understanding, because input files written
for solvers that demand it often carry such a subtraction. It is a snapshot: computed against one
mesh, one tie tolerance and one refinement state, and silently no longer correct when any of them
changes, with no diagnostic, because what remains is a plain list of node labels that no longer
records why those labels were chosen. It also interferes with :math:`h`-adaptivity, which extends a
node set to newly created nodes only where the *whole* parent face or edge already lies within the
set: every face touching a subtracted node stops propagating the boundary condition to its children,
so the boundary condition thins out with each refinement. Such subtractions should be removed rather
than maintained.

Dropped equations are reported in two classes, and the distinction matters:

*redundant*
    Every master carrying non-negligible weight is itself prescribed, and the weighted sum
    reproduces the value prescribed on the slave. The constraint equation and the boundary condition
    say the same thing, so dropping either changes nothing. Reported as a count, at journal level 1.
    This is the expected case for a symmetry plane cutting through a tied interface: the slave and
    its masters all lie on the plane and all carry the same prescribed value.

*not implied*
    They do not say the same thing -- some master is unprescribed, or the weighted sum disagrees.
    The boundary condition still wins, but the model has genuinely changed, so this is reported as a
    warning naming how much of the constraint weight rests on unprescribed masters. It is a signal
    worth chasing rather than silencing: the usual cause is a boundary condition whose node set is
    incomplete, so that masters which ought to carry the condition do not.

Masters are resolved transitively before classification -- a master may itself be a slave of another
constraint -- and weights are compared against a scaled tolerance, never against exact zero, since
clamped closest-point projections routinely leave weights at round-off magnitude.

``hangingnode`` - Hanging-node coupling for adaptive mesh refinement
--------------------------------------------------------------------

The kinematic multi-point constraint that couples a non-conforming 2:1 refined interface produced
by hanging-node :math:`h`-adaptivity (see :doc:`modelmodifiers`). Each hanging (slave) node is
tied to the coarse serendipity trace it lies on -- an 8-node QUAD8 face or a 3-node quadratic edge
-- by :math:`u_s = \sum_a N_a(\xi_s)\, u_{m_a}`, enforced by master-slave DOF elimination (no
Lagrange multipliers, no extra DOFs, no saddle point). Because the QUAD8 face-trace and quadratic
edge spaces are nested under octree refinement, the coupling is exact. All fields active on the
node (displacement, nonlocal damage, ...) are constrained with the same field-independent weights.
Multi-level chains are pre-flattened to independent masters, as required by the DOF-elimination
transformation.

The records (masters + weights per slave) are either loaded from a file (``recordsFile``, one line
per slave ``<slaveLabel> <masterLabel> <weight> ...``) for a statically pre-refined mesh, or set in
memory by the :doc:`hAdaptivity model modifier <modelmodifiers>` after each dynamic refinement.

Module ``edelweissfe.constraints.hangingnode``

.. automodule:: edelweissfe.constraints.hangingnode
    :members: __doc__

.. pprint:: constraint:hangingnode
    :caption: Options:

.. literalinclude:: ../../../testfiles/marmot/AMR_PatchTestU/test.inp
    :language: edelweiss
    :caption: Example (static 2:1 patch test): ``testfiles/marmot/AMR_PatchTestU/test.inp``

Implementing your own constraints
---------------------------------

Subclass from the constraint base class in module ``edelweissfe.constraints.base.constraintbase``

.. automodule:: edelweissfe.constraints.base.constraintbase
    :members:

Multi-point (DOF-elimination) constraints subclass from
``edelweissfe.constraints.base.multipointconstraintbase`` instead -- they contribute nothing to
the load vector or system matrix; they only declare linear dependency records, which the solvers
condense via :class:`~edelweissfe.numerics.mpctransformation.MultiPointConstraintTransformation`.

.. automodule:: edelweissfe.constraints.base.multipointconstraintbase
    :members:
