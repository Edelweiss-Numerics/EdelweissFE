Constraints
===========

.. automodule:: edelweissfe.config.constraints
    :members: __doc__

``embeddedregion`` - Embedded elements (perfect bond of reinforcement)
----------------------------------------------------------------------

Ties every node of the embedded elements (e.g., truss or beam elements of reinforcement bars or structural members,
meshed independently of the host) to the host continuum element it lies in,
:math:`u_s = \sum_a N_a(\xi_s)\, u_{a}`, by master-slave DOF elimination, like Abaqus' ``*EMBEDDED ELEMENT``.

The nodal ``P`` of an embedded (slave) node is reported on the slave node itself, before condensation: a reaction
summed over a node set must include the embedded nodes on that boundary.

Module ``edelweissfe.constraints.embeddedregion``

.. automodule:: edelweissfe.constraints.embeddedregion
    :members: __doc__

.. pprint:: constraint:embeddedregion
    :caption: Options:

.. literalinclude:: ../../../testfiles/marmot/EmbeddedRegionRebar2D/test.inp
    :language: edelweiss
    :caption: Example: ``testfiles/marmot/EmbeddedRegionRebar2D/test.inp``

.. literalinclude:: ../../../testfiles/marmot/EmbeddedBeamCantilever2D/test.inp
    :language: edelweiss
    :caption: Example (embedded beam, rotations tied): ``testfiles/marmot/EmbeddedBeamCantilever2D/test.inp``

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


``linearizedrigidbodympc`` - Linearized rigid body constraints by DOF elimination
-----------------------------------------------------------------------------------

Module ``edelweissfe.constraints.linearizedrigidbodympc``

.. automodule:: edelweissfe.constraints.linearizedrigidbodympc
    :members: __doc__

.. pprint:: constraint:linearizedrigidbodympc
    :caption: Options:

.. literalinclude:: ../../../testfiles/marmot/LinearizedRigidBodyMPCConstraint3D/test.inp
    :language: edelweiss
    :caption: Example: ``testfiles/marmot/LinearizedRigidBodyMPCConstraint3D/test.inp``


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

``nodetodeformablesurfacepenalty`` - Node-to-deformable-surface contact
-------------------------------------------------------------------------

The complete theory of this constraint -- surface discretization and triangulation, tributary
areas, finite- and small-sliding gap kinematics, penalty and augmented-Lagrange normal laws,
Coulomb friction with consistent tangents, solver integration, and the verification methodology
-- is documented in :doc:`/theory/contact`.

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

``surfacetodeformablesurfacepenalty`` - Integrated surface-to-surface contact
-----------------------------------------------------------------------------

The integrated (Gauss-point-to-segment) counterpart of the constraint above: contact is evaluated
at *quadrature points* over the slave surface's facets, and the resulting pressure is distributed
with each side's **parent element face** shape functions rather than with the flat facets' own.

Use this instead of ``nodetodeformablesurfacepenalty`` on quadratic (hexa20/quad8) contact
surfaces. There, a node-based penalty scheme cannot transmit a correct pressure at all -- the
consistent nodal load of a uniform pressure at a serendipity corner is *tensile*, which a
unilateral spring cannot exert, so the discrete solution opens every corner gap instead. The
mechanism, the measurements, and why no choice of nodal weights can repair it are documented in
:ref:`serendipity-liftoff`; the integrated formulation and its own limits are in
:ref:`integrated-contact`. On *matched* linear faces the parent-face basis coincides with the facet
basis, so the two constraints agree to 13 digits and there is nothing to gain -- but on
**non-matching** meshes the integrated formulation is a substantial improvement at any element
order, linear included, as the figure in :ref:`integrated-contact` shows.

The constraint is one contact for every kind of contact point (:ref:`contact-one-constraint`): the
quadrature points of the slave facets (``contactPoints=quadraturePoints``, the default) and the slave
nodes, each standing for its tributary area (``contactPoints=nodes``). Points that approach the master within
``searchDistance`` are paired with it; a point found inside without having been seen approaching gets no force, is
reported, and counts as found inside until it is seen outside again. A point in contact slides from facet to facet,
but not across a convex crease; in every update it walks over the master until its facet no longer changes, so that
an update repeated at the same configuration, as a resumed run repeats one, pairs it the same. The master may have
corners: at its creases -- a crest, a re-entrant root -- the gap is the distance to the crease, signed by its
pseudonormal (:ref:`contact-corners`). A new mesh of the master keeps what the pairing learnt about the points;
regenerated contact points are paired as at the first update. An overlap at the start within ``searchDistance`` (a
deeper one is found inside, and reported) is either an interference fit, pushed apart at once or
over ``initialOverlapRampTime`` (``initialOverlap=pushApart``, the default), or an error of the geometry, such as flat
faces in a curved hole, kept as the stress-free state (``initialOverlap=adjust``; :ref:`contact-initial-overlap`).

With ``type=tied`` the constraint ties its points to the master surface instead (:ref:`contact-tie`): each point within
``searchDistance`` at the first update is paired once, with the closest point of its closest facet, mapped onto the
facet's parent face, and held there in every direction, in tension and compression, by penalty springs; a gap or an
overlap at the pairing is part of the state at rest. It joins independently meshed bodies in implicit and explicit
analyses. A new mesh of either side hands the tie on: the same points stay tied to the same material points of the
master (on a curved master, to the nearest points of the new chords, mapped onto the parent face; the resting offset
takes up the difference, so the forces stay exact), displaced from rest as at the last accepted state (checkpointed),
interpolated onto the new points of a refined slave side. It is not the ``tie`` constraint below, which eliminates the
degrees of freedom of slave nodes exactly; the penalty tie is compliant by :math:`t/k`, adds to the frequencies of an
explicit analysis, works for both kinds of contact point on the parent faces, and does not compete with hanging nodes.
Both keep a gap at rest, the ``tie`` with ``adjust=False``. Which to use when: :ref:`Two ties <contact-two-ties>`.

Scope: frictionless normal penalty contact, or a tie, under ``sliding=small``, on both the implicit and the explicit
solver path. Coulomb friction and augmented Lagrange are **not** implemented here (the node-based
constraint has both), and ``sliding=finite`` is rejected rather than approximated. The master
surface must consist of a single element type, and so must a slave surface whose contact points are its quadrature
points.

Module ``edelweissfe.constraints.surfacetodeformablesurfacepenalty``

.. automodule:: edelweissfe.constraints.surfacetodeformablesurfacepenalty
    :members: __doc__

.. pprint:: constraint:surfacetodeformablesurfacepenalty
    :caption: Options:

.. literalinclude:: ../../../testfiles/edelweiss-only/SurfaceToDeformableSurfaceContactPatchHexa20/test.inp
    :language: edelweiss
    :caption: Example (the hexa20 acceptance test, with the measured results in its header):
              ``testfiles/edelweiss-only/SurfaceToDeformableSurfaceContactPatchHexa20/test.inp``

.. literalinclude:: ../../../testfiles/edelweiss-only/NEDSurfaceContact/test.inp
    :language: edelweiss
    :caption: Example (explicit dynamics):
              ``testfiles/edelweiss-only/NEDSurfaceContact/test.inp``

.. literalinclude:: ../../../testfiles/edelweiss-only/SurfaceTieHexa20AMR/test.inp
    :language: edelweiss
    :caption: Example (a tie with a gap at rest, across a refinement of its master side):
              ``testfiles/edelweiss-only/SurfaceTieHexa20AMR/test.inp``

``surfacetodiscreterigidbodypenalty`` - Integrated contact against a discrete rigid body
-----------------------------------------------------------------------------------------------

The rigid-body counterpart of ``surfacetodeformablesurfacepenalty``: contact is evaluated at the contact points of a
slave surface and distributed with their interpolation vectors -- by default at quadrature points, with the slave
parent-face shape functions --, against the triangulated, closed surface of a
:doc:`discrete rigid body </model/rigidbodies>`. The theory is documented in :ref:`integrated-contact-rigid-body`. Use
it for any contact against rigid supports or indenters, in particular with quadratic slave faces.

Its slave side is the same as that of ``surfacetodeformablesurfacepenalty`` (:ref:`contact-points`): the
quadrature points of the slave facets (``contactPoints=quadraturePoints``, the default) and the slave nodes, each
standing for its tributary area (``contactPoints=nodes``) -- so the nodes and the faces of finite elements hit a rigid
body through one constraint. Points farther than ``searchDistance`` from the rigid surface take no part until they
approach.

Scope: frictionless normal penalty contact under ``sliding=small``, 3D, implicit and explicit solver paths, live
AMR of the slave side. The slave surface must consist of a single element type.

**Moving from** ``nodeToDiscreteRigidBodyPenalty``. That node-based constraint is removed; this one with
``contactPoints=nodes`` takes the same nodes as contact points. An input moves in four steps:

1. The slave side is a surface, not a node set: generate the facets of the face that touches the body with the
   ``surfaceElementGenerator`` and give them as ``slaveSurface`` (``nSet=block_back`` becomes the facets of
   ``surface=block_back``), with ``contactPoints=nodes``. On quadratic (hexa20/quad8) faces, take the default
   quadrature points instead: nodes cannot carry the tensile corner loads there (:ref:`serendipity-liftoff`).
2. ``penalty`` is a stiffness per unit area: divide the penalty per node by the tributary area of a node, on a
   uniform mesh of linear faces of size :math:`h` by :math:`h^2`. A node on the edge of the surface then has half
   the stiffness of an interior one, at a corner a quarter -- its area -- where the penalty per node gave every
   node the same; and the stiffness of the interface no longer changes with the mesh
   (:ref:`integrated-contact-rigid-body`).
3. The gap is that of small sliding: at every contact search (every increment of an implicit analysis) a node is
   paired with its closest triangle, and until the next search it sees that triangle's plane, moving with the body.
   The removed constraint searched the closest point of the surface anew in every iteration (finite sliding). A
   node that slides over much of a triangle within one increment, past an edge or around a curved body, is
   therefore followed only from one increment to the next: take increments that are small against the triangles.
   ``searchDistance`` keeps its meaning: a node farther from the body takes no part.
4. The surface of the rigid body must be closed and consist of triangles, which is checked when the constraint is
   created: split the quadrilaterals of a surface mesh in two (for instance with PyVista's ``triangulate()``) and
   give the body the triangulated file.

The deck below is the former example of the node-based constraint, moved this way.

Module ``edelweissfe.constraints.surfacetodiscreterigidbodypenalty``

.. automodule:: edelweissfe.constraints.surfacetodiscreterigidbodypenalty
    :members: __doc__

.. pprint:: constraint:surfacetodiscreterigidbodypenalty
    :caption: Options:

.. literalinclude:: ../../../testfiles/edelweiss-only/SurfaceToDiscreteRigidBodyContactPatchHexa20/test.inp
    :language: edelweiss
    :caption: Example (hexa20 block on a rigid support):
              ``testfiles/edelweiss-only/SurfaceToDiscreteRigidBodyContactPatchHexa20/test.inp``

.. literalinclude:: ../../../testfiles/edelweiss-only/NEDSurfaceToDiscreteRigidBodyContact/test.inp
    :language: edelweiss
    :caption: Example (explicit dynamics):
              ``testfiles/edelweiss-only/NEDSurfaceToDiscreteRigidBodyContact/test.inp``

.. literalinclude:: ../../../testfiles/edelweiss-only/SurfaceToDiscreteRigidBodyContactAtNodes/test.inp
    :language: edelweiss
    :caption: Example (the nodes of a block against a rotating cylinder, ``contactPoints=nodes``):
              ``testfiles/edelweiss-only/SurfaceToDiscreteRigidBodyContactAtNodes/test.inp``

``tie`` - Surface-to-surface tie (DOF elimination)
--------------------------------------------------

An Abaqus-style tie constraint bonding a slave surface rigidly to a deformable master surface.
Unlike the penalty- and Lagrange-multiplier-based constraints above -- among them the penalty tie,
``surfaceToDeformableSurfacePenalty`` with ``type=tied``, compared with this one in
:ref:`Two ties <contact-two-ties>` -- the tie is enforced by master-slave DOF elimination (multi-point constraint
condensation): each slave node's
displacement is constrained to the frozen facet-linear interpolation of its closest master facet,
:math:`u_s = \sum_a N_a \, u_{m_a}`, and the slave DOFs are condensed out of the system of equations
-- exact enforcement, no penalty parameter, no additional DOFs, zero added stiffness. The same
mechanism serves the implicit solvers (system matrix transformation
:math:`\tilde{K} = T^T K \, T + I_s`, the identity :math:`I_s` filling the structurally zero
slave rows and columns, followed by the recovery :math:`\delta u_s \mathrel{+}= \sum_a N_a \,
\delta u_{m_a}` after each linear solve -- symmetric whenever :math:`K` is, and free of any scale of
its own, so it equilibrates cleanly also for a consistent mass matrix; see
:class:`~edelweissfe.numerics.mpctransformation.MultiPointConstraintTransformation`) and explicit
dynamics (mass-conserving row-sum folding of slave masses and forces onto the masters, direct
kinematic slaving -- the critical time step is untouched).

The implicit solver's ``useThreadedMPCCondensation`` option (:doc:`/analysis/solvers`, ``NIST``) computes
:math:`T^T K T + I_s` in place, with an OpenMP-threaded kernel
(:class:`~edelweissfe.numerics.mpccondensation.MPCCondensation`), instead of with SciPy's
single-threaded CSR routines, which build a second matrix: the system matrix is held once, not twice.
For this, the CSR pattern of :math:`K` gains, once per discrete system, the entries where a master
collects the couplings of its slaves
(:meth:`~edelweissfe.numerics.mpctransformation.MultiPointConstraintTransformation.condensedPatternFillIn`);
the slave columns of :math:`K` become stored zeros in turn. Only the rows a constraint touches are
computed, each by one thread from the rows of :math:`K` it combines, in a fixed order of summation, so
the result does not depend on the number of threads. Explicit zeros are pruned from a copy according to
``pruneCondensedMatrixZeros``. It is off by default.

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
by hanging-node :math:`h`-adaptivity (see :doc:`/analysis/modelmodifiers`). Each hanging (slave) node is
tied to the coarse serendipity trace it lies on -- an 8-node QUAD8 face or a 3-node quadratic edge
-- by :math:`u_s = \sum_a N_a(\xi_s)\, u_{m_a}`, enforced by master-slave DOF elimination (no
Lagrange multipliers, no extra DOFs, no saddle point). Because the QUAD8 face-trace and quadratic
edge spaces are nested under octree refinement, the coupling is exact. All fields active on the
node (displacement, nonlocal damage, ...) are constrained with the same field-independent weights.
Multi-level chains are pre-flattened to independent masters, as required by the DOF-elimination
transformation.

The records (masters + weights per slave) are either loaded from a file (``recordsFile``, one line
per slave ``<slaveLabel> <masterLabel> <weight> ...``) for a statically pre-refined mesh, or set in
memory by the :doc:`hAdaptivity model modifier </analysis/modelmodifiers>` after each dynamic refinement.

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
and implement its ``computeKernels``. A constraint enters the discrete system like an element does
(see :ref:`residual-and-tangent-convention`): it adds its residual contribution :math:`\mathbf{P}`
-- the negative of the force it exerts on its nodes -- and the tangent
:math:`\mathbf{K} = \partial \mathbf{P} / \partial \mathbf{U}`. Two examples:

* A **penalty** spring of stiffness :math:`k` that pulls a degree of freedom :math:`u` towards
  :math:`\bar{u}` exerts the force :math:`-k\,(u - \bar{u})` on its node, so it adds
  :math:`P \mathrel{+}= k\,(u - \bar{u})` and :math:`K \mathrel{+}= k`, exactly as a spring element
  would. ``directionalspringpenalty`` is this constraint.
* A **Lagrange multiplier** :math:`\boldsymbol{\lambda}` enforcing :math:`\mathbf{g}(\mathbf{u}) =
  \mathbf{0}` adds :math:`\boldsymbol{\lambda}^T \partial \mathbf{g} / \partial \mathbf{u}` to the rows
  of the nodes and :math:`\mathbf{g}` to the rows of the multipliers, with the tangent
  :math:`\begin{bmatrix} \boldsymbol{\lambda}^T \partial^2 \mathbf{g} / \partial \mathbf{u}^2 &
  (\partial \mathbf{g} / \partial \mathbf{u})^T \\ \partial \mathbf{g} / \partial \mathbf{u} &
  \mathbf{0} \end{bmatrix}`. ``equalvaluelagrangian`` and ``rigidbody`` are such constraints.

Because :math:`\mathbf{K}` is the derivative of exactly what is added to :math:`\mathbf{P}`, a
constraint can be checked by finite differences of its own ``computeKernels``: a wrong tangent, or a
wrong sign of :math:`\mathbf{P}` alone, shows up as a mismatch. :math:`\mathbf{P}` and :math:`\mathbf{K}` negated
together pass this check, so the test of a constraint also checks the direction of the forces it exerts on its nodes:
a penalty pulls them back towards what it prescribes, a contact pushes the bodies apart, a Lagrange multiplier acts on
them as :math:`\boldsymbol{\lambda}^T \partial \mathbf{g} / \partial \mathbf{u}` with the constraint's own
:math:`\mathbf{g}`. ``tests/test_residual_tangent_convention.py`` does both for the constraints shipped with
EdelweissFE (the two integrated surface contact constraints in their own test modules); add a case there for a new
one. The helpers it uses are part of the package, so that the tests of a package with constraints of its own use them
as well:

.. automodule:: edelweissfe.utils.residualtangentcheck
    :members:

A constraint takes part in the discrete system while it is ``active``. A constraint that is not active gets no
degrees of freedom, the solvers do not evaluate it, and its scalar variables -- the Lagrange multipliers of
``equalvaluelagrangian``, say -- leave the discrete system with it, so that they leave no unknowns without an equation
behind. Two states decide it, and ``active`` is true while both are:

* ``switchedOn``: the constraint is **switched on from outside**. Setting ``active`` switches it, e.g. a
  ``modelupdate`` step action that switches a constraint off for a step, as in
  ``testfiles/marmot/PenaltyBasedIndirectControl``. A switch-off from outside lasts until the constraint is switched
  on again from outside.
* ``withinReach``: the constraint finds itself within reach of what it constrains, in its ``updateConnectivity``.
  A contact constraint whose partner is out of reach sets it to false. It does not report the change of ``active``: the
  model compares which constraints take part before and after, and tells the solver (a
  :class:`~edelweissfe.models.change.Change`, see :ref:`model-discrete-system`). A constraint that does not check stays
  within reach.

A constraint reports from ``updateConnectivity`` only whether the nodes it couples changed. A constraint that
searches as the bodies move, e.g. a contact, which searches for the surface it touches, declares it with
``connectivityChangesWithMotion = True``, whether or not the nodes it couples change with the search (the contact
with a discrete rigid body without ``searchDistance`` locates every point and keeps its nodes, but finds their closest
triangles anew): it has to be updated again and again, so a solver
that never updates the connectivity (``NEST``) refuses it, and ``NED`` updates it only every
``contact-update-frequency`` increments. A constraint with fixed nodes and nothing to search leaves it False.

So a constraint switched off from outside stays off, whatever it finds in reach; switched on again, it takes part
while it is within reach. Switched off from outside, a constraint rests: the model does not locate it anew in
``updateConnectivity``, which it does for every constraint that is switched on, those out of reach included, so that
they can find themselves within reach again. A constraint that is not active accepts no state either
(``acceptLastState``): a contact switched off for a step keeps its multipliers and its frictional history as they
were. A solver that cannot handle a constraint, e.g. ``NEST`` a contact, refuses it only while it is switched on.

.. automodule:: edelweissfe.constraints.base.constraintbase
    :members:

In an explicit analysis, a penalty constraint adds the stiffness it has to the squared frequencies of the degrees of
freedom it couples, and so lowers the critical time step (``addBoundOfAbsoluteStiffnessTimes``,
:doc:`/theory/criticaltimestep`, "Penalty constraints"): by default from its tangent, also where it evaluates its
explicit forces without one. A constraint that has no tangent, or whose tangent is costly, or that needs the bound over
the next increment (``follow-critical-time-step``) while springs of it that close in different states share a degree
of freedom within its block, gives its springs instead, with one formula for the bound and one for the smallest gaps
within the next increment; a worked example from the tangent is ``directionalSpringPenalty``
(``tests/test_ned_penalty_constraints.py``), from the springs EdelweissMeshfree's
``ParticlePenaltyCartesianBoundaryConstraintExplicit``:

.. automodule:: edelweissfe.constraints.base.penaltysprings
    :members:

The penalty contacts are assembled from two pieces of their own: the *contact points* of the slave side
(:ref:`contact-points`) and the *law* that puts penalty springs at the paired points (:ref:`contact-law`). A contact
constraint takes its slave side as contact points: points that move with an interpolation vector over the nodes they
act on and stand for an area. The interface says what a constraint needs from them; the two kinds are the nodes of a
slave surface and the quadrature points of its faces. A new kind of contact point implements
:class:`~edelweissfe.constraints.base.contactpoints.ContactPoints`:

.. automodule:: edelweissfe.constraints.base.contactpoints
    :members:

.. automodule:: edelweissfe.constraints.base.contactpointsatslavenodes
    :members:

.. automodule:: edelweissfe.constraints.base.contactpointsonslavesurface
    :members:

A new law derives from :class:`~edelweissfe.constraints.base.penaltylaw.PenaltyLaw` (its springs, and whether it
pairs once), a new normal contact law from :class:`~edelweissfe.constraints.base.penaltylaw.ContactPenaltyLaw` (its
normal force and energy):

.. automodule:: edelweissfe.constraints.base.penaltylaw
    :members:

Multi-point (DOF-elimination) constraints subclass from
``edelweissfe.constraints.base.multipointconstraintbase`` instead -- they contribute nothing to
the load vector or system matrix; they only declare linear dependency records, which the solvers
condense via :class:`~edelweissfe.numerics.mpctransformation.MultiPointConstraintTransformation`.

.. automodule:: edelweissfe.constraints.base.multipointconstraintbase
    :members:
