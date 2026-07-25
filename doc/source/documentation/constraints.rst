Constraints
===========

.. automodule:: edelweissfe.config.constraints
    :members: __doc__

``equalvaluelagrangian`` - Constrain nodal values to equal values
-----------------------------------------------------------------

Module ``edelweissfe.constraints.equalvaluelagrangian``

.. automodule:: edelweissfe.constraints.equalvaluelagrangian
    :members: __doc__

.. pprint:: edelweissfe.constraints.equalvaluelagrangian.documentation
    :caption: Options:

.. literalinclude:: ../../../testfiles/marmot/EqualValueLagrangianConstraint/test.inp
    :language: edelweiss
    :caption: Example: ``testfiles/marmot/EqualValueLagrangianConstraint/test.inp``


``equalvaluepenalty`` - Constrain nodal values to equal values
--------------------------------------------------------------

Module ``edelweissfe.constraints.equalvaluepenalty``

.. automodule:: edelweissfe.constraints.equalvaluepenalty
    :members: __doc__

.. pprint:: edelweissfe.constraints.equalvaluepenalty.documentation
    :caption: Options:

.. literalinclude:: ../../../testfiles/marmot/EqualValuePenaltyConstraint/test.inp
    :language: edelweiss
    :caption: Example: ``testfiles/marmot/EqualValuePenaltyConstraint/test.inp``


``linearizedrigidbody`` - Linearized rigid body constraints
-----------------------------------------------------------

Module ``edelweissfe.constraints.linearizedrigidbody``

.. automodule:: edelweissfe.constraints.linearizedrigidbody
    :members: __doc__

.. pprint:: edelweissfe.constraints.linearizedrigidbody.documentation
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

.. pprint:: edelweissfe.constraints.rigidbody.documentation
    :caption: Options

.. literalinclude:: ../../../testfiles/marmot/RigidBodyConstraintLargeDeformations3D/test.inp
    :language: edelweiss
    :caption: Example: ``testfiles/marmot/RigidBodyConstraintLargeDeformations3D/test.inp``

``penaltyindirectcontrol`` - Penalty based indirect control
-----------------------------------------------------------

Module ``edelweissfe.constraints.penaltyindirectcontrol``

.. automodule:: edelweissfe.constraints.penaltyindirectcontrol
    :members: __doc__

.. pprint:: edelweissfe.constraints.penaltyindirectcontrol.documentation
    :caption: Options

.. literalinclude:: ../../../testfiles/marmot/PenaltyBasedIndirectControl/test.inp
    :language: edelweiss
    :caption: Example: ``testfiles/marmot/PenaltyBasedIndirectControl/test.inp``


``directionalspringpenalty`` - Assigning a stiffness to specific degrees of freedom
---------------------------------------------------------------------------------------

Module ``edelweissfe.constraints.directionalspringpenalty``

.. automodule:: edelweissfe.constraints.directionalspringpenalty
    :members: __doc__

.. pprint:: edelweissfe.constraints.directionalspringpenalty.documentation
    :caption: Options:

.. literalinclude:: ../../../testfiles/marmot/DirectionalSpringPenaltyConstraint/test.inp
    :language: edelweiss
    :caption: Example: ``testfiles/marmot/DirectionalSpringPenaltyConstraint/test.inp``

``nodetorigidsurfacepenalty`` - Preventing nodes from penetrating a defined rigid boundary
-----------------------------------------------------------------------------------------------

Module ``edelweissfe.constraints.nodetorigidsurfacepenalty``

.. automodule:: edelweissfe.constraints.nodetorigidsurfacepenalty
    :members: __doc__

.. pprint:: edelweissfe.constraints.nodetorigidsurfacepenalty.documentation
    :caption: Options:

.. literalinclude:: ../../../testfiles/marmot/NodeToRigidSurfacePenaltyConstraintLinear/test.inp
    :language: edelweiss
    :caption: Example: ``testfiles/marmot/NodeToRigidSurfacePenaltyConstraintLinear/test.inp``

``nodetodiscreterigidbodypenalty`` - Contact against a discrete rigid body
-----------------------------------------------------------------------------------------------

Module ``edelweissfe.constraints.nodetodiscreterigidbodypenalty``

.. automodule:: edelweissfe.constraints.nodetodiscreterigidbodypenalty
    :members: __doc__

.. pprint:: edelweissfe.constraints.nodetodiscreterigidbodypenalty.documentation
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

.. pprint:: edelweissfe.constraints.nodetodeformablesurfacepenalty.documentation
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

.. pprint:: edelweissfe.constraints.tie.documentation
    :caption: Options:

.. literalinclude:: ../../../testfiles/edelweiss-only/TieHexa20Patch/test.inp
    :language: edelweiss
    :caption: Example (hexa20, midside triangulation): ``testfiles/edelweiss-only/TieHexa20Patch/test.inp``

.. literalinclude:: ../../../testfiles/edelweiss-only/TieNED/test.inp
    :language: edelweiss
    :caption: Example (explicit dynamics): ``testfiles/edelweiss-only/TieNED/test.inp``

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

.. pprint:: edelweissfe.constraints.hangingnode.documentation
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
