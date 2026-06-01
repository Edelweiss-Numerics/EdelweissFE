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

Implementing your own constraints
---------------------------------

Subclass from the constraint base class in module ``edelweissfe.constraints.base.constraintbase``

.. automodule:: edelweissfe.constraints.base.constraintbase
    :members:
