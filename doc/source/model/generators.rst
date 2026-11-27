Generators
==========

Relevant module: ``edelweissfe.config.generators``

.. automodule:: edelweissfe.config.generators
   :members: __doc__

``boxgen`` - A 3D box mesh generator
------------------------------------

Relevant module ``edelweissfe.generators.boxgen``

.. automodule:: edelweissfe.generators.boxgen
    :members: __doc__

.. pprint:: generator:boxgen
   :caption: Options:

.. literalinclude:: ../../../testfiles/marmot/BoxGen/test.inp
    :language: edelweiss
    :caption: Example: ``testfiles/marmot/BoxGen/test.inp``

``planerectquad`` - A 2D plane rectangular mesh generator
---------------------------------------------------------

Relevant module ``edelweissfe.generators.planerectquad``

.. automodule:: edelweissfe.generators.planerectquad
   :members: __doc__,

.. pprint:: generator:planerectquad
   :caption: Options:

.. literalinclude:: ../../../testfiles/marmot/NodeForces/test.inp
    :language: edelweiss
    :caption: Example: ``testfiles/marmot/NodeForces/test.inp``

``pipegen`` - A 3D pipe mesh generator
--------------------------------------

Relevant module ``edelweissfe.generators.pipegen``

.. automodule:: edelweissfe.generators.pipegen
    :members: __doc__

.. pprint:: generator:pipegen
   :caption: Options:

.. literalinclude:: ../../../testfiles/marmot/PipeGen/test.inp
    :language: edelweiss
    :caption: Example: ``testfiles/marmot/PipeGen/test.inp``

``cubit`` - A cubit mesh generator
----------------------------------

Relevant module ``edelweissfe.generators.cubit``

.. automodule:: edelweissfe.generators.cubit
   :members: __doc__

.. pprint:: generator:cubit
   :caption: Options:

.. literalinclude:: ../../../testfiles/marmot/CubitGen/test.inp
    :language: edelweiss
    :caption: Example: ``testfiles/marmot/CubitGen/test.inp``

``findclosestnode`` - Find the closest node
-------------------------------------------

Relevant module ``edelweissfe.generators.findclosestnode``

.. automodule:: edelweissfe.generators.findclosestnode
   :members: __doc__

.. pprint:: generator:findclosestnode
   :caption: Options:


``cylindergenerator`` - A structured hex mesh generator for cylindrical geometries
------------------------------------------------------------------------------------

Relevant module ``edelweissfe.generators.cylindergenerator``

.. automodule:: edelweissfe.generators.cylindergenerator
    :members: __doc__

.. pprint:: generator:cylindergenerator
   :caption: Options:

.. literalinclude:: ../../../testfiles/marmot/CylinderGen/test.inp
    :language: edelweiss
    :caption: Example: ``testfiles/marmot/CylinderGen/test.inp``

``embeddedbond`` - Bond-slip of embedded reinforcement bars
-----------------------------------------------------------

Relevant module ``edelweissfe.generators.embeddedbond``

.. automodule:: edelweissfe.generators.embeddedbond
   :members: __doc__

.. pprint:: generator:embeddedbond
   :caption: Options:

.. literalinclude:: ../../../examples/EmbeddedBondPullOut/test.inp
    :language: edelweiss
    :caption: Example (realistic pull-out test): ``examples/EmbeddedBondPullOut/test.inp``

.. literalinclude:: ../../../testfiles/marmot/EmbeddedBeamPile2D/test.inp
    :language: edelweiss
    :caption: Example (embedded beam): ``testfiles/marmot/EmbeddedBeamPile2D/test.inp``

.. literalinclude:: ../../../testfiles/marmot/EmbeddedBondPullOutLargeSlipTilted2D/test.inp
    :language: edelweiss
    :caption: Example (large slip, bar at 45 degrees to the mesh): ``testfiles/marmot/EmbeddedBondPullOutLargeSlipTilted2D/test.inp``

**Verification.** Two checks a deck can be compared with (both in ``tests/test_embedded_bond.py``):

- A linear bond law (:math:`K_t`) and an elastic bar (:math:`EA`, perimeter :math:`p`, bonded length :math:`L`)
  pulled out of a rigid host: :math:`F/u = EA\,\beta\tanh(\beta L)`, :math:`\beta = \sqrt{K_t\,p/EA}`. The bar
  mesh converges to it with second order in the bar element length.
- A plastic bond law at full debonding (the pull-out example above): the peak force is
  :math:`F_{\max} = \tau_{\max}\, p\, L_b`.
- Large slip, a rigid bar pulled completely out of a rigid host with a linear bond law: the bonded length shrinks as
  :math:`L_b - s`, so :math:`F = K_t\, s\, p\, (L_b - s)` with the peak :math:`K_t\, p\, L_b^2 / 4`, up to the fade
  at the bar end, and the force vanishes once the bar has left the host.

``cuboidlatticegenerator`` - A cuboid lattice mesh generator
------------------------------------------------------------

Relevant module ``edelweissfe.generators.cuboidlatticegenerator``

.. automodule:: edelweissfe.generators.cuboidlatticegenerator
    :members: __doc__

.. pprint:: generator:cuboidlatticegenerator
   :caption: Options:

.. literalinclude:: ../../../testfiles/edelweiss-only/CuboidLatticeGenerator/test.inp
    :language: edelweiss
    :caption: Example: ``testfiles/edelweiss-only/CuboidLatticeGenerator/test.inp``

``microstructuregenerator`` - A unit-cell-based microstructure mesh generator
------------------------------------------------------------------------------

Relevant module ``edelweissfe.generators.microstructuregenerator``

.. automodule:: edelweissfe.generators.microstructuregenerator
    :members: __doc__

.. pprint:: generator:microstructuregenerator
   :caption: Options:

.. literalinclude:: ../../../testfiles/edelweiss-only/MicrostructureGenerator/test.inp
    :language: edelweiss
    :caption: Example: ``testfiles/edelweiss-only/MicrostructureGenerator/test.inp``

``discreterigidbodygenerator`` - A discrete rigid body from a surface mesh file
--------------------------------------------------------------------------------

Relevant module ``edelweissfe.generators.discreterigidbodygenerator``

.. automodule:: edelweissfe.generators.discreterigidbodygenerator
    :members: __doc__

.. pprint:: generator:discreterigidbodygenerator
   :caption: Options:

.. literalinclude:: ../../../testfiles/edelweiss-only/SurfaceToDiscreteRigidBodyContactAtNodes/test.inp
    :language: edelweiss
    :caption: Example: ``testfiles/edelweiss-only/SurfaceToDiscreteRigidBodyContactAtNodes/test.inp``

``executepythoncode`` - Script model generation using Python
------------------------------------------------------------

Relevant module ``edelweissfe.generators.executepythoncode``

.. automodule:: edelweissfe.generators.executepythoncode
    :members: __doc__

.. pprint:: generator:executepythoncode
   :caption: Options:

.. literalinclude:: ../../../testfiles/marmot/PythonCodeModelGeneration/test.inp
    :language: edelweiss
    :caption: Example: ``testfiles/marmot/PythonCodeModelGeneration/test.inp``

.. literalinclude:: ../../../testfiles/edelweiss-only/PythonCodeModelGenerationIndented/test.inp
    :language: edelweiss
    :caption: Example with indented code: ``testfiles/edelweiss-only/PythonCodeModelGenerationIndented/test.inp``

``surfaceElementGenerator`` - Contact facet elements from a *surface
----------------------------------------------------------------------

Relevant module ``edelweissfe.generators.surfaceelementgenerator``

.. automodule:: edelweissfe.generators.surfaceelementgenerator
    :members: __doc__

.. pprint:: generator:surfaceelementgenerator
   :caption: Options:

.. literalinclude:: ../../../testfiles/edelweiss-only/NodeToDeformableSurfaceContact/test.inp
    :language: edelweiss
    :caption: Example: ``testfiles/edelweiss-only/NodeToDeformableSurfaceContact/test.inp``
