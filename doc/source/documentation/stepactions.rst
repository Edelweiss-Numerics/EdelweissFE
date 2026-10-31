Step actions
============

.. automodule:: edelweissfe.config.stepactions
   :members: __doc__

``bodyforce`` - Bodyforce loads
-------------------------------

Relevant module: ``edelweissfe.stepactions.bodyforce``

.. automodule:: edelweissfe.stepactions.bodyforce
   :members: __doc__

.. pprint:: stepaction:bodyforce
   :caption: Options:

.. literalinclude:: ../../../testfiles/marmot/BodyForce3D/test.inp
   :language: edelweiss
   :caption: Example: ``testfiles/BodyForce3D/test.inp``

``dirichlet`` - Standard dirichlet BC
-------------------------------------

Relevant module: ``edelweissfe.stepactions.dirichlet``

.. automodule:: edelweissfe.stepactions.dirichlet
   :members: __doc__

.. pprint:: stepaction:dirichlet
   :caption: Options:

``distributedload`` - Distributed surface loads
-----------------------------------------------

Relevant module: ``edelweissfe.stepactions.distributedload``

.. automodule:: edelweissfe.stepactions.distributedload
    :members: __doc__

.. pprint:: stepaction:distributedload
   :caption: Options:


.. literalinclude:: ../../../testfiles/marmot/DLoad/test.inp
   :language: edelweiss
   :caption: Example: ``testfiles/marmot/DLoad/test.inp``

``geostatic`` - Geostatic stress states
---------------------------------------

Relevant module: ``edelweissfe.stepactions.geostatic``

.. automodule:: edelweissfe.stepactions.geostatic
    :members: __doc__

.. pprint:: stepaction:geostatic
   :caption: Options:


.. literalinclude:: ../../../testfiles/marmot/GeoStatic/test.inp
   :language: edelweiss
   :caption: Example: ``testfiles/marmot/GeoStatic/test.inp``

``indirectcontrol`` - Indirect displacement control
---------------------------------------------------

Relevant module: ``edelweissfe.stepactions.indirectcontrol``

.. automodule:: edelweissfe.stepactions.indirectcontrol
    :members: __doc__

.. pprint:: stepaction:indirectcontrol
   :caption: Options:


.. literalinclude:: ../../../testfiles/marmot/IndirectDisplacementControl/test.inp
   :language: edelweiss
   :caption: Example: ``testfiles/marmot/IndirectDisplacementControl/test.inp``

``indirectcontractioncontrol`` - Indirect displacement -- contraction ring control
----------------------------------------------------------------------------------

Relevant module: ``edelweissfe.stepactions.indirectcontrol``

.. automodule:: edelweissfe.stepactions.indirectcontractioncontrol
    :members: __doc__

.. pprint:: stepaction:indirectcontractioncontrol
   :caption: Options:

``initializematerial`` - Initialize materials
---------------------------------------------

Relevant module: ``edelweissfe.stepactions.initializematerial``

.. automodule:: edelweissfe.stepactions.initializematerial
    :members: __doc__

``modelupdate`` - Update the model
----------------------------------

Relevant module: ``edelweissfe.stepactions.modelupdate``

.. automodule:: edelweissfe.stepactions.modelupdate
    :members: __doc__

.. pprint:: stepaction:modelupdate
   :caption: Options:

``nodeforces`` - Concentrated node forces
-----------------------------------------

Relevant module: ``edelweissfe.stepactions.nodeforces``

.. automodule:: edelweissfe.stepactions.nodeforces
    :members: __doc__

.. pprint:: stepaction:nodeforces
   :caption: Options:


.. literalinclude:: ../../../testfiles/marmot/NodeForces/test.inp
   :language: edelweiss
   :caption: Example: ``testfiles/marmot/NodeForces/test.inp``

``setfield`` - Set a field to a prescribed value
------------------------------------------------

Relevant module: ``edelweissfe.stepactions.setfield``

.. automodule:: edelweissfe.stepactions.setfield
    :members: __doc__

.. pprint:: stepaction:setfield
   :caption: Options:

``setinitialconditions`` - Set initial conditions to elements
-------------------------------------------------------------

Relevant module: ``edelweissfe.stepactions.setinitialconditions``

.. automodule:: edelweissfe.stepactions.setinitialconditions
    :members: __doc__

.. pprint:: stepaction:setinitialconditions
   :caption: Options:

``changematerialproperty`` - Change a material property during a step
----------------------------------------------------------------------

Relevant module: ``edelweissfe.stepactions.changematerialproperty``

.. automodule:: edelweissfe.stepactions.changematerialproperty
    :members: __doc__

.. pprint:: stepaction:changematerialproperty
   :caption: Options:

.. literalinclude:: ../../../testfiles/marmot/ChangeMaterialProperties/test.inp
   :language: edelweiss
   :caption: Example: ``testfiles/marmot/ChangeMaterialProperties/test.inp``

Implementing your own step actions
----------------------------------

Subclass from the step action base class in module ``edelweissfe.stepactions.base.stepactionbase``

.. automodule:: edelweissfe.stepactions.base.stepactionbase
   :members:

Implementing your own loads
---------------------------

A load is a step action that derives from :class:`~edelweissfe.stepactions.base.loadbase.LoadBase` and assembles its
own contribution to the equation system in ``assembleLoad``: the *negative* of the external load it applies, and the
tangent of that contribution, nonzero only for a load that follows the deformation (see
:ref:`residual-and-tangent-convention`). The solvers do not need to know what a load acts on; their ``assembleLoads``
only loops over the loads of the step and lets each assemble itself.

Three bases implement ``assembleLoad`` once for all loads of their kind, so that a load deriving from them only says
how large it is at a time (``getCurrentLoad``) and where it acts:

* :class:`~edelweissfe.stepactions.base.nodalloadbase.NodalLoadBase`: concentrated loads on the degrees of freedom of
  a field on a node set, such as ``nodeforces``;
* :class:`~edelweissfe.stepactions.base.distributedloadbase.DistributedLoadBase`: loads per unit area on the faces of
  a surface, which every element integrates over its loaded face, such as ``distributedload``;
* :class:`~edelweissfe.stepactions.base.bodyloadbase.BodyLoadBase`: loads per unit volume, which every element of an
  element set integrates over its volume, such as ``bodyforce``.

A load that acts on something else -- the cells or particles of a meshfree model, say -- derives from
:class:`~edelweissfe.stepactions.base.loadbase.LoadBase` directly.

.. automodule:: edelweissfe.stepactions.base.loadbase
   :members:

.. automodule:: edelweissfe.stepactions.base.nodalloadbase
   :members:

.. automodule:: edelweissfe.stepactions.base.distributedloadbase
   :members:

.. automodule:: edelweissfe.stepactions.base.bodyloadbase
   :members:
