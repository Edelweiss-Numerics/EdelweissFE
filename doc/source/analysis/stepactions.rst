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
   :caption: Example: ``testfiles/marmot/BodyForce3D/test.inp``

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

``dissipationcontrol`` - Dissipation control
--------------------------------------------

Relevant module: ``edelweissfe.stepactions.dissipationcontrol``

An arc length controller (see :ref:`arc-length-control`): the solver option ``arcLengthController`` of
``NISTPArcLength`` selects it by its ``name``. It needs elements and materials that provide the gradients of their
dissipation (:attr:`~edelweissfe.elements.base.baseelement.BaseElement.providesEnergyGradients`); which Marmot elements,
sections and materials do is listed in Marmot's documentation of the
`energy gradients <https://materialmodelingtoolbox.github.io/Marmot/pages/features/energygradients.html>`_. An element
without the capability is rejected when the step action is defined; a section or a material that cannot provide the
gradients, when the step it controls starts.

.. automodule:: edelweissfe.stepactions.dissipationcontrol
    :members: __doc__

.. pprint:: stepaction:dissipationcontrol
   :caption: Options:

A bar of ten elements in uniaxial strain, whose three weak elements soften and harden again one after the other, is
pulled through three snap-backs; ``tests/test_dissipationcontrol.py`` compares its path with the analytical solution.
``testfiles/marmot/DissipationControlSnapBacksThreshold`` runs the same bar without a displacement limit, and
``testfiles/marmot/AMR_DissipationControl`` across a mesh refinement.

.. literalinclude:: ../../../testfiles/marmot/DissipationControlSnapBacks/test.inp
   :language: edelweiss
   :caption: Example: ``testfiles/marmot/DissipationControlSnapBacks/test.inp``

``indirectcontrol`` - Indirect displacement control
---------------------------------------------------

Relevant module: ``edelweissfe.stepactions.indirectcontrol``

An arc length controller (see :ref:`arc-length-control`): the solver option ``arcLengthController`` of
``NISTPArcLength`` selects it by its ``name``.

.. automodule:: edelweissfe.stepactions.indirectcontrol
    :members: __doc__

.. pprint:: stepaction:indirectcontrol
   :caption: Options:


.. literalinclude:: ../../../testfiles/marmot/IndirectDisplacementControl/test.inp
   :language: edelweiss
   :caption: Example: ``testfiles/marmot/IndirectDisplacementControl/test.inp``

``indirectcontractioncontrol`` - Indirect displacement -- contraction ring control
----------------------------------------------------------------------------------

Relevant module: ``edelweissfe.stepactions.indirectcontractioncontrol``

An arc length controller (see :ref:`arc-length-control`): the solver option ``arcLengthController`` of
``NISTPArcLength`` selects it by its ``name``.

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

``options`` - Change the options of a solver or an output manager
------------------------------------------------------------------

Changes options of a ``*solver`` or an ``*output`` from this step on. ``name`` is the name of that solver or output
manager; all other options are options of it, validated against its own options (see :doc:`solvers` and
:doc:`/output/outputmanagers`). Only the options given are changed.

.. code-block:: edelweiss

    *solver, name=theSolver, solver=NIST

    *step, solver=theSolver
    maxInc=1e-1, maxNumInc=100
    >>options, name=theSolver, extrapolation=linear

Implementing your own step actions
----------------------------------

Subclass from the step action base class in module ``edelweissfe.stepactions.base.stepactionbase``

.. automodule:: edelweissfe.stepactions.base.stepactionbase
   :members:

Implementing your own loads
---------------------------

A load is a step action that derives from :class:`~edelweissfe.stepactions.base.loadbase.LoadBase` and assembles its
own contribution to the discrete system in ``assembleLoad``: the *negative* of the external load it applies, and the
tangent of that contribution, nonzero only for a load that follows the deformation (see
:ref:`residual-and-tangent-convention`). The solvers do not need to know what a load acts on; their ``assembleLoads``
only loops over the loads of the step and lets each assemble itself. They find the loads by this base class, whatever
step action module defines them: a load module of its own, registered like any step action, needs no change to any
solver, and an arc length step scales it at its end like any other load.

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
