Sections
========

Relevant module: ``edelweissfe.config.sections``

.. automodule:: edelweissfe.config.sections
   :members: __doc__

``beam`` - Beam elements
------------------------

.. automodule:: edelweissfe.sections.beam
   :members: __doc__

.. pprint:: section:beam
   :caption: Options:

.. literalinclude:: ../../../testfiles/marmot/BeamCantilever2D/test.inp
    :language: edelweiss
    :caption: Example: ``testfiles/marmot/BeamCantilever2D/test.inp``

Section points (module ``edelweissfe.utils.beamsections``):

.. automodule:: edelweissfe.utils.beamsections
   :members: __doc__, lineRule, rectangle, tube, iProfile, polygon, genericPoints, readPoints, readPolygon

``truss`` - Truss (bar) elements
--------------------------------

.. automodule:: edelweissfe.sections.truss
   :members: __doc__

.. pprint:: section:truss
   :caption: Options:

.. literalinclude:: ../../../testfiles/marmot/TrussSnapThroughNeoHooke/test.inp
    :language: edelweiss
    :caption: Example (finite strain): ``testfiles/marmot/TrussSnapThroughNeoHooke/test.inp``
