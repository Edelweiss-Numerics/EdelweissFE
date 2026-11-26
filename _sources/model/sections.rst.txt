Sections
========

Relevant module: ``edelweissfe.config.sections``

.. automodule:: edelweissfe.config.sections
   :members: __doc__

``solid`` - Solid (3D) elements
-------------------------------

Assigns the material to three-dimensional continuum elements. The data lines list the element sets.

.. pprint:: section:solid
   :caption: Options:

.. code-block:: edelweiss

    *section, name=section1, material=linearelastic, type=solid
    all

``plane`` - Plane (2D) elements
-------------------------------

Assigns the material and the out-of-plane ``thickness`` to two-dimensional continuum elements. The data lines list
the element sets.

In a plane model (``*job, domain=2d``) the elements integrate over the slice of this thickness, and so do the contact
points on their faces, each standing for its length times the thickness (:ref:`contact-plane`). In an axisymmetric
model (``*job, domain=axisymmetric``) ``type=plane`` still requires a ``thickness``, but neither the axisymmetric
elements (``CAX4``, ``CAX8``, ``CAX8R``, ``GCAX4``, ``GCAX8``, ``GCAX8R``, ``CX8UL``, ``CX8RUL``) nor their contact
points use it: they integrate over the rings about the axis (:ref:`contact-axisymmetric`).

.. pprint:: section:plane
   :caption: Options:

.. code-block:: edelweiss

    *section, name=section1, thickness=1.0, material=myMaterial, type=plane
    all

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
