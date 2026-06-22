Analytical fields
=================

.. automodule:: edelweissfe.config.analyticalfields
    :members: __doc__

``scalarexpression`` - Field defined by expression
--------------------------------------------------

Relevant module ``edelweissfe.analyticalfields.scalarexpression``

.. automodule:: edelweissfe.analyticalfields.scalarexpression
    :members: __doc__

.. pprint:: edelweissfe.analyticalfields.scalarexpression.documentation
    :caption: Options

.. literalinclude:: ../../../testfiles/marmot/AnalyticalFieldsScalarExpression/test.inp
    :language: edelweiss
    :caption: Example: ``testfiles/marmot/AnalyticalFieldsScalarExpression/test.inp``

``randomscalar`` - A random field
---------------------------------

Relevant module ``edelweissfe.analyticalfields.randomscalar``

.. automodule:: edelweissfe.analyticalfields.randomscalar
     :members: __doc__

.. pprint:: edelweissfe.analyticalfields.randomscalar.documentation
     :caption: Options

.. literalinclude:: ../../../testfiles/marmot/AnalyticalFieldsRandomScalar/test.inp
     :language: edelweiss
     :caption: Example: ``testfiles/marmot/AnalyticalFieldsRandomScalar/test.inp``

``fromvtk`` - Field interpolated from VTK data
-----------------------------------------------

Relevant module ``edelweissfe.analyticalfields.fromvtk``

.. automodule:: edelweissfe.analyticalfields.fromvtk
     :members: __doc__

.. pprint:: edelweissfe.analyticalfields.fromvtk.documentation
     :caption: Options

Implementing your own fields
----------------------------

Subclass from the field base class in module ``edelweissfe.analyticalfields.base.analyticalfieldbase``

.. automodule:: edelweissfe.analyticalfields.base.analyticalfieldbase
   :members:
