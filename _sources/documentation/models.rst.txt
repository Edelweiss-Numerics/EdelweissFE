Models
======

``FEModel`` - A classical finite element model
----------------------------------------------

.. automodule:: edelweissfe.models.femodel
   :members:

``DiscreteSystem`` - What the discrete system of a model is made of
-------------------------------------------------------------------------

What a model answers a solver that (re)builds its discrete system: the numbering of the degrees of freedom, the
elements and constraints to compute, and how the solution passes between the model and the solver's vectors. See
:ref:`model-discrete-system`.

.. automodule:: edelweissfe.models.discretesystem
   :members:

``TopologyPipeline`` - How the mesh may change during a run
-----------------------------------------------------------

Held by every model as ``model.topology``. The model is the mesh and its data; the pipeline owns the
topology window, the number allocators, the fixed-point rounds of the model modifiers, the change log
and mesh-dependent consumers, and the recorded history that a restart replays. See
:doc:`topologypipeline` for the design.

.. automodule:: edelweissfe.models.topologypipeline
   :members:
