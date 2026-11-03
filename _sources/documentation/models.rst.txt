Models
======

``FEModel`` - A classical finite element model
----------------------------------------------

.. automodule:: edelweissfe.models.femodel
   :members:

``TopologyPipeline`` - How the mesh may change during a run
-----------------------------------------------------------

Held by every model as ``model.topology``. The model is the mesh and its data; the pipeline owns the
topology window, the number allocators, the fixed-point rounds of the model modifiers, the change log
and mesh-dependent consumers, and the recorded history that a restart replays. See
:doc:`topologypipeline` for the design.

.. automodule:: edelweissfe.models.topologypipeline
   :members:
