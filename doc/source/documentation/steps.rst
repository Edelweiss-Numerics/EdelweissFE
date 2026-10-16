Steps
=====

.. automodule:: edelweissfe.steps.base.stepbase
   :members:


Increment control of adaptive steps
-----------------------------------

An adaptive step cuts the increment back by ``cutbackFactor`` when an increment fails, and grows it by
``increaseFactor`` once ``goodIncrementsBeforeIncrease`` good increments have passed, counted from the step start
(the zero increment included) or from the last cutback, which itself counts as one; after that, every further good
increment lets the next one grow. With ``goodIncrementsBeforeIncrease=1`` the increment grows again right after the
increment retried after a cutback. An increment that needed at least
``criticalIter`` iterations does not let the next one grow. The defaults (``increaseFactor=1.1``,
``goodIncrementsBeforeIncrease=3``) recover slowly from a cutback: from a quarter of the previous size it takes about
17 increments to get back. In softening problems with frequent cutbacks, a faster growth is usually much cheaper,
with ``criticalIter`` keeping the increment from growing while Newton struggles, e.g.:

.. code-block:: edelweiss

    *step, solver=theSolver
    maxInc=5e-3, minInc=1e-6, maxIter=25, criticalIter=10, increaseFactor=1.5, goodIncrementsBeforeIncrease=1

``AdaptiveStep`` - A classical adpative step for transient simulations
----------------------------------------------------------------------

.. automodule:: edelweissfe.steps.adaptivestep
   :members:

``AdaptiveStepForExplicitSimulations`` - An adaptive step for explicit simulations
-----------------------------------------------------------------------------------

.. automodule:: edelweissfe.steps.adaptivestepforexplicitsimulations
   :members:
