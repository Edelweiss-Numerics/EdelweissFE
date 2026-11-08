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

An adaptive step begins with an increment of size zero, which brings the model into equilibrium with the state the
step starts from before time advances. ``makeZeroIncrementFirst=False`` leaves it out, e.g. for a model whose entities
integrate in time and cannot be evaluated over a time increment of zero. The zero increment cannot be cut back: if it
fails, the step fails. Any other increment, including a first increment of nonzero size, is cut back when it fails.
An explicit step (``AdaptiveStepForExplicitSimulations``) always begins with the zero increment, in which the explicit
solver builds its initial state, and refuses ``makeZeroIncrementFirst=False``.

Falling back to a restart checkpoint
------------------------------------

An increment that fails at the minimum increment size ``minInc`` fails the step, and so does a failing last increment
of the step that is smaller than ``minInc`` already, the remainder of the step. With ``maxCheckpointFallbacks=n``, the
step instead goes back to its latest restart checkpoint and continues from there with an increment cut back by
``cutbackFactor``, at most ``n`` times:

.. code-block:: edelweiss

    *output, type=restart, name=restart
    writeInterval=5, baseName=restart, numberOfFilesToKeep=3

    *step, solver=theSolver
    maxInc=1e-2, minInc=1e-6, maxCheckpointFallbacks=3

The increments accepted after the checkpoint are discarded, and the step continues exactly as a run resumed from that
checkpoint would (see :doc:`restart`), with the one difference that its next increment is smaller. Their output is
replaced: field output exports are cut back to the checkpoint, and the Ensight sequence continues from it. Path
dependent materials (plasticity, damage) then follow a different, finer path through the same load history, which may
pass where the coarser one got stuck. If the increment the checkpoint holds cannot be cut back either, the step goes
back to the checkpoint before it.

- The checkpoints are those of the restart output, of which the job must define exactly one: with none, or several,
  the step refuses to start. A resumed run may go back to the checkpoint it resumed from, and to the ones the
  interrupted run wrote before it, as long as their files still hold them.
- The step does not go back to a checkpoint that holds the state the failed increment started from, i.e., one written
  after the last increment it accepted: from there, every increment size down to ``minInc`` has just failed. It goes
  back to the checkpoint before it instead. With ``writeInterval=1``, the latest checkpoint always is such a one, so
  a fallback goes back at least one increment before the failed one.
- Every checkpoint serves one fallback: falling back again takes the checkpoint before it, unless the step has written
  newer ones since. How far back the step can go is set by ``writeInterval`` and ``numberOfFilesToKeep`` of the restart
  output.
- The step falls back to checkpoints of its own only, never into an earlier step: a failed zero increment, or a
  failure before the step has written a checkpoint after its zero increment, still fails the step.
- A mesh change cannot be undone, so a step whose mesh has changed since the checkpoint (e.g. by adaptive refinement)
  cannot go back to it, nor to any older one, and fails, naming the checkpoint and the mesh change it predates. A step
  with adaptive refinement can fall back only to checkpoints written after its last refinement.
- Each fallback is reported, together with the performance timing of the discarded part of the step.

``AdaptiveStep`` - A classical adpative step for transient simulations
----------------------------------------------------------------------

.. automodule:: edelweissfe.steps.adaptivestep
   :members:

``AdaptiveStepForExplicitSimulations`` - An adaptive step for explicit simulations
-----------------------------------------------------------------------------------

.. automodule:: edelweissfe.steps.adaptivestepforexplicitsimulations
   :members:
