Restart
=======

An analysis writes restart checkpoints with ``*output, type=restart`` (see :doc:`output`) and resumes
from one with ``*restart, readFrom=<checkpoint>`` (see :doc:`keywords`). This page describes the one
rule restart follows, and how each part of the code keeps it.

The rule
--------

**A checkpoint is written after an increment is accepted. A resumed run continues with the next
increment and performs exactly the operations the uninterrupted run performed from there.**

So a resumed run ends *bitwise identical* to the uninterrupted one -- not "close to" it. Nothing in
restart uses a tolerance. The rule has three consequences, each kept by construction:

1. **Everything carried from one increment to the next is written into the checkpoint, exactly as it
   is, and read back.** Nothing is re-derived or reconstructed. What is a pure function of the
   restored model needs nothing.
2. **The step start is not repeated.** Step-start actions (initial conditions, material
   initialization, prescribed fields, model updates), the explicit solver's step-start topology
   update and the first contact search all happened before the checkpoint was written. A resumed step
   skips them. ``step.isResumed`` is the one place the code asks whether it was resumed.
3. **Increment loops decide by increment number.** Topology checks, contact searches, output and
   checkpoints happen at the same increments whether a run started cold or was resumed.

Whatever cannot keep the rule refuses to resume, with a :class:`~edelweissfe.utils.exceptions.RestartError`:
a solver that does not checkpoint its state (the default of
:meth:`~edelweissfe.solvers.base.nonlinearsolverbase.NonlinearSolverBase.readRestart`), and a
checkpoint written by an older version that lacks state the current one needs.

One increment, and where the checkpoint is written
--------------------------------------------------

.. code-block:: text

    increment n                                                   (implicit: every increment;
    ├── topology update     model modifiers (e.g. AMR), then ties   explicit: every
    │                       and contact catch up                    topology-check-frequency)
    ├── solve               Newton iterations / explicit update
    ├── accept              material state, solver state (predictor, external work, ...)
    └── output              field outputs, then the CHECKPOINT
                                                ▲
    a resumed run starts here ──────────────────┘  with increment n + 1

An implicit increment retried after a cutback starts from the same accepted state, so the model
modifiers are not asked again: they decide once per accepted state.

Restoring a checkpoint
----------------------

The model is first rebuilt from the same input file, then restored in this order:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - What
     - How
   * - model time
     - read
   * - mesh
     - the recorded topology decisions are replayed through the model modifiers' own ``apply``
       (:meth:`~edelweissfe.models.topologypipeline.TopologyPipeline.replayHistory`), and the
       result is checked against the recorded fingerprint
   * - ties, contact surfaces
     - catch up with the replayed mesh, as they had in the uninterrupted run
   * - node fields, scalar variables, element state
     - read
   * - constraint state (e.g. a frozen contact search)
     - read; adopted only if it refers to the restored model's contact points, otherwise refused
   * - output managers' sequence bookkeeping
     - read
   * - time stepper
     - its progress within the step
   * - solver
     - its state between increments: the implicit solver's predictor (last accepted increment and
       its ``dU``), the explicit solver's last increment and accumulated external work

Adding something that carries state between increments
-------------------------------------------------------

Write it in ``writeRestart`` (or ``getRestartData`` for constraints and output managers) and read it
back in ``readRestart`` (``setRestartData``) -- the value itself, not something to rebuild it from.
Then add a scenario to ``tests/test_restart_exhaustive.py``: it resumes from every checkpoint of a
small run and requires the bitwise-identical end state, so a forgotten piece of state fails there.
