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
   initialization, prescribed fields), the zero increment, the step-start topology update and the
   first contact search all happened before the checkpoint was written. Whether a step is at its
   start is part of the time stepper's state
   (:meth:`~edelweissfe.timesteppers.base.timestepperbase.TimeStepperBase.isAtStepStart`): nothing
   accepted yet. A restored time stepper never is, so a resumed step skips them -- there is no
   separate "resumed" flag.
3. **The increment loop is written once, and decides by increment number.** Topology checks,
   contact searches, output and checkpoints happen at the same increments whether a run started cold
   or was resumed.

Whatever cannot keep the rule refuses to resume, with a :class:`~edelweissfe.utils.exceptions.RestartError`:
a solver that does not checkpoint its state (the default of
:meth:`~edelweissfe.solvers.base.nonlinearsolverbase.NonlinearSolverBase.readRestart`), and a
checkpoint of another format version (every checkpoint carries one, and a run reads its own only),
and a resume past a ``modelupdate`` -- it executes an arbitrary expression whose effect no checkpoint
records, so a run is resumed only from checkpoints written in steps before the first one.

One increment, and where the checkpoint is written
--------------------------------------------------

The step's increment loop (:meth:`~edelweissfe.steps.base.stepbase.StepBase.solve`) is the same for
every solver:

.. code-block:: text

    increment n
    ├── prepare      topology update when due: model modifiers   (implicit: every increment;
    │                (e.g. AMR), then ties and contact catch up    explicit: at the step start and
    │                                                               after every topology-check-
    │                                                               frequency-th increment)
    ├── propose      the time stepper proposes the increment
    ├── attempt      Newton iterations / explicit update; if it fails: reject, retry smaller
    ├── accept       solver: material state, solver state (predictor, external work, ...)
    │                time stepper: progress, size of the next increment
    └── output       field outputs, output managers, the CHECKPOINT last
                                                ▲
    a resumed run starts here ──────────────────┘  with increment n + 1

The topology update comes before the proposal, because a refinement may lower the stable time
increment of an explicit analysis. An increment retried after a cutback starts from the same
accepted state, so the model modifiers are not asked again: they decide once per accepted state.

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
   * - rigid bodies
     - their surface nodes are moved to the restored configuration
   * - constraint state (e.g. a frozen contact search)
     - read; adopted only if it refers to the restored model's contact points, otherwise refused
   * - output managers' sequence bookkeeping
     - read, including each restart writer's position in its write interval
   * - time stepper
     - its state after the last accepted increment, as it is: progress, increment counter, the size
       of the next increment, the enforced time increment, and whether the zero increment was done
   * - solver
     - its state between increments: the implicit solver's predictor (last accepted increment and
       its ``dU``), the explicit solver's last increment and accumulated external work

Adding something that carries state between increments
-------------------------------------------------------

Declare it. Every solver, time stepper, constraint and output manager states which of its attributes
it carries from one increment to the next, by name and type, and the checkpoint writes and reads
exactly those, as they are (:mod:`~edelweissfe.utils.checkpointedstate`):

.. code-block:: python

    class NIST(NonlinearSolverBase):
        #: The predictor's state between increments: the last accepted increment and its dU.
        checkpointedState = {"prevTimeStep": TimeStep, "dU": np.ndarray}

A component that carries nothing declares an empty mapping. One whose state is not a plain attribute
(Ensight's sequence bookkeeping, a frozen contact search) overrides ``getRestartData`` and
``setRestartData`` instead. A component that declares nothing cannot be checkpointed:
``tests/test_checkpointed_state_declared.py`` checks every class in the package, so a forgotten
declaration fails there, and writing a checkpoint refuses. A solver that is not restartable says so
by keeping the declaration None.

Step actions declare their state too: a load ramped over several steps accumulates from step to
step, and most actions switch themselves off at the end of their step. A resumed run skips the
steps before the checkpoint, so it restores that state rather than re-deriving it.

One exception is known: the history of a field output (``FieldOutput.timeHistory``) is not
checkpointed. Its exported CSV file is continued, but an output manager integrating over the
history in memory (the fracture energy integrator) sees only what was computed after the resume.

Then add a scenario to ``tests/test_restart_exhaustive.py``: it resumes from every checkpoint of a
small run and requires the bitwise-identical end state, the same checkpoints and the same output.
