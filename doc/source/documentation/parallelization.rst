Parallelization
===============

EdelweissFE makes use of OpenMP to parallelize the computation of finite elements and for certain solvers, such as SuperLU, UMFPACK, or PARDISO.

If a parallel solver (e.g, NISTParallel, NISTPArcLength) is selected in the .inp file, EdelweissFE  automatically determines the maximum number of threads,
depending on the host architecture.
However, it is RECOMMENDED to enforce a fixed number of threads by running

.. code-block:: console

    OMP_NUM_THREADS=XX python edelweiss.py INPUT.inp

This ensures that the same number of threads ``XX`` is employed both in EdelweissFE as well as in the underlying Intel MKL (e.g., if the PARDISO linear solver is used).

.. _parallelization_thread_pinning:

Do not set ``OMP_PROC_BIND`` or ``OMP_PLACES``
----------------------------------------------

EdelweissFE's element loop is a Python thread pool, not an OpenMP team, and the two do not mix
well. When ``OMP_PROC_BIND`` (or ``OMP_PLACES``) is set, the OpenMP runtime pins the thread that
loads it to a single place -- and it is loaded as soon as the first compiled extension that links
it is imported, which happens long before any solver starts. Every thread created after that
inherits the one-core mask, so the whole element loop ends up sharing a single core.

Nothing fails when this happens. The pool is created with the requested number of workers, the
results are correct, and the run is simply several times slower than it believes itself to be --
which is why it can survive a long time unnoticed on a cluster, where such variables are often set
by a site module or a job template rather than by the user.

The parallel solvers therefore compare the number of threads they are about to use against the
number of CPUs the process is actually permitted to run on
(:func:`~edelweissfe.numerics.parallelizationutilities.getNumberOfAvailableCpus`, which reads the
affinity mask and so also sees a batch scheduler's cgroup or a ``taskset``), and print a warning
naming ``OMP_PROC_BIND`` when there are fewer CPUs than threads. If you see that warning, unset
those variables for the run:

.. code-block:: console

    unset OMP_PROC_BIND OMP_PLACES
    OMP_NUM_THREADS=XX python edelweiss.py INPUT.inp
