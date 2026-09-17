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

Thread-local buffers in the element loop
----------------------------------------

On a free-threaded interpreter the element loop runs Python code on every core at once, and there
the cost of *sharing* an object becomes visible: a numpy view keeps a reference to the buffer it
looks into, so taking one view per element into a single buffer shared by all threads makes every
core update the same reference count, and they end up passing that one cache line back and forth
instead of computing. The loop looks parallel and does not scale.

The explicit element loop therefore gives each chunk of elements buffers of its own -- the gathered
solution and increment, and a force buffer the elements write into -- and places the finished chunk
into the shared scatter buffer in a single indexed assignment. Per element, nothing shared is
touched. The positions each chunk writes to are precomputed once, in
:func:`~edelweissfe.solvers.base.parallelelementcomputation._chunkedGatherPlan`, and reused for as
long as the mesh and the degree-of-freedom layout stay the same.

This is worth keeping in mind when adding a parallel loop of your own: prefer giving each task its
own buffer and merging once, over having every task write into one shared object as it goes.

Load balancing of the element loop
----------------------------------

The elements are handed to the threads in chunks. It is tempting to cut exactly one chunk per
thread, which minimises the bookkeeping, but that is only right when every element costs the same
-- and in a nonlinear analysis it does not. A quadrature point that is yielding or damaging pays
for a return mapping that an elastic one does not, so the elements along a propagating front are
several times more expensive than the bulk. Those elements are neighbours in the mesh and therefore
neighbours in element order, so chunks cut from that order are systematically unequal, and a
one-chunk-per-thread split makes the whole loop wait for whichever thread happened to draw the
front.

The element loop therefore cuts substantially more chunks than there are threads
(:data:`~edelweissfe.solvers.base.parallelelementcomputation._chunksPerThread`) and lets the thread
pool hand them out on demand: a thread that drew cheap elements comes back for more work rather
than idling while a neighbour finishes. This costs one plan entry per chunk and nothing per
element.
