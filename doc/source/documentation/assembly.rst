.. _system-matrix-assembly:

System matrix assembly
======================

Assembling a sparse system matrix from element, cell or particle contributions can be done two ways,
and EdelweissFE provides both. Which one is appropriate is a memory question far more than a speed one,
so this page describes what each costs as well as what each does.

Both paths are in ``edelweissfe.numerics.csrgeneratorv2``, backed by the C++ header
``edelweissfe/numerics/_csrcore.h``.

The two paths
-------------

**Stage, then gather** (:class:`~edelweissfe.numerics.csrgeneratorv2.CSRGenerator`). Every entity writes
its dense block into its own slice of one long *VIJ* (COO triplet) value array. Contributions to the
same matrix entry land in different slots, and a second pass then sums the duplicates into CSR. The
addressing is a *gather*: for each of the ``nnz`` output entries, read the value-array positions that
feed it.

**Scatter directly** (:class:`~edelweissfe.numerics.csrgeneratorv2.DirectCSRAssembler`). Each entity's
block is pushed straight into the CSR data array, because every COO pair knows its destination slot in
advance -- ``indptr[row] + offset``, where ``offset`` is the position of the column within that row. No
value array is ever materialised.

.. list-table:: What each path stores, per COO pair or per nnz
    :width: 100%
    :widths: 40 30 30
    :header-rows: 1

    * - Array
      - Stage-then-gather
      - Direct
    * - ``I`` / ``J`` COO indices (owned by the DofManager)
      - 4 + 4 bytes / pair
      - 4 + 4 bytes / pair
    * - VIJ value array
      - **8 bytes / pair**
      - --
    * - ``gather_sources``
      - **4 bytes / pair**
      - --
    * - ``assembly_ptr``
      - 4 bytes / nnz
      - --
    * - ``offsets`` (within-row position)
      - --
      - **2 bytes / pair**
    * - CSR pattern and data
      - 4 + 8 bytes / nnz
      - 4 + 8 bytes / nnz

The two entries in bold on the left are why this matters: the more duplicate contributions land on
the same matrix entry -- which grows with the number of entities touching a DOF, not with the DOF
count itself -- the larger the value array and ``gather_sources`` become relative to the CSR result
they produce. Their size is set by the number of COO pairs; the CSR pattern's is set by ``nnz``. A
discretisation where many entities contribute to the same entries (a wide stencil, or overlapping
supports) can have the staging array and gather map dwarf the actual result by a wide margin.

In the implicit solvers
-----------------------

The implicit solvers assemble their tangent through one interface,
:class:`~edelweissfe.numerics.systemmatrixassembly.SystemMatrixAssembly`, so that neither the solver nor
an element, constraint or load knows which path is in use. Every entity asks for its block --
``Ke = K[element]`` -- and writes it; the solver then asks for the assembled matrix. The solver option
``systemMatrixAssembly`` selects the path:

.. code-block:: edelweiss

    *solver, solver=NISTParallel, name=theSolver
        systemMatrixAssembly=direct

``vij`` (the default) is :class:`~edelweissfe.numerics.systemmatrixassembly.VIJSystemMatrixAssembly`,
stage then gather; ``direct`` is
:class:`~edelweissfe.numerics.systemmatrixassembly.DirectCSRSystemMatrixAssembly`, scatter directly.
``NIST``, ``NISTParallel`` and ``NISTPArcLength`` support both. ``NonlinearImplicitDynamic`` (and
``NIDParallel``) assemble on the VIJ path only, since they add the mass entry by entry to the VIJ value
array, and so does ``NEST``.

What the direct path has to know about the entities and the solvers:

- **Sparse blocks.** An entity may declare a VIJ pattern of its own, sparser than its dense
  ``nDof x nDof`` block (see :class:`~edelweissfe.numerics.vijentitybase.VIJEntityBase`): a constraint
  that couples only some of its degrees of freedom. Its block is then the list of values of its own
  ``(I, J)`` pairs, which the scatter places entry by entry. The entity says so
  (:attr:`~edelweissfe.numerics.vijentitybase.VIJEntityBase.couplesAllItsDofs` is ``False``); the number
  of its entries cannot tell, since a sparse pattern can have exactly ``nDof**2`` of them, as a rigid body
  contact with a single slave node in 3D has.
- **Worker threads.** Each thread of the parallel element loop writes through blocks of its own
  (:meth:`~edelweissfe.numerics.systemmatrixassembly.SystemMatrixAssembly.blocksOfWorker`), a scratch
  array, and adds every block to the one CSR matrix of the assembly with atomic additions, see
  `One CSR array, atomic additions`_. The loop deals the elements out to the workers in a fixed way
  (:func:`~edelweissfe.numerics.parallelizationutilities.mapChunksOnWorkers`), so that no two threads
  use the same scratch array at the same time.
- **More than one matrix.** The arc length method needs the tangents of its dead load and its reference
  load besides the tangent of the elements. It adds them to the tangent of the increment as blocks
  scaled by :math:`1 - \lambda` and :math:`\lambda`
  (:meth:`~edelweissfe.numerics.systemmatrixassembly.SystemMatrixAssembly.scaledBy`), on both paths, so
  that no further matrix is held.
- **The complete pattern.** The multi-point-constraint condensation adds entries to the CSR pattern. The
  assembly is built after them, on the complete pattern, whenever the equation system is (re)built.

Every block of the direct path is a zeroed scratch array, so an entity that is handed its block twice in
one assembly (an element, and a body load on it) has each scattered on its own; on the VIJ path the
second adds to the first in the same slice. Either way, an entity only adds to or overwrites its block,
and never reads it.

Why the direct path is also faster
----------------------------------

Three effects:

- **Nothing has to be cleared.** The staging array must be zeroed every Newton iteration; that cost
  scales with the number of COO pairs, not with ``nnz``. The direct path clears only the CSR data
  array, sized by ``nnz``.
- **Nothing has to be re-read.** The gather random-accesses the staging array, which can be many times
  larger than the CSR data the scatter touches instead.
- **The evaluation itself is not slower**, because each entity writes into a small reused scratch block
  that stays in cache instead of a slab of a much larger array.

The relative sizes of these three effects depend on how many duplicate contributions land per matrix
entry for a given discretisation and thread count -- measure on the model at hand rather than assuming
a fixed ratio.

The one definition of the pattern
---------------------------------

:class:`~edelweissfe.numerics.csrgeneratorv2.DirectCSRAssembler` **borrows** its pattern from an existing
:class:`~edelweissfe.numerics.csrgeneratorv2.CSRGenerator` rather than deriving its own. This is
deliberate: there is exactly one definition of what the CSR pattern is, and the two assembly paths
cannot drift apart. The generator must outlive the assembler.

The offset map is built by binary-searching each ``(row, col)`` pair in the borrowed pattern, and stores
the *within-row* position, which is why 2 bytes suffice: row length is set by the stencil, not by the
problem size, so it stays well within a ``uint16`` even for a wide stencil. A pair that cannot be
mapped, or an offset that does not fit, raises rather than being silently misplaced.

Building only the pattern
-------------------------

``CSRGenerator(systemMatrix, patternOnly=True)`` builds ``indptr`` and ``indices`` and nothing else.
Two reasons, and the second is the one that decides how large a model can be:

- **The sort payload halves.** With the gather map, the sort element is a key plus an origin index, which
  pads to 16 bytes per pair; without it, a bare 8-byte key. That array is the largest single allocation
  in the build.
- **It removes every 32-bit pair index.** ``gather_sources``, ``assembly_ptr`` and the sort element's
  origin field all index into the COO list, which is why a full build refuses more than ``INT32_MAX``
  pairs. For a discretisation with many duplicate contributions per matrix entry, that pair-count limit
  can bite well before memory does. With ``patternOnly`` the only remaining 32-bit quantity is ``nnz``,
  which grows only with the CSR result, not with the number of contributions.

A generator built this way, or one that has given its map back via
:meth:`~edelweissfe.numerics.csrgeneratorv2.CSRGenerator.releaseGatherMap`, **raises** if asked to
gather. That is deliberate: the alternative is a ``nogil`` loop reading released vectors, and a matrix
that is quietly wrong is worse than one that is loudly unavailable.

Dropping the gather map lowers peak memory and shortens the pattern build, since half the sort payload
and the 32-bit index arrays it removes never need writing in the first place.

One CSR array, atomic additions
-------------------------------

All threads scatter into the one CSR data array of the matrix being assembled, and every addition to an
entry is atomic (``#pragma omp atomic``), so threads that add to the same entry at the same time cannot
lose a contribution. Nothing else is held per thread but a scratch array the size of one block.

The alternative, a private copy of the CSR data per thread summed at the end, needs no synchronisation,
but it holds ``nThreads`` x ``nnz`` x 8 bytes and sums all copies once per assembly. Measured on real
assemblies, it was not faster than the atomic additions, and its memory grows with the number of threads
until it exceeds what the direct path saves by not staging the values. So there is one way, the atomic
one:

- **The cost of an atomic addition is the instruction, not contention.** Threads rarely add to the same
  entry at the same time, so the cost does not grow with the number of threads.
- **What the atomic additions give up is bit-reproducibility, not correctness.** The order in which the
  threads add to an entry depends on their timing, so the same assembly repeated with more than one
  thread agrees with itself to round-off, not bit for bit. A Newton iteration whose convergence hinges on
  the last bit (e.g., a frictional contact that sticks or slips by a hair) may take a different path than
  the VIJ path does.

Validation
----------

The addressing is validated **exactly, without a floating-point tolerance**, using an integer trick:
set every value in the staging array to ``1`` so each CSR entry becomes the *count* of contributing
pairs, then to its own index ``k`` so it becomes the *sum of their indices*. Both are integers far
inside 2\ :sup:`53`, so a bitwise comparison against
:meth:`~edelweissfe.numerics.csrgeneratorv2.CSRGenerator.updateCSR` validates the grouping exactly
rather than approximately. A transposition or an off-by-one shows up as an unmistakable mismatch instead
of a judgement call about how large a deviation is acceptable. Sums of such integers are exact in any
order, so the check stays exact with the threads adding in an order set by their timing.

:meth:`~edelweissfe.numerics.csrgeneratorv2.DirectCSRAssembler.assembleFromVIJ` exists for this purpose:
it pushes a whole staging array through the offset map, so the two paths can be compared on identical
values with the physics held fixed.

Reference
---------

.. automodule:: edelweissfe.numerics.systemmatrixassembly
   :members:

:class:`~edelweissfe.numerics.csrgeneratorv2.CSRGenerator` and the C++ engine behind it are described
under :doc:`utils`; only the direct-scatter side is documented here.

.. autoclass:: edelweissfe.numerics.csrgeneratorv2.DirectCSRAssembler
   :members:

.. autoclass:: edelweissfe.numerics.csrgeneratorv2.AliasedCSRMatrix
   :members:
