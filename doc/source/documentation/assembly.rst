.. _system-matrix-assembly:

System matrix assembly
======================

Assembling a sparse system matrix from element, cell or particle contributions can be done two ways,
and EdelweissFE provides both. Which one is appropriate is a memory question far more than a speed one,
so this page describes what each costs as well as what each does.

Both paths are in ``edelweissfe.numerics.csrgeneratorv2``, backed by the C++ header
``edelweissfe/numerics/_csrcore.h``.

The block of an entity
----------------------

Every element, constraint, cell or particle contributes a block to the system matrix, and knows it only in its own
numbering: row ``a`` and column ``b`` of the block are its ``a``-th and ``b``-th degree of freedom
(:class:`~edelweissfe.numerics.systemmatrixcontributor.SystemMatrixContributorBase`). Where these lie in the global
system is its global DOF list, which the :class:`~edelweissfe.numerics.dofmanager.DofManager` numbers; the DOF manager
knows nothing about the system matrix.

- By default an entity couples all its degrees of freedom: its block is the dense ``nDof x nDof`` matrix, stored
  column-major, as every element writes it.
- An entity that couples only some of its degrees of freedom (a contact constraint couples each slave node with the
  master side, but not the slave nodes with each other) lists the local row and column of every value of its block:
  it overrides
  :meth:`~edelweissfe.numerics.systemmatrixcontributor.SystemMatrixContributorBase.localEntriesOfBlock`, and its block
  is the list of the values of these entries
  (:attr:`~edelweissfe.numerics.systemmatrixcontributor.SystemMatrixContributorBase.couplesAllItsDofs` is then
  ``False``).

The number of values of a block is ``nDof**2`` for a dense block, and the length of the list otherwise. It cannot tell
the layout: a list can have exactly ``nDof**2`` entries, as a rigid body contact with a single slave node in 3D has
(36 + 9 + 18 + 18 = 81 = 9\ :sup:`2`), in an order that is not column-major. So the entity declares the layout, and an
assembly never guesses it. It declares it once, by overriding the list or not: a subclass that also declares
``couplesAllItsDofs`` is refused when it is defined, since the direct path reads the flag and the VIJ path the list,
and two declarations that disagree would assemble two different matrices. An extension type, which cannot inherit the
base class (the Marmot elements, cells and particles), declares a dense block itself, with the dense list.

The entity writes its tangent into the view
:meth:`~edelweissfe.numerics.systemmatrixcontributor.SystemMatrixContributorBase.shapeBlock` makes of the values of
its block -- a 2-D array for a dense block, a structured view for a list of its own.

The two paths
-------------

**Stage, then gather** (:class:`~edelweissfe.numerics.systemmatrixassembly.VIJSystemMatrixAssembly`), the textbook
path. Every value of every block has its own slot in one long *VIJ* (COO triplet) layout: the value ``V[k]``, its
global row ``I[k]`` and its global column ``J[k]``. The assembly derives ``I`` and ``J`` in one visible step, the
local-to-global step: the entity's local rows and columns looked up in its global DOF list, ``I = idcs[localRows]``,
``J = idcs[localColumns]``. The CSR pattern is then derived from ``I`` and ``J`` by sort-and-sum
(:class:`~edelweissfe.numerics.csrgeneratorv2.CSRGenerator`), together with a gather map: for each of the ``nnz`` CSR
entries, the VIJ slots that sum to it. Every entity writes its block into its own slice of ``V``, and the gather sums
the duplicates into CSR.

**Scatter directly** (:class:`~edelweissfe.numerics.systemmatrixassembly.DirectCSRSystemMatrixAssembly`), the
memory-efficient path, chosen with the solver option ``systemMatrixAssembly=direct``. It derives its own CSR pattern
from the entities' DOF lists, row by row, without any ``I``/``J`` pair, and knows for every value of every block its
destination in advance (:class:`~edelweissfe.numerics.csrgeneratorv2.DirectCSRAssembler`). Each block is pushed
straight into the CSR data array. Neither a value array nor index arrays of the VIJ layout are ever materialised. The
destination of a value is kept one of two ways:

- **by row**, for a dense block: the global row of each local row, and for every value ``offset``, the position of its
  column within its row, so that the destination is ``indptr[row] + offset``. The offset takes 16 bits, which reach the
  first 65,536 entries of a row.
- **by position**, for every other block: every value keeps its destination, a 32-bit position in the CSR data. That
  is a listed block, whose values share no rows to factor out, and a dense block with a degree of freedom in a row
  longer than 16-bit offsets reach.

.. list-table:: What each path stores, per value of all blocks, per entity DOF, per listed entry, or per nnz
    :width: 100%
    :widths: 40 30 30
    :header-rows: 1

    * - Array
      - Stage-then-gather
      - Direct
    * - ``I`` / ``J`` indices
      - **4 + 4 bytes / value**
      - --
    * - VIJ value array
      - **8 bytes / value**
      - --
    * - ``gather_sources``
      - **4 bytes / value**
      - --
    * - ``assembly_ptr``
      - 4 bytes / nnz
      - --
    * - ``offsets`` (within-row position), and the rows of the blocks
      - --
      - **2 bytes / value** of a block by row, 4 bytes / its entity DOF
    * - ``positions`` (position in the CSR data)
      - --
      - 4 bytes / value of a block by position
    * - entity table (DOF lists, listed entries), while the assembly is built
      - --
      - 4 bytes / entity DOF, 8 bytes / listed entry
    * - CSR pattern and data
      - 4 + 8 bytes / nnz
      - 4 + 8 bytes / nnz

The entries in bold are why this matters: the more duplicate contributions land on the same matrix entry -- which
grows with the number of entities touching a DOF, not with the DOF count itself -- the larger the VIJ arrays and
``gather_sources`` become relative to the CSR result they produce. Their size is set by the number of values of all
blocks; the CSR pattern's is set by ``nnz``. A discretisation where many entities contribute to the same entries (a
wide stencil, or overlapping supports) can have the staging arrays dwarf the actual result by a wide margin. For the
dense blocks of the elements, cells and particles, the direct path holds 2 bytes per value instead of 20.

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

- **Listed blocks.** The block of an entity that lists its entries itself (see `The block of an entity`_) is
  scattered entry by entry; a dense block row by row, unless it has a degree of freedom in a row too long for 16-bit
  offsets (see `Two independent derivations of the pattern, tested against each other`_).
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
  solver builds the condensation first and hands its
  :meth:`~edelweissfe.numerics.mpctransformation.MultiPointConstraintTransformation.condensedPatternFillIn` to the
  assembly, which adds the entries to its pattern before anything is mapped to it, whenever the equation system is
  (re)built. Both paths add them by the same merge.

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

Two independent derivations of the pattern, tested against each other
----------------------------------------------------------------------

The two paths derive the CSR pattern each in their own way, from the same entities:

- **Sort-and-sum** (VIJ path). Every ``(I[k], J[k])`` pair is packed into one 64-bit key; the keys are bucketed by
  row, sorted, and the duplicates collapsed. The sort also yields the gather map. Its sort element is a key plus the
  pair's origin index, and ``gather_sources`` and ``assembly_ptr`` index the pairs with 32-bit integers, so this path
  refuses more than ``INT32_MAX`` values of all blocks. For a discretisation with many duplicate contributions per
  matrix entry that limit can bite well before memory does.
- **Row by row** (direct path). The entities' DOF lists are transposed into, per row, the entities touching it and as
  which of their local rows (a counting sort). Each row is then the union of the columns these entities couple it
  with -- all DOFs of a dense block, the columns of the listed entries in that local row otherwise --, collected with
  a marker array of ``nDof`` integers per thread, and sorted. A count pass sets ``indptr``, a fill pass writes
  ``indices``; the result does not depend on the number of threads. No pair is ever materialised, and no 32-bit
  quantity but ``nnz`` remains: the direct path is the one for models beyond the VIJ path's limit.

The tests derive both patterns for random dense and listed entities (among them a list with exactly ``nDof**2``
entries), with and without added entries, on one and on several threads, and require them to be identical.

The map of the direct path is built by binary-searching each value's ``(row, column)`` in the pattern. For a dense
block it stores the *within-row* position, which is why 2 bytes suffice for the bulk of a model: the length of most
rows is set by the stencil, not by the problem size. Not every row's is: the row of a rigid body's reference point
has an entry for every slave node it is coupled with, and the condensation of a multi-point constraint can couple a
master degree of freedom with many others. Before mapping, the assembly looks at the rows of every dense block; a
block with a degree of freedom in a row longer than 65,536 entries keeps the positions of its values in the CSR data
instead, as every listed block does. So no row is too long for the direct path, and a model without such rows pays
nothing for them. A value whose entry is not in the pattern raises rather than being silently misplaced.

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

The addressing is validated **exactly, without a floating-point tolerance**, using an integer trick: the
tests give every value of every block ``1``, so each CSR entry becomes the *count* of its contributions, then
its own index ``k`` in the VIJ layout, so it becomes the *sum of their indices*. Both are integers far inside
2\ :sup:`53`, so the matrix the direct path scatters is compared bit for bit with the one
:meth:`~edelweissfe.numerics.csrgeneratorv2.CSRGenerator.updateInPlace` gathers from the same entities. A
transposition or an off-by-one shows up as an unmistakable mismatch instead of a judgement call about how
large a deviation is acceptable. Sums of such integers are exact in any order, so the check stays exact with
the threads adding in an order set by their timing.

Reference
---------

.. automodule:: edelweissfe.numerics.systemmatrixassembly
   :members:

:class:`~edelweissfe.numerics.csrgeneratorv2.CSRGenerator` and the C++ engine behind it are described
under :doc:`utils`; only the direct-scatter side is documented here.

.. autoclass:: edelweissfe.numerics.csrgeneratorv2.DirectCSRAssembler
   :members:

.. autofunction:: edelweissfe.numerics.csrgeneratorv2.csrPatternOfEntities

.. autoclass:: edelweissfe.numerics.csrgeneratorv2.AliasedCSRMatrix
   :members:
