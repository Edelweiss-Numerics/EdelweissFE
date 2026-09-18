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
    * - private CSR copies
      - --
      - ``nBuffers`` x 8 bytes / nnz
    * - CSR pattern and data
      - 4 + 8 bytes / nnz
      - 4 + 8 bytes / nnz

The two entries in bold on the left are why this matters: the more duplicate contributions land on
the same matrix entry -- which grows with the number of entities touching a DOF, not with the DOF
count itself -- the larger the value array and ``gather_sources`` become relative to the CSR result
they produce. Their size is set by the number of COO pairs; the CSR pattern's is set by ``nnz``. A
discretisation where many entities contribute to the same entries (a wide stencil, or overlapping
supports) can have the staging array and gather map dwarf the actual result by a wide margin.

Why the direct path is also faster
----------------------------------

Three effects:

- **Nothing has to be cleared.** The staging array must be zeroed every Newton iteration; that cost
  scales with the number of COO pairs, not with ``nnz``. The direct path clears only the private CSR
  copies, sized by ``nnz``.
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

Private copies, or atomics
--------------------------

Threads scattering into one CSR array must not collide. The default is one private copy of the CSR data
per thread, summed at the end: no synchronisation, a fixed summation order, and therefore
bit-reproducible results -- at ``nThreads`` x ``nnz`` x 8 bytes.
:meth:`~edelweissfe.numerics.csrgeneratorv2.DirectCSRAssembler.setNumBuffers` trades that memory for
atomics: fewer private copies means threads share one and synchronise the scatter with atomics
instead, down to a single, fully atomic copy at ``n == 1``.

Two things worth knowing, from measuring this tradeoff on a real assembly:

- **The penalty is the atomic instruction itself, not contention.** Most of the slowdown from going
  fully atomic already shows up after dropping from one copy per thread to just a handful of shared
  copies -- so an intermediate setting pays nearly all of the time cost for only part of the memory
  saving, and isn't a good middle ground in practice.
- **What atomics actually cost is exact reproducibility, not correctness.** The summation order
  becomes dependent on thread interleaving, so re-running the same computation no longer returns
  bit-identical values -- results still agree to round-off, verifiably: an internal control that
  re-evaluates the same kernel twice reports exactly zero difference with private copies and a
  round-off-scale difference with atomics.

.. note::

   Once the staging array is gone the assembly is usually no longer what sets peak memory, so the
   memory saving from atomics may buy nothing in practice while the reproducibility loss is real.
   Measure where your peak actually is before switching.

Validation
----------

The addressing is validated **exactly, without a floating-point tolerance**, using an integer trick:
set every value in the staging array to ``1`` so each CSR entry becomes the *count* of contributing
pairs, then to its own index ``k`` so it becomes the *sum of their indices*. Both are integers far
inside 2\ :sup:`53`, so a bitwise comparison against
:meth:`~edelweissfe.numerics.csrgeneratorv2.CSRGenerator.updateCSR` validates the grouping exactly
rather than approximately. A transposition or an off-by-one shows up as an unmistakable mismatch instead
of a judgement call about how large a deviation is acceptable.

:meth:`~edelweissfe.numerics.csrgeneratorv2.DirectCSRAssembler.assembleFromVIJ` exists for this purpose:
it pushes a whole staging array through the offset map, so the two paths can be compared on identical
values with the physics held fixed.

Reference
---------

:class:`~edelweissfe.numerics.csrgeneratorv2.CSRGenerator` and the C++ engine behind it are described
under :doc:`utils`; only the direct-scatter side is documented here.

.. autoclass:: edelweissfe.numerics.csrgeneratorv2.DirectCSRAssembler
   :members:

.. autoclass:: edelweissfe.numerics.csrgeneratorv2.AliasedCSRMatrix
   :members:
