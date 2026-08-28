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

The two entries in bold on the left are why this matters. At 43,350 DOF of a meshfree RKPM
discretisation -- 1.72e9 COO pairs against 71.2 M nnz, i.e. 24 duplicate contributions per matrix entry
-- the value array is 12.84 GiB and ``gather_sources`` a further 6.42 GiB, against a 0.53 GiB result.

Why the direct path is also faster
----------------------------------

Three effects, all measured at 43,350 DOF on 16 threads:

- **Nothing has to be cleared.** The staging array must be zeroed every Newton iteration; at this size
  that is 1.7 s of writing zeros. The direct path clears only the private CSR copies.
- **Nothing has to be re-read.** The gather random-accesses a multi-gigabyte array; the scatter touches
  only the CSR data, a ~23x smaller working set.
- **The evaluation itself is slightly faster**, because each entity writes into a small reused scratch
  block that stays in cache instead of a slab of a multi-gigabyte array.

Measured assembly, both paths timed in the same process on the same increment:

.. list-table::
    :width: 100%
    :widths: 40 20 20 20
    :header-rows: 1

    * - Component, per Newton iteration
      - Stage-then-gather
      - Direct
      - Ratio
    * - clear the staging array
      - 1.853 s
      - 0.157 s
      - 11.8x
    * - entity evaluation
      - 3.199 s
      - 3.046 s
      - 1.05x
    * - gather / reduce
      - 1.688 s
      - 0.099 s
      - 17.1x
    * - **total**
      - **6.704 s**
      - **3.301 s**
      - **2.03x**

.. note::

   That 2.03x is the *assembly in isolation*. Quote it with care: repeated later on the same data it
   came out at 1.75x, because the gather's cost over a multi-gigabyte array varies with the machine's
   memory state between runs, and it is the largest term the direct path removes. The honest figure is a
   range, 1.75-2.03x.

The one definition of the pattern
---------------------------------

:class:`~edelweissfe.numerics.csrgeneratorv2.DirectCSRAssembler` **borrows** its pattern from an existing
:class:`~edelweissfe.numerics.csrgeneratorv2.CSRGenerator` rather than deriving its own. This is
deliberate: there is exactly one definition of what the CSR pattern is, and the two assembly paths
cannot drift apart. The generator must outlive the assembler.

The offset map is built by binary-searching each ``(row, col)`` pair in the borrowed pattern, and stores
the *within-row* position, which is why 2 bytes suffice: row length is set by the stencil, not by the
problem size. Measured on an RKPM discretisation the largest within-row offset was 1,727 against the
65,535 a ``uint16`` allows, and the bound stays in that region even as the support radius grows. A pair
that cannot be mapped, or an offset that does not fit, raises rather than being silently misplaced.

Building only the pattern
-------------------------

``CSRGenerator(systemMatrix, patternOnly=True)`` builds ``indptr`` and ``indices`` and nothing else.
Two reasons, and the second is the one that decides how large a model can be:

- **The sort payload halves.** With the gather map, the sort element is a key plus an origin index, which
  pads to 16 bytes per pair; without it, a bare 8-byte key. That array is the largest single allocation
  in the build.
- **It removes every 32-bit pair index.** ``gather_sources``, ``assembly_ptr`` and the sort element's
  origin field all index into the COO list, which is why a full build refuses more than ``INT32_MAX``
  pairs. Measured on an RKPM problem, that limit bites at about **50,000 DOF with only a third of a
  187 GB machine in use** -- memory alone would have allowed roughly 121,000. With ``patternOnly`` the
  only remaining 32-bit quantity is ``nnz``, three orders of magnitude from its limit at these sizes.

A generator built this way, or one that has given its map back via
:meth:`~edelweissfe.numerics.csrgeneratorv2.CSRGenerator.releaseGatherMap`, **raises** if asked to
gather. That is deliberate: the alternative is a ``nogil`` loop reading released vectors, and a matrix
that is quietly wrong is worse than one that is loudly unavailable.

Measured effect at 43,350 DOF: peak resident memory 47.39 -> **34.59 GiB**, and the pattern build
19.36 -> **11.40 s**.

Private copies, or atomics
--------------------------

Threads scattering into one CSR array must not collide. The default is one private copy of the CSR data
per thread, summed at the end: no synchronisation, a fixed summation order, and therefore
bit-reproducible results -- at ``nThreads`` x ``nnz`` x 8 bytes.
:meth:`~edelweissfe.numerics.csrgeneratorv2.DirectCSRAssembler.setNumBuffers` trades that memory for
atomics.

Measured at 43,350 DOF, 16 threads, all three configurations in one process on one increment:

.. list-table::
    :width: 100%
    :widths: 20 25 20 35
    :header-rows: 1

    * - copies
      - assembler memory
      - assembly time
      - notes
    * - 16 (one per thread)
      - 11.72 GiB
      - 3.4289 s
      - default; bit-reproducible
    * - 4 (shared)
      - 5.35 GiB
      - 3.7800 s (+10.2%)
      - not worth having, see below
    * - 1 (fully atomic)
      - **3.76 GiB**
      - 3.8341 s (+11.8%)
      - reproducibility lost

Two things worth knowing. **Atomics cost 11.8%, not the 57% an earlier standalone benchmark suggested**
-- that benchmark timed the final reduction, which is 0.16 s of a 3.43 s assembly, rather than the
atomics inside the scatter. And **the penalty is the atomic instruction, not contention**: going from 16
copies to 4 already costs 10.2 of the 11.8 percentage points, so the intermediate setting pays nearly
all of the time penalty for only part of the memory saving.

What atomics cost is exact reproducibility. The summation order becomes dependent on thread
interleaving, so re-running the same computation no longer returns bit-identical values -- the results
stay correct to round-off, which is measurable: an internal control that re-evaluates the same kernel
twice reports exactly zero with private copies and 2.9e-16 with atomics.

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
