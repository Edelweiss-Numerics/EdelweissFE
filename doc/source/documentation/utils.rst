Utilities
=========

EdelweissFE contains several classes and modules,
which are not directly employed by the user,
but very handy if fundamental new functionalities should be added.

Matrix conversion from COO sparse format to CSR format
------------------------------------------------------

Matrix assembly is a critical performance bottleneck in Finite Element simulations, particularly during non-linear implicit or explicit iterations where the system matrix structure (non-zero pattern) remains constant, but its numerical values are updated at every iteration.

EdelweissFE offers two generators to convert COO (Coordinate) format sparse matrices to CSR (Compressed Sparse Row) format efficiently without re-analysing the sparsity pattern:

1. **Legacy Generator (``csrgenerator``)**: A Cython implementation utilizing a binary search algorithm for mapping.
2. **High-Performance Generator (``csrgeneratorv2``)**: A parallelized C++ engine with OpenMP support, thread-safe memory layouts, cache friendliness, and vectorized operations.

Legacy Generator
~~~~~~~~~~~~~~~~

Module ``edelweissfe.numerics.csrgenerator``

.. autoclass:: edelweissfe.numerics.csrgenerator.CSRGenerator
   :members:

High-Performance Generator
~~~~~~~~~~~~~~~~~~~~~~~~~~

Module ``edelweissfe.numerics.csrgeneratorv2``

.. autoclass:: edelweissfe.numerics.csrgeneratorv2.CSRGenerator
   :members:

Under the hood, ``csrgeneratorv2`` delegates sparse pattern building and assembly updates to a C++ engine class ``CSRCore`` (defined in ``_csrcore.h``).

Algorithm & Optimization Details
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Row-based Partitioning**:
   The rows are divided into partitions based on the number of threads (OpenMP max threads). By default, a heuristic of 4 partitions per thread is used to balance the load evenly across CPU cores.

2. **Parallel Histogram with Thread-Major Offset Layout**:
   To calculate prefix sums and sort edges without atomic contention or cache conflicts:
   - A thread-local count of rows per partition is calculated.
   - The thread counts are reduced to global partition counts.
   - The write offset array ``write_offsets`` is transposed to a **thread-major layout** (indexed as ``[thread][partition]``). Because each thread only writes to its own row in the offsets table, false sharing (cache line contention) is eliminated during the parallel scatter step under high thread counts.

3. **Parallel Scatter (Bucketing)**:
   COO elements (edges) are packed into 64-bit keys (High 32-bit: Row, Low 32-bit: Column) and scattered in parallel into partition buckets using the thread-major offsets.

4. **Parallel Sorting and Symbolic Counting**:
   Each OpenMP thread takes ownership of a subset of partitions, sorts the packed edges using ``std::sort``, and counts the unique non-zeros (nnz). Since partition boundaries map rows uniquely (the row partition assignment ensures no row overlaps partitions), thread-local writes to the ``indptr`` array are completely disjoint and synchronization-free.

5. **Vectorized Assembly Update**:
   When updating the matrix via ``updateInPlace(V)`` or ``updateCSR(V)``, values corresponding to duplicated coordinate entries are summed. Because the lookup array ``gather_sources`` is contiguous in memory, the compiler is able to autovectorize the reduction loops using OpenMP SIMD directives:

   .. code-block:: cpp

      #pragma omp simd reduction(+:sum)
      for (int32_t k = start; k < end; ++k) {
          sum += V_data[gather_sources[k]];
      }

Memory Safety and Lifetime Management
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Type Safety**:
  The constructor of ``csrgeneratorv2.CSRGenerator`` explicitly casts the input coordinate arrays ``I`` and ``J`` to contiguous memoryviews of type ``int`` (representing 32-bit signed integers, ``np.intc``). This ensures memory safety even if the coordinates originate from non-standard array types outside the `dofmanager`.

- **In-Place Matrix Updates**:
  ``updateInPlace(V)`` modifies the internal SciPy ``csr_matrix.data`` buffer directly using raw C++ pointers, avoiding any Python object creation or memory allocations during iteration loops. Note that the returned matrix is a live view; its contents are mutated on subsequent update calls.

- **Lifetime Extension**:
  To prevent Python from garbage-collecting the ``CSRGenerator`` object while the generated SciPy ``csr_matrix`` is still in use, the generator attaches itself as the private attribute ``csrMatrix._parent = self``. This ensures that the underlying C++ pointer to ``CSRCore`` remains valid as long as the matrix is referenced.


Gathering efficiently of element results
----------------------------------------

Module ``edelweissfe.utils.elementresultcollector``

.. autoclass:: edelweissfe.utils.elementresultcollector.ElementResultCollector
   :members:

Adaptive time stepping
----------------------

Module ``edelweissfe.timesteppers.adaptivetimestepper``

.. autoclass:: edelweissfe.timesteppers.adaptivetimestepper.AdaptiveTimeStepper
   :members:
