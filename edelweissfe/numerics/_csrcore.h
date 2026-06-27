#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <omp.h>
#include <stdexcept>
#include <vector>

struct PackedEdge {
  uint64_t key;  // High 32: Row, Low 32: Col
  int32_t  orig; // Original index

  // Default comparison for std::sort is now extremely fast
  bool operator<( const PackedEdge& other ) const { return key < other.key; }
};

class CSRCore {
public:
  // CSR Topology
  std::vector< int > indptr;
  std::vector< int > indices;
  int                nnz  = 0;
  int                nDof = 0;

  // Assembly Mapping
  std::vector< int32_t > gather_sources;
  std::vector< int32_t > assembly_ptr;

  CSRCore( const int* I, const int* J, int64_t n_pairs, int n_dof ) : nDof( n_dof )
  {
    if ( n_pairs == 0 ) {
      // Empty pattern: indptr is all-zeros, indices/gather_sources/assembly_ptr
      // remain empty. update() is guarded by nnz==0 and is a no-op in that case.
      indptr.assign( nDof + 1, 0 );
      return;
    }

    // --- SAFETY & CONFIG ---
    if ( n_pairs > std::numeric_limits< int32_t >::max() ) {
      throw std::overflow_error( "CSRCore: n_pairs exceeds 32-bit limit." );
    }

    // Determine Partitions based on threads
    int num_threads = omp_get_max_threads();

    // We partition by ROWS.
    // 4x partitions per thread is a good heuristic for load balancing.
    const int    num_partitions     = ( num_threads * 4 > nDof ) ? 1 : num_threads * 4;
    const double rows_per_partition = (double)nDof / num_partitions;

    // --- STEP 1: PARALLEL HISTOGRAM ---
    // Count how many edges fall into each partition.
    // We use thread-local counters to avoid atomic contention.

    std::vector< int64_t > partition_counts( num_partitions, 0 );
    // Thread-local counts avoid atomic contention: each thread accumulates
    // into its own private row, then the results are reduced serially.
    std::vector< std::vector< int64_t > > thread_local_counts( num_threads,
                                                               std::vector< int64_t >( num_partitions, 0 ) );

#pragma omp parallel
    {
      int   tid          = omp_get_thread_num();
      auto& local_counts = thread_local_counts[tid];

#pragma omp for schedule( static )
      for ( int64_t k = 0; k < n_pairs; ++k ) {
        int r    = I[k];
        int p_id = (int)( r / rows_per_partition );
        if ( p_id >= num_partitions )
          p_id = num_partitions - 1;
        local_counts[p_id]++;
      }
    }

    // Reduce thread counts to global partition counts & calculate offsets
    // This matrix transposition (thread x part -> part x thread) allows us
    // to calculate exactly where each thread should write its data.
    // Thread-major layout [thread][partition]: each thread's counters reside on
    // their own cache lines, eliminating false sharing during the parallel scatter.
    std::vector< std::vector< int64_t > > write_offsets( num_threads, std::vector< int64_t >( num_partitions ) );
    std::vector< int64_t >                partition_starts( num_partitions + 1, 0 );

    int64_t current_global_offset = 0;
    for ( int p = 0; p < num_partitions; ++p ) {
      partition_starts[p] = current_global_offset;
      for ( int t = 0; t < num_threads; ++t ) {
        write_offsets[t][p] = current_global_offset;
        current_global_offset += thread_local_counts[t][p];
      }
    }
    partition_starts[num_partitions] = current_global_offset; // Should equal n_pairs

    // --- STEP 2: PARALLEL SCATTER (BUCKETING) ---
    // Pack data into a structure that is fast to sort.
    std::vector< PackedEdge > edges( n_pairs );

#pragma omp parallel
    {
      int tid = omp_get_thread_num();

#pragma omp for schedule( static )
      for ( int64_t k = 0; k < n_pairs; ++k ) {
        int r = I[k];
        int c = J[k];

        // Pack (Row, Col) into 64-bit key
        uint64_t key = ( (uint64_t)r << 32 ) | (uint32_t)c;

        int p_id = (int)( r / rows_per_partition );
        if ( p_id >= num_partitions )
          p_id = num_partitions - 1;

        // Determine write position (no atomics needed — thread-major layout
        // means only this thread writes to write_offsets[tid][*]).
        int64_t pos = write_offsets[tid][p_id]++;

        edges[pos] = { key, (int32_t)k };
      }
    }

    // --- STEP 3: PARALLEL SORT & SYMBOLIC COMPRESSION ---
    // Each thread takes ownership of specific partitions, sorts them,
    // and calculates how many non-zeros (nnz) they will produce.

    std::vector< int32_t > partition_nnz( num_partitions, 0 );

    // Temporary indptr logic: Since indptr is cumulative globally,
    // we first fill it relative to the partition start, then fix it later.
    indptr.assign( nDof + 1, 0 );

#pragma omp parallel for schedule( dynamic, 1 )
    for ( int p = 0; p < num_partitions; ++p ) {
      int64_t start = partition_starts[p];
      int64_t end   = partition_starts[p + 1];

      if ( start == end )
        continue;

      // 3a. Sort the partition
      std::sort( edges.begin() + start, edges.begin() + end );

      // 3b. Symbolic Counting (Local CSR generation)
      // We iterate the sorted edges to count unique (r,c) pairs
      // and fill the indptr counts for rows in this partition.
      //
      // INVARIANT: the partition assignment p_id = r / rows_per_partition maps
      // every COO entry with the same row r to exactly one partition. Therefore
      // the indptr[r+1]++ increments below touch disjoint array positions across
      // different partitions — no synchronization is required.

      int32_t local_nnz = 0;
      if ( end > start ) {
        local_nnz = 1; // First entry is always new

        // Determine Row range for this partition
        // We must be careful: indptr is size nDof+1.
        // We extract row from the key.
        int prev_row = static_cast< int32_t >( edges[start].key >> 32 );

        // Mark first row count
        indptr[prev_row + 1]++;

        for ( int64_t k = start + 1; k < end; ++k ) {
          uint64_t curr_key = edges[k].key;
          uint64_t prev_key = edges[k - 1].key;

          int curr_row = static_cast< int32_t >( curr_key >> 32 );

          // If key is different, it's a new matrix entry
          if ( curr_key != prev_key ) {
            local_nnz++;
            indptr[curr_row + 1]++;
          }
        }
      }
      partition_nnz[p] = local_nnz;
    }

    // --- STEP 4: GLOBAL SCAN (OFFSET CALCULATION) ---
    // 4a. Indptr Prefix Sum (This is tricky)
    // Currently indptr[r+1] holds the COUNT of non-zeros in row r.
    // We need to do a standard cumulative sum over the whole array.
    // std::partial_sum is efficient enough here (serial, but cache-linear and fast for <10M rows).
    // For massive arrays, this can be parallelized, but usually not needed.
    std::partial_sum( indptr.begin(), indptr.end(), indptr.begin() );

    // 4b. NNZ Offsets for the other arrays
    std::vector< int32_t > global_nnz_offsets( num_partitions, 0 );
    int32_t                current_nnz = 0;
    for ( int p = 0; p < num_partitions; ++p ) {
      global_nnz_offsets[p] = current_nnz;
      current_nnz += partition_nnz[p];
    }
    this->nnz = current_nnz;

    // --- STEP 5: FINAL FILL ---
    indices.resize( nnz );
    gather_sources.resize( n_pairs );
    assembly_ptr.resize( nnz + 1 );
    assembly_ptr[nnz] = static_cast< int32_t >( n_pairs ); // Sentinel; safe: guarded by INT32_MAX check above

#pragma omp parallel for schedule( dynamic, 1 )
    for ( int p = 0; p < num_partitions; ++p ) {
      int64_t start = partition_starts[p];
      int64_t end   = partition_starts[p + 1];
      if ( start == end )
        continue;

      int32_t write_idx = global_nnz_offsets[p];

      // Fill first entry of the partition
      indices[write_idx]      = static_cast< int32_t >( edges[start].key & 0xFFFFFFFFu );
      assembly_ptr[write_idx] = start;
      gather_sources[start]   = edges[start].orig;

      int32_t internal_count = 0;

      for ( int64_t k = start + 1; k < end; ++k ) {
        uint64_t curr_key = edges[k].key;
        uint64_t prev_key = edges[k - 1].key;

        gather_sources[k] = edges[k].orig;

        if ( curr_key != prev_key ) {
          internal_count++;
          // Close previous assembly pointer
          // (assembly_ptr[i+1] is start of next, which is current k)
          assembly_ptr[write_idx + internal_count] = k;

          // Write new column index
          indices[write_idx + internal_count] = static_cast< int32_t >( curr_key & 0xFFFFFFFFu );
        }
      }
    }
  }

  void update( const double* V_data, double* csr_data ) const
  {
    if ( nnz == 0 )
      return;

#pragma omp      parallel for schedule( static )
    for ( int i = 0; i < nnz; ++i ) {
      int32_t start = assembly_ptr[i];
      int32_t end   = assembly_ptr[i + 1];

      double sum = 0.0;
// The compiler will autovectorize this reduction effectively
// because gather_sources is contiguous in memory now.
#pragma omp simd reduction( + : sum )
      for ( int32_t k = start; k < end; ++k ) {
        sum += V_data[gather_sources[k]];
      }
      csr_data[i] = sum;
    }
  }
};
