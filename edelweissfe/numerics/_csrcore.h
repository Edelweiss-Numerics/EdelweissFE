#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <omp.h>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

struct PackedEdge {
  uint64_t key;  // High 32: Row, Low 32: Col
  int32_t  orig; // Original index

  // Default comparison for std::sort is now extremely fast
  bool operator<( const PackedEdge& other ) const { return key < other.key; }
};

// The pattern needs only the (row, col) key; the gather map additionally needs to know which COO pair
// each key came from. These overloads let one build() serve both element types, so there is exactly one
// definition of how the pattern is derived and a pattern-only build cannot drift from the full one.
inline uint64_t keyOf( const PackedEdge& e )
{
  return e.key;
}
inline uint64_t keyOf( uint64_t k )
{
  return k;
}

class CSRCore {
public:
  // CSR Topology
  std::vector< int > indptr;
  std::vector< int > indices;
  int                nnz  = 0;
  int                nDof = 0;

  // Assembly Mapping. Only the gather needs these; see releaseGatherMap().
  std::vector< int32_t > gather_sources;
  std::vector< int32_t > assembly_ptr;
  bool                   gatherMapReleased = false;

  // patternOnly builds indptr/indices and nothing else. Two reasons, and the second is the one that
  // matters:
  //
  //  * The sort payload disappears. With the gather map, each pair is a PackedEdge -- 8 bytes of key
  //    plus a 4-byte origin index, padded to 16. Without it, a bare uint64_t key: 8 bytes. That array
  //    is the largest single allocation in the build (25.7 GiB at 43,350 DOF), so this halves it.
  //  * It removes every 32-bit pair index. `PackedEdge::orig`, `gather_sources` and `assembly_ptr` all
  //    hold indices into the COO list, which is why the full build has to refuse more than INT32_MAX
  //    pairs. Measured, that limit -- not memory -- is what caps the model: it bites at ~50,000 DOF
  //    with only a third of the machine in use, where memory alone would allow ~121,000. The
  //    pattern-only build has no such index, so the only 32-bit quantity left is nnz, which is three
  //    orders of magnitude from its limit at these sizes.
  //
  // A direct-to-CSR assembler borrows only the pattern, so it should always ask for this.
  CSRCore( const int* I, const int* J, int64_t n_pairs, int n_dof, bool patternOnly = false ) : nDof( n_dof )
  {
    if ( n_pairs == 0 ) {
      // Empty pattern: indptr is all-zeros, indices/gather_sources/assembly_ptr
      // remain empty. update() is guarded by nnz==0 and is a no-op in that case.
      indptr.assign( nDof + 1, 0 );
      gatherMapReleased = patternOnly;
      return;
    }

    if ( patternOnly )
      build< false >( I, J, n_pairs );
    else
      build< true >( I, J, n_pairs );
  }

private:
  template < bool WithGatherMap >
  void build( const int* I, const int* J, int64_t n_pairs )
  {
    // One key per pair, plus its origin index only when the gather map is wanted.
    using Elem = std::conditional_t< WithGatherMap, PackedEdge, uint64_t >;

    // --- SAFETY & CONFIG ---
    if constexpr ( WithGatherMap ) {
      // gather_sources and assembly_ptr are int32 indices into the COO list
      if ( n_pairs > std::numeric_limits< int32_t >::max() ) {
        throw std::overflow_error( "CSRCore: n_pairs exceeds 32-bit limit. Build the pattern only "
                                   "(patternOnly = true) if the gather map is not needed." );
      }
    }
    else {
      gatherMapReleased = true; // there never was one; update() must refuse
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
    std::vector< Elem > edges( n_pairs );

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

        if constexpr ( WithGatherMap )
          edges[pos] = { key, (int32_t)k };
        else
          edges[pos] = key;
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
        int prev_row = static_cast< int32_t >( keyOf( edges[start] ) >> 32 );

        // Mark first row count
        indptr[prev_row + 1]++;

        for ( int64_t k = start + 1; k < end; ++k ) {
          uint64_t curr_key = keyOf( edges[k] );
          uint64_t prev_key = keyOf( edges[k - 1] );

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
    if constexpr ( WithGatherMap ) {
      gather_sources.resize( n_pairs );
      assembly_ptr.resize( nnz + 1 );
      assembly_ptr[nnz] = static_cast< int32_t >( n_pairs ); // Sentinel; safe: guarded by INT32_MAX above
    }

#pragma omp parallel for schedule( dynamic, 1 )
    for ( int p = 0; p < num_partitions; ++p ) {
      int64_t start = partition_starts[p];
      int64_t end   = partition_starts[p + 1];
      if ( start == end )
        continue;

      int32_t write_idx = global_nnz_offsets[p];

      // Fill first entry of the partition
      indices[write_idx] = static_cast< int32_t >( keyOf( edges[start] ) & 0xFFFFFFFFu );
      if constexpr ( WithGatherMap ) {
        assembly_ptr[write_idx] = start;
        gather_sources[start]   = edges[start].orig;
      }

      int32_t internal_count = 0;

      for ( int64_t k = start + 1; k < end; ++k ) {
        uint64_t curr_key = keyOf( edges[k] );
        uint64_t prev_key = keyOf( edges[k - 1] );

        if constexpr ( WithGatherMap )
          gather_sources[k] = edges[k].orig;

        if ( curr_key != prev_key ) {
          internal_count++;
          if constexpr ( WithGatherMap ) {
            // Close previous assembly pointer
            // (assembly_ptr[i+1] is start of next, which is current k)
            assembly_ptr[write_idx + internal_count] = k;
          }

          // Write new column index
          indices[write_idx + internal_count] = static_cast< int32_t >( curr_key & 0xFFFFFFFFu );
        }
      }
    }
  }

public:
  // Give back the gather map, keeping the pattern. The direct-to-CSR assembler borrows only indptr
  // and indices; gather_sources is one int32 per COO pair, so at 43k DOF the map is 6.69 GiB -- the
  // second-largest allocation in the whole assembly, and larger than everything the direct path
  // holds. Leaving it around would cancel most of what that path saves, so a solver that has
  // committed to scattering releases it once the assembler has been built.
  //
  // update() must not be called afterwards. It cannot check cheaply enough to be worth it (it is a
  // nogil hot loop over nnz), so the guard lives in the Python binding, which raises. The swap idiom
  // is what actually returns the capacity; clear() alone would not.
  void releaseGatherMap()
  {
    std::vector< int32_t >().swap( gather_sources );
    std::vector< int32_t >().swap( assembly_ptr );
    gatherMapReleased = true;
  }

  // Add entries that no COO pair contributes to, assembled as stored zeros: room for values that a caller writes
  // after the assembly (the in-place multi-point-constraint condensation). They are given as a pattern of their own,
  // extraPtr (nDof + 1 entries) and extraIdx, with sorted rows and none already in the pattern. Each gets an empty
  // range of the gather map, so update() writes 0.0 there. The old pattern is moved into retiredIndptr and
  // retiredIndices, not freed: a matrix handed out before may still point into it.
  void addEntries( const int64_t*      extraPtr,
                   const int32_t*      extraIdx,
                   std::vector< int >& retiredIndptr,
                   std::vector< int >& retiredIndices )
  {
    const bool hasGatherMap = !gatherMapReleased;
    if ( hasGatherMap && assembly_ptr.empty() )
      assembly_ptr.assign( 1, 0 ); // an empty pattern: just the sentinel

    std::vector< int > newIndptr( nDof + 1 );
    newIndptr[0] = 0;
    for ( int r = 0; r < nDof; ++r ) {
      const int64_t length = static_cast< int64_t >( indptr[r + 1] - indptr[r] ) + extraPtr[r + 1] - extraPtr[r];
      if ( newIndptr[r] + length > std::numeric_limits< int >::max() )
        throw std::overflow_error( "CSRCore::addEntries: the pattern would exceed 2^31 - 1 entries." );
      newIndptr[r + 1] = static_cast< int >( newIndptr[r] + length );
    }
    const int newNnz = newIndptr[nDof];

    std::vector< int >     newIndices( newNnz );
    std::vector< int32_t > newAssemblyPtr( hasGatherMap ? newNnz + 1 : 0 );
    int                    alreadyThere = 0;

#pragma omp parallel for schedule( dynamic, 1024 ) reduction( + : alreadyThere )
    for ( int r = 0; r < nDof; ++r ) {
      int64_t k = indptr[r], e = extraPtr[r];
      int     w = newIndptr[r];
      // merge the two sorted rows; a new entry's gather range is the empty one where the next old entry's starts
      while ( k < indptr[r + 1] || e < extraPtr[r + 1] ) {
        const bool takeOld = e == extraPtr[r + 1] || ( k < indptr[r + 1] && indices[k] < extraIdx[e] );
        if ( !takeOld && k < indptr[r + 1] && indices[k] == extraIdx[e] )
          ++alreadyThere;
        newIndices[w] = takeOld ? indices[k] : extraIdx[e];
        if ( hasGatherMap )
          newAssemblyPtr[w] = assembly_ptr[k];
        ++w;
        takeOld ? ++k : ++e;
      }
    }

    if ( alreadyThere )
      throw std::invalid_argument( "CSRCore::addEntries: an entry to add is already in the pattern." );

    if ( hasGatherMap ) {
      newAssemblyPtr[newNnz] = assembly_ptr[nnz];
      assembly_ptr.swap( newAssemblyPtr );
    }
    retiredIndptr  = std::exchange( indptr, std::move( newIndptr ) );
    retiredIndices = std::exchange( indices, std::move( newIndices ) );
    nnz            = newNnz;
  }

  void update( const double* V_data, double* csr_data ) const
  {
    if ( nnz == 0 )
      return;

#pragma omp parallel for schedule( static )
    for ( int i = 0; i < nnz; ++i ) {
      int32_t start = assembly_ptr[i];
      int32_t end   = assembly_ptr[i + 1];

      double sum = 0.0;
// The compiler will autovectorize this reduction effectively
// because gather_sources is contiguous in memory now.
// MSVC's OpenMP 2.0 (/openmp) has no `omp simd` (only /openmp:experimental does); there the loop runs as a
// plain sequential sum.
#ifndef _MSC_VER
#  pragma omp simd reduction( + : sum )
#endif
      for ( int32_t k = start; k < end; ++k ) {
        sum += V_data[gather_sources[k]];
      }
      csr_data[i] = sum;
    }
  }

  // Bytes held by the CSR pattern and the gather map -- the counterpart to
  // CSRDirectAssembler::memoryBytes(), so the two assembly paths can be compared on the same
  // basis: what the object itself owns. That is the pattern (indptr, indices) plus the gather map,
  // which is the term the direct path does away with -- gather_sources is one int32 per COO pair,
  // so it scales with sizeVIJ rather than with nnz and is the second-largest array in the VIJ path
  // after the value array itself.
  //
  // Not counted here, because this object does not own them: the VIJ value and index arrays (held
  // by the DofManager) and the CSR data array (allocated by the Python-level CSRGenerator, which
  // adds it in its own memoryBytes).
  int64_t memoryBytes() const
  {
    return static_cast< int64_t >( indptr.size() ) * sizeof( int ) +
           static_cast< int64_t >( indices.size() ) * sizeof( int ) +
           static_cast< int64_t >( gather_sources.size() ) * sizeof( int32_t ) +
           static_cast< int64_t >( assembly_ptr.size() ) * sizeof( int32_t );
  }
};

// ---------------------------------------------------------------------------------------------
// Direct-to-CSR assembly.
//
// CSRCore stages values in a VIJ array of length sizeVIJ and then *gathers* duplicates into CSR.
// That staging array is the dominant memory cost of an assembly with many duplicate contributions
// per matrix entry (alongside I, J and gather_sources), and it exists only because assembly and
// reduction are separated in time.
//
// This assembler removes it by *scattering* instead: each entry knows its CSR slot directly, as
// indptr[row] + offset, where offset is the position of its column within that row. Row length is
// set by the stencil (for RKPM by the support radius), not by the DOF count, so the offset fits in
// uint16 -- half the width of the int32 gather index it replaces, and it makes the value array
// unnecessary entirely.
//
// All threads scatter into the ONE CSR data array, with `#pragma omp atomic` on every update. The
// alternative, a private CSR copy per thread summed at the end, costs nThreads * nnz * 8 bytes and
// a pass over all copies per assembly; measured on real assemblies it was no faster, and its memory
// grew with the thread count until it exceeded what the direct path saves. The atomic update is an
// instruction cost on every scattered entry, not a contention cost: threads rarely meet on the same
// entry at the same time.
//
// The order in which the threads add to an entry depends on their timing, so the assembled values
// are not bit-reproducible from run to run; they agree to round-off.
//
// IMPORTANT: scatterBlock is called from EdelweissMeshfree's particle kernel, which includes this
// header and therefore *inlines* this code into its own extension. Any change here needs BOTH
// extensions rebuilt; a stale meshfree build keeps scattering the old way.
struct CSRDirectAssembler {
  const int*              indptr;  // borrowed from the shared pattern
  const int*              indices; // borrowed
  double*                 data;    // the CSR data array of the shared pattern, borrowed; what is assembled
  int                     nnz      = 0;
  int                     nDof     = 0;
  int64_t                 nPairs   = 0;
  int                     nThreads = 1;
  std::vector< uint16_t > offsets; // within-row offset per entry; the whole map

  // Builds the map against an existing pattern, so there is exactly one definition of what the
  // CSR pattern is and the two assembly paths cannot drift apart.
  CSRDirectAssembler( const int* indptr_,
                      const int* indices_,
                      double*    data_,
                      int        nnz_,
                      int        nDof_,
                      const int* I,
                      const int* J,
                      int64_t    nPairs_,
                      int        nThreads_ )
    : indptr( indptr_ ),
      indices( indices_ ),
      data( data_ ),
      nnz( nnz_ ),
      nDof( nDof_ ),
      nPairs( nPairs_ ),
      nThreads( nThreads_ > 0 ? nThreads_ : 1 )
  {
    offsets.resize( nPairs );
    int64_t tooWide = 0;

#pragma omp parallel for schedule( static ) reduction( + : tooWide )
    for ( int64_t k = 0; k < nPairs; ++k ) {
      const int  r     = I[k];
      const int  start = indptr[r];
      const int  end   = indptr[r + 1];
      const int* found = std::lower_bound( indices + start, indices + end, J[k] );
      if ( found == indices + end || *found != J[k] ) {
        tooWide += 1; // reported as a build failure below; no entry may be unmapped
        offsets[k] = 0;
        continue;
      }
      const int64_t off = found - ( indices + start );
      if ( off > std::numeric_limits< uint16_t >::max() ) {
        tooWide += 1;
        offsets[k] = 0;
        continue;
      }
      offsets[k] = static_cast< uint16_t >( off );
    }

    if ( tooWide > 0 )
      throw std::runtime_error( "CSRDirectAssembler: entry unmapped, or within-row offset exceeds "
                                "the 16-bit range (row longer than 65535); a uint32 offset variant "
                                "is required for this pattern." );
  }

  // Scatter a whole VIJ value array through the map. This is the equivalence path used to validate
  // the addressing against CSRCore::update; the fused path hands each entity's block straight in.
  void assembleFromVIJ( const double* V, const int* I )
  {
    beginAssembly();

#pragma omp parallel for schedule( static ) num_threads( nThreads )
    for ( int64_t k = 0; k < nPairs; ++k ) {
      double* dst = data + indptr[I[k]] + offsets[k];
#pragma omp atomic
      *dst += V[k];
    }
  }

  // ---- fused path -------------------------------------------------------------------------
  //
  // beginAssembly() / scatterBlock() per entity. The entity writes its block into a small
  // thread-local scratch buffer -- which stays in cache -- and scatterBlock adds it straight into
  // the CSR data array. No sizeVIJ-sized value array is ever materialized, which is the entire point.
  //
  // Registration gives each entity its slice of the map. Two layouts of a block are supported:
  //
  //   dense    the nDofE x nDofE block, column-major (vijSize == nDofE^2), as every element writes
  //            it. For local row a, every pair (a, b) shares the same global row, so the row is
  //            I[mapStart + a] and only nDofE rows per entity need keeping rather than nDofE^2.
  //   sparse   any other vijSize: the entity declared its own list of (I, J) pairs (a constraint
  //            coupling only some of its degrees of freedom), and its block is the list of values
  //            of these pairs. Each entry is scattered on its own, so each keeps its global row.
  std::vector< int64_t >            entityMapStart;  // into offsets
  std::vector< int64_t >            entityRowStart;  // into entityRows
  std::vector< int >                entityNDof;
  std::vector< int64_t >            entityVIJSize;   // nDofE^2 for a dense block, else the number of pairs
  std::vector< int >                entityRows;      // per entity: the row of each local row (dense) or entry (sparse)
  std::vector< std::vector< int > > rowStartScratch; // per thread, size maxNDof

  bool isDense( int entity ) const
  {
    return entityVIJSize[entity] == static_cast< int64_t >( entityNDof[entity] ) * entityNDof[entity];
  }

  void registerEntities( const int64_t* mapStarts,
                         const int*     nDofs,
                         const int64_t* vijSizes,
                         int            nEntities,
                         const int*     I )
  {
    entityMapStart.assign( mapStarts, mapStarts + nEntities );
    entityNDof.assign( nDofs, nDofs + nEntities );
    entityVIJSize.assign( vijSizes, vijSizes + nEntities );
    entityRowStart.resize( nEntities );
    int64_t total = 0;
    for ( int e = 0; e < nEntities; ++e ) {
      entityRowStart[e] = total;
      total += isDense( e ) ? nDofs[e] : vijSizes[e];
    }
    entityRows.resize( total );
    // A dense block is stored COLUMN-major: initializeVIJContribution writes I[k] = idcs[k % nDof]
    // and J[k] = idcs[k / nDof], so the row index varies fastest and the first nDof entries of I are
    // exactly the entity's global row list. A sparse block keeps the row of every entry.
#pragma omp parallel for schedule( static )
    for ( int e = 0; e < nEntities; ++e ) {
      const int64_t ms   = entityMapStart[e];
      const int64_t rows = isDense( e ) ? entityNDof[e] : entityVIJSize[e];
      for ( int64_t a = 0; a < rows; ++a )
        entityRows[entityRowStart[e] + a] = I[ms + a];
    }

    int maxNDof = 0;
    for ( int e = 0; e < nEntities; ++e )
      maxNDof = std::max( maxNDof, nDofs[e] );
    rowStartScratch.assign( nThreads, std::vector< int >( maxNDof, 0 ) );
  }

  // Zero the CSR data array, with all threads. Call once before the entity loop.
  void beginAssembly()
  {
#pragma omp parallel for schedule( static ) num_threads( nThreads )
    for ( int i = 0; i < nnz; ++i )
      data[i] = 0.0;
  }

  // Called from inside the entity loop, once per entity, by the thread that computed the block. tid
  // selects the thread's own row scratch (0 <= tid < nThreads); no two concurrent callers may pass
  // the same tid. The additions to the CSR data are atomic, so any number of threads scatter at once.
  void scatterBlock( int tid, int entity, const double* block ) noexcept
  {
    if ( isDense( entity ) )
      scatterDenseBlock( tid, entity, block );
    else
      scatterEntries( entity, block );
  }

private:
  void scatterDenseBlock( int tid, int entity, const double* block ) noexcept
  {
    const int64_t ms = entityMapStart[entity];
    const int64_t rs = entityRowStart[entity];
    const int     nd = entityNDof[entity];

    // indptr[row] for each local row, resolved once per entity into a small cache-resident array:
    // the block is column-major, so the row index varies along the inner loop and the lookup cannot
    // be hoisted out of it directly.
    int* __restrict rowStart = rowStartScratch[tid].data();
    for ( int a = 0; a < nd; ++a )
      rowStart[a] = indptr[entityRows[rs + a]];

    // Traverse by ROW even though the block is stored column-major. Iterating rows outer confines
    // all writes of an inner loop to a single CSR row (cache-resident once touched) and pays a
    // stride-nDof read of the block and of the offsets instead; both are small enough to stay in
    // cache, whereas scattered writes across nDof different CSR rows would not be.
    //
    // That is the reasoning, NOT a measurement: an earlier A/B of the two traversals was withdrawn,
    // because its harness distorted the ranking and the run-to-run spread was wider than the gap.
    // Anyone reordering these loops should measure it properly first -- with the block already
    // resident, and with enough repeats to see the variance -- rather than trust this comment.
    //
    // Concurrent writers to the same CSR data are the point, so there is no __restrict on it.
    const uint16_t* off0 = offsets.data() + ms;
    for ( int a = 0; a < nd; ++a ) {
      double* row = data + rowStart[a];
      for ( int b = 0; b < nd; ++b ) {
        const int64_t k   = static_cast< int64_t >( b ) * nd + a;
        double*       dst = row + off0[k];
#pragma omp atomic
        *dst += block[k];
      }
    }
  }

  // A sparse block: the values of the entity's own (I, J) pairs, each scattered on its own.
  void scatterEntries( int entity, const double* block ) noexcept
  {
    const int64_t   rs   = entityRowStart[entity];
    const int64_t   size = entityVIJSize[entity];
    const uint16_t* off0 = offsets.data() + entityMapStart[entity];

    for ( int64_t k = 0; k < size; ++k ) {
      double* dst = data + indptr[entityRows[rs + k]] + off0[k];
#pragma omp atomic
      *dst += block[k];
    }
  }

public:
  // Bytes held by the map and the entity tables -- the figure to compare against the VIJ value array
  // and the gather map. The CSR data array is not counted: it belongs to the shared pattern, whose
  // generator counts it.
  int64_t memoryBytes() const
  {
    return static_cast< int64_t >( offsets.size() ) * 2 + static_cast< int64_t >( entityRows.size() ) * 4 +
           static_cast< int64_t >( entityMapStart.size() ) * 28;
  }
};
