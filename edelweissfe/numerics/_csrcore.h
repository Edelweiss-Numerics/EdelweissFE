#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <omp.h>
#include <stdexcept>
#include <utility>
#include <vector>

struct PackedEdge {
  uint64_t key;  // High 32: Row, Low 32: Col
  int32_t  orig; // Original index

  // Default comparison for std::sort is now extremely fast
  bool operator<( const PackedEdge& other ) const { return key < other.key; }
};

/**
 * @brief The entities that contribute a block to the system matrix, as the direct assembly sees them: borrowed
 * views of flat arrays, one entry per entity unless stated otherwise.
 *
 * Entity e has the global DOF list dofs[dofStart[e] .. dofStart[e + 1]). Its block is either dense (it couples
 * all its DOFs, column-major: value k lies in local row k % nDofE and local column k / nDofE, the rule of
 * edelweissfe.numerics.systemmatrixcontributor.denseBlockLocalEntries) or a list of entries of its own, given in
 * its local DOF numbering in localRows and localColumns [localEntryStart[e] .. localEntryStart[e + 1]). A dense
 * entity lists no entries.
 */
struct SystemMatrixEntities {
  int64_t        nEntities;         ///< the number of entities
  const int64_t* dofStart;          ///< nEntities + 1: where the DOF list of each entity begins in dofs
  const int32_t* dofs;              ///< the global DOF lists of all entities, one after the other
  const uint8_t* couplesAllItsDofs; ///< nonzero for a dense block
  const int64_t* localEntryStart;   ///< nEntities + 1: where the listed entries of each entity begin
  const int32_t* localRows;         ///< the local row of every listed entry
  const int32_t* localColumns;      ///< the local column of every listed entry

  /** @brief The number of DOFs of entity @p e. */
  int nDofOf( int64_t e ) const { return static_cast< int >( dofStart[e + 1] - dofStart[e] ); }

  /** @brief Whether the block of entity @p e is dense. */
  bool isDense( int64_t e ) const { return couplesAllItsDofs[e] != 0; }
};

/**
 * @brief The CSR pattern of a square matrix: for every row, the sorted columns of its entries.
 *
 * Both assembly paths keep their pattern in one, derived independently: CSRCore sorts and sums the (I, J) pairs of
 * the VIJ path, CSRPattern::ofEntities unions the entities' columns row by row for the direct path. Entries that no
 * entity contributes (the fill-in of the multi-point-constraint condensation) are added to either by addEntries.
 */
struct CSRPattern {
  std::vector< int > indptr;   ///< nDof + 1: where the entries of each row begin in indices
  std::vector< int > indices;  ///< the column of every entry, sorted within each row
  int                nnz  = 0; ///< the number of entries
  int                nDof = 0; ///< the number of rows and columns

  /** @brief The observer of addEntries for a pattern without anything to keep in step with it. */
  struct NoMergeObserver {
    /** @brief Called once, before the merge, with the number of entries after it. */
    void beginMerge( int ) {}
    /** @brief Called for every entry after the merge, in parallel: its position, and the position of the next old
     * entry at or after it (the old entry itself, if it is one). */
    void entryMerged( int, int64_t ) {}
  };

  /**
   * @brief Add entries, merged into the sorted rows.
   *
   * The old pattern is moved into @p retiredIndptr and @p retiredIndices, not freed: a matrix handed out before may
   * still point into it.
   *
   * @tparam MergeObserver keeps a structure indexed by entry in step (see NoMergeObserver).
   * @param extraPtr nDof + 1: where the entries to add of each row begin in @p extraIdx.
   * @param extraIdx the columns of the entries to add, sorted within each row, none already in the pattern.
   * @param retiredIndptr receives the old indptr.
   * @param retiredIndices receives the old indices.
   * @param observer is told about the merge.
   */
  template < class MergeObserver = NoMergeObserver >
  void addEntries( const int64_t*      extraPtr,
                   const int32_t*      extraIdx,
                   std::vector< int >& retiredIndptr,
                   std::vector< int >& retiredIndices,
                   MergeObserver&&     observer = MergeObserver{} )
  {
    std::vector< int > newIndptr( nDof + 1 );
    newIndptr[0] = 0;
    for ( int r = 0; r < nDof; ++r ) {
      const int64_t length = static_cast< int64_t >( indptr[r + 1] - indptr[r] ) + extraPtr[r + 1] - extraPtr[r];
      if ( newIndptr[r] + length > std::numeric_limits< int >::max() )
        throw std::overflow_error( "CSRPattern::addEntries: the pattern would exceed 2^31 - 1 entries." );
      newIndptr[r + 1] = static_cast< int >( newIndptr[r] + length );
    }
    const int newNnz = newIndptr[nDof];

    std::vector< int > newIndices( newNnz );
    observer.beginMerge( newNnz );
    int alreadyThere = 0;

#pragma omp parallel for schedule( dynamic, 1024 ) reduction( + : alreadyThere )
    for ( int r = 0; r < nDof; ++r ) {
      int64_t k = indptr[r], e = extraPtr[r];
      int     w = newIndptr[r];
      while ( k < indptr[r + 1] || e < extraPtr[r + 1] ) {
        const bool takeOld = e == extraPtr[r + 1] || ( k < indptr[r + 1] && indices[k] < extraIdx[e] );
        if ( !takeOld && k < indptr[r + 1] && indices[k] == extraIdx[e] )
          ++alreadyThere;
        newIndices[w] = takeOld ? indices[k] : extraIdx[e];
        observer.entryMerged( w, k );
        ++w;
        takeOld ? ++k : ++e;
      }
    }

    if ( alreadyThere )
      throw std::invalid_argument( "CSRPattern::addEntries: an entry to add is already in the pattern." );

    retiredIndptr  = std::exchange( indptr, std::move( newIndptr ) );
    retiredIndices = std::exchange( indices, std::move( newIndices ) );
    nnz            = newNnz;
  }

  /**
   * @brief The pattern of the entities' blocks, derived row by row from their DOF lists.
   *
   * The entity-to-DOF lists are transposed into, per row, the entities touching it and as which of their local
   * rows (a counting sort). Each row is then the union of the columns these entities couple it with -- all of the
   * entity's DOFs for a dense block, the columns of the listed entries in that local row otherwise --, collected
   * with a marker array of nDof ints per thread, and sorted. A count pass sets indptr, a fill pass writes indices.
   * The result does not depend on the number of threads.
   *
   * @param entities the entities.
   * @param nDof the number of rows and columns.
   * @param nThreads the number of threads.
   * @return the pattern.
   */
  static CSRPattern ofEntities( const SystemMatrixEntities& entities, int nDof, int nThreads );

private:
  /**
   * @brief The listed entries of every sparse entity, grouped by local row, as global columns.
   *
   * @param entities the entities.
   * @param groupOffset nEntities + 1: where the groups of each entity begin in @p groupStart (nDofE + 1 of them for a
   * sparse entity, none for a dense one).
   * @param groupStart receives, per sparse entity and local row, where its columns begin in @p groupedColumns.
   * @param groupedColumns receives the global column of every listed entry, grouped by local row.
   * @param nThreads the number of threads.
   */
  static void groupListedEntriesByLocalRow( const SystemMatrixEntities&   entities,
                                            const std::vector< int64_t >& groupOffset,
                                            std::vector< int64_t >&       groupStart,
                                            std::vector< int32_t >&       groupedColumns,
                                            int                           nThreads );
};

inline void CSRPattern::groupListedEntriesByLocalRow( const SystemMatrixEntities&   entities,
                                                      const std::vector< int64_t >& groupOffset,
                                                      std::vector< int64_t >&       groupStart,
                                                      std::vector< int32_t >&       groupedColumns,
                                                      int                           nThreads )
{
  // each sparse entity on its own: disjoint parts of groupStart and groupedColumns
#pragma omp parallel for schedule( dynamic, 16 ) num_threads( nThreads )
  for ( int64_t e = 0; e < entities.nEntities; ++e ) {
    if ( entities.isDense( e ) )
      continue;
    const int      n     = entities.nDofOf( e );
    const int32_t* d     = entities.dofs + entities.dofStart[e];
    const int64_t  first = entities.localEntryStart[e];
    const int64_t  last  = entities.localEntryStart[e + 1];
    int64_t*       group = groupStart.data() + groupOffset[e];

    std::fill( group, group + n + 1, 0 );
    for ( int64_t k = first; k < last; ++k )
      ++group[entities.localRows[k] + 1];
    std::partial_sum( group, group + n + 1, group );
    std::vector< int64_t > next( group, group + n );
    for ( int64_t k = first; k < last; ++k )
      groupedColumns[first + next[entities.localRows[k]]++] = d[entities.localColumns[k]];
    for ( int a = 0; a <= n; ++a )
      group[a] += first;
  }
}

inline CSRPattern CSRPattern::ofEntities( const SystemMatrixEntities& entities, int nDof, int nThreads )
{
  const int64_t nEntities = entities.nEntities;
  nThreads                = nThreads > 0 ? nThreads : 1;

  CSRPattern pattern;
  pattern.nDof = nDof;

  // --- the entities touching each row, and as which of their local rows: entity -> DOF, transposed
  std::vector< int64_t > touchStart( static_cast< size_t >( nDof ) + 1, 0 );
  for ( int64_t i = 0; i < entities.dofStart[nEntities]; ++i ) {
    const int32_t dof = entities.dofs[i];
    if ( dof < 0 || dof >= nDof )
      throw std::out_of_range( "CSRPattern::ofEntities: a DOF of an entity lies outside [0, nDof)." );
    ++touchStart[dof + 1];
  }
  std::partial_sum( touchStart.begin(), touchStart.end(), touchStart.begin() );

  std::vector< int64_t > touchEntity( touchStart[nDof] );
  std::vector< int32_t > touchLocalRow( touchStart[nDof] );
  {
    std::vector< int64_t > next( touchStart.begin(), touchStart.end() - 1 );
    for ( int64_t e = 0; e < nEntities; ++e )
      for ( int a = 0; a < entities.nDofOf( e ); ++a ) {
        const int64_t t  = next[entities.dofs[entities.dofStart[e] + a]]++;
        touchEntity[t]   = e;
        touchLocalRow[t] = a;
      }
  }

  // --- the listed entries of every sparse entity, grouped by local row, as global columns
  for ( int64_t e = 0; e < nEntities; ++e )
    for ( int64_t k = entities.localEntryStart[e]; k < entities.localEntryStart[e + 1]; ++k )
      if ( entities.localRows[k] < 0 || entities.localRows[k] >= entities.nDofOf( e ) || entities.localColumns[k] < 0 ||
           entities.localColumns[k] >= entities.nDofOf( e ) )
        throw std::out_of_range( "CSRPattern::ofEntities: a listed entry lies outside the DOFs of its entity." );

  std::vector< int64_t > groupOffset( nEntities + 1, 0 );
  for ( int64_t e = 0; e < nEntities; ++e )
    groupOffset[e + 1] = groupOffset[e] + ( entities.isDense( e ) ? 0 : entities.nDofOf( e ) + 1 );
  std::vector< int64_t > groupStart( groupOffset[nEntities] );
  std::vector< int32_t > groupedColumns( entities.localEntryStart[nEntities] );
  groupListedEntriesByLocalRow( entities, groupOffset, groupStart, groupedColumns, nThreads );

  // --- row by row: the union of the columns of the entities touching the row
  // a dense entity couples the row with all its DOFs, a sparse one with the columns of its entries in that local row
  auto collectRow = [&]( int r, std::vector< int >& marker, std::vector< int >& row ) {
    row.clear();
    for ( int64_t t = touchStart[r]; t < touchStart[r + 1]; ++t ) {
      const int64_t  e        = touchEntity[t];
      const int32_t* columns  = entities.dofs + entities.dofStart[e];
      int64_t        nColumns = entities.nDofOf( e );
      if ( !entities.isDense( e ) ) {
        const int64_t* group = groupStart.data() + groupOffset[e] + touchLocalRow[t];
        columns              = groupedColumns.data() + group[0];
        nColumns             = group[1] - group[0];
      }
      for ( int64_t i = 0; i < nColumns; ++i )
        if ( marker[columns[i]] != r ) {
          marker[columns[i]] = r;
          row.push_back( columns[i] );
        }
    }
  };

  std::vector< int64_t > rowLength( nDof );
#pragma omp parallel num_threads( nThreads )
  {
    std::vector< int > marker( nDof, -1 );
    std::vector< int > row;
#pragma omp for schedule( dynamic, 64 )
    for ( int r = 0; r < nDof; ++r ) {
      collectRow( r, marker, row );
      rowLength[r] = static_cast< int64_t >( row.size() );
    }
  }

  pattern.indptr.assign( static_cast< size_t >( nDof ) + 1, 0 );
  int64_t nnz = 0;
  for ( int r = 0; r < nDof; ++r ) {
    nnz += rowLength[r];
    if ( nnz > std::numeric_limits< int >::max() )
      throw std::overflow_error( "CSRPattern::ofEntities: the pattern would exceed 2^31 - 1 entries." );
    pattern.indptr[r + 1] = static_cast< int >( nnz );
  }
  pattern.nnz = static_cast< int >( nnz );
  pattern.indices.resize( nnz );

#pragma omp parallel num_threads( nThreads )
  {
    std::vector< int > marker( nDof, -1 );
    std::vector< int > row;
#pragma omp for schedule( dynamic, 64 )
    for ( int r = 0; r < nDof; ++r ) {
      collectRow( r, marker, row );
      std::sort( row.begin(), row.end() );
      std::copy( row.begin(), row.end(), pattern.indices.begin() + pattern.indptr[r] );
    }
  }

  return pattern;
}

/**
 * @brief The CSR pattern of the VIJ path, derived from the (I, J) pairs by sort-and-sum, and its gather map: for every
 * entry of the pattern, the VIJ values that sum to it.
 */
class CSRCore : public CSRPattern {
public:
  // Assembly Mapping. Only the gather needs these; see releaseGatherMap().
  std::vector< int32_t > gather_sources;
  std::vector< int32_t > assembly_ptr;

  // gather_sources and assembly_ptr are int32 indices into the COO list, so the build refuses more than INT32_MAX
  // pairs; the direct path, which derives its pattern from the entities, has no such limit.
  CSRCore( const int* I, const int* J, int64_t n_pairs, int n_dof )
  {
    nDof = n_dof;
    if ( n_pairs == 0 ) {
      // Empty pattern: indptr is all-zeros, indices/gather_sources/assembly_ptr
      // remain empty. update() is guarded by nnz==0 and is a no-op in that case.
      indptr.assign( nDof + 1, 0 );
      return;
    }

    build( I, J, n_pairs );
  }

private:
  void build( const int* I, const int* J, int64_t n_pairs )
  {
    // --- SAFETY & CONFIG ---
    // gather_sources and assembly_ptr are int32 indices into the COO list
    if ( n_pairs > std::numeric_limits< int32_t >::max() ) {
      throw std::overflow_error( "CSRCore: n_pairs exceeds 32-bit limit. Assemble directly into CSR "
                                 "(systemMatrixAssembly=direct), which has no such limit." );
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
    assembly_ptr[nnz] = static_cast< int32_t >( n_pairs ); // Sentinel; safe: guarded by INT32_MAX above

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

public:
  // Give back the gather map, keeping the pattern: the generator does so when it goes, while a matrix it handed out
  // may still use the pattern. The swap idiom is what actually returns the capacity; clear() alone would not.
  void releaseGatherMap()
  {
    std::vector< int32_t >().swap( gather_sources );
    std::vector< int32_t >().swap( assembly_ptr );
  }

  // Add entries that no COO pair contributes to, assembled as stored zeros: room for values that a caller writes
  // after the assembly (the in-place multi-point-constraint condensation). The pattern merges them in
  // (CSRPattern::addEntries); each new entry gets an empty range of the gather map, so update() writes 0.0 there.
  void addEntries( const int64_t*      extraPtr,
                   const int32_t*      extraIdx,
                   std::vector< int >& retiredIndptr,
                   std::vector< int >& retiredIndices )
  {
    if ( assembly_ptr.empty() )
      assembly_ptr.assign( 1, 0 ); // an empty pattern: just the sentinel

    /** @brief Keeps assembly_ptr in step with the merge: a new entry's gather range is the empty one where the range
     * of the next old entry starts. */
    struct GatherRangesOfMergedEntries {
      const std::vector< int32_t >& old;
      std::vector< int32_t >        merged;
      void                          beginMerge( int newNnz ) { merged.resize( static_cast< size_t >( newNnz ) + 1 ); }
      void entryMerged( int newPosition, int64_t oldPosition ) { merged[newPosition] = old[oldPosition]; }
    } ranges{ assembly_ptr, {} };

    const int32_t end = assembly_ptr.back();
    CSRPattern::addEntries( extraPtr, extraIdx, retiredIndptr, retiredIndices, ranges );
    ranges.merged[nnz] = end;
    assembly_ptr.swap( ranges.merged );
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
// unnecessary entirely. It needs no I/J pair either: the pattern is derived from the entities
// (CSRPattern::ofEntities), and so are the offsets and the rows of every block.
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
// IMPORTANT: a compiled entity loop in another extension may include this header and call scatterBlock,
// which inlines this code into that extension. Any change here needs such an extension rebuilt as well.

/**
 * @brief Scatters the blocks of the entities straight into the data array of its own CSR pattern.
 *
 * Built in three steps: the pattern from the entities (constructor), entries added to it (CSRPattern::addEntries, the
 * fill-in of the multi-point-constraint condensation), then the map of every value of every block to its CSR entry
 * (mapEntities). Two layouts of a block, as the entity declares it (SystemMatrixEntities::couplesAllItsDofs):
 *
 * - dense: the nDofE x nDofE block, column-major, as every element writes it. All values of a local row share the
 *   global row, so only nDofE rows per entity are kept.
 * - listed: the entity lists its entries itself (a constraint coupling only some of its degrees of freedom); each
 *   value is scattered on its own, and keeps its global row.
 *
 * The layout is never guessed from the number of values: a list can have exactly nDofE^2 entries (a rigid body
 * contact with one slave node in 3D: 36 + 9 + 18 + 18 = 81 = 9^2) in an order that is not column-major.
 */
struct CSRDirectAssembler : CSRPattern {
  double*                           data     = nullptr; ///< the CSR data array, borrowed; what is assembled
  int                               nThreads = 1;       ///< how many threads may scatter at the same time
  std::vector< int64_t >            entryStart;         ///< nEntities + 1: where the values of each block begin
  std::vector< uint16_t >           offsets;            ///< per value of every block: its position within its CSR row
  std::vector< int64_t >            entityRowStart;  ///< nEntities: where the rows of each entity begin in entityRows
  std::vector< int >                entityNDof;      ///< per entity: its number of DOFs
  std::vector< char >               entityIsDense;   ///< per entity: the layout it declared
  std::vector< int >                entityRows;      ///< per entity: the global row of each local row (dense) or value
  std::vector< std::vector< int > > rowStartScratch; ///< per thread, size of the largest entity

  /**
   * @brief The pattern of the entities' blocks; the map follows with mapEntities, once the pattern is complete.
   *
   * @param entities the entities.
   * @param nDof_ the number of rows and columns.
   * @param nThreads_ how many threads build the pattern and the map, and may scatter at the same time.
   */
  CSRDirectAssembler( const SystemMatrixEntities& entities, int nDof_, int nThreads_ )
    : CSRPattern( CSRPattern::ofEntities( entities, nDof_, nThreads_ ) ), nThreads( nThreads_ > 0 ? nThreads_ : 1 )
  {
  }

  /**
   * @brief Map every value of every block to its CSR entry, on the complete pattern.
   *
   * @param entities the entities, the same as given to the constructor.
   * @param data_ the CSR data array of the pattern, nnz values, which the assembler scatters into.
   */
  void mapEntities( const SystemMatrixEntities& entities, double* data_ )
  {
    const int64_t nEntities = entities.nEntities;
    data                    = data_;

    entryStart.assign( nEntities + 1, 0 );
    entityRowStart.assign( nEntities, 0 );
    entityNDof.resize( nEntities );
    entityIsDense.resize( nEntities );
    int64_t nRows   = 0;
    int     maxNDof = 0;
    for ( int64_t e = 0; e < nEntities; ++e ) {
      const int64_t n       = entities.nDofOf( e );
      const int64_t nListed = entities.localEntryStart[e + 1] - entities.localEntryStart[e];
      entityNDof[e]         = static_cast< int >( n );
      entityIsDense[e]      = entities.isDense( e );
      entityRowStart[e]     = nRows;
      entryStart[e + 1]     = entryStart[e] + ( entities.isDense( e ) ? n * n : nListed );
      nRows += entities.isDense( e ) ? n : nListed;
      maxNDof = std::max( maxNDof, entityNDof[e] );
    }
    offsets.resize( entryStart[nEntities] );
    entityRows.resize( nRows );
    rowStartScratch.assign( nThreads, std::vector< int >( maxNDof, 0 ) );

    int64_t unmapped = 0;
#pragma omp parallel for schedule( dynamic, 64 ) num_threads( nThreads ) reduction( + : unmapped )
    for ( int64_t e = 0; e < nEntities; ++e ) {
      const int32_t* d  = entities.dofs + entities.dofStart[e];
      const int64_t  n  = entityNDof[e];
      const int64_t  rs = entityRowStart[e];
      const int64_t  es = entryStart[e];
      if ( entityIsDense[e] ) {
        // value k of the column-major block: local row k % n, local column k / n
        for ( int64_t a = 0; a < n; ++a )
          entityRows[rs + a] = d[a];
        for ( int64_t k = 0; k < n * n; ++k )
          unmapped += mapValue( d[k % n], d[k / n], offsets[es + k] ) ? 0 : 1;
      }
      else {
        const int64_t ls = entities.localEntryStart[e];
        for ( int64_t k = 0; k < entryStart[e + 1] - es; ++k ) {
          const int row      = d[entities.localRows[ls + k]];
          entityRows[rs + k] = row;
          unmapped += mapValue( row, d[entities.localColumns[ls + k]], offsets[es + k] ) ? 0 : 1;
        }
      }
    }

    if ( unmapped > 0 )
      throw std::runtime_error( "CSRDirectAssembler: entry unmapped, or within-row offset exceeds "
                                "the 16-bit range (row longer than 65535); a uint32 offset variant "
                                "is required for this pattern." );
  }

  /** @brief Zero the CSR data array, with all threads. Call once before the entity loop. */
  void beginAssembly()
  {
#pragma omp parallel for schedule( static ) num_threads( nThreads )
    for ( int i = 0; i < nnz; ++i )
      data[i] = 0.0;
  }

  /**
   * @brief Add the block of an entity to the CSR data, atomically.
   *
   * Called from inside the entity loop, once per entity, by the thread that computed the block. No two concurrent
   * callers may pass the same @p tid; any number of threads scatter at once.
   *
   * @param tid the thread, which selects its own row scratch (0 <= tid < nThreads).
   * @param entity the number of the entity.
   * @param block its values, entryStart[entity + 1] - entryStart[entity] of them.
   */
  void scatterBlock( int tid, int entity, const double* block ) noexcept
  {
    if ( entityIsDense[entity] )
      scatterDenseBlock( tid, entity, block );
    else
      scatterEntries( entity, block );
  }

private:
  /**
   * @brief The position of entry (@p row, @p column) within its CSR row.
   *
   * @param row the global row.
   * @param column the global column.
   * @param offset receives the position.
   * @return whether the entry is in the pattern, and its position fits in 16 bits.
   */
  bool mapValue( int row, int column, uint16_t& offset ) const
  {
    const int* first = indices.data() + indptr[row];
    const int* last  = indices.data() + indptr[row + 1];
    const int* found = std::lower_bound( first, last, column );
    if ( found == last || *found != column || found - first > std::numeric_limits< uint16_t >::max() ) {
      offset = 0;
      return false;
    }
    offset = static_cast< uint16_t >( found - first );
    return true;
  }

  /**
   * @brief Add a dense, column-major block to the CSR data, row by row.
   *
   * @param tid the thread, which selects its row scratch.
   * @param entity the number of the entity.
   * @param block its nDofE^2 values.
   */
  void scatterDenseBlock( int tid, int entity, const double* block ) noexcept
  {
    const int64_t es = entryStart[entity];
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
    const uint16_t* off0 = offsets.data() + es;
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

  /**
   * @brief Add a listed block to the CSR data: the values of the entity's own entries, each on its own.
   *
   * @param entity the number of the entity.
   * @param block its values, one per listed entry.
   */
  void scatterEntries( int entity, const double* block ) noexcept
  {
    const int64_t   rs   = entityRowStart[entity];
    const int64_t   size = entryStart[entity + 1] - entryStart[entity];
    const uint16_t* off0 = offsets.data() + entryStart[entity];

    for ( int64_t k = 0; k < size; ++k ) {
      double* dst = data + indptr[entityRows[rs + k]] + off0[k];
#pragma omp atomic
      *dst += block[k];
    }
  }

public:
  /**
   * @brief Bytes held by the pattern, the map and the entity tables -- the figure to compare against the VIJ arrays
   * and the gather map. The CSR data array is not counted: it belongs to the Python matrix.
   */
  int64_t memoryBytes() const
  {
    return static_cast< int64_t >( indptr.size() + indices.size() ) * sizeof( int ) +
           static_cast< int64_t >( offsets.size() ) * sizeof( uint16_t ) +
           static_cast< int64_t >( entityRows.size() ) * sizeof( int ) +
           static_cast< int64_t >( entityNDof.size() ) * ( 2 * sizeof( int64_t ) + sizeof( int ) + sizeof( char ) );
  }
};
