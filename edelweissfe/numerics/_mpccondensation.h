#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <omp.h>
#include <stdexcept>
#include <vector>

// The multi-point-constraint condensation of a system matrix, Kt = T^T K T + C, computed in one threaded pass over
// the rows of the result, without any intermediate matrix. Row i of the result is
//
//     Kt[i,:] = sum_a  Tt[i,a] * ( sum_b K[a,b] * T[b,:] )  +  C[i,:],        Tt = T^T,
//
// built by one thread with a sparse accumulator (Gustavson's row-by-row product). Two passes: countPattern() counts
// the entries of every row, the caller allocates the result, fill() writes the sorted columns and the values.
//
// The sums are taken in a fixed order, that of evaluating T^T (K T) + C row by row: the inner sum over b in the
// stored order of row a of K, the outer sum over a in the stored order of row i of T^T, the first term of each sum
// assigned and the following ones added. Each row is computed by a single thread, so the result does not depend on
// the number of threads.
//
// Most rows need no arithmetic: a row of T^T that is a single unit entry (an independent DOF that is no master of
// any slave), paired with a row of K that has no slave column, gives a row of Kt equal to that row of K. Such rows
// are copied.
//
// Explicit zeros are kept: the pattern is the structural pattern of T^T K T + C. Whether to prune them is the
// solver's decision (pruneCondensedMatrixZeros).
//
// T, T^T and C are borrowed, not copied: the caller keeps them alive and unchanged for the lifetime of this object.
// countPattern() and fill() are called back to back with the same K; the per-row copy flags travel between them in
// a caller-owned buffer, so one object can serve concurrent calls.
//
// Nothing inside the parallel regions allocates: an exception cannot leave an OpenMP region (it would end in
// std::terminate), so every per-thread buffer is allocated before the region, where a failed allocation raises
// normally. A row of Kt, or of K T, has at most n entries, which bounds every buffer.

class MPCCondensationCore {

  // Zero-initialized scratch arrays, n entries for each of nThreads threads. calloc hands out untouched zero pages
  // for large blocks, so the memory is only paid for where a thread actually writes.
  template < typename V >
  struct PerThreadArray {
    V*      p;
    int64_t n;
    PerThreadArray( int nThreads, int64_t n_ )
      : p( static_cast< V* >( std::calloc( std::max< int64_t >( nThreads * n_, 1 ), sizeof( V ) ) ) ), n( n_ )
    {
      if ( !p )
        throw std::bad_alloc();
    }
    ~PerThreadArray() { std::free( p ); }
    PerThreadArray( const PerThreadArray& )            = delete;
    PerThreadArray& operator=( const PerThreadArray& ) = delete;
    // the slice of the calling thread
    V* ofThisThread() const { return p + omp_get_thread_num() * n; }
  };

  // one entry of a row being sorted
  struct ColumnValue {
    int32_t column;
    double  value;
  };

  int            n;
  const int32_t *Tp, *Ti;
  const double*  Tv;
  const int32_t *Ttp, *Tti;
  const double*  Ttv;
  const int32_t *Cp, *Ci;
  const double*  Cv;

  // tRowIsUnit[b]: row b of T is the unit row e_b (b is an independent DOF).
  std::vector< uint8_t > tRowIsUnit;

  static int threadCount( int numThreads ) { return numThreads > 0 ? numThreads : omp_get_max_threads(); }

  // Row i of T^T is the single unit entry Tt[i,a] = 1 and row i of C is empty: row i of Kt is row a of K T.
  bool isSingleUnitRow( int64_t i ) const
  {
    return Ttp[i + 1] - Ttp[i] == 1 && Ttv[Ttp[i]] == 1.0 && Cp[i + 1] == Cp[i];
  }

public:
  MPCCondensationCore( int            n_,
                       const int32_t* Tp_,
                       const int32_t* Ti_,
                       const double*  Tv_,
                       const int32_t* Ttp_,
                       const int32_t* Tti_,
                       const double*  Ttv_,
                       const int32_t* Cp_,
                       const int32_t* Ci_,
                       const double*  Cv_ )
    : n( n_ ),
      Tp( Tp_ ),
      Ti( Ti_ ),
      Tv( Tv_ ),
      Ttp( Ttp_ ),
      Tti( Tti_ ),
      Ttv( Ttv_ ),
      Cp( Cp_ ),
      Ci( Ci_ ),
      Cv( Cv_ ),
      tRowIsUnit( n_ )
  {
    for ( int64_t b = 0; b < n; ++b )
      tRowIsUnit[b] = Tp[b + 1] - Tp[b] == 1 && Ti[Tp[b]] == b && Tv[Tp[b]] == 1.0;
  }

  // Count the entries of every row of Kt into outPtr (n + 1 entries, exclusive prefix sum on return), and flag the
  // rows that are plain copies of a row of K in rowIsCopy (n entries). Returns nnz(Kt).
  int64_t countPattern( const int32_t* Kp, const int32_t* Ki, int32_t* outPtr, uint8_t* rowIsCopy, int numThreads )
  {
    const int              nThreads = threadCount( numThreads );
    std::vector< int64_t > rowNnz( n );
    // mark[c] == i + 1: column c is already counted for row i
    PerThreadArray< int32_t > markOfThread( nThreads, n );

#pragma omp parallel num_threads( nThreads )
    {
      int32_t* mark = markOfThread.ofThisThread();

#pragma omp for schedule( dynamic, 256 )
      for ( int64_t i = 0; i < n; ++i ) {
        rowIsCopy[i] = 0;

        if ( isSingleUnitRow( i ) ) {
          const int64_t a    = Tti[Ttp[i]];
          bool          copy = true;
          for ( int64_t jb = Kp[a]; jb < Kp[a + 1] && copy; ++jb )
            copy = tRowIsUnit[Ki[jb]] && ( jb == Kp[a] || Ki[jb - 1] < Ki[jb] );
          if ( copy ) {
            rowIsCopy[i] = 1;
            rowNnz[i]    = Kp[a + 1] - Kp[a];
            continue;
          }
        }

        const int32_t stamp = static_cast< int32_t >( i + 1 );
        int64_t       count = 0;
        for ( int64_t ja = Ttp[i]; ja < Ttp[i + 1]; ++ja ) {
          const int64_t a = Tti[ja];
          for ( int64_t jb = Kp[a]; jb < Kp[a + 1]; ++jb ) {
            const int64_t b = Ki[jb];
            for ( int64_t jt = Tp[b]; jt < Tp[b + 1]; ++jt ) {
              const int32_t c = Ti[jt];
              if ( mark[c] != stamp ) {
                mark[c] = stamp;
                ++count;
              }
            }
          }
        }
        for ( int64_t jc = Cp[i]; jc < Cp[i + 1]; ++jc )
          if ( mark[Ci[jc]] != stamp ) {
            mark[Ci[jc]] = stamp;
            ++count;
          }
        rowNnz[i] = count;
      }
    }

    int64_t nnz = 0;
    outPtr[0]   = 0;
    for ( int64_t i = 0; i < n; ++i ) {
      nnz += rowNnz[i];
      if ( nnz > std::numeric_limits< int32_t >::max() )
        throw std::overflow_error( "MPC condensation: the condensed matrix has more than 2^31 - 1 entries, which "
                                   "exceeds its 32-bit index type." );
      outPtr[i + 1] = static_cast< int32_t >( nnz );
    }
    return nnz;
  }

  // Write the columns (sorted) and values of Kt into outIdx/outVal, for the K, outPtr and rowIsCopy of the
  // preceding countPattern() call.
  void fill( const int32_t* Kp,
             const int32_t* Ki,
             const double*  Kv,
             const int32_t* outPtr,
             const uint8_t* rowIsCopy,
             int32_t*       outIdx,
             double*        outVal,
             int            numThreads )
  {
    const int           nThreads = threadCount( numThreads );
    std::atomic< bool > countMismatch( false );

    // Accumulator of the row of Kt: mark[c] == i + 1 says column c sits at position pos[c] of row i.
    PerThreadArray< int32_t > markOfThread( nThreads, n );
    PerThreadArray< int32_t > posOfThread( nThreads, n );
    // Accumulator of one row a of K T, for the rows of T^T that combine several rows of K: tMark[c] == tStamp says
    // column c sits at position tPos[c] of tCols/tVals.
    PerThreadArray< int32_t >     tMarkOfThread( nThreads, n );
    PerThreadArray< int32_t >     tPosOfThread( nThreads, n );
    PerThreadArray< int32_t >     tColsOfThread( nThreads, n );
    PerThreadArray< double >      tValsOfThread( nThreads, n );
    PerThreadArray< ColumnValue > sortBufferOfThread( nThreads, n );

#pragma omp parallel num_threads( nThreads )
    {
      int32_t*     mark       = markOfThread.ofThisThread();
      int32_t*     pos        = posOfThread.ofThisThread();
      int32_t*     tMark      = tMarkOfThread.ofThisThread();
      int32_t*     tPos       = tPosOfThread.ofThisThread();
      int32_t*     tCols      = tColsOfThread.ofThisThread();
      double*      tVals      = tValsOfThread.ofThisThread();
      ColumnValue* sortBuffer = sortBufferOfThread.ofThisThread();
      int32_t      tStamp     = 0;

#pragma omp for schedule( dynamic, 256 )
      for ( int64_t i = 0; i < n; ++i ) {
        const int64_t rowBegin = outPtr[i];

        if ( rowIsCopy[i] ) {
          const int64_t a      = Tti[Ttp[i]];
          const int64_t length = Kp[a + 1] - Kp[a];
          if ( rowBegin + length != outPtr[i + 1] ) {
            countMismatch = true;
            continue;
          }
          std::memcpy( outIdx + rowBegin, Ki + Kp[a], sizeof( int32_t ) * length );
          std::memcpy( outVal + rowBegin, Kv + Kp[a], sizeof( double ) * length );
          continue;
        }

        const int32_t stamp  = static_cast< int32_t >( i + 1 );
        int32_t       length = 0;
        int32_t*      idx    = outIdx + rowBegin;
        double*       val    = outVal + rowBegin;

        // first touch assigns, later touches add; a column beyond the length counted by countPattern() is not written
        // (K changed in between) and reported after the loop
        auto add = [&]( int32_t c, double v ) {
          if ( mark[c] == stamp ) {
            val[pos[c]] += v;
            return;
          }
          if ( rowBegin + length == outPtr[i + 1] ) {
            countMismatch = true;
            return;
          }
          mark[c]     = stamp;
          pos[c]      = length;
          idx[length] = c;
          val[length] = v;
          ++length;
        };

        if ( Ttp[i + 1] - Ttp[i] == 1 && Ttv[Ttp[i]] == 1.0 ) {
          // a single unit entry of T^T: row i of Kt is row a of K T, accumulated directly
          const int64_t a = Tti[Ttp[i]];
          for ( int64_t jb = Kp[a]; jb < Kp[a + 1]; ++jb ) {
            const int64_t b   = Ki[jb];
            const double  kab = Kv[jb];
            for ( int64_t jt = Tp[b]; jt < Tp[b + 1]; ++jt )
              add( Ti[jt], kab * Tv[jt] );
          }
        }
        else {
          // several (or weighted) rows of K T: build each row a of K T on its own, then add it with its weight
          for ( int64_t ja = Ttp[i]; ja < Ttp[i + 1]; ++ja ) {
            const int64_t a = Tti[ja];
            const double  w = Ttv[ja];
            ++tStamp;
            int32_t tLength = 0;
            for ( int64_t jb = Kp[a]; jb < Kp[a + 1]; ++jb ) {
              const int64_t b   = Ki[jb];
              const double  kab = Kv[jb];
              for ( int64_t jt = Tp[b]; jt < Tp[b + 1]; ++jt ) {
                const int32_t c = Ti[jt];
                if ( tMark[c] != tStamp ) {
                  tMark[c]       = tStamp;
                  tPos[c]        = tLength;
                  tCols[tLength] = c;
                  tVals[tLength] = kab * Tv[jt];
                  ++tLength;
                }
                else
                  tVals[tPos[c]] += kab * Tv[jt];
              }
            }
            for ( int32_t k = 0; k < tLength; ++k )
              add( tCols[k], w * tVals[k] );
          }
        }

        for ( int64_t jc = Cp[i]; jc < Cp[i + 1]; ++jc )
          add( Ci[jc], Cv[jc] );

        if ( rowBegin + length != outPtr[i + 1] ) {
          countMismatch = true;
          continue;
        }

        if ( std::is_sorted( idx, idx + length ) )
          continue;
        for ( int32_t k = 0; k < length; ++k )
          sortBuffer[k] = { idx[k], val[k] };
        std::sort( sortBuffer, sortBuffer + length, []( const ColumnValue& x, const ColumnValue& y ) {
          return x.column < y.column;
        } );
        for ( int32_t k = 0; k < length; ++k ) {
          idx[k] = sortBuffer[k].column;
          val[k] = sortBuffer[k].value;
        }
      }
    }

    if ( countMismatch )
      throw std::logic_error( "MPC condensation: fill() found a row length different from countPattern(); "
                              "K must not change between the two calls." );
  }
};
