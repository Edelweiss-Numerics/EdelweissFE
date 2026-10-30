#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <limits>
#include <omp.h>
#include <stdexcept>
#include <vector>

// The multi-point-constraint condensation of a system matrix, Kt = T^T K T + C, computed in place: the values of
// Kt overwrite those of K, in K's own sparsity pattern. Row i of the result is
//
//     Kt[i,:] = sum_a  Tt[i,a] * ( sum_b K[a,b] * T[b,:] )  +  C[i,:],        Tt = T^T,
//
// built by one thread with a sparse accumulator (Gustavson's row-by-row product). The pattern of K must hold the
// structural pattern of Kt: countFillIn() and writeFillIn() give the entries it lacks, which the caller adds to K's
// pattern once, as stored zeros, before the first condenseInPlace(). An entry of K's pattern that is not in the pattern
// of Kt becomes a stored zero (the slave columns of the rows next to a slave, and the slave rows apart from their
// diagonal).
//
// The sums are taken in a fixed order, that of evaluating T^T (K T) + C row by row: the inner sum over b in the
// stored order of row a of K, the outer sum over a in the stored order of row i of T^T, the first term of each sum
// assigned and the following ones added. Each row is computed by a single thread, so the result does not depend on
// the number of threads.
//
// Most rows need no work at all: a row of T^T that is the unit entry Tt[i,i] = 1 (an independent DOF that is no
// master of any slave), with no entry of C and a row of K that has no slave column, gives a row of Kt equal to that
// row of K. Such rows are left untouched. Of the others, a row that reads only its own row of K (a row next to a
// slave, or a slave row) is computed and written back by its thread; a row that reads other rows of K too (a
// master, which collects the rows of its slaves) is computed first, into a buffer, and written back only after all
// rows have been read.
//
// The rows of K must be sorted, without duplicates: an entry is found in its row by bisection.
//
// T, T^T and C are borrowed, not copied: the caller keeps them alive and unchanged for the lifetime of this object.
//
// Nothing inside the parallel regions allocates: an exception cannot leave an OpenMP region (it would end in
// std::terminate), so every per-thread buffer is allocated before the region, where a failed allocation raises
// normally.

class MPCCondensationCore {

  // What a row of K needs for its row of Kt.
  enum RowWork : uint8_t {
    unchanged,     // the row of Kt equals the row of K
    readsOwnRow,   // computed from the row of K itself (or from nothing, a slave row)
    readsOtherRows // computed from other rows of K, which may be overwritten in the meantime
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

  // Row i of T^T is the single unit entry Tt[i,a] = 1.
  bool isSingleUnitRow( int64_t i ) const { return Ttp[i + 1] - Ttp[i] == 1 && Ttv[Ttp[i]] == 1.0; }

  RowWork rowWork( int64_t i, const int32_t* Kp, const int32_t* Ki ) const
  {
    for ( int64_t ja = Ttp[i]; ja < Ttp[i + 1]; ++ja )
      if ( Tti[ja] != i )
        return readsOtherRows;
    if ( !isSingleUnitRow( i ) || Cp[i + 1] != Cp[i] )
      return readsOwnRow;
    for ( int64_t jb = Kp[i]; jb < Kp[i + 1]; ++jb )
      if ( !tRowIsUnit[Ki[jb]] )
        return readsOwnRow;
    return unchanged;
  }

  // The rows that need work, split into those reading only their own row and those reading other rows too.
  void classifyRows( const int32_t*          Kp,
                     const int32_t*          Ki,
                     int                     nThreads,
                     std::vector< int32_t >& ownRows,
                     std::vector< int32_t >& otherRows ) const
  {
    std::vector< uint8_t > work( n );
#pragma omp parallel for num_threads( nThreads ) schedule( dynamic, 1024 )
    for ( int64_t i = 0; i < n; ++i )
      work[i] = rowWork( i, Kp, Ki );

    for ( int64_t i = 0; i < n; ++i ) {
      if ( work[i] == readsOwnRow )
        ownRows.push_back( static_cast< int32_t >( i ) );
      else if ( work[i] == readsOtherRows )
        otherRows.push_back( static_cast< int32_t >( i ) );
    }
  }

  // The scratch arrays of computeRow(), one row of K long for each thread.
  struct RowBuffers {
    int64_t                rowLength;
    std::vector< uint8_t > touched, kTTouched;
    std::vector< double >  kTValues, rowValues;
    std::vector< int32_t > kTTouchedList;

    RowBuffers( int nThreads, int64_t rowLength_ )
      : rowLength( std::max< int64_t >( rowLength_, 1 ) ),
        touched( nThreads * rowLength ),
        kTTouched( nThreads * rowLength ),
        kTValues( nThreads * rowLength ),
        rowValues( nThreads * rowLength ),
        kTTouchedList( nThreads * rowLength )
    {
    }

    // the slice of the calling thread
    int64_t offsetOfThisThread() const { return omp_get_thread_num() * rowLength; }
  };

  static int64_t longestRow( const int32_t* Kp, const std::vector< int32_t >& rows )
  {
    int64_t length = 0;
    for ( int32_t i : rows )
      length = std::max< int64_t >( length, Kp[i + 1] - Kp[i] );
    return length;
  }

  // Compute row i of Kt from the (unmodified) rows of K it reads, into out[0, length of row i of K), aligned with
  // the columns of row i of K; entries of that row that are not in the pattern of Kt are set to zero. Returns false
  // if the pattern of row i of K misses an entry of Kt, or is not sorted.
  bool computeRow( int64_t i, const int32_t* Kp, const int32_t* Ki, const double* Kv, double* out, RowBuffers& buffers )
    const
  {
    const int64_t  offset    = buffers.offsetOfThisThread();
    uint8_t*       touched   = buffers.touched.data() + offset;
    uint8_t*       kTTouched = buffers.kTTouched.data() + offset;
    double*        kTValues  = buffers.kTValues.data() + offset;
    int32_t*       kTList    = buffers.kTTouchedList.data() + offset;
    const int64_t  rowBegin  = Kp[i];
    const int64_t  length    = Kp[i + 1] - rowBegin;
    const int32_t* columns   = Ki + rowBegin;

    for ( int64_t p = 1; p < length; ++p )
      if ( columns[p - 1] >= columns[p] )
        return false;

    bool found = true;
    // the position of column c in row i of K, -1 if it is not there
    auto positionOf = [&]( int32_t c ) -> int64_t {
      const int32_t* at = std::lower_bound( columns, columns + length, c );
      if ( at == columns + length || *at != c ) {
        found = false;
        return -1;
      }
      return at - columns;
    };
    // the position of column c of row a of K T, reached through entry jb of row a of K and entry jt of row b of T:
    // the entry's own position if it is row i itself and T passes column b through unchanged
    auto positionOfKT = [&]( int64_t a, int64_t jb, int64_t jt ) -> int64_t {
      if ( a == i && tRowIsUnit[Ki[jb]] )
        return jb - rowBegin;
      return positionOf( Ti[jt] );
    };

    std::fill( touched, touched + length, 0 );
    // first touch assigns, later touches add
    auto add = [&]( int64_t p, double v ) {
      if ( p < 0 )
        return;
      if ( touched[p] )
        out[p] += v;
      else {
        touched[p] = 1;
        out[p]     = v;
      }
    };

    if ( isSingleUnitRow( i ) ) {
      // a single unit entry of T^T: row i of Kt is row a of K T, accumulated directly
      const int64_t a = Tti[Ttp[i]];
      for ( int64_t jb = Kp[a]; jb < Kp[a + 1]; ++jb ) {
        const int64_t b   = Ki[jb];
        const double  kab = Kv[jb];
        for ( int64_t jt = Tp[b]; jt < Tp[b + 1]; ++jt )
          add( positionOfKT( a, jb, jt ), kab * Tv[jt] );
      }
    }
    else {
      // several (or weighted) rows of K T: build each row a of K T on its own, then add it with its weight
      std::fill( kTTouched, kTTouched + length, 0 );
      for ( int64_t ja = Ttp[i]; ja < Ttp[i + 1]; ++ja ) {
        const int64_t a          = Tti[ja];
        const double  w          = Ttv[ja];
        int64_t       nKTTouched = 0;
        for ( int64_t jb = Kp[a]; jb < Kp[a + 1]; ++jb ) {
          const int64_t b   = Ki[jb];
          const double  kab = Kv[jb];
          for ( int64_t jt = Tp[b]; jt < Tp[b + 1]; ++jt ) {
            const int64_t p = positionOfKT( a, jb, jt );
            if ( p < 0 )
              continue;
            if ( kTTouched[p] )
              kTValues[p] += kab * Tv[jt];
            else {
              kTTouched[p]         = 1;
              kTValues[p]          = kab * Tv[jt];
              kTList[nKTTouched++] = static_cast< int32_t >( p );
            }
          }
        }
        for ( int64_t k = 0; k < nKTTouched; ++k ) {
          add( kTList[k], w * kTValues[kTList[k]] );
          kTTouched[kTList[k]] = 0;
        }
      }
    }

    for ( int64_t jc = Cp[i]; jc < Cp[i + 1]; ++jc )
      add( positionOf( Ci[jc] ), Cv[jc] );

    for ( int64_t p = 0; p < length; ++p )
      if ( !touched[p] )
        out[p] = 0.0;

    return found;
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

  // The entries of the structural pattern of Kt that the pattern of K lacks, as a pattern with sorted rows, in two
  // calls: countFillIn() writes the row pointer fillPtr (n + 1 entries) and returns the number of entries,
  // writeFillIn() writes their columns fillIdx for that fillPtr.
  int64_t countFillIn( const int32_t* Kp, const int32_t* Ki, int64_t* fillPtr, int numThreads ) const
  {
    std::vector< int64_t > rowFill( n, 0 );
    visitFillIn( Kp, Ki, nullptr, rowFill.data(), nullptr, numThreads );
    fillPtr[0] = 0;
    for ( int64_t i = 0; i < n; ++i )
      fillPtr[i + 1] = fillPtr[i] + rowFill[i];
    return fillPtr[n];
  }

  void writeFillIn( const int32_t* Kp,
                    const int32_t* Ki,
                    const int64_t* fillPtr,
                    int32_t*       fillIdx,
                    int            numThreads ) const
  {
    visitFillIn( Kp, Ki, fillPtr, nullptr, fillIdx, numThreads );
  }

  // Condense K in place: overwrite the values Kv of K (pattern Kp, Ki) with those of Kt.
  void condenseInPlace( const int32_t* Kp, const int32_t* Ki, double* Kv, int numThreads ) const
  {
    const int              nThreads = threadCount( numThreads );
    std::vector< int32_t > ownRows, otherRows;
    classifyRows( Kp, Ki, nThreads, ownRows, otherRows );

    // the rows of Kt that read other rows of K, buffered until every row has been read
    std::vector< int64_t > bufferStart( otherRows.size() + 1, 0 );
    for ( size_t k = 0; k < otherRows.size(); ++k )
      bufferStart[k + 1] = bufferStart[k] + Kp[otherRows[k] + 1] - Kp[otherRows[k]];
    std::vector< double > otherRowValues( bufferStart.back() );

    RowBuffers          buffers( nThreads, std::max( longestRow( Kp, ownRows ), longestRow( Kp, otherRows ) ) );
    std::atomic< bool > patternMismatch( false );

#pragma omp parallel num_threads( nThreads )
    {
#pragma omp for schedule( dynamic, 1 )
      for ( int64_t k = 0; k < static_cast< int64_t >( otherRows.size() ); ++k ) {
        if ( !computeRow( otherRows[k], Kp, Ki, Kv, otherRowValues.data() + bufferStart[k], buffers ) )
          patternMismatch = true;
      }
      // (the implicit barrier: every row that reads other rows has read them)

#pragma omp for schedule( dynamic, 64 )
      for ( int64_t k = 0; k < static_cast< int64_t >( ownRows.size() ); ++k ) {
        const int64_t i   = ownRows[k];
        double*       out = buffers.rowValues.data() + buffers.offsetOfThisThread();
        if ( !computeRow( i, Kp, Ki, Kv, out, buffers ) ) {
          patternMismatch = true;
          continue;
        }
        std::memcpy( Kv + Kp[i], out, sizeof( double ) * ( Kp[i + 1] - Kp[i] ) );
      }

#pragma omp for schedule( dynamic, 1 )
      for ( int64_t k = 0; k < static_cast< int64_t >( otherRows.size() ); ++k )
        std::memcpy( Kv + Kp[otherRows[k]],
                     otherRowValues.data() + bufferStart[k],
                     sizeof( double ) * ( bufferStart[k + 1] - bufferStart[k] ) );
    }

    if ( patternMismatch )
      throw std::logic_error( "MPC condensation: the pattern of K misses entries of T^T K T + C, or a row of K is "
                              "not sorted; add the fill-in pattern to K first. The values of K are now undefined." );
  }

private:
  // The fill-in of every row: counted into rowFill, or (rowFill == nullptr) written to fillIdx at fillPtr.
  void visitFillIn( const int32_t* Kp,
                    const int32_t* Ki,
                    const int64_t* fillPtr,
                    int64_t*       rowFill,
                    int32_t*       fillIdx,
                    int            numThreads ) const
  {
    const int              nThreads = threadCount( numThreads );
    std::vector< int32_t > ownRows, otherRows;
    classifyRows( Kp, Ki, nThreads, ownRows, otherRows );
    std::vector< int32_t > rows( ownRows );
    rows.insert( rows.end(), otherRows.begin(), otherRows.end() );

    // mark[c] == i + 1: column c is in row i of K, or already visited for row i
    std::vector< int32_t > markOfAll( static_cast< int64_t >( nThreads ) * n, 0 );

#pragma omp parallel num_threads( nThreads )
    {
      int32_t* mark = markOfAll.data() + static_cast< int64_t >( omp_get_thread_num() ) * n;

#pragma omp for schedule( dynamic, 64 )
      for ( int64_t k = 0; k < static_cast< int64_t >( rows.size() ); ++k ) {
        const int64_t i     = rows[k];
        const int32_t stamp = static_cast< int32_t >( i + 1 );
        int64_t       count = 0;
        int32_t*      out   = rowFill ? nullptr : fillIdx + fillPtr[i];
        auto          visit = [&]( int32_t c ) {
          if ( mark[c] == stamp )
            return;
          mark[c] = stamp;
          if ( out )
            out[count] = c;
          ++count;
        };

        for ( int64_t jb = Kp[i]; jb < Kp[i + 1]; ++jb )
          mark[Ki[jb]] = stamp;
        for ( int64_t ja = Ttp[i]; ja < Ttp[i + 1]; ++ja ) {
          const int64_t a = Tti[ja];
          for ( int64_t jb = Kp[a]; jb < Kp[a + 1]; ++jb ) {
            const int64_t b = Ki[jb];
            for ( int64_t jt = Tp[b]; jt < Tp[b + 1]; ++jt )
              visit( Ti[jt] );
          }
        }
        for ( int64_t jc = Cp[i]; jc < Cp[i + 1]; ++jc )
          visit( Ci[jc] );

        if ( out )
          std::sort( out, out + count );
        else
          rowFill[i] = count;
      }
    }
  }
};
