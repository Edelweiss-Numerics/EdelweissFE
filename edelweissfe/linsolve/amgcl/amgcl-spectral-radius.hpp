/*
 * ---------------------------------------------------------------------
 *
 *  _____    _      _              _         _____ _____
 * | ____|__| | ___| |_      _____(_)___ ___|  ___| ____|
 * |  _| / _` |/ _ \ \ \ /\ / / _ \ / __/ __| |_  |  _|
 * | |__| (_| |  __/ |\ V  V /  __/ \__ \__ \  _| | |___
 * |_____\__,_|\___|_| \_/\_/ \___|_|___/___/_|   |_____|
 *
 *
 *  Unit of Strength of Materials and Structural Analysis
 *  University of Innsbruck,
 *  2017 - today
 *
 *  Matthias Neuner matthias.neuner@uibk.ac.at
 *
 *  This file is part of EdelweissFE.
 *
 *  This library is free software; you can redistribute it and/or
 *  modify it under the terms of the GNU Lesser General Public
 *  License as published by the Free Software Foundation; either
 *  version 2.1 of the License, or (at your option) any later version.
 *
 *  The full text of the license can be found in the file LICENSE.md at
 *  the top level directory of EdelweissFE.
 * ---------------------------------------------------------------------
 */
/// The estimate of the spectral radius of an operator that every Chebyshev smoother and every prolongation smoothing
/// of EdelweissFE uses: spectralRadiusSteps Arnoldi steps from a start vector seeded per row.
///
/// Why Arnoldi. A Chebyshev polynomial damps the eigenvalues in [lower, higher] times the estimate; an estimate below
/// the spectral radius leaves the largest eigenvalues outside that interval, where the polynomial amplifies them. The
/// power method converges with the ratio of the two largest eigenvalues, which is close to one for finite element
/// operators: AMGCL's 50 power iterations could stop well below the radius. Arnoldi builds the Krylov space of the same
/// number of products, and its largest Ritz value converges far faster -- about 10 steps reach what 300 power
/// iterations do. Our operators are not symmetric (contact, softening, non-associated plasticity, the multi-point
/// constraint condensation), so symmetric Lanczos has no theory for them; Arnoldi holds for any matrix, and its
/// orthogonalization (k^2/2 vector operations for k steps) is cheap against the k sparse products.
///
/// Why a start vector seeded per row. AMGCL seeds its start vector per OpenMP thread, so its estimate -- and with it
/// the outer iterations -- depended on the thread count. A value computed from the row index alone gives the same start
/// vector at any thread count, and the inner products sum over a fixed partition of the rows in order: the estimate is
/// the same, bit for bit, at any thread count.
///
/// The estimate returned is the modulus |theta| of the Ritz value of largest modulus plus the residual norm r of its
/// Ritz pair, |h_{k+1,k} y_k|: the spectral radius is the largest modulus of the eigenvalues, so a dominant eigenvalue
/// that is negative or complex (an indefinite or strongly nonsymmetric level, which contact or softening can make) is
/// not missed, as the largest real part would miss it. For a symmetric matrix some eigenvalue lies within r of theta,
/// and since |theta| is below the spectral radius, |theta| + r is close to it from above once the dominant Ritz value
/// has converged; for a nonsymmetric matrix (theta, x) is an exact eigenpair of a matrix within r of A (in the
/// 2-norm), which bounds the error only with the conditioning of the eigenvectors. In neither case is |theta| + r a
/// guaranteed upper bound -- the Chebyshev smoother's own margin (higher, 1.1) stays -- but it lifts an unconverged
/// estimate towards the radius instead of leaving it below.
///
/// The memory: Arnoldi keeps its k basis vectors, k doubles per DOF (10 by default, 80 bytes per DOF), while the
/// estimate runs; the power method needed two. Against the operator itself (about 80 stored entries per DOF of a 3D
/// field) that is small, and it is freed before the hierarchy's next level is built.
#pragma once

#include <algorithm>
#include <amgcl/backend/builtin.hpp>
#include <amgcl/value_type/static_matrix.hpp>
#include <boost/property_tree/ptree.hpp>
#include <cmath>
#include <complex>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace amgcl {

  /// @brief How a spectral radius is estimated: spectralRadiusSteps Arnoldi steps from a start vector seeded per row.
  struct spectral_radius_estimate {
    /// @brief The Arnoldi steps (spectralRadiusSteps), at least 1.
    int steps = 10;

    /// @brief Take the estimate out of a parameter tree: spectralRadiusSteps, 10 if the tree has none. power_iters,
    /// the steps of AMGCL's power iteration, which this estimate replaces, is refused.
    /// @param p the tree, without spectralRadiusSteps afterwards
    /// @param where the tree's name in the messages, e.g. "relax"
    /// @throws std::invalid_argument for power_iters, or fewer than one step
    static spectral_radius_estimate take( boost::property_tree::ptree& p, const std::string& where )
    {
      if ( p.count( "power_iters" ) ) {
        throw std::invalid_argument( where + ".power_iters: the spectral radius is estimated by Arnoldi steps "
                                             "(spectralRadiusSteps), not by power iterations" );
      }
      spectral_radius_estimate how;
      how.steps = p.get( "spectralRadiusSteps", how.steps );
      p.erase( "spectralRadiusSteps" );
      if ( how.steps < 1 ) {
        throw std::invalid_argument( where + ".spectralRadiusSteps must be at least 1, got " +
                                     std::to_string( how.steps ) );
      }
      return how;
    }
  };

  namespace backend {

    /// @brief The start-vector entry of row i, from i alone (splitmix64, uniform in [-1, 1)): the same at any thread
    /// count.
    inline double row_seeded_value( uint64_t i )
    {
      uint64_t z = i + 0x9e3779b97f4a7c15ULL;
      z          = ( z ^ ( z >> 30 ) ) * 0xbf58476d1ce4e5b9ULL;
      z          = ( z ^ ( z >> 27 ) ) * 0x94d049bb133111ebULL;
      z          = z ^ ( z >> 31 );
      return 2.0 * ( static_cast< double >( z >> 11 ) * ( 1.0 / 9007199254740992.0 ) ) - 1.0;
    }

    namespace detail {

      typedef std::complex< double > complex;

      /// @brief The eigenvalues of the upper Hessenberg matrix H (m x m, row-major), by the shifted QR algorithm with
      /// Wilkinson shifts and Givens rotations, in complex arithmetic (m is the number of Arnoldi steps: small).
      /// A subdiagonal entry is negligible relative to its two diagonal neighbours, or to the norm of H where those are
      /// (nearly) zero; every tenth step without a deflation takes an exceptional shift, as LAPACK does, to break the
      /// cycles a Wilkinson shift can fall into.
      inline std::vector< complex > hessenberg_eigenvalues( const std::vector< double >& H, int m )
      {
        std::vector< complex > A( H.begin(), H.end() );
        auto                   at    = [&]( int i, int j ) -> complex& { return A[i * m + j]; };
        double                 normH = 0.0;
        for ( double h : H )
          normH += h * h;
        normH = std::sqrt( normH );
        std::vector< complex > eigenvalues( m );
        int                    hi = m - 1, iterations = 0;
        while ( hi >= 0 ) {
          if ( hi == 0 ) {
            eigenvalues[0] = at( 0, 0 );
            break;
          }
          // the start of the active block: the last negligible subdiagonal entry
          int lo = hi;
          while ( lo > 0 &&
                  std::abs( at( lo, lo - 1 ) ) >
                    std::max( 1e-14 * ( std::abs( at( lo, lo ) ) + std::abs( at( lo - 1, lo - 1 ) ) ), 1e-16 * normH ) )
            --lo;
          if ( lo == hi ) {
            eigenvalues[hi] = at( hi, hi );
            --hi;
            iterations = 0;
            continue;
          }
          if ( ++iterations > 100 * m )
            throw std::runtime_error(
              "spectral radius estimate: the QR algorithm on the Arnoldi matrix did not converge" );
          // Wilkinson shift: the eigenvalue of the trailing 2x2 block closer to its last diagonal entry
          const complex a = at( hi - 1, hi - 1 ), b = at( hi - 1, hi ), c = at( hi, hi - 1 ), d = at( hi, hi );
          const complex half = 0.5 * ( a - d ), root = std::sqrt( half * half + b * c );
          const complex mu1 = d + half - root, mu2 = d + half + root;
          const complex mu = iterations % 10 == 0 ? d + 1.5 * std::abs( c )
                                                  : ( std::abs( mu1 - d ) < std::abs( mu2 - d ) ? mu1 : mu2 );
          // one QR step on the active block: H - mu I = Q R, H <- R Q + mu I
          std::vector< complex > cs( hi - lo ), sn( hi - lo );
          for ( int k = lo; k <= hi; ++k )
            at( k, k ) -= mu;
          for ( int k = lo; k < hi; ++k ) {
            const complex x = at( k, k ), y = at( k + 1, k );
            const double  r = std::sqrt( std::norm( x ) + std::norm( y ) );
            cs[k - lo]      = r == 0.0 ? complex( 1.0 ) : x / r;
            sn[k - lo]      = r == 0.0 ? complex( 0.0 ) : y / r;
            for ( int j = k; j < m; ++j ) {
              const complex u = at( k, j ), v = at( k + 1, j );
              at( k, j )     = std::conj( cs[k - lo] ) * u + std::conj( sn[k - lo] ) * v;
              at( k + 1, j ) = -sn[k - lo] * u + cs[k - lo] * v;
            }
          }
          for ( int k = lo; k < hi; ++k ) {
            for ( int i = 0; i <= std::min( k + 2, hi ); ++i ) {
              const complex u = at( i, k ), v = at( i, k + 1 );
              at( i, k )     = u * cs[k - lo] + v * sn[k - lo];
              at( i, k + 1 ) = -u * std::conj( sn[k - lo] ) + v * std::conj( cs[k - lo] );
            }
          }
          for ( int k = lo; k <= hi; ++k )
            at( k, k ) += mu;
        }
        return eigenvalues;
      }

      /// @brief An eigenvector of H (m x m, row-major) for its eigenvalue lambda, by two steps of inverse iteration
      /// with a slightly perturbed shift (Gaussian elimination with partial pivoting), normalized.
      inline std::vector< complex > hessenberg_eigenvector( const std::vector< double >& H, int m, complex lambda )
      {
        double scale = 0.0;
        for ( double h : H )
          scale = std::max( scale, std::abs( h ) );
        const complex          shift = lambda + complex( 1e-10 * ( scale > 0.0 ? scale : 1.0 ), 0.0 );
        std::vector< complex > y( m, complex( 1.0 ) );
        for ( int step = 0; step < 2; ++step ) {
          std::vector< complex > M( H.begin(), H.end() ), b( y );
          for ( int i = 0; i < m; ++i )
            M[i * m + i] -= shift;
          for ( int k = 0; k < m; ++k ) {
            int pivot = k;
            for ( int i = k + 1; i < m; ++i )
              if ( std::abs( M[i * m + k] ) > std::abs( M[pivot * m + k] ) )
                pivot = i;
            if ( pivot != k ) {
              for ( int j = 0; j < m; ++j )
                std::swap( M[k * m + j], M[pivot * m + j] );
              std::swap( b[k], b[pivot] );
            }
            if ( M[k * m + k] == complex( 0.0 ) )
              M[k * m + k] = complex( 1e-300 );
            for ( int i = k + 1; i < m; ++i ) {
              const complex f = M[i * m + k] / M[k * m + k];
              for ( int j = k; j < m; ++j )
                M[i * m + j] -= f * M[k * m + j];
              b[i] -= f * b[k];
            }
          }
          for ( int i = m - 1; i >= 0; --i ) {
            complex s = b[i];
            for ( int j = i + 1; j < m; ++j )
              s -= M[i * m + j] * y[j];
            y[i] = s / M[i * m + i];
          }
          double norm = 0.0;
          for ( const complex& v : y )
            norm += std::norm( v );
          norm = std::sqrt( norm );
          for ( complex& v : y )
            v /= norm;
        }
        return y;
      }

    } // namespace detail

    /// @brief The largest modulus |theta| of the Ritz values of the leading m x m block of the Arnoldi matrix H (row
    /// width `width`), and the residual norm r = |h_{m+1,m} y_m| of its Ritz pair, y its normalized eigenvector of H.
    inline std::pair< double, double > largest_ritz_value( const std::vector< double >& H, int width, int m )
    {
      std::vector< double > Hm( static_cast< size_t >( m ) * m );
      for ( int i = 0; i < m; ++i )
        for ( int j = 0; j < m; ++j )
          Hm[i * m + j] = H[i * width + j];
      const std::vector< detail::complex > ritz    = detail::hessenberg_eigenvalues( Hm, m );
      detail::complex                      largest = ritz[0];
      for ( const detail::complex& value : ritz )
        if ( std::abs( value ) > std::abs( largest ) )
          largest = value;
      const std::vector< detail::complex > y = detail::hessenberg_eigenvector( Hm, m, largest );
      return std::make_pair( std::abs( largest ), std::abs( H[m * width + m - 1] * y[m - 1] ) );
    }

    /// @brief The spectral radius estimate of the module comment on the operator `apply`, from the start vector `v`
    /// (n entries, not all zero): how.steps Arnoldi steps, and |theta| + r of the last step's Ritz pair of largest
    /// modulus. Stops early, with an exact estimate, if the Krylov space becomes invariant.
    /// @param apply y <- A x, for double arrays of n entries
    /// @param stepsTaken if not null, the Arnoldi steps taken
    template < class Apply >
    double arnoldi_spectral_radius( ptrdiff_t                       n,
                                    std::vector< double >           v,
                                    const spectral_radius_estimate& how,
                                    Apply                           apply,
                                    int*                            stepsTaken = nullptr )
    {
      const int maxSteps = how.steps;
      // inner products over a fixed partition of the rows, the partial sums added in order: the same bits at any thread
      // count, and from run to run
      const ptrdiff_t       nChunks = std::max< ptrdiff_t >( 1, std::min< ptrdiff_t >( n, 4096 ) );
      const ptrdiff_t       chunk   = ( n + nChunks - 1 ) / nChunks;
      std::vector< double > partial( nChunks );
      auto                  dot = [&]( const double* a, const double* b ) {
#pragma omp parallel for schedule( static )
        for ( ptrdiff_t k = 0; k < nChunks; ++k ) {
          double s = 0.0;
          for ( ptrdiff_t i = k * chunk, e = std::min( n, ( k + 1 ) * chunk ); i < e; ++i )
            s += a[i] * b[i];
          partial[k] = s;
        }
        double s = 0.0;
        for ( ptrdiff_t k = 0; k < nChunks; ++k )
          s += partial[k];
        return s;
      };
      std::vector< std::vector< double > > V;
      const double                         startNorm = std::sqrt( dot( v.data(), v.data() ) );
      if ( !( startNorm > 0.0 ) )
        return 0.0;
#pragma omp parallel for
      for ( ptrdiff_t i = 0; i < n; ++i )
        v[i] /= startNorm;
      V.push_back( std::move( v ) );

      // H, (maxSteps + 1) x maxSteps, row-major in an array of width maxSteps
      std::vector< double >       H( static_cast< size_t >( maxSteps + 1 ) * maxSteps, 0.0 );
      std::pair< double, double > ritz( 0.0, 0.0 );
      std::vector< double >       w( n );
      int                         m = 0;
      for ( int j = 0; j < maxSteps; ++j ) {
        apply( V[j].data(), w.data() );
        // modified Gram-Schmidt
        for ( int i = 0; i <= j; ++i ) {
          const double h      = dot( w.data(), V[i].data() );
          H[i * maxSteps + j] = h;
          const double* vi    = V[i].data();
#pragma omp parallel for
          for ( ptrdiff_t r = 0; r < n; ++r )
            w[r] -= h * vi[r];
        }
        const double h              = std::sqrt( dot( w.data(), w.data() ) );
        H[( j + 1 ) * maxSteps + j] = h;
        m                           = j + 1;
        // invariant if what is left of A v_j after the orthogonalization is rounding: compared with the whole column
        // of H, |A v_j| -- not with H_jj alone, which is zero on an indefinite or skew part however large A v_j is
        double columnSquared = h * h;
        for ( int i = 0; i <= j; ++i )
          columnSquared += H[i * maxSteps + j] * H[i * maxSteps + j];
        const bool invariant = !( h > 1e-14 * std::sqrt( columnSquared ) );
        if ( invariant || m == maxSteps ) {
          ritz = largest_ritz_value( H, maxSteps, m );
          break;
        }
        std::vector< double > next( n );
#pragma omp parallel for
        for ( ptrdiff_t r = 0; r < n; ++r )
          next[r] = w[r] / h;
        V.push_back( std::move( next ) );
      }
      if ( stepsTaken )
        *stepsTaken = m;
      return ritz.first + ritz.second;
    }

    /// @brief The block of a matrix entry as rows x columns doubles (a scalar entry is 1 x 1).
    template < class Value >
    struct entry_block {
      static const int rows = 1;
      static double    at( const Value& v, int, int ) { return static_cast< double >( v ); }
    };

    /// @brief See entry_block.
    template < class T, int N, int M >
    struct entry_block< static_matrix< T, N, M > > {
      static const int rows = N;
      static double at( const static_matrix< T, N, M >& v, int r, int c ) { return static_cast< double >( v( r, c ) ); }
    };

    /// @brief The spectral radius estimate (arnoldi_spectral_radius()) of a CSR matrix A of scalar or B x B block
    /// entries, or of D^-1 A given the inverse diagonal entries (blocks) of A, as AMGCL's backend::diagonal(A, true)
    /// computes them for the smoother (so that the estimate is of the operator the smoother applies), from the start
    /// vector seeded per row; rows with zeroStart set (may be null) start at zero -- the Dirichlet rows, which a cycle
    /// keeps zero.
    ///
    /// The product is written out rather than AMGCL's backend::spmv, because the stored entries may be float (the
    /// float backends) while the Arnoldi vectors are double; entry_block reads a scalar or a block entry alike.
    template < class Matrix >
    double estimated_spectral_radius(
      const Matrix&                                                      A,
      const spectral_radius_estimate&                                    how,
      const numa_vector< typename backend::value_type< Matrix >::type >* inverseDiagonal,
      const char*                                                        zeroStart  = nullptr,
      int*                                                               stepsTaken = nullptr )
    {
      typedef typename backend::value_type< Matrix >::type value_type;
      const int                                            B     = entry_block< value_type >::rows;
      const ptrdiff_t                                      nRows = static_cast< ptrdiff_t >( backend::rows( A ) );
      const ptrdiff_t                                      n     = nRows * B;

      std::vector< double > start( n );
#pragma omp parallel for
      for ( ptrdiff_t i = 0; i < n; ++i )
        start[i] = zeroStart && zeroStart[i] ? 0.0 : row_seeded_value( static_cast< uint64_t >( i ) );

      return arnoldi_spectral_radius(
        n,
        std::move( start ),
        how,
        [&]( const double* x, double* y ) {
#pragma omp parallel for
          for ( ptrdiff_t i = 0; i < nRows; ++i ) {
            double s[entry_block< value_type >::rows] = {};
            for ( ptrdiff_t j = A.ptr[i]; j < A.ptr[i + 1]; ++j ) {
              const double* xc = x + static_cast< ptrdiff_t >( A.col[j] ) * B;
              for ( int r = 0; r < B; ++r )
                for ( int c = 0; c < B; ++c )
                  s[r] += entry_block< value_type >::at( A.val[j], r, c ) * xc[c];
            }
            for ( int r = 0; r < B; ++r ) {
              double value = s[r];
              if ( inverseDiagonal ) {
                value = 0.0;
                for ( int c = 0; c < B; ++c )
                  value += entry_block< value_type >::at( ( *inverseDiagonal )[i], r, c ) * s[c];
              }
              y[i * B + r] = value;
            }
          }
        },
        stepsTaken );
    }

  } // namespace backend
} // namespace amgcl
