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
/// The estimate returned is |theta| + r: the modulus of the Ritz value theta of largest modulus, plus the residual norm
/// r = |h_{k+1,k} y_k| of its Ritz pair. The Ritz values theta and their vectors y are the eigenpairs of the small
/// k x k Arnoldi matrix H_k, which Eigen's dense eigensolver computes. The modulus, because the spectral radius is the
/// largest modulus of the eigenvalues: a dominant eigenvalue that is negative or complex (an indefinite or strongly
/// nonsymmetric level, which contact or softening can make) is not missed, as the largest real part would miss it. For
/// a symmetric matrix some eigenvalue lies within r of theta, and since |theta| is below the spectral radius,
/// |theta| + r is close to it from above once the dominant Ritz value has converged; for a nonsymmetric matrix
/// (theta, x), x = V_k y the Ritz vector, is an exact eigenpair of a matrix within r of A (in the 2-norm), which bounds
/// the error only with the conditioning of the eigenvectors. In neither case is |theta| + r a guaranteed upper bound --
/// the Chebyshev smoother's own margin (higher, 1.1) stays -- but it lifts an unconverged estimate towards the radius
/// instead of leaving it below.
///
/// The memory: Arnoldi keeps its k basis vectors, k doubles per DOF (10 by default, 80 bytes per DOF), while the
/// estimate runs; the power method needed two. Against the operator itself (about 80 stored entries per DOF of a 3D
/// field) that is small, and it is freed before the hierarchy's next level is built.
#pragma once

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <amgcl/backend/builtin.hpp>
#include <amgcl/value_type/static_matrix.hpp>
#include <boost/property_tree/ptree.hpp>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace EdelweissFE {
  namespace linsolve {

    /// @brief How a spectral radius is estimated: spectralRadiusSteps Arnoldi steps from a start vector seeded per row.
    struct SpectralRadiusEstimate {
      /// @brief The Arnoldi steps (spectralRadiusSteps), at least 1.
      int steps = 10;

      /// @brief Take the estimate out of a parameter tree: spectralRadiusSteps, 10 if the tree has none. power_iters,
      /// the steps of AMGCL's power iteration, which this estimate replaces, is refused.
      /// @param p the tree, without spectralRadiusSteps afterwards
      /// @param where the tree's name in the messages, e.g. "relax"
      /// @throws std::invalid_argument for power_iters, or fewer than one step
      static SpectralRadiusEstimate take( boost::property_tree::ptree& p, const std::string& where )
      {
        if ( p.count( "power_iters" ) ) {
          throw std::invalid_argument( where + ".power_iters: the spectral radius is estimated by Arnoldi steps "
                                               "(spectralRadiusSteps), not by power iterations" );
        }
        SpectralRadiusEstimate estimate;
        estimate.steps = p.get( "spectralRadiusSteps", estimate.steps );
        p.erase( "spectralRadiusSteps" );
        if ( estimate.steps < 1 ) {
          throw std::invalid_argument( where + ".spectralRadiusSteps must be at least 1, got " +
                                       std::to_string( estimate.steps ) );
        }
        return estimate;
      }
    };

    /// @brief The start-vector entry of row i, from i alone (splitmix64, uniform in [-1, 1)): the same at any thread
    /// count.
    inline double rowSeededValue( uint64_t i )
    {
      uint64_t z = i + 0x9e3779b97f4a7c15ULL;
      z          = ( z ^ ( z >> 30 ) ) * 0xbf58476d1ce4e5b9ULL;
      z          = ( z ^ ( z >> 27 ) ) * 0x94d049bb133111ebULL;
      z          = z ^ ( z >> 31 );
      return 2.0 * ( static_cast< double >( z >> 11 ) * ( 1.0 / 9007199254740992.0 ) ) - 1.0;
    }

    /// @brief The Ritz value of largest modulus of an Arnoldi process and the residual norm of its Ritz pair.
    struct RitzValue {
      /// @brief |theta|, the modulus of the Ritz value.
      double modulus = 0.0;
      /// @brief r = |h_{m+1,m} y_m|: the norm of A x - theta x for the Ritz vector x = V_m y.
      double residualNorm = 0.0;
    };

    /// @brief The Ritz value theta of largest modulus of the Arnoldi matrix H_m (m x m) and the residual norm of its
    /// Ritz pair, r = |h_{m+1,m} y_m|, y the unit eigenvector of H_m for theta and y_m its last entry. Eigen solves the
    /// small dense eigenproblem (EigenSolver: the real Schur form of H_m, then its eigenvectors, normalized). It runs
    /// on the calling thread (EIGEN_DONT_PARALLELIZE, set for this extension in setup.py), so the result is the same
    /// at any thread count.
    /// @param Hm the Arnoldi matrix H_m
    /// @param subdiagonal h_{m+1,m}, its entry below the last row
    /// @throws std::runtime_error if Eigen's eigensolver fails
    inline RitzValue largestRitzValue( const Eigen::MatrixXd& Hm, double subdiagonal )
    {
      const Eigen::EigenSolver< Eigen::MatrixXd > eigensolver( Hm );
      if ( eigensolver.info() != Eigen::Success )
        throw std::runtime_error( "spectral radius estimate: Eigen found no eigenvalues of the Arnoldi matrix" );
      const Eigen::VectorXcd theta   = eigensolver.eigenvalues();
      Eigen::Index           largest = 0;
      for ( Eigen::Index k = 1; k < theta.size(); ++k )
        if ( std::abs( theta( k ) ) > std::abs( theta( largest ) ) )
          largest = k;
      const Eigen::MatrixXcd y = eigensolver.eigenvectors();
      RitzValue              ritz;
      ritz.modulus      = std::abs( theta( largest ) );
      ritz.residualNorm = std::abs( subdiagonal * y( Hm.rows() - 1, largest ) );
      return ritz;
    }

    /// @brief The spectral radius estimate of the module comment on the operator `apply`, from the start vector `v`
    /// (n entries, not all zero): estimate.steps Arnoldi steps, and |theta| + r of the last step's Ritz pair of largest
    /// modulus. Stops early, with an exact estimate, if the Krylov space becomes invariant.
    /// @param apply y <- A x, for double arrays of n entries
    template < class Apply >
    double arnoldiSpectralRadius( ptrdiff_t                     n,
                                  std::vector< double >         v,
                                  const SpectralRadiusEstimate& estimate,
                                  Apply                         apply )
    {
      const int maxSteps = estimate.steps;
      // inner products over a fixed partition of the rows, the partial sums added in order: the same bits at any thread
      // count, and from run to run; chunk k holds the rows [k n / nChunks, (k + 1) n / nChunks), none of them empty
      const ptrdiff_t       nChunks = std::max< ptrdiff_t >( 1, std::min< ptrdiff_t >( n, 4096 ) );
      std::vector< double > partial( nChunks );
      auto                  dot = [&]( const double* a, const double* b ) {
#pragma omp parallel for schedule( static )
        for ( ptrdiff_t k = 0; k < nChunks; ++k ) {
          double s = 0.0;
          for ( ptrdiff_t i = k * n / nChunks, e = ( k + 1 ) * n / nChunks; i < e; ++i )
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

      // the Arnoldi matrix H, (maxSteps + 1) x maxSteps, upper Hessenberg
      Eigen::MatrixXd       H = Eigen::MatrixXd::Zero( maxSteps + 1, maxSteps );
      RitzValue             ritz;
      std::vector< double > w( n );
      for ( int j = 0; j < maxSteps; ++j ) {
        apply( V[j].data(), w.data() );
        // modified Gram-Schmidt
        for ( int i = 0; i <= j; ++i ) {
          const double h   = dot( w.data(), V[i].data() );
          H( i, j )        = h;
          const double* vi = V[i].data();
#pragma omp parallel for
          for ( ptrdiff_t r = 0; r < n; ++r )
            w[r] -= h * vi[r];
        }
        const double h = std::sqrt( dot( w.data(), w.data() ) );
        H( j + 1, j )  = h;
        const int m    = j + 1;
        // invariant if what is left of A v_j after the orthogonalization is rounding: compared with the whole column
        // of H, |A v_j| -- not with H_jj alone, which is zero on an indefinite or skew part however large A v_j is
        double columnSquared = h * h;
        for ( int i = 0; i <= j; ++i )
          columnSquared += H( i, j ) * H( i, j );
        // a non-finite entry of A: no estimate (estimatedSpectralRadius() refuses it)
        if ( !std::isfinite( columnSquared ) )
          return std::numeric_limits< double >::quiet_NaN();
        const bool invariant = !( h > 1e-14 * std::sqrt( columnSquared ) );
        if ( invariant || m == maxSteps ) {
          const Eigen::MatrixXd Hm = H.topLeftCorner( m, m );
          ritz                     = largestRitzValue( Hm, H( m, m - 1 ) );
          break;
        }
        std::vector< double > next( n );
#pragma omp parallel for
        for ( ptrdiff_t r = 0; r < n; ++r )
          next[r] = w[r] / h;
        V.push_back( std::move( next ) );
      }
      return ritz.modulus + ritz.residualNorm;
    }

    /// @brief The block of a matrix entry as rows x columns doubles (a scalar entry is 1 x 1).
    template < class Value >
    struct EntryBlock {
      static const int rows = 1;
      static double    at( const Value& v, int, int ) { return static_cast< double >( v ); }
    };

    /// @brief See EntryBlock.
    template < class T, int N, int M >
    struct EntryBlock< amgcl::static_matrix< T, N, M > > {
      static const int rows = N;
      static double    at( const amgcl::static_matrix< T, N, M >& v, int r, int c )
      {
        return static_cast< double >( v( r, c ) );
      }
    };

    /// @brief The spectral radius estimate (arnoldiSpectralRadius()) of a CSR matrix A of scalar or B x B block
    /// entries, or of D^-1 A given the inverse diagonal entries (blocks) of A, as AMGCL's backend::diagonal(A, true)
    /// computes them for the smoother (so that the estimate is of the operator the smoother applies), from the start
    /// vector seeded per row; rows with zeroStart set (may be null) start at zero -- the Dirichlet rows, which a cycle
    /// keeps zero.
    ///
    /// The product is written out rather than AMGCL's backend::spmv, because the stored entries may be float (the
    /// float backends) while the Arnoldi vectors are double; EntryBlock reads a scalar or a block entry alike.
    /// @throws std::runtime_error for an estimate that is not positive and finite -- a zero or non-finite operator --,
    /// since the Chebyshev polynomial and the prolongation smoothing weight 4/3 / rho divide by it
    template < class Matrix >
    double estimatedSpectralRadius(
      const Matrix&                                                                             A,
      const SpectralRadiusEstimate&                                                             estimate,
      const amgcl::backend::numa_vector< typename amgcl::backend::value_type< Matrix >::type >* inverseDiagonal,
      const char*                                                                               zeroStart = nullptr )
    {
      typedef typename amgcl::backend::value_type< Matrix >::type value_type;
      const int                                                   B = EntryBlock< value_type >::rows;
      const ptrdiff_t nRows = static_cast< ptrdiff_t >( amgcl::backend::rows( A ) );
      const ptrdiff_t n     = nRows * B;

      std::vector< double > start( n );
#pragma omp parallel for
      for ( ptrdiff_t i = 0; i < n; ++i )
        start[i] = zeroStart && zeroStart[i] ? 0.0 : rowSeededValue( static_cast< uint64_t >( i ) );

      // y <- A x, or D^-1 A x with the inverse diagonal
      auto apply = [&]( const double* x, double* y ) {
#pragma omp parallel for
        for ( ptrdiff_t i = 0; i < nRows; ++i ) {
          double s[EntryBlock< value_type >::rows] = {};
          for ( ptrdiff_t j = A.ptr[i]; j < A.ptr[i + 1]; ++j ) {
            const double* xc = x + static_cast< ptrdiff_t >( A.col[j] ) * B;
            for ( int r = 0; r < B; ++r )
              for ( int c = 0; c < B; ++c )
                s[r] += EntryBlock< value_type >::at( A.val[j], r, c ) * xc[c];
          }
          for ( int r = 0; r < B; ++r ) {
            double value = s[r];
            if ( inverseDiagonal ) {
              value = 0.0;
              for ( int c = 0; c < B; ++c )
                value += EntryBlock< value_type >::at( ( *inverseDiagonal )[i], r, c ) * s[c];
            }
            y[i * B + r] = value;
          }
        }
      };
      const double rho = arnoldiSpectralRadius( n, std::move( start ), estimate, apply );
      if ( !( rho > 0.0 ) || !std::isfinite( rho ) ) {
        throw std::runtime_error( "spectral radius estimate: got " + std::to_string( rho ) +
                                  ", but a Chebyshev smoother and a prolongation smoothing weight need a positive, "
                                  "finite spectral radius -- is the operator zero, or has it a non-finite entry?" );
      }
      return rho;
    }

  } // namespace linsolve
} // namespace EdelweissFE
