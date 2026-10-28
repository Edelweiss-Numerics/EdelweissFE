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
/// The Chebyshev polynomial of every Chebyshev smoother of EdelweissFE (ChebyshevPolynomial), and the AMGCL
/// hierarchies of EdelweissFE with the spectral radius estimate of amgcl-spectral-radius.hpp: a Chebyshev relaxation
/// and a smoothed aggregation of our own, which estimate it with Arnoldi steps (SpectralRadiusEstimate) instead of
/// AMGCL's power iteration (seeded per OpenMP thread), and the runtime selections of the relaxation and the coarsening
/// that use them (amg_hierarchy) -- everything else, every other relaxation and coarsening, is AMGCL's. The relaxation
/// and the coarsening are EdelweissFE's; they live in AMGCL's namespaces because AMGCL's amg takes them as its own.
#pragma once

#include "amgcl-smoothed-prolongation.hpp"
#include "amgcl-spectral-radius.hpp"
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/runtime.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/runtime.hpp>
#include <boost/property_tree/ptree.hpp>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace EdelweissFE {
  namespace linsolve {

    /// @brief The polynomial of a Chebyshev smoother: `degree` steps of the Chebyshev iteration on A x = b for the
    /// interval [l, h] = [lower, higher] x rho, rho the spectral radius of A -- AMGCL's relaxation::chebyshev. With the
    /// centre d = (h + l) / 2 and the half-width c = (h - l) / 2 of the interval, step k is
    ///
    ///   p_k = alpha_k s_k + beta_k p_{k-1},   x <- x + p_k,   s_k = b - A x,
    ///
    ///   alpha_0 = 1 / d,   alpha_1 = 2 d / (2 d^2 - c^2),   alpha_k = 1 / (d - alpha_{k-1} c^2 / 4),
    ///   beta_0 = 0,        beta_k = alpha_k d - 1.
    ///
    /// The polynomial is smallest on the interval, so it damps the error components whose eigenvalues lie there:
    /// lower leaves the smallest eigenvalues to the coarse-grid correction, and higher puts a margin above the estimate
    /// of rho, since an eigenvalue above the interval is amplified, not damped.
    struct ChebyshevPolynomial {
      /// @brief The degree: the steps, one product with A each.
      int degree = 5;
      /// @brief The interval relative to the spectral radius, as AMGCL's float parameters, so that it rounds as
      /// AMGCL's does.
      float lower = 1.0f / 30, higher = 1.1f;

      /// @brief AMGCL's defaults, but the margin higher 1.1 (see chebyshev_with_estimate::params).
      ChebyshevPolynomial() {}

      /// @brief The polynomial, checked.
      /// @throws std::invalid_argument for a degree below 1, or an interval that does not start above zero and reach
      /// the spectral radius, 0 < lower < 1 <= higher (the parameters named as a tree's relax spells them)
      ChebyshevPolynomial( int degree, float lower, float higher ) : degree( degree ), lower( lower ), higher( higher )
      {
        if ( degree < 1 )
          throw std::invalid_argument( "relax.degree must be at least 1, got " + std::to_string( degree ) );
        if ( !( lower > 0.0f && lower < 1.0f ) || !( higher >= 1.0f ) ) {
          throw std::invalid_argument( "needs 0 < relax.lower < 1 <= relax.higher, got " + std::to_string( lower ) +
                                       " and " + std::to_string( higher ) );
        }
      }

      /// @brief The coefficients alpha_k and beta_k of the steps k = 0, ..., degree - 1 for the spectral radius rho
      /// (positive and finite, as estimatedSpectralRadius() returns it).
      /// @tparam T the scalar type of the loop that applies them: double, or float on a single-precision backend --
      /// the arithmetic is that of AMGCL's relaxation::chebyshev, operation for operation
      template < class T >
      void coefficients( double rho, std::vector< T >& alpha, std::vector< T >& beta ) const
      {
        T       h = static_cast< T >( rho );
        const T l = h * lower;
        h *= higher;
        const T d = static_cast< T >( 0.5 * ( h + l ) );
        const T c = static_cast< T >( 0.5 * ( h - l ) );
        alpha.assign( degree, T( 0 ) );
        beta.assign( degree, T( 0 ) );
        for ( int k = 0; k < degree; ++k ) {
          if ( k == 0 )
            alpha[k] = T( 1 ) / d;
          else if ( k == 1 )
            alpha[k] = 2 * d * ( T( 1 ) / ( 2 * d * d - c * c ) );
          else
            alpha[k] = static_cast< T >( 1.0 / ( d - 0.25 * alpha[k - 1] * c * c ) );
          beta[k] = k == 0 ? T( 0 ) : alpha[k] * d - T( 1 );
        }
      }
    };

  } // namespace linsolve
} // namespace EdelweissFE

namespace amgcl {

  namespace relaxation {

    /// @brief AMGCL's Chebyshev relaxation (amgcl/relaxation/chebyshev.hpp), step for step, with the spectral radius
    /// estimate of amgcl-spectral-radius.hpp: the ChebyshevPolynomial for rho, rho estimated by Arnoldi steps
    /// (SpectralRadiusEstimate; of A, or with scale of D^-1 A, D its diagonal).
    template < class Backend >
    class chebyshev_with_estimate {
    public:
      typedef typename Backend::value_type                 value_type;
      typedef typename Backend::vector                     vector;
      typedef typename math::scalar_of< value_type >::type scalar_type;

      /// @brief The parameters, AMGCL's names and defaults, but the Arnoldi steps of SpectralRadiusEstimate for
      /// power_iters, and a default margin `higher` of 1.1 instead of 1.0: AMGCL's 1.0 sits on its Gershgorin bound,
      /// which is never below rho; the Arnoldi estimate can be, slightly, and an eigenvalue above the interval is
      /// amplified, not damped.
      struct params {
        /// @brief The polynomial: degree, lower and higher.
        EdelweissFE::linsolve::ChebyshevPolynomial polynomial;
        /// @brief How the spectral radius is estimated.
        EdelweissFE::linsolve::SpectralRadiusEstimate spectralRadius;
        /// @brief Smooth D^-1 A instead of A.
        bool scale = false;

        params() {}

        /// @brief From a parameter tree, checked, rejecting what it does not know.
        /// @throws std::invalid_argument for a key it does not know, or a value it cannot use
        params( boost::property_tree::ptree p )
        {
          spectralRadius = EdelweissFE::linsolve::SpectralRadiusEstimate::take( p, "relax" );
          const EdelweissFE::linsolve::ChebyshevPolynomial defaults;
          polynomial = EdelweissFE::linsolve::ChebyshevPolynomial( p.get( "degree", defaults.degree ),
                                                                   p.get( "lower", defaults.lower ),
                                                                   p.get( "higher", defaults.higher ) );
          scale      = p.get( "scale", scale );
          for ( const auto& entry : p ) {
            if ( entry.first != "degree" && entry.first != "higher" && entry.first != "lower" &&
                 entry.first != "scale" )
              throw std::invalid_argument( "relax: unknown parameter of the Chebyshev smoother: " + entry.first );
          }
        }
      } prm;

      /// @brief Estimate the spectral radius of A, and set up the polynomial's coefficients for it.
      template < class Matrix >
      chebyshev_with_estimate( const Matrix& A, const params& prm, const typename Backend::params& backend_prm )
        : prm( prm ),
          p( Backend::create_vector( backend::rows( A ), backend_prm ) ),
          r( Backend::create_vector( backend::rows( A ), backend_prm ) )
      {
        // with scale, the inverse diagonal of A: computed once, for the estimate of rho(D^-1 A) and the smoother
        auto inverseDiagonal = prm.scale ? backend::diagonal( A, /*invert*/ true ) : nullptr;
        if ( prm.scale )
          M = Backend::copy_vector( inverseDiagonal, backend_prm );
        prm.polynomial.coefficients( EdelweissFE::linsolve::estimatedSpectralRadius( A,
                                                                                     prm.spectralRadius,
                                                                                     inverseDiagonal.get() ),
                                     alpha,
                                     beta );
      }

      template < class Matrix, class VectorRHS, class VectorX, class VectorTMP >
      void apply_pre( const Matrix& A, const VectorRHS& rhs, VectorX& x, VectorTMP& ) const
      {
        solve( A, rhs, x );
      }

      template < class Matrix, class VectorRHS, class VectorX, class VectorTMP >
      void apply_post( const Matrix& A, const VectorRHS& rhs, VectorX& x, VectorTMP& ) const
      {
        solve( A, rhs, x );
      }

      template < class Matrix, class VectorRHS, class VectorX >
      void apply( const Matrix& A, const VectorRHS& rhs, VectorX& x ) const
      {
        backend::clear( x );
        solve( A, rhs, x );
      }

      size_t bytes() const
      {
        size_t b = backend::bytes( *p ) + backend::bytes( *r );
        if ( prm.scale )
          b += backend::bytes( *M );
        return b;
      }

    private:
      std::shared_ptr< typename Backend::matrix_diagonal > M;
      mutable std::shared_ptr< vector >                    p, r;
      /// @brief The coefficients of the polynomial's steps (ChebyshevPolynomial::coefficients()).
      std::vector< scalar_type > alpha, beta;

      /// @brief The polynomial's steps on A x = b from x.
      template < class Matrix, class VectorB, class VectorX >
      void solve( const Matrix& A, const VectorB& b, VectorX& x ) const
      {
        static const scalar_type one  = math::identity< scalar_type >();
        static const scalar_type zero = math::zero< scalar_type >();
        for ( size_t k = 0; k < alpha.size(); ++k ) {
          backend::residual( b, A, x, *r );
          if ( prm.scale )
            backend::vmul( one, *M, *r, zero, *r );
          backend::axpby( alpha[k], *r, beta[k], *p );
          backend::axpby( one, *p, one, x );
        }
      }
    };

    /// @brief AMGCL's runtime choice of the relaxation (relax.type), with chebyshev_with_estimate for "chebyshev".
    template < class Backend >
    class runtime_with_chebyshev {
    public:
      typedef boost::property_tree::ptree params;

      template < class Matrix >
      runtime_with_chebyshev( const Matrix&                   A,
                              params                          prm  = params(),
                              const typename Backend::params& bprm = typename Backend::params() )
      {
        if ( prm.get( "type", std::string() ) == "chebyshev" ) {
          prm.erase( "type" );
          chebyshev_.reset(
            new chebyshev_with_estimate< Backend >( A,
                                                    typename chebyshev_with_estimate< Backend >::params( prm ),
                                                    bprm ) );
        }
        else
          other_.reset( new runtime::relaxation::wrapper< Backend >( A, prm, bprm ) );
      }

      template < class Matrix, class VectorRHS, class VectorX, class VectorTMP >
      void apply_pre( const Matrix& A, const VectorRHS& rhs, VectorX& x, VectorTMP& tmp ) const
      {
        if ( chebyshev_ )
          chebyshev_->apply_pre( A, rhs, x, tmp );
        else
          other_->apply_pre( A, rhs, x, tmp );
      }

      template < class Matrix, class VectorRHS, class VectorX, class VectorTMP >
      void apply_post( const Matrix& A, const VectorRHS& rhs, VectorX& x, VectorTMP& tmp ) const
      {
        if ( chebyshev_ )
          chebyshev_->apply_post( A, rhs, x, tmp );
        else
          other_->apply_post( A, rhs, x, tmp );
      }

      template < class Matrix, class VectorRHS, class VectorX >
      void apply( const Matrix& A, const VectorRHS& rhs, VectorX& x ) const
      {
        if ( chebyshev_ )
          chebyshev_->apply( A, rhs, x );
        else
          other_->apply( A, rhs, x );
      }

      size_t bytes() const { return chebyshev_ ? chebyshev_->bytes() : other_->bytes(); }

    private:
      std::unique_ptr< chebyshev_with_estimate< Backend > >      chebyshev_;
      std::unique_ptr< runtime::relaxation::wrapper< Backend > > other_;
    };

  } // namespace relaxation

  namespace coarsening {

    /// @brief AMGCL's smoothed aggregation parameters, with the Arnoldi steps of SpectralRadiusEstimate for
    /// power_iters: the estimate of rho(D^-1 A) the prolongation smoothing weight takes when estimate_spectral_radius
    /// is set.
    template < class Backend >
    struct smoothed_aggregation_estimate_params : public smoothed_aggregation< Backend >::params {
      typedef typename smoothed_aggregation< Backend >::params Base;

      /// @brief How rho(D^-1 A) is estimated.
      EdelweissFE::linsolve::SpectralRadiusEstimate spectralRadius;

      smoothed_aggregation_estimate_params() {}

      /// @brief From a parameter tree: the keys of the estimate to spectralRadius, the rest to AMGCL's parameters.
      /// (Each member is initialized from p itself: a base-class initializer that wrote to spectralRadius would be
      /// overwritten by the member's own initialization, which runs after it.)
      smoothed_aggregation_estimate_params( const boost::property_tree::ptree& p )
        : Base( without_estimate( p ) ), spectralRadius( estimate_of( p ) )
      {
      }

    private:
      /// @brief How p says rho(D^-1 A) is estimated.
      static EdelweissFE::linsolve::SpectralRadiusEstimate estimate_of( boost::property_tree::ptree p )
      {
        return EdelweissFE::linsolve::SpectralRadiusEstimate::take( p, "coarsening" );
      }

      /// @brief p without the keys of the estimate (AMGCL's parameters would warn about them).
      static boost::property_tree::ptree without_estimate( boost::property_tree::ptree p )
      {
        EdelweissFE::linsolve::SpectralRadiusEstimate::take( p, "coarsening" );
        return p;
      }
    };

    /// @brief AMGCL's smoothed aggregation (amgcl/coarsening/smoothed_aggregation.hpp) with the prolongation smoothing
    /// weight from the spectral radius estimate of amgcl-spectral-radius.hpp (prolongation_smoothing_weight()) and the
    /// prolongation smoothing of amgcl-smoothed-prolongation.hpp, which is AMGCL's step for step.
    template < class Backend >
    struct smoothed_aggregation_with_estimate : public smoothed_aggregation< Backend > {
      typedef smoothed_aggregation< Backend >                 Base;
      typedef smoothed_aggregation_estimate_params< Backend > params;

      /// @brief The parameters, with the steps of the estimate.
      params estimatePrm;

      smoothed_aggregation_with_estimate( const params& prm = params() ) : Base( prm ), estimatePrm( prm ) {}

      template < class Matrix >
      std::tuple< std::shared_ptr< Matrix >, std::shared_ptr< Matrix > > transfer_operators( const Matrix& A )
      {
        const size_t              n = backend::rows( A );
        typename Base::Aggregates aggr( A, this->prm.aggr, this->prm.nullspace.cols );
        this->prm.aggr.eps_strong *= 0.5;
        auto       P_tent = tentative_prolongation< Matrix >( n,
                                                        aggr.count,
                                                        aggr.id,
                                                        this->prm.nullspace,
                                                        this->prm.aggr.block_size );
        const auto omega  = prolongation_smoothing_weight( A,
                                                          this->prm.relax,
                                                          this->prm.estimate_spectral_radius,
                                                          estimatePrm.spectralRadius );
        auto       P      = smoothed_prolongation( A, aggr.strong_connection, *P_tent, omega );
        return std::make_tuple( P, transpose( *P ) );
      }
    };

    /// @brief AMGCL's runtime choice of the coarsening (coarsening.type), with smoothed_aggregation_with_estimate for
    /// "smoothed_aggregation". (AMGCL's coarsening of a block value type with a near null-space, as_scalar, is not
    /// used by EdelweissFE: its block backends take the translations through
    /// smoothed_aggregation_with_translations; it is refused.)
    template < class Backend >
    class runtime_with_smoothed_aggregation {
    public:
      typedef boost::property_tree::ptree params;

      runtime_with_smoothed_aggregation( params prm = params() )
      {
        typedef typename backend::value_type< Backend >::type value_type;
        if ( prm.get( "type", std::string( "smoothed_aggregation" ) ) == "smoothed_aggregation" ) {
          if ( math::static_rows< value_type >::value > 1 && prm.get( "nullspace.cols", 0 ) > 0 ) {
            throw std::invalid_argument( "coarsening: smoothed aggregation of blocks with a near null-space is not "
                                         "supported here (the block backends take the translations)" );
          }
          prm.erase( "type" );
          smoothedAggregation_.reset( new smoothed_aggregation_with_estimate< Backend >(
            typename smoothed_aggregation_with_estimate< Backend >::params( prm ) ) );
        }
        else
          other_.reset( new runtime::coarsening::wrapper< Backend >( prm ) );
      }

      template < class Matrix >
      std::tuple< std::shared_ptr< Matrix >, std::shared_ptr< Matrix > > transfer_operators( const Matrix& A )
      {
        return smoothedAggregation_ ? smoothedAggregation_->transfer_operators( A ) : other_->transfer_operators( A );
      }

      template < class Matrix >
      std::shared_ptr< Matrix > coarse_operator( const Matrix& A, const Matrix& P, const Matrix& R ) const
      {
        return smoothedAggregation_ ? smoothedAggregation_->coarse_operator( A, P, R )
                                    : other_->coarse_operator( A, P, R );
      }

    private:
      std::unique_ptr< smoothed_aggregation_with_estimate< Backend > > smoothedAggregation_;
      std::unique_ptr< runtime::coarsening::wrapper< Backend > >       other_;
    };

  } // namespace coarsening

  /// @brief An AMG hierarchy of EdelweissFE: AMGCL's amg, its coarsening and relaxation chosen by the parameter tree,
  /// with the Chebyshev relaxation and the smoothed aggregation of this file.
  template < class Backend >
  using amg_hierarchy = amg< Backend,
                             coarsening::runtime_with_smoothed_aggregation,
                             relaxation::runtime_with_chebyshev >;

} // namespace amgcl
