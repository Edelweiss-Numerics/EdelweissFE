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
/// The AMGCL hierarchies of EdelweissFE with the spectral radius estimate of amgcl-spectral-radius.hpp: a Chebyshev
/// relaxation and a smoothed aggregation of our own, which estimate it with Arnoldi steps (spectral_radius_estimate)
/// instead of AMGCL's power iteration (seeded per OpenMP thread), and the runtime selections of the relaxation and the
/// coarsening that use them (amg_hierarchy) -- everything else, every other relaxation and coarsening, is AMGCL's.
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

namespace amgcl {

  namespace relaxation {

    /// @brief AMGCL's Chebyshev relaxation (amgcl/relaxation/chebyshev.hpp), step for step, with the spectral radius
    /// estimate of amgcl-spectral-radius.hpp: the polynomial of the given degree that is smallest on [lower, higher] x
    /// rho, rho estimated by Arnoldi steps (spectral_radius_estimate; of A, or with scale of D^-1 A, D its diagonal).
    template < class Backend >
    class chebyshev_with_estimate {
    public:
      typedef typename Backend::value_type                 value_type;
      typedef typename Backend::vector                     vector;
      typedef typename math::scalar_of< value_type >::type scalar_type;

      /// @brief The parameters, AMGCL's names and defaults, but the Arnoldi steps of spectral_radius_estimate for
      /// power_iters.
      struct params {
        /// @brief The degree of the polynomial.
        unsigned degree = 5;
        /// @brief The upper end of the interval, relative to the estimate: a margin above it.
        float higher = 1.0f;
        /// @brief The lower end of the interval, relative to the estimate.
        float lower = 1.0f / 30;
        /// @brief How the spectral radius is estimated.
        spectral_radius_estimate spectralRadius = spectral_radius_estimate::byDefault();
        /// @brief Smooth D^-1 A instead of A.
        bool scale = false;

        params() {}

        /// @brief From a parameter tree, rejecting what it does not know.
        params( boost::property_tree::ptree p )
        {
          spectralRadius = spectral_radius_estimate::take( p, "relax" );
          degree         = p.get( "degree", degree );
          higher         = p.get( "higher", higher );
          lower          = p.get( "lower", lower );
          scale          = p.get( "scale", scale );
          for ( const auto& entry : p ) {
            if ( entry.first != "degree" && entry.first != "higher" && entry.first != "lower" &&
                 entry.first != "scale" )
              throw std::invalid_argument( "relax: unknown parameter of the Chebyshev smoother: " + entry.first );
          }
        }
      } prm;

      /// @brief Estimate the spectral radius of A, and set up the polynomial's interval.
      template < class Matrix >
      chebyshev_with_estimate( const Matrix& A, const params& prm, const typename Backend::params& backend_prm )
        : prm( prm ),
          p( Backend::create_vector( backend::rows( A ), backend_prm ) ),
          r( Backend::create_vector( backend::rows( A ), backend_prm ) )
      {
        if ( prm.scale )
          M = Backend::copy_vector( backend::diagonal( A, /*invert*/ true ), backend_prm );
        scalar_type hi = static_cast< scalar_type >(
          backend::estimated_spectral_radius( A, prm.spectralRadius, prm.scale ) );
        const scalar_type lo = hi * prm.lower;
        hi *= prm.higher;
        d = 0.5 * ( hi + lo );
        c = 0.5 * ( hi - lo );
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
      scalar_type                                          c, d;

      /// @brief degree Chebyshev steps on A x = b from x (AMGCL's recurrence).
      template < class Matrix, class VectorB, class VectorX >
      void solve( const Matrix& A, const VectorB& b, VectorX& x ) const
      {
        static const scalar_type one   = math::identity< scalar_type >();
        static const scalar_type zero  = math::zero< scalar_type >();
        scalar_type              alpha = zero, beta = zero;
        for ( unsigned k = 0; k < prm.degree; ++k ) {
          backend::residual( b, A, x, *r );
          if ( prm.scale )
            backend::vmul( one, *M, *r, zero, *r );
          if ( k == 0 ) {
            alpha = math::inverse( d );
            beta  = zero;
          }
          else if ( k == 1 ) {
            alpha = 2 * d * math::inverse( 2 * d * d - c * c );
            beta  = alpha * d - one;
          }
          else {
            alpha = math::inverse( d - 0.25 * alpha * c * c );
            beta  = alpha * d - one;
          }
          backend::axpby( alpha, *r, beta, *p );
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

    /// @brief AMGCL's smoothed aggregation parameters, with the Arnoldi steps of spectral_radius_estimate for
    /// power_iters: the estimate of rho(D^-1 A) the prolongation smoothing weight takes when estimate_spectral_radius
    /// is set.
    template < class Backend >
    struct smoothed_aggregation_estimate_params : public smoothed_aggregation< Backend >::params {
      typedef typename smoothed_aggregation< Backend >::params Base;

      /// @brief How rho(D^-1 A) is estimated.
      spectral_radius_estimate spectralRadius = spectral_radius_estimate::byDefault();

      smoothed_aggregation_estimate_params() {}

      smoothed_aggregation_estimate_params( boost::property_tree::ptree p )
        : Base( without_estimate( p, spectralRadius ) )
      {
      }

    private:
      /// @brief p without the keys of the estimate, which go to how (AMGCL's parameters would warn about them).
      static boost::property_tree::ptree without_estimate( boost::property_tree::ptree p,
                                                           spectral_radius_estimate&   how )
      {
        how = spectral_radius_estimate::take( p, "coarsening" );
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
