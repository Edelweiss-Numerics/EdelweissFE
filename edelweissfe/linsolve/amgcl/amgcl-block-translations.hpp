/*
 * Smoothed aggregation on a block backend with the translations of the block as its near null-space
 * (EdelweissFE, AMG(blockTranslationNullspace=True) of edelweissfe/linsolve/composition/amg.py).
 *
 * smoothed_aggregation_with_translations::transfer_operators() is the one of
 * amgcl/coarsening/smoothed_aggregation.hpp (AMGCL 1.4.x) except for the tentative prolongation and the minimum
 * aggregate size; its prolongation smoothing is shared with the rigid-body coarse level
 * (amgcl-smoothed-prolongation.hpp). The license of AMGCL, which that code is taken from, follows verbatim.
 *
 * The MIT License
 *
 * Copyright (c) 2012-2022 Denis Demidov <dennis.demidov@gmail.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#pragma once

#include "amgcl-smoothed-prolongation.hpp"
#include <amgcl/backend/builtin.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/util.hpp>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace amgcl {
  namespace coarsening {

    /// @brief Throw unless w holds the B translations of n nodes: (n B) x B values, row-major, per node a diagonal
    /// B x B block of positive, finite weights (see translation_tentative_prolongation).
    /// @param w the translations
    /// @param n the number of nodes
    /// @param B the number of components per node
    /// @throws std::runtime_error naming the first offending entry -- an entry off a block's diagonal (e.g. a
    /// rotation), or a weight that is not positive and finite
    inline void check_translation_weights( const double* w, size_t n, int B )
    {
      for ( size_t i = 0; i < n; ++i ) {
        for ( int k = 0; k < B; ++k ) {
          for ( int l = 0; l < B; ++l ) {
            const double entry = w[( i * B + k ) * B + l];
            if ( l != k ? entry != 0.0 : !( entry > 0.0 && std::isfinite( entry ) ) ) {
              throw std::runtime_error( "block row " + std::to_string( i ) +
                                        " of the near null-space is not a diagonal block of positive, finite "
                                        "translation weights (entry " +
                                        std::to_string( k ) + "," + std::to_string( l ) + " = " +
                                        std::to_string( entry ) + "); only the translations are supported" );
            }
          }
        }
      }
    }

    /// @brief The tentative prolongation of a matrix of B x B blocks (one block row per node) whose near
    /// null-space are the B translations of its nodes.
    ///
    /// The translations are given per node i as a diagonal block diag(w_i0, ..., w_i,B-1): block row i of the
    /// row-major (n B) x B array translations.B, with zeros off the diagonal of every block. For an equilibrated
    /// block D A D they are D^-1 t, i.e. w_ik = 1 / s_ik with s the equilibration (``translationNullspace`` in
    /// edelweissfe/linsolve/nullspace.py).
    ///
    /// Column k of an aggregate's stacked blocks is nonzero only in the component-k rows, so the columns are
    /// orthogonal and the QR factorization of the stack is a normalization per component: node i of aggregate a
    /// gets the tentative block P_i = diag(w_ik / |w_k|_a), with |w_k|_a the 2-norm of component k over the nodes
    /// of the aggregate, and the coarse node a gets the translations R_a = diag(|w_k|_a), which replace
    /// translations.B for the next level (same layout). So P^T P = I on every aggregate, and P R reproduces the
    /// fine translations exactly. Every block stays a diagonal B x B block, whatever the level.
    ///
    /// A node in no aggregate (aggregate id < 0, e.g. one without strong connections) gets an empty row, as in
    /// AMGCL's own tentative_prolongation.
    ///
    /// @tparam Matrix the block CSR type of the hierarchy build (AMGCL's build_matrix)
    /// @param n the number of block rows (nodes)
    /// @param nAggregates the number of aggregates, i.e. of coarse nodes
    /// @param aggregateOfNode the aggregate of every node, < 0 for none
    /// @param translations in: the translations of the n nodes; out: those of the nAggregates coarse nodes
    /// @return the n x nAggregates block tentative prolongation
    /// @throws std::runtime_error on translations not of this form (a wrong size; an entry off a block's
    /// diagonal, e.g. a rotation; a weight that is not positive and finite), or on a component whose norm over an
    /// aggregate is not positive and finite (impossible for positive finite weights: a loud invariant, no
    /// fallback)
    template < class Matrix >
    std::shared_ptr< Matrix > translation_tentative_prolongation( size_t                          n,
                                                                  size_t                          nAggregates,
                                                                  const std::vector< ptrdiff_t >& aggregateOfNode,
                                                                  nullspace_params&               translations )
    {
      typedef typename backend::value_type< Matrix >::type value_type;
      typedef typename backend::col_type< Matrix >::type   col_type;
      const int                                            B = math::static_rows< value_type >::value;

      const std::vector< double >& w = translations.B;
      if ( translations.cols != B || w.size() != n * B * B ) {
        throw std::runtime_error( "translation_tentative_prolongation: the near null-space must be the " +
                                  std::to_string( B ) + " translations of all " + std::to_string( n ) + " nodes, " +
                                  std::to_string( n * B ) + " x " + std::to_string( B ) + ", got " +
                                  std::to_string( w.size() ) + " values in " + std::to_string( translations.cols ) +
                                  " columns" );
      }
      check_translation_weights( w.data(), n, B );
      // the weight of component k of node i: entry (k, k) of block row i
      auto weight = [&]( size_t i, int k ) { return w[( i * B + k ) * B + k]; };

      // per aggregate and component: the 2-norm of the stacked translation weights
      std::vector< double > norm( nAggregates * B, 0.0 );
      for ( size_t i = 0; i < n; ++i ) {
        if ( aggregateOfNode[i] < 0 )
          continue;
        for ( int k = 0; k < B; ++k )
          norm[aggregateOfNode[i] * B + k] += weight( i, k ) * weight( i, k );
      }
      for ( size_t a = 0; a < nAggregates * B; ++a ) {
        norm[a] = std::sqrt( norm[a] );
        if ( !( norm[a] > 0.0 && std::isfinite( norm[a] ) ) ) {
          throw std::runtime_error( "translation_tentative_prolongation: aggregate " + std::to_string( a / B ) +
                                    " has the translation norm " + std::to_string( norm[a] ) + " in component " +
                                    std::to_string( a % B ) );
        }
      }

      // one diagonal block per node, in its aggregate's column
      auto P = std::make_shared< Matrix >();
      P->set_size( n, nAggregates, true );
#pragma omp parallel for
      for ( ptrdiff_t i = 0; i < static_cast< ptrdiff_t >( n ); ++i )
        P->ptr[i + 1] = aggregateOfNode[i] >= 0 ? 1 : 0;
      P->set_nonzeros( P->scan_row_sizes() );
#pragma omp parallel for
      for ( ptrdiff_t i = 0; i < static_cast< ptrdiff_t >( n ); ++i ) {
        if ( aggregateOfNode[i] < 0 )
          continue;
        value_type block = math::zero< value_type >();
        for ( int k = 0; k < B; ++k )
          block( k, k ) = weight( i, k ) / norm[aggregateOfNode[i] * B + k];
        P->col[P->ptr[i]] = static_cast< col_type >( aggregateOfNode[i] );
        P->val[P->ptr[i]] = block;
      }

      // the coarse nodes' translations, R_a = diag(|w_k|_a), in the same layout
      std::vector< double > coarseTranslations( nAggregates * B * B, 0.0 );
      for ( size_t a = 0; a < nAggregates; ++a )
        for ( int k = 0; k < B; ++k )
          coarseTranslations[( a * B + k ) * B + k] = norm[a * B + k];
      translations.B.swap( coarseTranslations );
      return P;
    }

    /// @brief AMGCL's smoothed aggregation on a block backend, with the translations of the block as its near
    /// null-space instead of AMGCL's identity blocks.
    ///
    /// For a block value type AMGCL's own near null-space is the identity block: the constants of the variables
    /// the hierarchy is built on. For an equilibrated block D A D those are D-weighted translations, not its
    /// translations D^-1 t, and the two differ wherever the diagonal of A is not uniform. This coarsening builds
    /// the tentative prolongation from the translations given as params::nullspace instead
    /// (translation_tentative_prolongation). Everything else is smoothed_aggregation's: its parameters, the
    /// aggregation (AMGCL's block path, which keeps aggregates of every size), the prolongation smoothing and the
    /// Galerkin product.
    ///
    /// @tparam Backend the AMGCL backend of the hierarchy
    template < class Backend >
    struct smoothed_aggregation_with_translations : public smoothed_aggregation< Backend > {
      /// @brief AMGCL's smoothed aggregation: the parameters, the aggregates and the coarse operator.
      typedef smoothed_aggregation< Backend > Base;

      /// @brief Construct with smoothed_aggregation's parameters.
      /// @param prm the parameters; prm.nullspace must hold the translations of the finest level
      smoothed_aggregation_with_translations( const typename Base::params& prm = typename Base::params() ) : Base( prm )
      {
      }

      /// @brief The prolongation P = (I - omega diag_B(A_F)^-1 A_F) P_tent and the restriction R = P^T, formed as
      /// smoothed_aggregation::transfer_operators() forms them, with the tentative prolongation P_tent from the
      /// translations (which this replaces by the coarse level's).
      /// @tparam Matrix the block CSR type of the hierarchy build
      /// @param A the operator of the level
      /// @return P and R
      template < class Matrix >
      std::tuple< std::shared_ptr< Matrix >, std::shared_ptr< Matrix > > transfer_operators( const Matrix& A )
      {
        typedef typename backend::value_type< Matrix >::type value_type;
        typedef typename math::scalar_of< value_type >::type scalar_type;

        const size_t n = backend::rows( A );

        AMGCL_TIC( "aggregates" );
        // minimum aggregate size 0, as AMGCL's own block path without a near null-space: a single node already
        // carries all B translations
        typename Base::Aggregates aggr( A, this->prm.aggr, 0 );
        this->prm.aggr.eps_strong *= 0.5;
        AMGCL_TOC( "aggregates" );

        AMGCL_TIC( "tentative" );
        auto P_tent = translation_tentative_prolongation< Matrix >( n, aggr.count, aggr.id, this->prm.nullspace );
        AMGCL_TOC( "tentative" );

        // from here on smoothed_aggregation::transfer_operators(): P = (I - omega diag_B(A_F)^-1 A_F) P_tent
        const scalar_type omega = prolongation_smoothing_weight( A,
                                                                 this->prm.relax,
                                                                 this->prm.estimate_spectral_radius,
                                                                 this->prm.power_iters );
        auto              P     = smoothed_prolongation( A, aggr.strong_connection, *P_tent, omega );

        return std::make_tuple( P, transpose( *P ) );
      }
    };

  } // namespace coarsening
} // namespace amgcl
