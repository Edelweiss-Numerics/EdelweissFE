/*
 * The prolongation smoothing of AMGCL's smoothed aggregation, for a tentative prolongation of any block shape
 * (EdelweissFE): shared by the translations of a block backend (amgcl-block-translations.hpp) and the
 * rigid-body coarse level (amgcl-rigid-body-coarse-level.hpp).
 *
 * prolongation_smoothing_weight() and smoothed_prolongation() are the prolongation smoothing of
 * smoothed_aggregation::transfer_operators() in amgcl/coarsening/smoothed_aggregation.hpp (AMGCL 1.4.x), taken out of
 * it so that the tentative prolongation may have other blocks than the operator: B x M blocks for M near-null-space
 * vectors per aggregate on B x B blocks. The license of AMGCL, which that code is taken from, follows verbatim.
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

#include <amgcl/backend/builtin.hpp>
#include <amgcl/util.hpp>
#include <memory>
#include <vector>

namespace amgcl {
  namespace coarsening {

    /// @brief The damping weight omega of the prolongation smoothing P = (I - omega diag_B(A_F)^-1 A_F) P_tent (see
    /// smoothed_prolongation()), chosen as AMGCL's smoothed aggregation chooses it.
    ///
    /// One damped block-Jacobi step reduces the error components with the largest eigenvalues of D^-1 A most: with
    /// omega = 4/3 / rho(D^-1 A), the component at the spectral radius rho is multiplied by 1 - 4/3 = -1/3. Without the
    /// estimate, AMGCL takes omega = 2/3, which is the same for rho = 2.
    ///
    /// @tparam Matrix the block CSR type of the operator
    /// @param A the operator of the level
    /// @param relax a factor on omega (AMGCL's coarsening parameter relax, 1 by default)
    /// @param estimateSpectralRadius estimate rho(D^-1 A) by power iteration (AMGCL's estimate_spectral_radius), with
    /// D the diagonal blocks of A; else take omega = relax * 2/3
    /// @param powerIters the power iterations of the estimate (AMGCL's power_iters; 0 bounds rho by Gershgorin's
    /// theorem instead)
    /// @return omega
    template < class Matrix >
    typename math::scalar_of< typename backend::value_type< Matrix >::type >::type prolongation_smoothing_weight(
      const Matrix& A,
      float         relax,
      bool          estimateSpectralRadius,
      int           powerIters )
    {
      typedef typename math::scalar_of< typename backend::value_type< Matrix >::type >::type scalar_type;

      scalar_type omega = relax;
      if ( estimateSpectralRadius ) {
        omega *= static_cast< scalar_type >( 4.0 / 3 ) / backend::spectral_radius< true >( A, powerIters );
      }
      else {
        omega *= static_cast< scalar_type >( 2.0 / 3 );
      }
      return omega;
    }

    /// @brief The smoothed prolongation P = (I - omega diag_B(A_F)^-1 A_F) P_tent of smoothed aggregation: one damped
    /// block-Jacobi step applied to every column of the tentative prolongation, diag_B(A_F) the node-wise B x B
    /// diagonal blocks of A_F. (A is the level's operator: the diagonally scaled \hat A = D A D on the first level of
    /// a field's hierarchy.)
    ///
    /// A column of P_tent is a near-null-space vector cut off at the boundary of its aggregate, so it jumps there, and
    /// its energy is large. The Jacobi step smooths the jump away and widens the column by one layer of nodes, so that
    /// the coarse basis functions overlap: the coarse space then holds smooth error with a far smaller energy, which is
    /// what the coarse-grid correction is for.
    ///
    /// The step is taken with the filtered operator A_F: its off-diagonal blocks are those of the strong connections
    /// only (from the aggregation), and the blocks of the weak ones are lumped onto the diagonal block,
    /// diag_B(A_F)_ii = a_ii + sum of the weak a_ij of the row, so that A_F keeps the row sums of A. Filtering keeps P
    /// as sparse as the aggregates are connected, instead of as wide as every (also numerically tiny) entry of A
    /// reaches.
    ///
    /// @tparam Matrix the block CSR type of the operator, B x B blocks
    /// @tparam Prolongation the block CSR type of the tentative prolongation, B x M blocks (M = B for the
    /// translations, M = 6 for the rigid-body modes of a 3D field)
    /// @param A the operator of the level
    /// @param strongConnection per stored block of A, whether it is a strong connection (AMGCL's aggregates)
    /// @param P_tent the tentative prolongation
    /// @param omega the damping weight, see prolongation_smoothing_weight()
    /// @return P, the same block type and shape as P_tent, its rows in the order the aggregates are first reached (not
    /// sorted)
    template < class Matrix, class Prolongation >
    std::shared_ptr< Prolongation > smoothed_prolongation(
      const Matrix&                                                                  A,
      const std::vector< char >&                                                     strongConnection,
      const Prolongation&                                                            P_tent,
      typename math::scalar_of< typename backend::value_type< Matrix >::type >::type omega )
    {
      typedef typename backend::value_type< Matrix >::type       value_type;
      typedef typename backend::value_type< Prolongation >::type prolongation_value_type;
      typedef typename math::scalar_of< value_type >::type       scalar_type;

      const size_t n = backend::rows( A );

      auto P = std::make_shared< Prolongation >();
      P->set_size( backend::rows( P_tent ), backend::cols( P_tent ), true );

      AMGCL_TIC( "smoothing" );
#pragma omp parallel
      {
        std::vector< ptrdiff_t > marker( P->ncols, -1 );

        // the number of entries of every row of P
#pragma omp for
        for ( ptrdiff_t i = 0; i < static_cast< ptrdiff_t >( n ); ++i ) {
          for ( ptrdiff_t ja = A.ptr[i], ea = A.ptr[i + 1]; ja < ea; ++ja ) {
            ptrdiff_t ca = A.col[ja];
            // weak off-diagonal connections are filtered out
            if ( ca != i && !strongConnection[ja] )
              continue;
            for ( ptrdiff_t jp = P_tent.ptr[ca], ep = P_tent.ptr[ca + 1]; jp < ep; ++jp ) {
              ptrdiff_t cp = P_tent.col[jp];
              if ( marker[cp] != i ) {
                marker[cp] = i;
                ++( P->ptr[i + 1] );
              }
            }
          }
        }
      }

      P->scan_row_sizes();
      P->set_nonzeros();

#pragma omp parallel
      {
        std::vector< ptrdiff_t > marker( P->ncols, -1 );

#pragma omp for
        for ( ptrdiff_t i = 0; i < static_cast< ptrdiff_t >( n ); ++i ) {
          // the diagonal block of the filtered matrix: the diagonal block plus the weak connections, lumped onto it
          value_type dia = math::zero< value_type >();
          for ( ptrdiff_t j = A.ptr[i], e = A.ptr[i + 1]; j < e; ++j ) {
            if ( A.col[j] == i || !strongConnection[j] )
              dia += A.val[j];
          }
          if ( !math::is_zero( dia ) )
            dia = -omega * math::inverse( dia );

          ptrdiff_t row_beg = P->ptr[i];
          ptrdiff_t row_end = row_beg;
          for ( ptrdiff_t ja = A.ptr[i], ea = A.ptr[i + 1]; ja < ea; ++ja ) {
            ptrdiff_t ca = A.col[ja];
            if ( ca != i && !strongConnection[ja] )
              continue;

            value_type va = ( ca == i ) ? static_cast< value_type >( static_cast< scalar_type >( 1 - omega ) *
                                                                     math::identity< value_type >() )
                                        : dia * A.val[ja];

            for ( ptrdiff_t jp = P_tent.ptr[ca], ep = P_tent.ptr[ca + 1]; jp < ep; ++jp ) {
              ptrdiff_t                      cp = P_tent.col[jp];
              const prolongation_value_type& vp = P_tent.val[jp];
              if ( marker[cp] < row_beg ) {
                marker[cp]      = row_end;
                P->col[row_end] = cp;
                P->val[row_end] = va * vp;
                ++row_end;
              }
              else {
                P->val[marker[cp]] += va * vp;
              }
            }
          }
        }
      }
      AMGCL_TOC( "smoothing" );

      return P;
    }

  } // namespace coarsening
} // namespace amgcl
