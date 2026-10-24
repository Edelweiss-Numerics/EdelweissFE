/*
 * The rigid-body coarse level of a vector field on a block backend (EdelweissFE blockamg, key rigidBodyCoarseLevel):
 * the building blocks of a smoothed-aggregation coarse level whose coarse space is spanned by the rigid-body modes --
 * translations and rotations -- of every aggregate, under a fine level stored in node-wise B x B blocks.
 * RigidBodyCoarseLevelPreconditionerT (amgcl-wrapper.hpp) assembles them into a two-grid cycle.
 *
 * Why. The near null-space of linear elasticity -- the displacements of (almost) no strain energy -- are the rigid-body
 * modes: B translations and B (B - 1) / 2 rotations. A smoothed-aggregation coarse level represents a smooth error
 * well only where it is close to a combination of near-null-space vectors on every aggregate. AMGCL's hierarchy on a
 * block backend takes the translations alone (amgcl-block-translations.hpp): its coarse space holds displacements
 * that are constant on every aggregate. In a bending-dominated structure the smooth error is bending, which rotates
 * every aggregate almost as a rigid body; the coarse space cannot represent that, and the Chebyshev smoother does not
 * reduce it either (its spectral window starts at a fraction of the spectral radius by construction), so it converges
 * slowly. With the rotations in the coarse space, a rigid rotation of an aggregate is represented exactly.
 *
 * How. On the field's equilibrated block A = D K D (D = diag(scaling)):
 *   1. the aggregates: AMGCL's plain aggregation of the node-wise B x B blocks, as AMGCL's own first level on the
 *      block backend;
 *   2. the tentative prolongation (rigidBodyTentativeProlongation()): per aggregate, its rigid-body modes, orthonormal
 *      by a QR factorization; M = B (B + 1) / 2 coarse dofs per aggregate;
 *   3. the smoothed prolongation P = (I - omega D_F^-1 A_F) P_tent, AMGCL's (amgcl-smoothed-prolongation.hpp);
 *   4. the Galerkin coarse operator A_c = P^T A P (galerkinCoarseOperator()) in M x M blocks, and its near null-space,
 *      the coefficients of the field's rigid-body modes in the coarse basis: an AMGCL hierarchy continues below.
 */
#pragma once

#include "amgcl-smoothed-prolongation.hpp"
#include <amgcl/backend/builtin.hpp>
#include <amgcl/detail/sort_row.hpp>
#include <amgcl/value_type/static_matrix.hpp>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

/// @brief The rigid-body modes of a vector field of B components per node: the B translations and the B (B - 1) / 2
/// infinitesimal rotations, the displacements of zero strain energy of linear elasticity -- the same modes, in the
/// same order, as edelweissfe/linsolve/nullspace.py's rigidBodyNullspace().
/// @tparam B the number of components per node, 2 or 3
template < int B >
struct RigidBodyModes;

/// @brief The rigid-body modes in 3D: the translations t_x, t_y, t_z, and the rotations about the x, y and z axes,
/// r_x = (0, -z, y), r_y = (z, 0, -x), r_z = (-y, x, 0) -- the cross products e_axis x p of the axis with the position
/// p = (x, y, z) relative to the centre of rotation.
template <>
struct RigidBodyModes< 3 > {
  /// @brief The number of modes: three translations and three rotations.
  enum { count = 6 };

  /// @brief Component c of mode k at the position p relative to the centre of rotation.
  static double value( int k, int c, const double* p )
  {
    switch ( k ) {
    case 3:  // rotation about x: (0, -z, y)
      return c == 1 ? -p[2] : ( c == 2 ? p[1] : 0.0 );
    case 4:  // rotation about y: (z, 0, -x)
      return c == 0 ? p[2] : ( c == 2 ? -p[0] : 0.0 );
    case 5:  // rotation about z: (-y, x, 0)
      return c == 0 ? -p[1] : ( c == 1 ? p[0] : 0.0 );
    default: // translation along axis k
      return c == k ? 1.0 : 0.0;
    }
  }
};

/// @brief The rigid-body modes in 2D: the translations t_x, t_y, and the in-plane rotation r = (-y, x).
template <>
struct RigidBodyModes< 2 > {
  /// @brief The number of modes: two translations and one rotation.
  enum { count = 3 };

  /// @brief Component c of mode k at the position p relative to the centre of rotation.
  static double value( int k, int c, const double* p )
  {
    if ( k == 2 )              // the in-plane rotation: (-y, x)
      return c == 0 ? -p[1] : p[0];
    return c == k ? 1.0 : 0.0; // translation along axis k
  }
};

/// @brief The relative remainder at or below which rigidBodyTentativeProlongation() takes a mode of an aggregate as
/// linearly dependent on the modes before it: sqrt(machine epsilon), 1.5e-8.
///
/// Not a tuned value. An exactly dependent mode leaves a remainder at the rounding level of the Gram-Schmidt sweeps,
/// about 1e-16 of its norm; an independent one leaves a remainder of the order of the aggregate's own geometry (a
/// rotation about an axis across a thin aggregate: its thickness over its length, and no element is 1e8 times longer
/// than thick). Any threshold between the two separates them; half the digits of double precision is the
/// conventional one.
inline double dependentModeTolerance()
{
  return std::sqrt( std::numeric_limits< double >::epsilon() );
}

/// @brief The tentative prolongation of the rigid-body coarse level, and what the coarse level takes over from its
/// construction, see rigidBodyTentativeProlongation().
/// @tparam B the number of components per node, 2 or 3
template < int B >
struct RigidBodyTentativeProlongationT {
  /// @brief The coarse dofs per aggregate: one per rigid-body mode.
  enum { M = RigidBodyModes< B >::count };
  /// @brief A block of P: the B components of a node times the M coarse dofs of its aggregate.
  typedef amgcl::static_matrix< double, B, M > Block;
  /// @brief The block CSR type of P.
  typedef amgcl::backend::crs< Block, ptrdiff_t, ptrdiff_t > Matrix;

  /// @brief P_tent: one block per aggregated node, in the column of its aggregate; empty rows for the other nodes.
  std::shared_ptr< Matrix > P;
  /// @brief The near null-space of the coarse level: the coefficients of the field's M rigid-body modes in the coarse
  /// basis, (nAggregates M) x M, row-major -- per aggregate the M x M block R_a T_a.
  std::vector< double > coarseNullspace;
  /// @brief Per coarse dof, whether its mode is in the coarse space; one that is not has a zero column of P.
  std::vector< char > keptMode;
  /// @brief The number of modes dropped as linearly dependent.
  ptrdiff_t droppedModes;
};

/// @brief The tentative prolongation of the rigid-body coarse level: the rigid-body modes of every aggregate,
/// orthonormalized.
///
/// Per aggregate a, with the centroid c_a of the coordinates of its nodes:
///   * The modes V_a. Column k stacks mode k (RigidBodyModes) at x_i - c_a over the nodes i of the aggregate, divided
///     by the equilibration: the near null-space of the equilibrated block D K D is D^-1 times that of K, as in
///     rigidBodyNullspace(). A mode about the aggregate's own centroid, rather than about the field's, is not
///     dominated by the translation it would carry far from the centre, so the QR factorization below loses no digits
///     to cancellation.
///   * The Dirichlet rows. Their entries are zero: the cycle solves a Dirichlet row -- or the identity row of a
///     multi-point-constraint slave -- exactly, apart from the coarse-grid correction, and a rigid-body mode is
///     near-null on the free dofs only. So P^T A P = P_F^T A_FF P_F, and the coarse correction leaves those rows zero.
///   * The QR factorization V_a = Q_a R_a, by modified Gram-Schmidt with a second sweep, which makes Q_a orthonormal to
///     rounding also for nearly dependent columns. The rows of Q_a are the blocks of P_tent: the columns of P_tent are
///     orthonormal, and P_tent R_a reproduces the modes on the aggregate exactly.
///   * The dependent modes. A mode the Gram-Schmidt sweeps reduce to at most dependentModeTolerance() of its norm lies
///     in the span of the modes before it: a rotation of an aggregate of a single node (zero about its own centroid),
///     the rotation about the line through a collinear aggregate, a mode whose components are all constrained. Its
///     normalized remainder would be rounding noise, not a direction of the aggregate, so it is dropped: a zero column
///     of P_tent and a zero row of R_a. (AMGCL's own Householder QR keeps such a column in an aggregate of at least
///     as many dofs as modes: an orthonormal completion, unrelated to the modes.)
///   * The coarse near null-space. The field's rigid-body modes, about the field's centroid g, are V_a T_a on the
///     aggregate, with T_a the identity plus, in its translation rows, the rotations evaluated at c_a - g: a rotation
///     about g is the rotation about c_a plus a translation of the whole aggregate. So they are P_tent R_a T_a, and
///     their coefficients in the coarse basis are R_a T_a: the near null-space of the AMGCL hierarchy below (AMGCL's
///     own recursion takes R_a as the next near null-space in the same way).
///
/// A node in no aggregate (aggregate id < 0: no strong connection, e.g. a fully constrained one) gets an empty row, as
/// in AMGCL's tentative prolongation: only the smoother acts on it.
///
/// @tparam B the number of components per node, 2 or 3
/// @param nNodes the number of nodes, the block rows of the fine operator
/// @param nAggregates the number of aggregates
/// @param aggregateOfNode the aggregate of every node, < 0 for none
/// @param coordinates the node coordinates, nNodes x B, row-major
/// @param scaling the equilibration of every dof, nNodes B entries
/// @param isDirichlet per dof, whether its row is a Dirichlet (identity) row
/// @param modesInCoarseSpace the modes that span the coarse space, the first ones: all M, or the B translations
/// alone, which is AMGCL's own coarse space on the block backend (the control, see
/// RigidBodyCoarseLevelPreconditioner: the other modes are dropped)
/// @return P_tent, the coarse near null-space, the kept modes
template < int B >
RigidBodyTentativeProlongationT< B > rigidBodyTentativeProlongation( size_t                          nNodes,
                                                                     size_t                          nAggregates,
                                                                     const std::vector< ptrdiff_t >& aggregateOfNode,
                                                                     const double*                   coordinates,
                                                                     const double*                   scaling,
                                                                     const char*                     isDirichlet,
                                                                     int modesInCoarseSpace )
{
  typedef RigidBodyTentativeProlongationT< B > Result;
  const int                                    M = RigidBodyModes< B >::count;

  // the nodes of every aggregate, in node order: a counting sort over their aggregate ids
  std::vector< ptrdiff_t > aggregateStart( nAggregates + 1, 0 );
  for ( size_t i = 0; i < nNodes; ++i )
    if ( aggregateOfNode[i] >= 0 )
      ++aggregateStart[aggregateOfNode[i] + 1];
  std::partial_sum( aggregateStart.begin(), aggregateStart.end(), aggregateStart.begin() );
  std::vector< ptrdiff_t > members( aggregateStart[nAggregates] );
  std::vector< ptrdiff_t > next( aggregateStart.begin(), aggregateStart.end() - 1 );
  for ( size_t i = 0; i < nNodes; ++i )
    if ( aggregateOfNode[i] >= 0 )
      members[next[aggregateOfNode[i]]++] = static_cast< ptrdiff_t >( i );

  // the field's centroid: the centre of rotation of the field's rigid-body modes (as in rigidBodyNullspace())
  double fieldCentroid[B] = {};
  for ( size_t i = 0; i < nNodes; ++i )
    for ( int c = 0; c < B; ++c )
      fieldCentroid[c] += coordinates[B * i + c];
  for ( int c = 0; c < B; ++c )
    fieldCentroid[c] /= nNodes > 0 ? static_cast< double >( nNodes ) : 1.0;

  Result result;
  result.P = std::make_shared< typename Result::Matrix >();
  result.P->set_size( nNodes, nAggregates, true );
#pragma omp parallel for
  for ( ptrdiff_t i = 0; i < static_cast< ptrdiff_t >( nNodes ); ++i )
    result.P->ptr[i + 1] = aggregateOfNode[i] >= 0 ? 1 : 0;
  result.P->set_nonzeros( result.P->scan_row_sizes() );
  result.coarseNullspace.assign( nAggregates * M * M, 0.0 );
  result.keptMode.assign( nAggregates * M, 0 );

  const double tolerance = dependentModeTolerance();
  ptrdiff_t    dropped   = 0;
#pragma omp parallel reduction( + : dropped )
  {
    // the aggregate's modes, column-major: entry (row, k) at V[row + d k], row = B q + c for component c of its
    // node q, d = B m rows for its m nodes
    std::vector< double > V;

#pragma omp for schedule( dynamic, 16 )
    for ( ptrdiff_t a = 0; a < static_cast< ptrdiff_t >( nAggregates ); ++a ) {
      const ptrdiff_t first = aggregateStart[a], m = aggregateStart[a + 1] - first, d = B * m;

      double centroid[B] = {};
      for ( ptrdiff_t q = 0; q < m; ++q )
        for ( int c = 0; c < B; ++c )
          centroid[c] += coordinates[B * members[first + q] + c];
      for ( int c = 0; c < B; ++c )
        centroid[c] /= static_cast< double >( m );

      // the modes about the centroid, divided by the equilibration, zero on the Dirichlet rows
      V.assign( d * M, 0.0 );
      for ( ptrdiff_t q = 0; q < m; ++q ) {
        const ptrdiff_t node = members[first + q];
        double          position[B];
        for ( int c = 0; c < B; ++c )
          position[c] = coordinates[B * node + c] - centroid[c];
        for ( int c = 0; c < B; ++c ) {
          const ptrdiff_t dof = B * node + c;
          if ( isDirichlet[dof] )
            continue;
          for ( int k = 0; k < M; ++k )
            V[B * q + c + d * k] = RigidBodyModes< B >::value( k, c, position ) / scaling[dof];
        }
      }

      // V = Q R by modified Gram-Schmidt with a second sweep; the columns of V become those of Q
      double R[M * M] = {};
      char   kept[M]  = {};
      for ( int k = 0; k < M; ++k ) {
        double* v           = &V[d * k];
        double  normSquared = 0.0;
        for ( ptrdiff_t r = 0; r < d; ++r )
          normSquared += v[r] * v[r];
        const double norm = std::sqrt( normSquared );

        for ( int sweep = 0; sweep < 2; ++sweep ) {
          for ( int j = 0; j < k; ++j ) {
            if ( !kept[j] )
              continue;
            const double* qj         = &V[d * j];
            double        projection = 0.0;
            for ( ptrdiff_t r = 0; r < d; ++r )
              projection += qj[r] * v[r];
            for ( ptrdiff_t r = 0; r < d; ++r )
              v[r] -= projection * qj[r];
            R[M * j + k] += projection;
          }
        }

        double remainderSquared = 0.0;
        for ( ptrdiff_t r = 0; r < d; ++r )
          remainderSquared += v[r] * v[r];
        const double remainder = std::sqrt( remainderSquared );

        if ( k < modesInCoarseSpace && norm > 0.0 && remainder > tolerance * norm ) {
          kept[k]          = 1;
          R[M * k + k]     = remainder;
          const double inv = 1.0 / remainder;
          for ( ptrdiff_t r = 0; r < d; ++r )
            v[r] *= inv;
        }
        else {
          // dropped: linearly dependent on the modes before it (or not in the coarse space)
          if ( k < modesInCoarseSpace )
            ++dropped;
          for ( ptrdiff_t r = 0; r < d; ++r )
            v[r] = 0.0;
          for ( int j = 0; j < M; ++j )
            R[M * k + j] = 0.0;
        }
      }

      // the blocks of P_tent: the rows of Q
      for ( ptrdiff_t q = 0; q < m; ++q ) {
        typename Result::Block block;
        for ( int c = 0; c < B; ++c )
          for ( int k = 0; k < M; ++k )
            block( c, k ) = V[B * q + c + d * k];
        const ptrdiff_t node               = members[first + q];
        result.P->col[result.P->ptr[node]] = a;
        result.P->val[result.P->ptr[node]] = block;
      }

      // the coarse near null-space R T: T is the identity plus the rotations at c_a - g in the translation rows
      double T[M * M] = {};
      double offset[B];
      for ( int c = 0; c < B; ++c )
        offset[c] = centroid[c] - fieldCentroid[c];
      for ( int k = 0; k < M; ++k )
        T[M * k + k] = 1.0;
      for ( int k = B; k < M; ++k )
        for ( int c = 0; c < B; ++c )
          T[M * c + k] = RigidBodyModes< B >::value( k, c, offset );
      for ( int i = 0; i < M; ++i ) {
        for ( int j = 0; j < M; ++j ) {
          double sum = 0.0;
          for ( int k = 0; k < M; ++k )
            sum += R[M * i + k] * T[M * k + j];
          result.coarseNullspace[( M * a + i ) * M + j] = sum;
        }
        result.keptMode[M * a + i] = kept[i];
      }
    }
  }
  result.droppedModes = dropped;
  return result;
}

/// @brief The transpose of a sparse matrix of N x K blocks: K x N blocks, each block transposed (AMGCL's
/// backend::transpose keeps the block type, so it transposes square blocks only). Every row of the transpose has its
/// columns in increasing order.
/// @tparam Matrix the block CSR type
/// @param A the matrix
/// @return A^T
template < class Matrix >
std::shared_ptr< amgcl::backend::crs<
  typename amgcl::math::adjoint_impl< typename amgcl::backend::value_type< Matrix >::type >::return_type,
  ptrdiff_t,
  ptrdiff_t > >
transposedBlocks( const Matrix& A )
{
  typedef typename amgcl::backend::value_type< Matrix >::type          Block;
  typedef typename amgcl::math::adjoint_impl< Block >::return_type     TransposedBlock;
  typedef amgcl::backend::crs< TransposedBlock, ptrdiff_t, ptrdiff_t > Transposed;

  auto T = std::make_shared< Transposed >();
  T->set_size( A.ncols, A.nrows, true );
  for ( ptrdiff_t j = 0; j < static_cast< ptrdiff_t >( A.ptr[A.nrows] ); ++j )
    ++T->ptr[A.col[j] + 1];
  T->set_nonzeros( T->scan_row_sizes() );

  std::vector< ptrdiff_t > head( T->ptr, T->ptr + T->nrows );
  for ( ptrdiff_t i = 0; i < static_cast< ptrdiff_t >( A.nrows ); ++i ) {
    for ( ptrdiff_t j = A.ptr[i], e = A.ptr[i + 1]; j < e; ++j ) {
      const ptrdiff_t position = head[A.col[j]]++;
      T->col[position]         = i;
      T->val[position]         = amgcl::math::adjoint( A.val[j] );
    }
  }
  return T;
}

/// @brief The sparse product C = A B of block matrices whose block shapes chain -- an N x K block of A times a K x L
/// block of B gives an N x L block of C -- row by row (Gustavson's algorithm): each row of C is gathered with a marker
/// over the columns of B, in two OpenMP-parallel passes, the row widths and then the entries. AMGCL's own product
/// (backend::product) takes one block type for A, B and C, which the Galerkin product of the rigid-body coarse level
/// (B x B, B x M and M x B blocks) does not have.
/// @tparam CBlock the block type of C
/// @tparam AMatrix, BMatrix the block CSR types of A and B
/// @param A, B the factors
/// @return C, the columns of each row in the order they are first reached (not sorted)
template < class CBlock, class AMatrix, class BMatrix >
std::shared_ptr< amgcl::backend::crs< CBlock, ptrdiff_t, ptrdiff_t > > blockProduct( const AMatrix& A,
                                                                                     const BMatrix& B )
{
  auto C = std::make_shared< amgcl::backend::crs< CBlock, ptrdiff_t, ptrdiff_t > >();
  C->set_size( A.nrows, B.ncols, true );
  const ptrdiff_t nRows = static_cast< ptrdiff_t >( A.nrows );

#pragma omp parallel
  {
    std::vector< ptrdiff_t > marker( B.ncols, -1 );
#pragma omp for schedule( dynamic, 64 )
    for ( ptrdiff_t i = 0; i < nRows; ++i ) {
      ptrdiff_t width = 0;
      for ( ptrdiff_t ja = A.ptr[i], ea = A.ptr[i + 1]; ja < ea; ++ja ) {
        for ( ptrdiff_t jb = B.ptr[A.col[ja]], eb = B.ptr[A.col[ja] + 1]; jb < eb; ++jb ) {
          if ( marker[B.col[jb]] != i ) {
            marker[B.col[jb]] = i;
            ++width;
          }
        }
      }
      C->ptr[i + 1] = width;
    }
  }
  C->set_nonzeros( C->scan_row_sizes() );

#pragma omp parallel
  {
    // marker[column]: the position of the column in the row being gathered, or -1 if the row has not reached it. The
    // columns a row reached are reset after it, so that every row starts from a clean marker whatever order the
    // schedule hands the rows to this thread in. (AMGCL's own products test the marker against the row's first
    // position instead, which holds only if each thread visits its rows in increasing order: true of the static
    // schedule AMGCL's loops run on, not of a dynamic one, which is nonmonotonic since OpenMP 5.0 and is reordered by
    // libomp's work stealing.)
    std::vector< ptrdiff_t > marker( B.ncols, -1 );
#pragma omp for schedule( dynamic, 64 )
    for ( ptrdiff_t i = 0; i < nRows; ++i ) {
      const ptrdiff_t rowBegin = C->ptr[i];
      ptrdiff_t       rowEnd   = rowBegin;
      for ( ptrdiff_t ja = A.ptr[i], ea = A.ptr[i + 1]; ja < ea; ++ja ) {
        const auto& a = A.val[ja];
        for ( ptrdiff_t jb = B.ptr[A.col[ja]], eb = B.ptr[A.col[ja] + 1]; jb < eb; ++jb ) {
          const ptrdiff_t column = B.col[jb];
          if ( marker[column] < 0 ) {
            marker[column] = rowEnd;
            C->col[rowEnd] = column;
            C->val[rowEnd] = a * B.val[jb];
            ++rowEnd;
          }
          else {
            C->val[marker[column]] += a * B.val[jb];
          }
        }
      }
      for ( ptrdiff_t j = rowBegin; j < rowEnd; ++j )
        marker[C->col[j]] = -1;
    }
  }
  return C;
}

/// @brief The Galerkin coarse operator A_c = R (A P), R = P^T, in the order AMGCL forms it
/// (amgcl/coarsening/detail/galerkin.hpp), with every row sorted by column: the aggregation of the AMGCL hierarchy
/// below reads the columns in order.
/// @tparam CoarseBlock the block type of A_c, M x M
/// @param A the fine operator, B x B blocks
/// @param P the prolongation, B x M blocks
/// @param R the restriction P^T, M x B blocks
/// @return A_c
template < class CoarseBlock, class Matrix, class PMatrix, class RMatrix >
std::shared_ptr< amgcl::backend::crs< CoarseBlock, ptrdiff_t, ptrdiff_t > > galerkinCoarseOperator( const Matrix&  A,
                                                                                                    const PMatrix& P,
                                                                                                    const RMatrix& R )
{
  typedef typename amgcl::backend::value_type< PMatrix >::type PBlock;

  auto Ac = blockProduct< CoarseBlock >( R, *blockProduct< PBlock >( A, P ) );
#pragma omp parallel for
  for ( ptrdiff_t i = 0; i < static_cast< ptrdiff_t >( Ac->nrows ); ++i )
    amgcl::detail::sort_row( Ac->col + Ac->ptr[i],
                             Ac->val + Ac->ptr[i],
                             static_cast< int >( Ac->ptr[i + 1] - Ac->ptr[i] ) );
  return Ac;
}

/// @brief The Galerkin coarse operator A_c (M x M blocks) as the scalar CSR matrix that the AMGCL hierarchy below the
/// rigid-body coarse level is built on, without its exact zeros.
///
/// The coarse dof of a dropped mode has a zero column of P, hence a zero row and column in A_c. It gets the diagonal
/// entry padding and nothing else, which keeps A_c regular (positive definite where A is) for the hierarchy below.
/// For the two-grid cycle the padding is exact, whatever its value: P^T maps every residual to zero there, and the
/// zero column of P never prolongates what the hierarchy below computes there. Within the hierarchy below the padded
/// dof is not entirely inert: where an aggregate of the next level is rank-deficient (it holds dropped modes), AMGCL's
/// QR completes the orthonormal basis of that aggregate with directions that may include padded dofs, which then take
/// a share of that level's coarse space -- a cost in efficiency only, the operator stays positive definite. The
/// padding, the mean of the diagonal entries of the kept modes, lies inside the spectrum of A_c (each diagonal entry
/// is a Rayleigh quotient), so it does not move the spectral-radius estimate of the hierarchy's smoother.
/// @tparam CoarseMatrix the block CSR type of A_c
/// @param Ac A_c, every row sorted by column
/// @param keptMode per coarse dof, whether its mode is in the coarse space
/// @return the scalar A_c, M rows per block row, every row sorted by column
/// @throws std::overflow_error if it does not fit 32-bit indices
template < class CoarseMatrix >
std::shared_ptr< amgcl::backend::crs< double, int, int > > paddedScalarCoarseOperator(
  const CoarseMatrix&        Ac,
  const std::vector< char >& keptMode )
{
  typedef typename amgcl::backend::value_type< CoarseMatrix >::type CoarseBlock;
  const int                                                         M = amgcl::math::static_rows< CoarseBlock >::value;
  const ptrdiff_t nBlockRows = static_cast< ptrdiff_t >( Ac.nrows ), n = M * nBlockRows;

  double    keptDiagonalSum = 0.0;
  ptrdiff_t nKept           = 0;
  for ( ptrdiff_t a = 0; a < nBlockRows; ++a )
    for ( ptrdiff_t j = Ac.ptr[a]; j < Ac.ptr[a + 1]; ++j )
      if ( Ac.col[j] == a )
        for ( int k = 0; k < M; ++k )
          if ( keptMode[M * a + k] ) {
            keptDiagonalSum += Ac.val[j]( k, k );
            ++nKept;
          }
  const double padding = nKept > 0 ? keptDiagonalSum / static_cast< double >( nKept ) : 1.0;

  // an entry of the scalar row (a, k) is kept if it is not zero, and the diagonal always
  auto isKept = [&]( ptrdiff_t a, int k, ptrdiff_t b, int l, double value ) {
    return value != 0.0 || ( b == a && l == k );
  };

  std::vector< ptrdiff_t > width( n + 1, 0 );
#pragma omp parallel for
  for ( ptrdiff_t a = 0; a < nBlockRows; ++a )
    for ( int k = 0; k < M; ++k )
      for ( ptrdiff_t j = Ac.ptr[a]; j < Ac.ptr[a + 1]; ++j )
        for ( int l = 0; l < M; ++l )
          if ( isKept( a, k, Ac.col[j], l, Ac.val[j]( k, l ) ) )
            ++width[M * a + k + 1];
  std::partial_sum( width.begin(), width.end(), width.begin() );
  if ( width[n] > static_cast< ptrdiff_t >( std::numeric_limits< int >::max() ) ) {
    throw std::overflow_error( "rigid-body coarse level: the Galerkin coarse operator has " +
                               std::to_string( width[n] ) + " entries, more than 32-bit indices hold" );
  }

  auto S = std::make_shared< amgcl::backend::crs< double, int, int > >();
  S->set_size( n, n, false );
  for ( ptrdiff_t i = 0; i <= n; ++i )
    S->ptr[i] = static_cast< int >( width[i] );
  S->set_nonzeros( S->ptr[n] );
#pragma omp parallel for
  for ( ptrdiff_t a = 0; a < nBlockRows; ++a ) {
    for ( int k = 0; k < M; ++k ) {
      int position = S->ptr[M * a + k];
      for ( ptrdiff_t j = Ac.ptr[a]; j < Ac.ptr[a + 1]; ++j ) {
        for ( int l = 0; l < M; ++l ) {
          const double value = Ac.val[j]( k, l );
          if ( !isKept( a, k, Ac.col[j], l, value ) )
            continue;
          const bool droppedDiagonal = Ac.col[j] == a && l == k && !keptMode[M * a + k];
          S->col[position]           = static_cast< int >( M * Ac.col[j] + l );
          S->val[position]           = droppedDiagonal ? padding : value;
          ++position;
        }
      }
    }
  }
  return S;
}

/// @brief y <- A x, or y <- y + A x with addToY, for a matrix A of N x K blocks, stored in any scalar type (double,
/// or float for half the bytes), and plain double vectors, every row summed in double: the restriction (M x B blocks)
/// and the prolongation (B x M blocks) of the rigid-body coarse level. AMGCL's own spmv takes square blocks only (it
/// views both vectors in blocks of the block's row count).
/// @tparam Matrix the block CSR type
/// @param A the matrix
/// @param x the vector A acts on, K entries per block column
/// @param y the result, N entries per block row
/// @param addToY add A x to y instead of overwriting y
template < class Matrix >
void blockMatrixVectorProduct( const Matrix& A, const double* x, double* y, bool addToY )
{
  typedef typename amgcl::backend::value_type< Matrix >::type Block;
  const int                                                   N     = amgcl::math::static_rows< Block >::value;
  const int                                                   K     = amgcl::math::static_cols< Block >::value;
  const ptrdiff_t                                             nRows = static_cast< ptrdiff_t >( A.nrows );

#pragma omp parallel for schedule( dynamic, 64 )
  for ( ptrdiff_t i = 0; i < nRows; ++i ) {
    double sum[N] = {};
    for ( ptrdiff_t j = A.ptr[i], e = A.ptr[i + 1]; j < e; ++j ) {
      const Block&  block = A.val[j];
      const double* xj    = x + static_cast< ptrdiff_t >( K ) * A.col[j];
      for ( int r = 0; r < N; ++r )
        for ( int c = 0; c < K; ++c )
          sum[r] += block( r, c ) * xj[c];
    }
    double* yi = y + static_cast< ptrdiff_t >( N ) * i;
    for ( int r = 0; r < N; ++r )
      yi[r] = addToY ? yi[r] + sum[r] : sum[r];
  }
}
