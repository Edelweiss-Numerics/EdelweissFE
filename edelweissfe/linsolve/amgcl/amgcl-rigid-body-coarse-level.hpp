/*
 * The rigid-body coarse level of a vector field on a block backend (EdelweissFE,
 * TwoLevel(coarseSpace=RigidBodyModes())): the building blocks of a smoothed-aggregation coarse level whose coarse
 * space is spanned, on every aggregate, by a near null-space of M vectors -- the rigid-body modes, translations and
 * rotations -- under a fine level stored in node-wise B x B blocks. RigidBodyCoarseLevelPreconditionerT
 * (amgcl-wrapper.hpp) assembles them into a two-level cycle.
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
 * How. On the field's equilibrated block \hat A = D A D (D = diag(scaling)), with the near null-space B of \hat A --
 * n x M, the field's rigid-body modes divided by D, which Python builds (nullspace.py's rigidBodyNullspace(), the one
 * definition of the modes) -- the steps, one function each:
 *   1. the aggregates: AMGCL's plain aggregation of the node-wise B x B blocks, as AMGCL's own first level on the
 *      block backend (nodesOfAggregates() lists the nodes of each);
 *   2. the tentative prolongation (tentativeProlongation()), as AMGCL's tentative_prolongation does it: per aggregate,
 *      the rows of B on its dofs (aggregateModes()), orthonormalized by a QR factorization (orthonormalizeModes()),
 *      the rows of Q the blocks of P_tent (writeProlongationBlocks()); M coarse dofs per aggregate;
 *   3. the dependent modes, dropped by that QR factorization;
 *   4. the smoothed prolongation P = (I - omega diag_B(\hat A_F)^-1 \hat A_F) P_tent, AMGCL's
 *      (amgcl-smoothed-prolongation.hpp), diag_B the node-wise B x B diagonal blocks;
 *   5. the Galerkin coarse operator A_c = P^T \hat A P (galerkinCoarseOperator()) in M x M blocks, handed to AMGCL as
 *      a scalar matrix (unblockedCoarseOperator());
 *   6. the AMGCL hierarchy below, with the R factors of the QR factorizations as its near null-space (AMGCL's
 *      convention).
 * Steps 2 to 5 run outside AMGCL because its own building blocks do not fit: AMGCL's coarsening on a block backend
 * takes the translations alone, and its as_scalar route re-blocks the prolongation into B x B blocks (3 modes do not
 * fit 2 x 2 blocks), keeps an orthonormal completion of dependent modes in its Householder QR, and leaves the modes on
 * the Dirichlet rows; its sparse products take one block type, where the Galerkin product chains B x B, B x M and
 * M x B blocks.
 */
#pragma once

#include "amgcl-smoothed-prolongation.hpp"
#include <algorithm>
#include <amgcl/adapter/block_matrix.hpp>
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

/// @brief The relative remainder at or below which orthonormalizeModes() takes a mode of an aggregate as linearly
/// dependent on the modes before it: sqrt(machine epsilon), 1.5e-8.
///
/// Not a tuned value. An exactly dependent mode leaves a remainder at the rounding level of the Gram-Schmidt sweeps,
/// about 1e-16 of its norm; an independent one leaves a remainder of the order of the aggregate's own geometry relative
/// to the mode: a rotation about the field's centroid g carries a translation of the order of |c_a - g| across an
/// aggregate of extent h_a around c_a, and leaves a remainder of the order of h_a once the translations are projected
/// out, h_a / |c_a - g| of its norm -- 1e-2 to 1e-4 on a mesh of elements a hundred to ten thousand times smaller than
/// the structure, so a rotation does not need to be centred on the aggregate to be kept. Any threshold between the two
/// separates them; half the digits of double precision is the conventional one.
inline double dependentModeTolerance()
{
  return std::sqrt( std::numeric_limits< double >::epsilon() );
}

/// @brief The nodes of every aggregate, in node order: those of aggregate a are nodes[start[a]], ...,
/// nodes[start[a + 1] - 1].
struct AggregateMembers {
  /// @brief Where the nodes of each aggregate begin in `nodes`, nAggregates + 1 entries.
  std::vector< ptrdiff_t > start;
  /// @brief The aggregated nodes, aggregate by aggregate.
  std::vector< ptrdiff_t > nodes;

  /// @brief The number of nodes of aggregate a.
  ptrdiff_t size( ptrdiff_t a ) const { return start[a + 1] - start[a]; }
  /// @brief The q-th node of aggregate a.
  ptrdiff_t node( ptrdiff_t a, ptrdiff_t q ) const { return nodes[start[a] + q]; }
};

/// @brief The nodes of every aggregate: a counting sort over their aggregate ids (a node of id < 0 is in none).
/// @param aggregateOfNode the aggregate of every node
/// @param nAggregates the number of aggregates
/// @return the nodes of each aggregate
inline AggregateMembers nodesOfAggregates( const std::vector< ptrdiff_t >& aggregateOfNode, size_t nAggregates )
{
  AggregateMembers members;
  members.start.assign( nAggregates + 1, 0 );
  for ( ptrdiff_t aggregate : aggregateOfNode )
    if ( aggregate >= 0 )
      ++members.start[aggregate + 1];
  std::partial_sum( members.start.begin(), members.start.end(), members.start.begin() );
  members.nodes.resize( members.start[nAggregates] );
  std::vector< ptrdiff_t > next( members.start.begin(), members.start.end() - 1 );
  for ( size_t i = 0; i < aggregateOfNode.size(); ++i )
    if ( aggregateOfNode[i] >= 0 )
      members.nodes[next[aggregateOfNode[i]]++] = static_cast< ptrdiff_t >( i );
  return members;
}

/// @brief The M modes of one aggregate as the columns of a d x M matrix -- column k mode k on the aggregate's
/// d = B m dofs, row B q + c the component c of its q-th node -- with the column operations of the Gram-Schmidt sweeps
/// of orthonormalizeModes().
/// @tparam M the number of modes
template < int M >
class ModeColumns {
public:
  /// @brief Make it d x M, zero.
  void reset( ptrdiff_t d )
  {
    d_ = d;
    values_.assign( static_cast< size_t >( d ) * M, 0.0 );
  }

  /// @brief The rows d.
  ptrdiff_t rows() const { return d_; }

  /// @brief The entry in row `row` of column k.
  double& operator()( ptrdiff_t row, int k ) { return values_[row + d_ * k]; }
  /// @copydoc operator()(ptrdiff_t,int)
  double operator()( ptrdiff_t row, int k ) const { return values_[row + d_ * k]; }

  /// @brief The scalar product of the columns j and k.
  double dot( int j, int k ) const
  {
    const double* vj  = column( j );
    const double* vk  = column( k );
    double        sum = 0.0;
    for ( ptrdiff_t r = 0; r < d_; ++r )
      sum += vj[r] * vk[r];
    return sum;
  }

  /// @brief The norm of column k.
  double norm( int k ) const { return std::sqrt( dot( k, k ) ); }

  /// @brief column k <- column k - factor column j
  void subtract( int k, double factor, int j )
  {
    double*       vk = column( k );
    const double* vj = column( j );
    for ( ptrdiff_t r = 0; r < d_; ++r )
      vk[r] -= factor * vj[r];
  }

  /// @brief column k <- factor column k
  void scale( int k, double factor )
  {
    double* vk = column( k );
    for ( ptrdiff_t r = 0; r < d_; ++r )
      vk[r] *= factor;
  }

  /// @brief column k <- 0
  void zero( int k ) { std::fill( column( k ), column( k ) + d_, 0.0 ); }

private:
  ptrdiff_t             d_ = 0;
  std::vector< double > values_;

  double*       column( int k ) { return values_.data() + d_ * k; }
  const double* column( int k ) const { return values_.data() + d_ * k; }
};

/// @brief 2. The modes V_a of aggregate a: the rows of the near null-space on its dofs, zero on the Dirichlet rows.
///
/// The Dirichlet rows are zero because the cycle solves a Dirichlet row -- or the identity row of a
/// multi-point-constraint slave -- exactly, apart from the coarse-grid correction, and a rigid-body mode is near-null
/// on the free dofs only. So P^T \hat A P = P_F^T \hat A_FF P_F, and the coarse correction leaves those rows zero.
/// @tparam B the number of components per node
/// @tparam M the number of modes, the columns of the near null-space
/// @param members the nodes of the aggregates
/// @param a the aggregate
/// @param nullspace the near null-space, n x M, row-major
/// @param isDirichlet per dof, whether its row is a Dirichlet row
/// @param V the modes of the aggregate, set here
template < int B, int M >
void aggregateModes( const AggregateMembers& members,
                     ptrdiff_t               a,
                     const double*           nullspace,
                     const char*             isDirichlet,
                     ModeColumns< M >&       V )
{
  V.reset( B * members.size( a ) );
  for ( ptrdiff_t q = 0; q < members.size( a ); ++q ) {
    for ( int c = 0; c < B; ++c ) {
      const ptrdiff_t dof = B * members.node( a, q ) + c;
      if ( isDirichlet[dof] )
        continue;
      for ( int k = 0; k < M; ++k )
        V( B * q + c, k ) = nullspace[M * dof + k];
    }
  }
}

/// @brief The QR factorization V_a = Q_a R_a of the modes of an aggregate (orthonormalizeModes()).
/// @tparam M the number of modes
template < int M >
struct OrthonormalizedModes {
  /// @brief R_a, upper triangular; the row of a dropped mode is zero.
  amgcl::static_matrix< double, M, M > R = amgcl::math::zero< amgcl::static_matrix< double, M, M > >();
  /// @brief Per mode, whether it is in the coarse space (not dropped as dependent).
  char kept[M] = {};
  /// @brief The number of modes dropped.
  int dropped = 0;
};

/// @brief 2. and 3. V_a = Q_a R_a, by modified Gram-Schmidt with a second sweep, which makes Q_a orthonormal to
/// rounding also for nearly dependent columns; the columns of V become those of Q_a. The columns of P_tent are then
/// orthonormal, and P_tent R_a reproduces the modes on the aggregate exactly.
///
/// The dependent modes. A mode the sweeps reduce to at most `tolerance` of its norm (dependentModeTolerance()) lies in
/// the span of the modes before it: a rotation of an aggregate of a single node, the rotation about the line through a
/// collinear aggregate, a mode whose components are all constrained. Its normalized remainder would be rounding noise,
/// not a direction of the aggregate, so it is dropped: a zero column of Q_a and a zero row of R_a. (AMGCL's own
/// Householder QR keeps such a column in an aggregate of at least as many dofs as modes: an orthonormal completion,
/// unrelated to the modes.)
/// @tparam M the number of modes
/// @param V the modes, which become the columns of Q
/// @param tolerance the relative remainder at or below which a mode is dependent
/// @return R and the modes kept
template < int M >
OrthonormalizedModes< M > orthonormalizeModes( ModeColumns< M >& V, double tolerance )
{
  OrthonormalizedModes< M > qr;
  for ( int k = 0; k < M; ++k ) {
    const double norm = V.norm( k );
    for ( int sweep = 0; sweep < 2; ++sweep ) {
      for ( int j = 0; j < k; ++j ) {
        if ( !qr.kept[j] )
          continue;
        const double r_jk = V.dot( j, k );
        V.subtract( k, r_jk, j );
        qr.R( j, k ) += r_jk;
      }
    }
    const double remainder = V.norm( k );
    if ( norm > 0.0 && remainder > tolerance * norm ) {
      qr.kept[k]   = 1;
      qr.R( k, k ) = remainder;
      V.scale( k, 1.0 / remainder );
    }
    else {
      ++qr.dropped;
      V.zero( k );
      for ( int j = 0; j < M; ++j )
        qr.R( k, j ) = 0.0;
    }
  }
  return qr;
}

/// @brief 2. The blocks of P_tent of aggregate a: the rows of Q_a, one B x M block per node, in the column of the
/// aggregate.
/// @tparam B the number of components per node
/// @tparam M the number of modes
/// @param P P_tent, its row pointers set
/// @param members the nodes of the aggregates
/// @param a the aggregate
/// @param Q the orthonormal modes of the aggregate
template < int B, int M, class Matrix >
void writeProlongationBlocks( Matrix& P, const AggregateMembers& members, ptrdiff_t a, const ModeColumns< M >& Q )
{
  for ( ptrdiff_t q = 0; q < members.size( a ); ++q ) {
    amgcl::static_matrix< double, B, M > block;
    for ( int c = 0; c < B; ++c )
      for ( int k = 0; k < M; ++k )
        block( c, k ) = Q( B * q + c, k );
    const ptrdiff_t node = members.node( a, q );
    P.col[P.ptr[node]]   = a;
    P.val[P.ptr[node]]   = block;
  }
}

/// @brief The tentative prolongation of the coarse level, and what the coarse level takes over from its construction,
/// see tentativeProlongation().
/// @tparam B the number of components per node
/// @tparam M the number of modes, the coarse dofs per aggregate
template < int B, int M >
struct TentativeProlongationT {
  /// @brief A block of P: the B components of a node times the M coarse dofs of its aggregate.
  typedef amgcl::static_matrix< double, B, M > Block;
  /// @brief The block CSR type of P.
  typedef amgcl::backend::crs< Block, ptrdiff_t, ptrdiff_t > Matrix;

  /// @brief P_tent: one block per aggregated node, in the column of its aggregate; empty rows for the other nodes.
  std::shared_ptr< Matrix > P;
  /// @brief The near null-space of the coarse level: the coefficients of the near null-space in the coarse basis,
  /// (nAggregates M) x M, row-major -- per aggregate the M x M block R_a (P_tent R_a = V_a).
  std::vector< double > coarseNullspace;
  /// @brief Per coarse dof, whether its mode is in the coarse space; one that is not has a zero column of P.
  std::vector< char > keptMode;
  /// @brief The number of modes dropped as linearly dependent.
  ptrdiff_t droppedModes = 0;
};

/// @brief 2. and 3. The tentative prolongation of the coarse level: the near null-space of every aggregate,
/// orthonormalized, as AMGCL's tentative_prolongation does it -- QR of the near null-space restricted to each
/// aggregate, with R_a as the coarse near null-space -- with the Dirichlet rows zero and the dependent modes dropped
/// (aggregateModes(), orthonormalizeModes()). A node in no aggregate (aggregate id < 0: no strong connection, e.g. a
/// fully constrained one) gets an empty row, as in AMGCL's tentative prolongation: only the smoother acts on it.
/// @tparam B the number of components per node, 2 or 3
/// @tparam M the number of modes: B (the translations) or B (B + 1) / 2 (all rigid-body modes)
/// @param nNodes the number of nodes, the block rows of the fine operator
/// @param nAggregates the number of aggregates
/// @param aggregateOfNode the aggregate of every node, < 0 for none
/// @param nullspace the near null-space of the equilibrated block, n x M, row-major (n = B nNodes)
/// @param isDirichlet per dof, whether its row is a Dirichlet (identity) row
/// @return P_tent, the coarse near null-space, the kept modes
template < int B, int M >
TentativeProlongationT< B, M > tentativeProlongation( size_t                          nNodes,
                                                      size_t                          nAggregates,
                                                      const std::vector< ptrdiff_t >& aggregateOfNode,
                                                      const double*                   nullspace,
                                                      const char*                     isDirichlet )
{
  typedef TentativeProlongationT< B, M > Result;
  const AggregateMembers                 members = nodesOfAggregates( aggregateOfNode, nAggregates );

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
    ModeColumns< M > V;
#pragma omp for schedule( dynamic, 16 )
    for ( ptrdiff_t a = 0; a < static_cast< ptrdiff_t >( nAggregates ); ++a ) {
      aggregateModes< B, M >( members, a, nullspace, isDirichlet, V );
      const OrthonormalizedModes< M > qr = orthonormalizeModes< M >( V, tolerance );
      dropped += qr.dropped;
      writeProlongationBlocks< B, M >( *result.P, members, a, V );
      for ( int i = 0; i < M; ++i ) {
        for ( int j = 0; j < M; ++j )
          result.coarseNullspace[( M * a + i ) * M + j] = qr.R( i, j );
        result.keptMode[M * a + i] = qr.kept[i];
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

/// @brief 5. The Galerkin coarse operator A_c (M x M blocks) as the scalar matrix the AMGCL hierarchy below is built
/// on: AMGCL's unblock_matrix, with the diagonal entry `padding` for the coarse dof of every dropped mode.
///
/// The coarse dof of a dropped mode has a zero column of P, hence a zero row and column in A_c. Its diagonal entry
/// becomes the padding, which keeps A_c regular (positive definite where \hat A is) for the hierarchy below. For the
/// two-level cycle the padding is exact, whatever its value: P^T maps every residual to zero there, and the zero column
/// of P never prolongates what the hierarchy below computes there. Within the hierarchy below the padded dof is not
/// entirely inert: where an aggregate of the next level is rank-deficient (it holds dropped modes), AMGCL's QR
/// completes the orthonormal basis of that aggregate with directions that may include padded dofs, which then take a
/// share of that level's coarse space -- a cost in efficiency only, the operator stays positive definite. The padding,
/// the mean of the diagonal entries of the kept modes, lies inside the spectrum of A_c (each diagonal entry is a
/// Rayleigh quotient), so it does not move the spectral-radius estimate of the hierarchy's smoother.
///
/// The zeros of a dropped mode's row and column stay stored: AMGCL's aggregation below is point-wise (the M dofs of an
/// aggregate one point, aggr.block_size), with the largest entry of every M x M block as the strength of a
/// connection, which they do not change, and a block of zeros is never a strong connection -- so the aggregates are
/// those of the operator without them. Filtering them out would only save stored zeros in the levels below.
/// @tparam CoarseMatrix the block CSR type of A_c
/// @param Ac A_c, every row sorted by column
/// @param keptMode per coarse dof, whether its mode is in the coarse space
/// @return the scalar A_c, M rows per block row, every row sorted by column
template < class CoarseMatrix >
std::shared_ptr< amgcl::backend::crs< double, ptrdiff_t, ptrdiff_t > > unblockedCoarseOperator(
  const CoarseMatrix&        Ac,
  const std::vector< char >& keptMode )
{
  auto            S = amgcl::adapter::unblock_matrix( Ac );
  const ptrdiff_t n = static_cast< ptrdiff_t >( S->nrows );

  // the diagonal entry of every row (the columns of a row are sorted), and the mean of those of the kept modes
  std::vector< ptrdiff_t > diagonal( n, -1 );
  double                   keptDiagonalSum = 0.0;
  ptrdiff_t                nKept           = 0;
#pragma omp parallel for reduction( + : keptDiagonalSum, nKept )
  for ( ptrdiff_t i = 0; i < n; ++i ) {
    const ptrdiff_t* found = std::lower_bound( S->col + S->ptr[i], S->col + S->ptr[i + 1], i );
    if ( found != S->col + S->ptr[i + 1] && *found == i ) {
      diagonal[i] = found - S->col;
      if ( keptMode[i] ) {
        keptDiagonalSum += S->val[diagonal[i]];
        ++nKept;
      }
    }
  }
  const double padding = nKept > 0 ? keptDiagonalSum / static_cast< double >( nKept ) : 1.0;
  for ( ptrdiff_t i = 0; i < n; ++i ) {
    if ( keptMode[i] )
      continue;
    if ( diagonal[i] < 0 )
      throw std::logic_error( "rigid-body coarse level: a coarse dof without a diagonal entry in A_c" );
    S->val[diagonal[i]] = padding;
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
