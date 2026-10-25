#pragma once

#include "amgcl-block-translations.hpp"
#include "amgcl-rigid-body-coarse-level.hpp"
#include <algorithm>
#include <amgcl/adapter/block_matrix.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/backend/builtin.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/preconditioner/runtime.hpp>
#include <amgcl/relaxation/runtime.hpp>
#include <amgcl/solver/fgmres.hpp>
#include <amgcl/solver/lgmres.hpp>
#include <amgcl/solver/runtime.hpp>
#include <amgcl/value_type/static_matrix.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

/// The interface PyAMGCLSolver drives, whatever the backend (scalar or BxB blocks, double or float):
/// one pointer on the Cython side instead of one per backend. The matrix always arrives as a plain
/// scalar double CSR and rhs/x are always double; each backend converts internally.
class AMGCLSolverInterface {
public:
  virtual ~AMGCLSolverInterface() {}

  /// @brief Supply near null-space vectors for smoothed-aggregation coarsening (a block backend takes only the
  /// translations, see LinearSolverBlockT).
  /// @param B rows x cols, row-major (column j is the j-th vector); cols == 0 clears it.
  virtual void set_nullspace( const double* B, int rows, int cols ) = 0;

  /// @brief Build the AMG hierarchy for the n x n CSR matrix (ptr, col, val) once, for repeated
  /// applyPreconditioner() calls.
  virtual void build( int n, const int* ptr, const int* col, const double* val ) = 0;

  /// @brief x <- M^-1 rhs: one cycle of the hierarchy built by build().
  virtual void applyPreconditioner( int n, const double* rhs, double* x ) = 0;

  /// @brief Human-readable hierarchy report of the last build() or solve().
  virtual std::string report() const = 0;

  /// @brief Build a hierarchy for (ptr, col, val) and solve with the configured Krylov solver.
  /// @param iters, error the iteration count and relative residual the solver reports.
  virtual void solve( int           n,
                      const int*    ptr,
                      const int*    col,
                      const double* val,
                      const double* rhs,
                      double*       x,
                      int&          iters,
                      double&       error ) = 0;
};

/// @brief A JSON parameter tree as a boost property tree (empty for an empty string).
inline boost::property_tree::ptree parseJsonParameters( const char* jsonParameters )
{
  boost::property_tree::ptree parameters;
  const std::string           json( jsonParameters );
  if ( !json.empty() ) {
    std::stringstream stream( json );
    boost::property_tree::read_json( stream, parameters );
  }
  return parameters;
}

/// @brief A copy of the subtree of tree at path, empty if there is none.
inline boost::property_tree::ptree childOrEmpty( const boost::property_tree::ptree& tree, const std::string& path )
{
  boost::optional< const boost::property_tree::ptree& > child = tree.get_child_optional( path );
  return child ? *child : boost::property_tree::ptree();
}

/// @brief The human-readable report of an AMGCL solver (levels, operator complexity, coarse size).
template < typename Solver >
std::string hierarchyReport( const std::unique_ptr< Solver >& solver )
{
  if ( !solver ) {
    throw std::runtime_error( "report(): no hierarchy built yet -- call build() or solve() first" );
  }
  std::ostringstream report;
  report << *solver;
  return report.str();
}

/// @brief The matrix values in the backend's value type: the caller's own array for double, a
/// rounded copy (kept in storage) for float.
inline const double* valuesAs( const double* val, int, std::vector< double >& )
{
  return val;
}

/// @copydoc valuesAs(const double*, int, std::vector<double>&)
inline const float* valuesAs( const double* val, int nnz, std::vector< float >& storage )
{
  storage.assign( val, val + nnz );
  return storage.data();
}

// Templated on the AMGCL backend's value type, so the same wrapper serves both the default
// double-precision hierarchy and a float32 one (half the memory traffic in the smoother apply, the
// dominant cost on large coupled solves). The outer
// Krylov solve (blockamg's GMRES) always stays double; only the preconditioner's own storage and
// arithmetic narrow. The matrix values and rhs/x at the build()/applyPreconditioner()/solve()
// boundary are always double -- the value-type-dependent conversion happens inside this class, not at
// the Cython/Python boundary.
template < typename ValueType, typename BackendType = amgcl::backend::builtin< ValueType > >
class LinearSolverT : public AMGCLSolverInterface {
public:
  typedef BackendType Backend;
  typedef amgcl::make_solver< amgcl::runtime::preconditioner< Backend >, amgcl::runtime::solver::wrapper< Backend > >
    Solver;

  boost::property_tree::ptree prm;

  // Cached solver and matrix structure information
  std::unique_ptr< Solver > solver_;

  // Near null-space vectors, kept alive here because the property tree only stores a raw pointer to
  // them (AMGCL copies from it when the hierarchy is built). Must outlive every solver construction.
  // Always double, independent of ValueType/Backend: AMGCL's coarsening::nullspace_params::B is
  // hardcoded std::vector<double> (amgcl/coarsening/tentative_prolongation.hpp) -- the tentative
  // prolongation's QR factorization stays double-precision regardless of backend, since it runs once
  // at hierarchy build time, not in the per-iteration smoother apply this backend split targets.
  std::vector< double > nullspace_;

  // Constructor: Just stores the parameters
  LinearSolverT( const char* json_params ) : prm( parseJsonParameters( json_params ) ), solver_() {}

  // Supply near null-space vectors for smoothed-aggregation coarsening. B is a rows-by-cols matrix in
  // row-major order (column j is the j-th near null-space vector). AMGCL takes these as a raw pointer
  // in its property tree -- they cannot travel through the JSON parameter string -- so this is a
  // separate entry point. Must be called before the first solve(); the pointer is read when the AMG
  // hierarchy is constructed. Passing cols == 0 clears any previously set null-space.
  void set_nullspace( const double* B, int rows, int cols ) override
  {
    if ( cols <= 0 ) {
      nullspace_.clear();
      boost::optional< boost::property_tree::ptree& > coarsening = prm.get_child_optional( "precond.coarsening" );
      if ( coarsening ) {
        coarsening->erase( "nullspace" );
      }
      return;
    }
    nullspace_.assign( B, B + static_cast< size_t >( rows ) * cols );
    prm.put( "precond.coarsening.nullspace.cols", cols );
    prm.put( "precond.coarsening.nullspace.rows", rows );
    prm.put( "precond.coarsening.nullspace.B", nullspace_.data() );
  }

  // Build the solver (and thus the AMG hierarchy) once for A, so it can be applied repeatedly as a
  // preconditioner without rebuilding. This is the build-once / apply-many split that :meth:`solve`
  // fuses: :meth:`solve` reconstructs the hierarchy on every call, which is fine for a one-shot solve
  // but ruinous for an inner block preconditioner applied on every outer Krylov iteration. Uses the
  // current property tree (including any near null-space set via set_nullspace). A float backend
  // rounds the values first.
  void build( int n, const int* ptr, const int* col, const double* val ) override
  {
    int                      nnz = ptr[n];
    std::vector< ValueType > rounded;
    const ValueType*         values = valuesAs( val, nnz, rounded );
    auto                     A      = std::make_tuple( n,
                              amgcl::make_iterator_range( ptr, ptr + n + 1 ),
                              amgcl::make_iterator_range( col, col + nnz ),
                              amgcl::make_iterator_range( values, values + nnz ) );
    solver_.reset( new Solver( A, prm ) );
  }

  // Apply one preconditioner (AMG) cycle to rhs: x <- M^-1 rhs, where M is the hierarchy built by
  // :meth:`build`. This is the operation a block Gauss-Seidel / field-split preconditioner performs on
  // each field per outer iteration. Cheap relative to the build. rhs/x are double regardless of
  // ValueType -- for the float backend this narrows on the way in and widens on the way out; for the
  // (default) double backend the extra copy is a no-op-equivalent memcpy, not a behaviour change.
  void applyPreconditioner( int n, const double* rhs, double* x ) override
  {
    std::vector< ValueType > rhs_v( rhs, rhs + n );
    std::vector< ValueType > x_v( n, ValueType( 0 ) );
    auto                     rhs_rng = amgcl::make_iterator_range( rhs_v.data(), rhs_v.data() + n );
    auto                     x_rng   = amgcl::make_iterator_range( x_v.data(), x_v.data() + n );
    solver_->precond().apply( rhs_rng, x_rng );
    std::copy( x_v.begin(), x_v.end(), x );
  }

  // Human-readable hierarchy report (levels, operator complexity, coarse size) for the AMG built by
  // build() or solve(). Streams AMGCL's own operator<< for the preconditioner + solver. Must be
  // called after a successful build()/solve() -- throws otherwise.
  std::string report() const override { return hierarchyReport( solver_ ); }

  void solve( int           n,
              const int*    ptr,
              const int*    col,
              const double* val,
              const double* rhs,
              double*       x,
              int&          iters,
              double&       error ) override
  {

    int                      nnz = ptr[n];
    std::vector< ValueType > rounded;
    const ValueType*         values = valuesAs( val, nnz, rounded );

    auto ptr_rng = amgcl::make_iterator_range( ptr, ptr + n + 1 );
    auto col_rng = amgcl::make_iterator_range( col, col + nnz );
    auto val_rng = amgcl::make_iterator_range( values, values + nnz );

    auto A = std::make_tuple( n, ptr_rng, col_rng, val_rng );

    // Rebuilt unconditionally, and the sparsity pattern is deliberately not consulted: amgcl's
    // make_solver binds the matrix *values* at construction and ( *solver_ )( rhs, x ) takes no
    // matrix, so a solver kept across calls would solve the system it was built for. Values change
    // on every Newton iteration even when the pattern does not, so an unchanged pattern buys nothing
    // here -- reusing on that basis would silently return the previous iteration's answer. (Callers
    // that do want to control refresh timing use build()/apply() instead of solve().)
    solver_.reset( new Solver( A, prm ) );

    std::vector< ValueType > rhs_v( rhs, rhs + n );
    std::vector< ValueType > x_v( n, ValueType( 0 ) );
    std::tie( iters, error ) = ( *solver_ )( amgcl::make_iterator_range( rhs_v.data(), rhs_v.data() + n ),
                                             amgcl::make_iterator_range( x_v.data(), x_v.data() + n ) );
    std::copy( x_v.begin(), x_v.end(), x );
  }
};

// --- float-stored blocks, double hierarchy setup ------------------------------------------------
//
// AMGCL's own single-precision backend (builtin<float>, LinearSolverFloat) builds the whole hierarchy
// in float: amg's build_matrix is backend::builtin<Backend::value_type>::matrix (amgcl/amg.hpp), so
// the coarsening and every level's relaxation setup -- the Chebyshev smoother's spectral-radius power
// iteration included, which runs on that build matrix in the relaxation's constructor -- see float
// values. That was measured to cost noticeably more outer iterations. builtin_float_matrices keeps
// Backend::value_type double (blocks), so the build matrices, coarsening, Galerkin products,
// spectral-radius estimates and the coarsest direct solve are all double; only Backend::matrix --
// the level operators, prolongations and restrictions the cycle reads -- is stored in float, via
// copy_matrix(), the same hook AMGCL's own builtin_hybrid backend uses to store a scalar-built
// hierarchy in block form. Vectors stay double.

/// @brief A copy of A with every entry converted to StoredType -- double to float, a double 3x3 block to a
/// float one, or unchanged -- and the same structure with 32-bit indices, in one OpenMP-parallel pass (a
/// parallel first touch, so the copy is spread over the NUMA nodes like A). The index type never enters the
/// arithmetic: a product with the copy is bitwise the one with A's own indices.
template < typename StoredType, typename ValueType, typename Col, typename Ptr >
std::shared_ptr< amgcl::backend::crs< StoredType, int, int > > convertedCopy(
  const amgcl::backend::crs< ValueType, Col, Ptr >& A )
{
  const size_t maxIndex = static_cast< size_t >( std::numeric_limits< int >::max() );
  if ( A.nnz > maxIndex || A.ncols > maxIndex ) {
    throw std::overflow_error( "convertedCopy: a matrix of " + std::to_string( A.nnz ) +
                               " entries does not fit 32-bit indices" );
  }
  std::shared_ptr< amgcl::backend::crs< StoredType, int, int > >
    stored = std::make_shared< amgcl::backend::crs< StoredType, int, int > >();
  stored->set_size( A.nrows, A.ncols, false );
  stored->set_nonzeros( A.nnz );
  stored->ptr[0]        = 0;
  const ptrdiff_t nRows = static_cast< ptrdiff_t >( A.nrows );
#pragma omp parallel for
  for ( ptrdiff_t i = 0; i < nRows; ++i ) {
    stored->ptr[i + 1] = static_cast< int >( A.ptr[i + 1] );
    for ( Ptr j = A.ptr[i]; j < A.ptr[i + 1]; ++j ) {
      stored->col[j] = static_cast< int >( A.col[j] );
      stored->val[j] = A.val[j];
    }
  }
  return stored;
}

namespace amgcl {

  /// @brief A float-stored 3x3 block times a double vector block, accumulated in double. Without it,
  /// AMGCL's generic static_matrix product would return -- and accumulate in -- the matrix's value
  /// type, float; with it, every sparse product of the cycle accumulates in double (a non-template
  /// overload is preferred over the generic template).
  inline static_matrix< double, 3, 1 > operator*( const static_matrix< float, 3, 3 >&  a,
                                                  const static_matrix< double, 3, 1 >& b )
  {
    static_matrix< double, 3, 1 > c;
    for ( int i = 0; i < 3; ++i ) {
      double sum = 0.0;
      for ( int k = 0; k < 3; ++k )
        sum += a( i, k ) * b( k );
      c( i ) = sum;
    }
    return c;
  }

  static_assert( std::is_same< decltype( std::declval< const static_matrix< float, 3, 3 >& >() *
                                         std::declval< const static_matrix< double, 3, 1 >& >() ),
                               static_matrix< double, 3, 1 > >::value,
                 "a float-stored 3x3 block times a double vector block must accumulate in double" );

  namespace backend {

    /// @brief The builtin backend over double B x B blocks, with the level matrices the cycle reads --
    /// operators, prolongations, restrictions -- stored in float blocks with 32-bit indices. The
    /// hierarchy is set up as builtin< BlockType > sets it up -- coarsening, Galerkin products, every
    /// Chebyshev spectral-radius estimate and the coarsest direct solve stay double; only
    /// Backend::matrix, via copy_matrix(), narrows. 32-bit indices: builtin< BlockType > itself would
    /// store every level with 64-bit indices (its default ptrdiff_t), 16 instead of 12 bytes per scalar
    /// double entry. Used only for LinearSolverBlock3Float below -- the scalar and double-block backends
    /// stay plain builtin< BlockType >, so relax.type = "gauss_seidel" and the exact ILU0/ILUk/ILUT/ILUP
    /// triangular solve keep working for them (see LinearSolverT's, LinearSolverBlockT's docs).
    template < typename BlockType >
    struct builtin_float_matrices : public builtin< BlockType > {
      /// @brief The backend this one stores the level matrices of.
      typedef builtin< BlockType > Base;
      /// @brief BlockType with float entries.
      typedef typename math::replace_scalar< BlockType, float >::type StoredType;
      /// @brief A level matrix as the cycle reads it.
      typedef crs< StoredType, int, int > matrix;

      /// @brief No row iterator over the stored matrices (they differ from the build matrices, and are
      /// float where the build matrices are double). This is also what makes AMGCL reject
      /// relax.type = "gauss_seidel" for this backend (relaxation_is_supported) and route ilu0/iluk/
      /// ilut/ilup to the generic approximate triangular solve (ilu_solve.hpp's primary template)
      /// instead of the exact one plain amgcl::backend::builtin gets.
      struct provides_row_iterator : std::false_type {};

      /// @brief The level matrix as the cycle reads it: A's entries as StoredType, its indices as int.
      static std::shared_ptr< matrix > copy_matrix( std::shared_ptr< typename Base::matrix > A,
                                                    const typename Base::params& )
      {
        return ::convertedCopy< StoredType >( *A );
      }
    };

    template < typename B1, typename B2 >
    struct backends_compatible< builtin_float_matrices< B1 >, builtin_float_matrices< B2 > > : std::true_type {};

  } // namespace backend
} // namespace amgcl

// Block-valued backend: the per-field hierarchy stores/operates on B×B nodal blocks
// (amgcl::static_matrix<double,B,B>) instead of scalar entries. Two motivations, in confidence order:
// (i) the CSR index arrays shrink by ~B² (one column index per block instead of per scalar entry --
// index traffic, not values, is typically the larger share of hierarchy bandwidth, which is exactly
// what this attacks); (ii) block-aware smoothers (block-ILU0, block-GS) invert each node's B×B
// coupling exactly, AMGCL's own canonical recipe for vector-PDE (elasticity) operators -- true for the
// plain-builtin instantiations below (LinearSolverBlock2, LinearSolverBlock3); LinearSolverBlock3Float's
// float-stored level matrices lose both (see builtin_float_matrices above).
//
// This is a *separate* class from LinearSolverT, not another instantiation of it, because the
// construction/apply pattern genuinely differs: the matrix arrives from Python as a plain scalar CSR
// (Cython has no reason to know about block layout), so it must be wrapped with
// amgcl::adapter::block_matrix<BlockType> before reaching the block Backend, and rhs/x must be
// reinterpreted (amgcl::backend::reinterpret_as_rhs<BlockType>, a zero-copy amgcl::reinterpret_cast
// over a same-sized contiguous double buffer -- amgcl/backend/builtin.hpp -- not a per-element
// conversion) rather than element-cast like the float backend's std::vector<ValueType> narrowing.
// LinearSolverT stays untouched to keep the validated scalar/float paths at zero regression risk.
//
// The near null-space. Without one, the hierarchy is AMGCL's runtime-configured one (Solver), whose
// near null-space for a block value type is the identity block: constants per component of the
// variables it is built on -- for blockamg's equilibrated blocks D A D, D-weighted translations, not
// the translations D^-1 t of D A D. set_nullspace() takes exactly those B translations instead, one
// diagonal B×B block per node (edelweissfe/linsolve/nullspace.py's translationNullspace), and the
// next build() then builds smoothed aggregation with a tentative prolongation from them
// (TranslationSolver, see amgcl-block-translations.hpp). Any other near null-space -- the rotations
// in particular -- is refused: that hierarchy only knows diagonal blocks of translation weights, and
// the rotations are carried by the scalar backend. (AMGCL's runtime coarsening, given a near null-space
// on a block value type, switches the whole coarsening to scalar matrices, as_scalar: it coarsens the
// unblocked matrix and re-blocks P and R. This hierarchy does not use that route, and the rotations
// through it are untested here. The TODO in amgcl/coarsening/tentative_prolongation.hpp -- "this is
// just a workaround to make non-scalar value types compile. Most probably this won't actually work."
// -- concerns the compile-time smoothed_aggregation<BlockBackend> path, whose tentative prolongation
// TranslationSolver replaces.)
template < typename BlockType, typename BackendType = amgcl::backend::builtin< BlockType > >
class LinearSolverBlockT : public AMGCLSolverInterface {
public:
  typedef BackendType                                Backend;
  typedef amgcl::runtime::solver::wrapper< Backend > KrylovSolver;
  /// @brief AMGCL's runtime-configured hierarchy: precond.class, precond.coarsening.type and
  /// precond.relax.type are chosen by the parameter tree; the near null-space is the identity block.
  typedef amgcl::make_solver< amgcl::runtime::preconditioner< Backend >, KrylovSolver > Solver;
  /// @brief Smoothed aggregation whose tentative prolongation is built from the translations given to
  /// set_nullspace(); only precond.relax.type stays a runtime choice.
  typedef amgcl::make_solver< amgcl::amg< Backend,
                                          amgcl::coarsening::smoothed_aggregation_with_translations,
                                          amgcl::runtime::relaxation::wrapper >,
                              KrylovSolver >
    TranslationSolver;

  static const int BlockSize = amgcl::math::static_rows< BlockType >::value;

  boost::property_tree::ptree prm;
  /// @brief The hierarchy of the last build() or solve() without translations, else null.
  std::unique_ptr< Solver > solver_;
  /// @brief The hierarchy of the last build() or solve() with translations, else null.
  std::unique_ptr< TranslationSolver > translationSolver_;
  /// @brief The translations given to set_nullspace(), n x BlockSize row-major; empty for the identity blocks.
  /// Kept alive here: the parameter tree of TranslationSolver only holds a pointer to them.
  std::vector< double > translations_;

  LinearSolverBlockT( const char* json_params )
    : prm( parseJsonParameters( json_params ) ), solver_(), translationSolver_()
  {
  }

  /// @brief Set the near null-space of the next build()/solve(): the BlockSize translations of the block, or
  /// (cols == 0) none, i.e. AMGCL's identity blocks.
  /// @param B rows x cols, row-major: per node a diagonal BlockSize x BlockSize block of positive weights
  /// @param rows, cols the shape of B: rows = n, cols = BlockSize, or cols = 0
  /// @throws std::runtime_error for another shape, for an entry off the diagonal blocks or a weight that is not
  /// positive and finite, or for a parameter tree whose hierarchy is not smoothed aggregation
  void set_nullspace( const double* B, int rows, int cols ) override
  {
    if ( cols <= 0 ) {
      translations_.clear();
      return;
    }
    if ( cols != BlockSize || rows <= 0 || rows % BlockSize != 0 ) {
      throw std::runtime_error( "set_nullspace(): a block-valued AMGCL backend takes the " +
                                std::to_string( BlockSize ) + " translations as its near null-space, n x " +
                                std::to_string( BlockSize ) + " with one diagonal block per node, got " +
                                std::to_string( rows ) + " x " + std::to_string( cols ) +
                                " (the rotations are supported on the scalar backend)" );
    }
    try {
      amgcl::coarsening::check_translation_weights( B, static_cast< size_t >( rows / BlockSize ), BlockSize );
    }
    catch ( const std::runtime_error& error ) {
      throw std::runtime_error( std::string( "set_nullspace(): " ) + error.what() );
    }
    const std::string precondClass   = prm.get< std::string >( "precond.class", "amg" );
    const std::string coarseningType = prm.get< std::string >( "precond.coarsening.type", "smoothed_aggregation" );
    if ( precondClass != "amg" || coarseningType != "smoothed_aggregation" ) {
      throw std::runtime_error( "set_nullspace(): the translations of a block-valued AMGCL backend are a near "
                                "null-space for smoothed aggregation, but the parameters ask for precond.class '" +
                                precondClass + "' with precond.coarsening.type '" + coarseningType + "'" );
    }
    translations_.assign( B, B + static_cast< size_t >( rows ) * cols );
  }

  /// @brief The parameters of TranslationSolver for an n-row matrix: prm without the runtime selectors
  /// precond.class and precond.coarsening.type (fixed at compile time), with the translations as
  /// precond.coarsening.nullspace.
  /// @throws std::runtime_error if the translations were set for another number of rows
  boost::property_tree::ptree translationParameters( int n )
  {
    if ( translations_.size() != static_cast< size_t >( n ) * BlockSize ) {
      throw std::runtime_error( "block-valued AMGCL backend: set_nullspace() gave the translations of " +
                                std::to_string( translations_.size() / BlockSize ) + " rows, the matrix has " +
                                std::to_string( n ) );
    }
    boost::property_tree::ptree                     parameters = prm;
    boost::optional< boost::property_tree::ptree& > precond    = parameters.get_child_optional( "precond" );
    if ( precond ) {
      precond->erase( "class" );
      boost::optional< boost::property_tree::ptree& > coarsening = precond->get_child_optional( "coarsening" );
      if ( coarsening ) {
        coarsening->erase( "type" );
      }
    }
    const int cols = BlockSize; // a copy: put() takes a reference, and BlockSize has no out-of-class definition
    parameters.put( "precond.coarsening.nullspace.cols", cols );
    parameters.put( "precond.coarsening.nullspace.rows", n );
    parameters.put( "precond.coarsening.nullspace.B", translations_.data() );
    return parameters;
  }

  /// @brief Build the hierarchy for the block matrix A: TranslationSolver if set_nullspace() gave the
  /// translations, Solver otherwise. The other one is released; a build that throws leaves both as they were.
  template < class BlockMatrix >
  void buildHierarchy( const BlockMatrix& A, int n )
  {
    if ( translations_.empty() ) {
      solver_.reset( new Solver( A, prm ) );
      translationSolver_.reset();
    }
    else {
      translationSolver_.reset( new TranslationSolver( A, translationParameters( n ) ) );
      solver_.reset();
    }
  }

  // n must be divisible by BlockSize (node-major DOF layout: BlockSize contiguous scalar DOFs per
  // node). amgcl::adapter::block_matrix asserts this too, but only via assert(), which -DNDEBUG (this
  // extension's build flag) compiles out -- so this check is the only one that actually runs.
  void checkBlockDivisible( int n ) const
  {
    if ( n % BlockSize != 0 ) {
      throw std::runtime_error( "block-valued AMGCL backend: n=" + std::to_string( n ) +
                                " is not divisible by the block size " + std::to_string( BlockSize ) +
                                " -- the DOF layout must be node-major with BlockSize contiguous scalar "
                                "DOFs per node." );
    }
  }

  void build( int n, const int* ptr, const int* col, const double* val ) override
  {
    checkBlockDivisible( n );
    int  nnz = ptr[n];
    auto A   = std::make_tuple( n,
                              amgcl::make_iterator_range( ptr, ptr + n + 1 ),
                              amgcl::make_iterator_range( col, col + nnz ),
                              amgcl::make_iterator_range( val, val + nnz ) );
    buildHierarchy( amgcl::adapter::block_matrix< BlockType >( A ), n );
  }

  void applyPreconditioner( int n, const double* rhs, double* x ) override
  {
    std::fill( x, x + n, 0.0 );
    std::vector< double > rhs_v( rhs, rhs + n );
    std::vector< double > x_v( x, x + n );
    auto                  rhs_rng = amgcl::backend::reinterpret_as_rhs< BlockType >( rhs_v );
    auto                  x_rng   = amgcl::backend::reinterpret_as_rhs< BlockType >( x_v );
    if ( translationSolver_ )
      translationSolver_->precond().apply( rhs_rng, x_rng );
    else
      solver_->precond().apply( rhs_rng, x_rng );
    std::copy( x_v.begin(), x_v.end(), x );
  }

  std::string report() const override
  {
    return translationSolver_ ? hierarchyReport( translationSolver_ ) : hierarchyReport( solver_ );
  }

  void solve( int           n,
              const int*    ptr,
              const int*    col,
              const double* val,
              const double* rhs,
              double*       x,
              int&          iters,
              double&       error ) override
  {
    checkBlockDivisible( n );
    int  nnz = ptr[n];
    auto A   = std::make_tuple( n,
                              amgcl::make_iterator_range( ptr, ptr + n + 1 ),
                              amgcl::make_iterator_range( col, col + nnz ),
                              amgcl::make_iterator_range( val, val + nnz ) );

    // Rebuilt unconditionally, and the sparsity pattern is deliberately not consulted: amgcl's
    // make_solver binds the matrix *values* at construction and ( *solver_ )( rhs, x ) takes no
    // matrix, so a solver kept across calls would solve the system it was built for. Values change
    // on every Newton iteration even when the pattern does not, so an unchanged pattern buys nothing
    // here -- reusing on that basis would silently return the previous iteration's answer. (Callers
    // that do want to control refresh timing use build()/apply() instead of solve().)
    buildHierarchy( amgcl::adapter::block_matrix< BlockType >( A ), n );

    std::vector< double > rhs_v( rhs, rhs + n );
    std::vector< double > x_v( n, 0.0 );
    auto                  rhs_rng = amgcl::backend::reinterpret_as_rhs< BlockType >( rhs_v );
    auto                  x_rng   = amgcl::backend::reinterpret_as_rhs< BlockType >( x_v );
    std::tie( iters, error )      = translationSolver_ ? ( *translationSolver_ )( rhs_rng, x_rng )
                                                       : ( *solver_ )( rhs_rng, x_rng );
    std::copy( x_v.begin(), x_v.end(), x );
  }
};

/// @brief The diagonal entry a_ii of row i of a CSR matrix (the sum of its stored entries in column i, 0 if
/// none is stored).
inline double rowDiagonal( const int* ptr, const int* col, const double* val, ptrdiff_t i )
{
  double diagonal = 0.0;
  for ( int j = ptr[i]; j < ptr[i + 1]; ++j )
    if ( col[j] == i )
      diagonal += val[j];
  return diagonal;
}

/// The equilibrated, field-split system of blockamg.py's outer solve, built in one OpenMP-threaded
/// pass over A's CSR arrays instead of a chain of serial SciPy copies and slices.
///
/// blockamg solves A x = b as (D A D)(D^-1 x) = D b, D = diag(scaling), and its block Gauss-Seidel
/// sweep needs every field's equilibrated diagonal block (for the per-field preconditioners, on a
/// hierarchy rebuild) and the equilibrated coupling blocks between fields (on every solve). This
/// class holds
///
///   * the outer Krylov operator: matvec(), residual(), and the operator OuterKrylovSolverT::solve()
///     iterates on -- the equilibrated operator As = D A D or, built with `unscaled`, A itself. Only that one
///     is stored. An entry of As is formed exactly as SciPy's former
///     `As.data *= scaling[As.indices]; As.data *= scaling[rows]` did, (a_ij * s_j) * s_i, s = scaling, so the
///     operator is bitwise the former one; with A stored, the entries of As the field split reads are formed
///     from it by that same expression (scaledValue()), bitwise the same;
///   * the coupling blocks As[field i, field j], i != j, rectangular, with field-local column
///     indices: couplingMatvec();
///   * the per-row, per-field entry counts that let a diagonal block As[field i, field i] be written
///     straight into caller-owned (NumPy) arrays on demand: diagonalBlockNnz() / fillDiagonalBlock().
///
/// The copy is a parallel first touch, so the operator is spread over the NUMA nodes like every other
/// AMGCL matrix (the input arrays come from a single-threaded SciPy allocation). Columns are stored as
/// int, as in the input: 12 instead of 16 bytes per entry in every outer matrix-vector product.
class EquilibratedSystemT {
public:
  typedef amgcl::backend::crs< double, int, int > Matrix;

  /// the stored operator: As = D A D, or A itself if built with `unscaled`
  std::shared_ptr< Matrix > operator_;
  /// whether operator_ holds A itself (build()'s `unscaled`)
  bool unscaled_ = false;
  /// D = diag(scaling_), to form the entries of As when A itself is stored (scaledValue())
  std::vector< double >                    scaling_;
  std::vector< std::shared_ptr< Matrix > > couplings_;   // nFields x nFields, row-major; null on the diagonal
  std::vector< int >                       fieldStart_, fieldStop_;
  std::vector< signed char >               fieldOfDof_;  // -1 for a DOF in no field
  std::vector< int >                       diagonalPtr_; // per field, concatenated: local row pointers
  std::vector< size_t >                    diagonalPtrOffset_;
  int                                      n_ = 0, nFields_ = 0;

  /// @brief |a_ii| for every row (0 where no diagonal entry is stored), the basis of the equilibration.
  static void absoluteDiagonal( int n, const int* ptr, const int* col, const double* val, double* out )
  {
#pragma omp parallel for
    for ( ptrdiff_t i = 0; i < n; ++i )
      out[i] = std::abs( rowDiagonal( ptr, col, val, i ) );
  }

  /// @brief Build As = D A D and the coupling blocks. The new system is assembled completely before it
  /// replaces the previous one, so a build that throws (an invalid field layout, an allocation) leaves the
  /// previous system intact.
  /// @param n, ptr, col, val A, square CSR in canonical form: sorted columns, no duplicates -- the diagonal
  /// blocks inherit its column order, and AMGCL's block backends require it sorted.
  /// @param scaling the equilibration, D = diag(scaling).
  /// @param nFields, fieldBounds the fields' DOF ranges, [fieldBounds[2k], fieldBounds[2k+1]),
  /// disjoint.
  /// @param unscaled store A itself (without its stored zeros) instead of As, for an outer Krylov solve on
  /// the unscaled system: matvec() and residual() with unscaled, OuterKrylovSolverT::solve(). The blocks of
  /// As are the same either way.
  void build( int           n,
              const int*    ptr,
              const int*    col,
              const double* val,
              const double* scaling,
              int           nFields,
              const int*    fieldBounds,
              bool          unscaled = false )
  {
    EquilibratedSystemT assembled;
    assembled.assemble( n, ptr, col, val, scaling, nFields, fieldBounds, unscaled );
    *this = std::move( assembled );
  }

  /// @brief Keep A itself, as given (its stored zeros too), as the operator of an outer Krylov solve on the unscaled
  /// system, without fields or blocks: what matrix( true ), matvec() and residual() with unscaled, and
  /// OuterKrylovSolverT::solve() read (blockamg's condensed solves). Kept with A's own sparsity pattern, its values
  /// can be replaced in place for the next matrix of that pattern, see refreshOperatorValues().
  /// @param n, ptr, col, val A, square CSR
  void buildOperator( int n, const int* ptr, const int* col, const double* val )
  {
    auto stored = std::make_shared< Matrix >();
    stored->set_size( n, n, false );
    std::copy( ptr, ptr + n + 1, stored->ptr );
    stored->set_nonzeros( ptr[n] );
    std::copy( col, col + ptr[n], stored->col );
    std::copy( val, val + ptr[n], stored->val );
    EquilibratedSystemT built;
    built.operator_ = stored;
    built.unscaled_ = true;
    built.scaling_.assign( n, 1.0 );
    built.n_ = n;
    *this    = std::move( built );
  }

  /// @brief Replace the values of the operator kept by buildOperator() by those of a matrix of the same sparsity
  /// pattern, in place: one pass over the values, without an allocation. The caller guarantees the pattern; only the
  /// number of values is checked here.
  /// @param nnz, val the values of the matrix, in the order of the kept pattern
  void refreshOperatorValues( int nnz, const double* val )
  {
    checkBuilt();
    if ( nFields_ != 0 || !unscaled_ ) {
      throw std::runtime_error( "EquilibratedSystemT::refreshOperatorValues(): only for an operator kept by "
                                "buildOperator()" );
    }
    if ( static_cast< ptrdiff_t >( nnz ) != static_cast< ptrdiff_t >( operator_->ptr[n_] ) ) {
      throw std::runtime_error( "EquilibratedSystemT::refreshOperatorValues(): " + std::to_string( nnz ) +
                                " values for an operator of " + std::to_string( operator_->ptr[n_] ) + " entries" );
    }
#pragma omp parallel for
    for ( ptrdiff_t j = 0; j < static_cast< ptrdiff_t >( nnz ); ++j )
      operator_->val[j] = val[j];
  }

  /// @brief The work of build(), on this (fresh) object.
  void assemble( int           n,
                 const int*    ptr,
                 const int*    col,
                 const double* val,
                 const double* scaling,
                 int           nFields,
                 const int*    fieldBounds,
                 bool          unscaled )
  {
    if ( nFields > std::numeric_limits< signed char >::max() ) {
      throw std::runtime_error( "EquilibratedSystemT: at most " +
                                std::to_string( std::numeric_limits< signed char >::max() ) + " fields." );
    }
    std::vector< signed char > fieldOfDof( n, -1 );
    for ( int f = 0; f < nFields; ++f ) {
      const int start = fieldBounds[2 * f], stop = fieldBounds[2 * f + 1];
      if ( start < 0 || stop > n || start > stop ) {
        throw std::runtime_error( "EquilibratedSystemT: field " + std::to_string( f ) + " has invalid bounds." );
      }
      for ( int i = start; i < stop; ++i ) {
        if ( fieldOfDof[i] != -1 ) {
          throw std::runtime_error( "EquilibratedSystemT: fields overlap at DOF " + std::to_string( i ) + "." );
        }
        fieldOfDof[i] = static_cast< signed char >( f );
      }
    }
    n_       = n;
    nFields_ = nFields;
    fieldOfDof_.swap( fieldOfDof );
    fieldStart_.assign( nFields, 0 );
    fieldStop_.assign( nFields, 0 );
    for ( int f = 0; f < nFields; ++f ) {
      fieldStart_[f] = fieldBounds[2 * f];
      fieldStop_[f]  = fieldBounds[2 * f + 1];
    }

    // As = D A D, with the same rounding as SciPy's two in-place scalings, and without A's explicitly
    // stored zeros: the Dirichlet elimination (solvers/base/dirichlet.pyx) keeps the eliminated
    // entries as stored zeros, so a constrained row is only recognizable as one -- a single entry, its
    // diagonal -- once they are gone, and every product reads fewer entries (several percent of A on
    // a condensed model). A sum loses only exact zero terms, so every product is bitwise the one with
    // them (up to the sign of a zero).
    operator_ = std::make_shared< Matrix >();
    operator_->set_size( n, n, false );
    operator_->ptr[0] = 0;
#pragma omp parallel for
    for ( ptrdiff_t i = 0; i < n; ++i ) {
      int width = 0;
      for ( int j = ptr[i]; j < ptr[i + 1]; ++j )
        if ( val[j] != 0.0 )
          ++width;
      operator_->ptr[i + 1] = width;
    }
    operator_->set_nonzeros( operator_->scan_row_sizes() );
    unscaled_ = unscaled;
    scaling_.assign( scaling, scaling + n );
#pragma omp parallel for
    for ( ptrdiff_t i = 0; i < n; ++i ) {
      int head = operator_->ptr[i];
      for ( int j = ptr[i]; j < ptr[i + 1]; ++j ) {
        if ( val[j] != 0.0 ) {
          operator_->col[head]   = col[j];
          operator_->val[head++] = unscaled ? val[j] : ( val[j] * scaling[col[j]] ) * scaling[i];
        }
      }
    }
    const int* const filteredPtr = operator_->ptr;
    const int* const filteredCol = operator_->col;

    // per row, the number of entries in each field's columns: row pointers of the diagonal blocks
    // (kept) and of the coupling blocks (consumed below)
    std::vector< std::vector< int > > rowCounts( static_cast< size_t >( nFields ) * nFields );
    for ( int f = 0; f < nFields; ++f )
      for ( int g = 0; g < nFields; ++g )
        rowCounts[f * nFields + g].assign( fieldStop_[f] - fieldStart_[f] + 1, 0 );
#pragma omp parallel for
    for ( ptrdiff_t i = 0; i < n; ++i ) {
      const int f = fieldOfDof_[i];
      if ( f < 0 )
        continue;
      const int local = static_cast< int >( i ) - fieldStart_[f];
      for ( int j = filteredPtr[i]; j < filteredPtr[i + 1]; ++j ) {
        const int g = fieldOfDof_[filteredCol[j]];
        if ( g >= 0 )
          ++rowCounts[f * nFields + g][local + 1];
      }
    }
    for ( auto& counts : rowCounts )
      std::partial_sum( counts.begin(), counts.end(), counts.begin() );

    diagonalPtrOffset_.assign( nFields + 1, 0 );
    for ( int f = 0; f < nFields; ++f )
      diagonalPtrOffset_[f + 1] = diagonalPtrOffset_[f] + rowCounts[f * nFields + f].size();
    diagonalPtr_.resize( diagonalPtrOffset_[nFields] );
    for ( int f = 0; f < nFields; ++f )
      std::copy( rowCounts[f * nFields + f].begin(),
                 rowCounts[f * nFields + f].end(),
                 diagonalPtr_.begin() + diagonalPtrOffset_[f] );

    couplings_.assign( static_cast< size_t >( nFields ) * nFields, nullptr );
    for ( int f = 0; f < nFields; ++f ) {
      for ( int g = 0; g < nFields; ++g ) {
        if ( f == g )
          continue;
        const std::vector< int >& counts   = rowCounts[f * nFields + g];
        const int                 nRows    = fieldStop_[f] - fieldStart_[f];
        std::shared_ptr< Matrix > coupling = std::make_shared< Matrix >();
        coupling->set_size( nRows, fieldStop_[g] - fieldStart_[g], false );
        std::copy( counts.begin(), counts.end(), coupling->ptr );
        coupling->set_nonzeros( counts.back() );
        fillBlock( f, g, coupling->ptr, coupling->col, coupling->val );
        couplings_[f * nFields + g] = coupling;
      }
    }
  }

  /// @brief The entry of As at the stored operator's position j, in row i: stored, or formed from A's entry
  /// by the expression As was assembled with.
  double scaledValue( ptrdiff_t i, int j ) const
  {
    if ( !unscaled_ )
      return operator_->val[j];
    return ( operator_->val[j] * scaling_[operator_->col[j]] ) * scaling_[i];
  }

  /// @brief Write As[field f, field g] (field-local columns) into ptr/col/val -- ptr already holding
  /// the block's row pointers.
  void fillBlock( int f, int g, const int* blockPtr, int* blockCol, double* blockVal ) const
  {
    const int start = fieldStart_[f], nRows = fieldStop_[f] - fieldStart_[f], columnStart = fieldStart_[g];
#pragma omp parallel for
    for ( ptrdiff_t local = 0; local < nRows; ++local ) {
      const ptrdiff_t i    = start + local;
      int             head = blockPtr[local];
      for ( int j = operator_->ptr[i]; j < operator_->ptr[i + 1]; ++j ) {
        const int c = operator_->col[j];
        if ( fieldOfDof_[c] == g ) {
          blockCol[head]   = c - columnStart;
          blockVal[head++] = scaledValue( i, j );
        }
      }
    }
  }

  /// @brief Throw unless build() has run.
  void checkBuilt() const
  {
    if ( !operator_ ) {
      throw std::runtime_error( "EquilibratedSystemT: nothing built yet -- call build() first" );
    }
  }

  /// @brief Throw unless build() has run and f is one of its fields.
  void checkField( int f ) const
  {
    checkBuilt();
    if ( f < 0 || f >= nFields_ ) {
      throw std::runtime_error( "EquilibratedSystemT: no field " + std::to_string( f ) );
    }
  }

  /// @brief The number of DOFs of field f.
  int diagonalBlockSize( int f ) const
  {
    checkField( f );
    return fieldStop_[f] - fieldStart_[f];
  }

  /// @brief The number of entries of As[field f, field f] (stored zeros of A dropped).
  int diagonalBlockNnz( int f ) const
  {
    checkField( f );
    return diagonalPtr_[diagonalPtrOffset_[f + 1] - 1];
  }

  /// @brief As[field f, field f] into caller-owned arrays of diagonalBlockSize(f) + 1 and
  /// diagonalBlockNnz(f) entries -- written in parallel, so freshly allocated (untouched) arrays get
  /// the same NUMA spread as the operator.
  void fillDiagonalBlock( int f, int* ptr, int* col, double* val ) const
  {
    checkField( f );
    std::copy( diagonalPtr_.begin() + diagonalPtrOffset_[f], diagonalPtr_.begin() + diagonalPtrOffset_[f + 1], ptr );
    fillBlock( f, f, ptr, col, val );
  }

  /// @brief The operator a product runs on: As, or with unscaled A itself -- the one build() stored.
  const Matrix& matrix( bool unscaled ) const
  {
    checkBuilt();
    if ( unscaled != unscaled_ ) {
      throw std::runtime_error( unscaled ? "EquilibratedSystemT: the unscaled operator was not kept -- "
                                           "build( ..., unscaled )"
                                         : "EquilibratedSystemT: the equilibrated operator was not kept -- "
                                           "build() without unscaled" );
    }
    return *operator_;
  }

  /// @brief y <- As x, or with unscaled y <- A x
  void matvec( const double* x, double* y, bool unscaled = false ) const
  {
    const Matrix& M      = matrix( unscaled );
    auto          xRange = amgcl::make_iterator_range( x, x + n_ );
    auto          yRange = amgcl::make_iterator_range( y, y + n_ );
    amgcl::backend::spmv( 1.0, M, xRange, 0.0, yRange );
  }

  /// @brief r <- rhs - As x, or with unscaled r <- rhs - A x
  void residual( const double* rhs, const double* x, double* r, bool unscaled = false ) const
  {
    const Matrix& M        = matrix( unscaled );
    auto          rhsRange = amgcl::make_iterator_range( rhs, rhs + n_ );
    auto          xRange   = amgcl::make_iterator_range( x, x + n_ );
    auto          rRange   = amgcl::make_iterator_range( r, r + n_ );
    amgcl::backend::residual( rhsRange, M, xRange, rRange );
  }

  /// @brief y (field f) <- As[field f, field f] x (field f), read straight from the operator's rows
  /// (the diagonal blocks are only materialized for a hierarchy rebuild). Only the block Gauss-Seidel
  /// sweep's revisits of a field need it -- sweeps > 1 or symmetric.
  void diagonalBlockMatvec( int f, const double* x, double* y ) const
  {
    checkField( f );
    const int       start = fieldStart_[f];
    const ptrdiff_t nRows = fieldStop_[f] - fieldStart_[f];
#pragma omp parallel for
    for ( ptrdiff_t local = 0; local < nRows; ++local ) {
      const ptrdiff_t i   = start + local;
      double          sum = 0.0;
      for ( int j = operator_->ptr[i]; j < operator_->ptr[i + 1]; ++j ) {
        const int c = operator_->col[j];
        if ( fieldOfDof_[c] == f )
          sum += scaledValue( i, j ) * x[c - start];
      }
      y[local] = sum;
    }
  }

  /// @brief y (field f) <- As[field f, field g] x (field g), f != g
  void couplingMatvec( int f, int g, const double* x, double* y ) const
  {
    checkField( f );
    checkField( g );
    if ( f == g ) {
      throw std::runtime_error( "EquilibratedSystemT::couplingMatvec(): a coupling block needs two different fields." );
    }
    const Matrix& coupling = *couplings_[f * nFields_ + g];
    auto          xRange   = amgcl::make_iterator_range( x, x + coupling.ncols );
    auto          yRange   = amgcl::make_iterator_range( y, y + coupling.nrows );
    amgcl::backend::spmv( 1.0, coupling, xRange, 0.0, yRange );
  }
};

// Bridges AMGCL's own native amgcl::solver::lgmres (an outer Krylov solve running entirely on
// the OpenMP-threaded builtin backend) to blockamg.py's Python-level block Gauss-Seidel preconditioner
// (the same closure the shipped scipy.sparse.linalg.gmres path already uses), via a plain C
// function-pointer callback + opaque context set from Cython.
//
// Motivation: with the outer operator SpMV already threaded (EquilibratedSystemT above), the single
// largest remaining bucket in the shipped arm's own solve wall (measured as a substantial share of total
// wall-clock on a real reference model) is scipy's *own* GMRES orchestration -- Arnoldi/Gram-Schmidt/
// restart bookkeeping -- which is unavoidably serial CPython regardless of any matvec/preconditioner
// threading, and scales with the square of the restart length.
//
// Why a C function pointer + void* context, not a Cython-overridden C++ virtual method: lgmres's
// Precond parameter (see amgcl/solver/lgmres.hpp's operator()) is a compile-time template argument,
// not a runtime-polymorphic base pointer -- nothing inside lgmres itself ever needs to swap
// preconditioner *implementations* through a common interface, so a virtual base class would only add
// a vtable/RTTI and an extra indirection for no benefit. A bare function pointer needs neither and
// keeps this header entirely free of any Python/Cython dependency, matching every other class here.

// Callback signature: one preconditioner application, x <- M^-1 rhs, where M is whatever
// blockamg.py's own block Gauss-Seidel closure computes for this solve. `ctx` is an opaque pointer to
// whatever Cython-side state the callback needs (the calling PyAMGCLLGMRESSolver instance, in
// practice) -- this header never interprets it as anything other than a pass-through argument. `n` is
// duplicated here (also known to the caller) purely so the callback does not need to reach back into
// any C++-side state to learn it. It returns 0 on success and anything else if it failed (a Python
// exception, which the caller keeps and re-raises).
typedef int ( *PyPrecondApplyFn )( void* ctx, int n, const double* rhs, double* x );

// Adapts a PyPrecondApplyFn callback to amgcl::solver::lgmres's Precond interface: a single const
// apply(rhs, x) method (see amgcl/solver/precond_side.hpp's spmv() -- with the default
// preconditioning side `right` that this class assumes, apply() is the *only* thing lgmres ever calls
// on a Precond object; see OuterKrylovSolverT's own comment for why `right` is not overridden here).
//
// rhs/x are written directly against amgcl::backend::builtin<double>::vector (numa_vector<double>),
// not kept as a generic template parameter: lgmres calls apply() exclusively on its own internal
// Arnoldi/augmentation-vector storage (vs[j], outer_v entries, the final correction dx), which is
// always exactly this concrete type for the Backend this class is built for -- there is no second
// vector type apply() is ever actually called with in this codebase's usage, so templating it, as
// amgcl::preconditioner::dummy does for genuine backend-portability, would cost readability for no
// real flexibility gained here.
class PyKrylovPrecondT {
public:
  PyKrylovPrecondT( PyPrecondApplyFn applyFn, void* ctx, int n ) : applyFn_( applyFn ), ctx_( ctx ), n_( n ) {}

  /// @brief x <- M^-1 rhs through the callback.
  /// @throw std::runtime_error if the callback failed: the exception unwinds lgmres at once, instead of
  /// letting it run to maxiter on a meaningless x.
  void apply( const amgcl::backend::builtin< double >::vector& rhs, amgcl::backend::builtin< double >::vector& x ) const
  {
    if ( applyFn_( ctx_, n_, rhs.data(), x.data() ) != 0 )
      throw std::runtime_error( "OuterKrylovSolverT: the preconditioner callback failed" );
  }

private:
  PyPrecondApplyFn applyFn_;
  void*            ctx_;
  int              n_;
};

/// Wraps amgcl::solver::lgmres<builtin<double>> (or fgmres) as blockamg.py's outer Krylov solve, in place of
/// scipy.sparse.linalg.gmres, at both of that path's call sites (the main solve and the true-residual
/// continuation retry -- see blockamg.py's __call__).
///
/// One instance lives as long as the system size does: lgmres preallocates every scratch vector
/// (Arnoldi basis, augmentation vectors) for a fixed n in its own constructor, so a size change (AMR)
/// needs a fresh instance, and blockamg.py rebuilds it then -- and when its own settings change. Every
/// call starts from an empty augmentation subspace (lgmres's always_reset, forced here): carrying the
/// augmentation vectors across calls ("recycling") was measured to buy nothing on a set of
/// representative captured systems and, on a live trajectory, to compound a struggling solve's subspace into much more
/// expensive and once NaN subsequent ones, so it is not offered. The intra-call restart-cycle
/// augmentation (K) is unaffected.
///
/// tol/maxiter are set on every call on the solver's own public prm member -- blockamg's forcing and
/// continuation tolerances change per call -- and every call iterates on the EquilibratedSystemT it is
/// given, built fresh for that solve.
/// The outer Krylov solve of either method, as Cython holds it (see makeOuterKrylovSolver()).
class OuterKrylovSolver {
public:
  virtual ~OuterKrylovSolver() {}

  /// @brief x <- A^-1 rhs, right-preconditioned by the callback (applyFn/ctx), see OuterKrylovSolverT::solve().
  virtual void solve( const EquilibratedSystemT& system,
                      const double*              rhs,
                      double*                    x,
                      double                     tol,
                      int                        maxiter,
                      PyPrecondApplyFn           applyFn,
                      void*                      ctx,
                      int&                       iters,
                      double&                    error,
                      bool                       onUnscaledOperator ) = 0;
};

/// @brief The parameters LGMRES is always run with, whatever the caller's tree says: always_reset (see the class
/// comment) and right preconditioning -- lgmres's own convergence test is then on the residual of the operator it
/// iterates on, which blockamg relies on (see its cap-hit and true-residual logic); left preconditioning would test
/// the preconditioned residual instead.
inline void forceOuterKrylovParameters( boost::property_tree::ptree& prm,
                                        const amgcl::solver::lgmres< amgcl::backend::builtin< double > >* )
{
  prm.put( "always_reset", true );
  prm.put( "pside", "right" );
}

/// @brief FGMRES has nothing to force: it is right-preconditioned by construction (x = Z_m y_m) and keeps no
/// subspace across calls.
inline void forceOuterKrylovParameters( boost::property_tree::ptree&,
                                        const amgcl::solver::fgmres< amgcl::backend::builtin< double > >* )
{
}

template < class SolverType >
class OuterKrylovSolverT : public OuterKrylovSolver {
public:
  typedef amgcl::backend::builtin< double > Backend;
  typedef SolverType                        Solver;

  std::unique_ptr< Solver > solver_;
  int                       n_;

  /// @brief Construct for a fixed problem size n.
  /// @param json_params lgmres::params' own sub-tree, e.g. {"M": 30, "K": 3}, forwarded verbatim;
  /// always_reset is forced to true (see the class comment).
  OuterKrylovSolverT( int n, const char* json_params ) : n_( n )
  {
    boost::property_tree::ptree prm = parseJsonParameters( json_params );
    forceOuterKrylovParameters( prm, static_cast< const Solver* >( nullptr ) );
    typename Solver::params p( prm );
    solver_.reset( new Solver( static_cast< size_t >( n ), p ) );
  }

  /// @brief x <- A^-1 rhs, right-preconditioned by the callback (applyFn/ctx). x holds the initial
  /// guess on input (blockamg's warm-started continuation retry pre-fills it) and the solution on
  /// output; iters/error return lgmres's iteration count and its own relative residual -- of the
  /// equilibrated system As, or, onUnscaledOperator, of A itself (the right preconditioner then has
  /// to map an A-residual to an A-correction).
  void solve( const EquilibratedSystemT& system,
              const double*              rhs,
              double*                    x,
              double                     tol,
              int                        maxiter,
              PyPrecondApplyFn           applyFn,
              void*                      ctx,
              int&                       iters,
              double&                    error,
              bool                       onUnscaledOperator ) override
  {
    system.checkBuilt();
    const int n = system.n_;
    if ( n != n_ ) {
      throw std::runtime_error( "OuterKrylovSolverT::solve(): n=" + std::to_string( n ) + " does not match the size " +
                                std::to_string( n_ ) + " this instance was constructed for." );
    }

    solver_->prm.tol     = tol;
    solver_->prm.maxiter = static_cast< size_t >( maxiter );

    PyKrylovPrecondT precond( applyFn, ctx, n );

    auto rhs_rng = amgcl::make_iterator_range( rhs, rhs + n );
    auto x_rng   = amgcl::make_iterator_range( x, x + n );

    size_t itersOut;
    double errorOut;
    std::tie( itersOut, errorOut ) = ( *solver_ )( system.matrix( onUnscaledOperator ), precond, rhs_rng, x_rng );
    iters                          = static_cast< int >( itersOut );
    error                          = errorOut;
  }
};

/// @brief A new outer Krylov solve of `method` ("lgmres" or "fgmres") for size n, with AMGCL's parameters of that
/// method as JSON; the caller owns it.
inline OuterKrylovSolver* makeOuterKrylovSolver( const std::string& method, int n, const char* json_params )
{
  typedef amgcl::backend::builtin< double > Backend;
  if ( method == "lgmres" )
    return new OuterKrylovSolverT< amgcl::solver::lgmres< Backend > >( n, json_params );
  if ( method == "fgmres" )
    return new OuterKrylovSolverT< amgcl::solver::fgmres< Backend > >( n, json_params );
  throw std::runtime_error( "makeOuterKrylovSolver(): unknown method '" + method + "' (lgmres, fgmres)" );
}

// --- two-grid preconditioners: the p-two-grid (P1) and the rigid-body coarse level -------------------------------
//
// Two preconditioners of one vector field's diagonal block share the two-grid cycle of TwoGridPreconditioner, in
// C++ throughout: the p-multigrid two-grid cycle of edelweissfe/linsolve/blockamg/ptwogrid.py (PTwoGridPreconditionerT,
// its coarse level the P1 corner nodes) and the rigid-body coarse level of
// edelweissfe/linsolve/blockamg/rigidbodycoarselevel.py (RigidBodyCoarseLevelPreconditionerT, its coarse level a
// smoothed aggregation spanned by the rigid-body modes of the aggregates). The cycle: nu fine Chebyshev sweeps, the
// fine residual, restriction through P^T, one AMGCL V-cycle on the Galerkin coarse operator A_c = P^T A P,
// prolongation through P, nu more fine Chebyshev sweeps. Every stage of build() and apply() is OpenMP-threaded;
// apply() runs without any Python in between.
//
// Dirichlet handling: the field's diagonal block arrives *with* its Dirichlet identity rows (a row
// whose only nonzero is its diagonal; stored zeros, which the Dirichlet elimination keeps, do not
// count). The multi-point-constraint condensation fills its slave rows and columns with the identity
// as well (the slaves are recovered from their masters after the solve); such a row is the same row
// to the cycle, so "Dirichlet" below means either.
// The former path extracted the free submatrix A_FF (Dirichlet rows *and* columns removed) and
// smoothed on that. The cycle keeps the full block layout instead and makes the Dirichlet DOFs
// decouple exactly:
//
//   * the fine right-hand side and the fine iterate are zero on Dirichlet DOFs, and a Dirichlet row
//     carries nothing but its diagonal, so every residual r_D = 0 - a_DD * 0 = 0, hence every
//     Chebyshev update leaves x_D = 0. On the free rows, A x = A_FF x_F + A_FD x_D = A_FF x_F,
//     whether or not the upstream Dirichlet elimination also cleared the columns A_FD;
//   * the power iteration starts from a vector that is zero on Dirichlet DOFs, which stays zero by
//     the same argument, so it estimates the spectral radius of A_FF, not of A;
//   * P has nothing in the rows of Dirichlet fine DOFs (the p-two-grid no entries, the rigid-body coarse level zero
//     ones) and nothing in the columns of Dirichlet coarse DOFs (the p-two-grid has none of them), so
//     P^T A P = P_F^T A_FF P_F exactly.
//
// This is what makes a node-wise block storage of the fine operator possible at all: a node may
// be partially constrained, which a free submatrix cannot represent in node-wise blocks, but the
// full layout always can. For the scalar storage it simply saves the free-submatrix extraction. The
// result is the free-submatrix cycle up to floating-point summation order (verified in
// tests/test_ptwogrid.py).
//
// The fine Chebyshev smoother reimplements amgcl::relaxation::chebyshev's polynomial (same degree,
// lower/higher conventions, same recurrence; no Jacobi scaling) rather than wrapping it,
// because AMGCL estimates the spectral radius inside its constructor and offers no way to supply
// one. Here the estimate is either computed by power iteration -- from a deterministic, thread-count
// independent start vector -- or handed in, e.g. the value of a previous build on the same field,
// skipping the 300 sparse matrix-vector products of the estimate altogether (fineSpectralRadius()).
// The start vector is drawn like the one of AMGCL's own power iteration on a single thread (mt19937
// seeded 0, uniform on [-1, 1]), but only for the free DOFs, in order -- AMGCL draws one per row, so the
// two coincide only without Dirichlet DOFs. Unlike AMGCL's (per-thread seeds), the estimate is independent
// of the thread count and reproducible run to run. (AMGCL's Chebyshev default, power_iters = 0, does no power
// iteration at all: it bounds the radius by Gershgorin's theorem.)

/// The fine smoother as the two-grid cycle sees it, independent of the operator's storage format
/// (scalar entries or node-wise B x B blocks) and value type (double or float). Vectors are always plain
/// double arrays of the field's scalar size.
class TwoGridFineSmoother {
public:
  virtual ~TwoGridFineSmoother() {}

  /// @brief One Chebyshev sweep (degree steps) on A x = b, continuing from x -- or, with xIsZero, from
  /// x = 0 without reading x at all (its content is then irrelevant).
  virtual void smooth( const double* b, double* x, bool xIsZero ) = 0;

  /// @brief r <- b - A x
  virtual void residual( const double* b, const double* x, double* r ) const = 0;
};

/// @brief r <- b - A x for the fine operator, with AMGCL's own kernel: AMGCL reinterprets the scalar
/// vectors as node-wise ones for a block matrix (amgcl/backend/detail/matrix_ops.hpp), and every sum is
/// accumulated in the vectors' type, double, whatever the matrix's -- float scalar entries by the
/// usual promotion (`sum += float * double`), float 3x3 blocks by the mixed block product defined with
/// builtin_float_matrices above. A float operator therefore differs from the double one only by the
/// rounding of its stored entries.
template < typename MatrixValueType >
inline void twoGridFineResidual( const amgcl::backend::crs< MatrixValueType, int, int >& A,
                                 const double*                                           b,
                                 const double*                                           x,
                                 double*                                                 r )
{
  const size_t n      = A.nrows * amgcl::math::static_rows< MatrixValueType >::value;
  auto         bRange = amgcl::make_iterator_range( b, b + n );
  auto         xRange = amgcl::make_iterator_range( x, x + n );
  auto         rRange = amgcl::make_iterator_range( r, r + n );
  amgcl::backend::residual( bRange, A, xRange, rRange );
}

/// The fine operator in one storage format (scalar entries, or node-wise BxB blocks) and one value
/// type, with the Chebyshev smoother acting on it. ValueType is the double entry type (double, or a
/// double B x B block); MatrixValueType is how the operator is stored for the smoother and the residual:
/// ValueType itself, or its float counterpart -- half the bytes per entry, the dominant traffic of
/// the cycle. The spectral-radius estimate only ever runs on a double operator: a float level is made
/// from a double one (makeFloatCopy()), which has estimated or re-checked the radius before.
template < typename ValueType, typename MatrixValueType = ValueType >
class TwoGridFineLevelT : public TwoGridFineSmoother {
public:
  /// @brief The fine operator's storage: CSR with 32-bit indices and MatrixValueType entries.
  typedef amgcl::backend::crs< MatrixValueType, int, int > Matrix;
  /// @brief The vector entry type matching ValueType (double, or a double B-vector).
  typedef typename amgcl::math::rhs_of< ValueType >::type RhsType;
  /// @brief The scalar type of ValueType (double).
  typedef typename amgcl::math::scalar_of< ValueType >::type ScalarType;
  /// @brief Scalar rows per stored row: 1 for scalar entries, B for BxB blocks.
  static const int BlockSize = amgcl::math::static_rows< ValueType >::value;

  /// @brief The fine operator the smoother and the residual act on.
  std::shared_ptr< Matrix > A_;
  /// @brief Work vectors of the Chebyshev iteration (residual r and search direction p), one entry per scalar row.
  std::vector< double > r_, p_;
  /// @brief Chebyshev polynomial degree, i.e. matrix products per smoothing sweep.
  unsigned degree_ = 0;
  /// @brief Centre d and half-width c of the Chebyshev interval [d - c, d + c] (setPolynomial()).
  double d_ = 0.0, c_ = 0.0;

  /// @brief Take over the operator A (already in this storage format).
  void setOperator( std::shared_ptr< Matrix > A )
  {
    A_ = A;
    r_.assign( A_->nrows * BlockSize, 0.0 );
    p_.assign( A_->nrows * BlockSize, 0.0 );
  }

  /// @brief The same level with the operator's entries rounded to float (same structure, same
  /// polynomial, same double inverse diagonal), in one parallel pass.
  template < typename FloatValueType >
  std::unique_ptr< TwoGridFineLevelT< ValueType, FloatValueType > > makeFloatCopy() const
  {
    std::unique_ptr< TwoGridFineLevelT< ValueType, FloatValueType > > level(
      new TwoGridFineLevelT< ValueType, FloatValueType >() );
    level->A_      = convertedCopy< FloatValueType >( *A_ );
    level->degree_ = degree_;
    level->d_      = d_;
    level->c_      = c_;
    level->r_.assign( r_.size(), 0.0 );
    level->p_.assign( p_.size(), 0.0 );
    return level;
  }

  /// @brief Set up the Chebyshev polynomial for the given spectral radius.
  /// @param lower, higher amgcl::relaxation::chebyshev's own `float` parameters, kept as `float` so
  /// the ellipse bounds round exactly as AMGCL's do.
  void setPolynomial( unsigned degree, float lower, float higher, double spectralRadius )
  {
    degree_   = degree;
    double hi = spectralRadius;
    double lo = hi * lower;
    hi *= higher;
    d_ = 0.5 * ( hi + lo );
    c_ = 0.5 * ( hi - lo );
  }

  /// @brief Power iteration for the spectral radius of A: the loop of
  /// amgcl::backend::spectral_radius (estimate sum_i |(A b)_i b_i| per scalar component, b normalized),
  /// with two differences that make the estimate a deterministic function of the matrix alone: the
  /// start vector is the caller's instead of per-thread random numbers, and the reductions run over a
  /// fixed partition of the rows, summed in order, instead of whichever order the threads finish in.
  /// Accumulating per scalar component also makes a block operator's estimate the scalar one's, up to
  /// summation order. The last normalized iterate -- the estimate of the dominant eigenvector -- is
  /// returned in dominantVector, as the start of a later, short re-check of the estimate.
  double estimateSpectralRadius( int                          powerIters,
                                 const std::vector< double >& startVector,
                                 std::vector< double >&       dominantVector ) const
  {
    const Matrix&   A       = *A_;
    const ptrdiff_t n       = static_cast< ptrdiff_t >( A.nrows );
    const ptrdiff_t nChunks = std::min< ptrdiff_t >( n, 4096 );
    const ptrdiff_t chunk   = ( n + nChunks - 1 ) / nChunks;

    std::vector< double > b0Scalar( startVector );
    double                startNormSquared = 0.0;
    for ( size_t i = 0; i < b0Scalar.size(); ++i )
      startNormSquared += b0Scalar[i] * b0Scalar[i];
    const double startScale = 1.0 / std::sqrt( startNormSquared );
    for ( size_t i = 0; i < b0Scalar.size(); ++i )
      b0Scalar[i] = startScale * b0Scalar[i];

    std::vector< double > b1Scalar( b0Scalar.size(), 0.0 );
    RhsType*              b0 = reinterpret_cast< RhsType* >( b0Scalar.data() );
    RhsType*              b1 = reinterpret_cast< RhsType* >( b1Scalar.data() );
    std::vector< double > chunkNorm( nChunks ), chunkRadius( nChunks );

    double radius = 0.0;
    for ( int iter = 0; iter < powerIters; ) {
#pragma omp parallel for schedule( static )
      for ( ptrdiff_t k = 0; k < nChunks; ++k ) {
        double localNorm = 0.0, localRadius = 0.0;
        for ( ptrdiff_t i = k * chunk, e = std::min( n, ( k + 1 ) * chunk ); i < e; ++i ) {
          RhsType s = amgcl::math::zero< RhsType >();
          for ( int j = A.ptr[i], end = A.ptr[i + 1]; j < end; ++j )
            s += A.val[j] * b0[A.col[j]];
          const double* sScalar  = reinterpret_cast< const double* >( &s );
          const double* b0Scalar = reinterpret_cast< const double* >( &b0[i] );
          for ( int component = 0; component < BlockSize; ++component ) {
            localNorm += sScalar[component] * sScalar[component];
            localRadius += std::abs( sScalar[component] * b0Scalar[component] );
          }
          b1[i] = s;
        }
        chunkNorm[k]   = localNorm;
        chunkRadius[k] = localRadius;
      }
      double b1NormSquared = 0.0;
      radius               = 0.0;
      for ( ptrdiff_t k = 0; k < nChunks; ++k ) {
        b1NormSquared += chunkNorm[k];
        radius += chunkRadius[k];
      }
      const double b1Scale = 1.0 / std::sqrt( b1NormSquared );
#pragma omp parallel for
      for ( ptrdiff_t i = 0; i < n; ++i )
        b0[i] = b1Scale * b1[i];
      ++iter;
    }
    dominantVector.swap( b0Scalar );
    return radius < 0 ? 2.0 : radius;
  }

  /// @brief `degree_` Chebyshev steps on A x = b, continuing from the given x -- the recurrence of
  /// amgcl::relaxation::chebyshev::solve(), step for step, with two economies that leave the result
  /// bitwise unchanged:
  ///   * from x = 0 (xIsZero, the pre-smoothing), the first residual is b itself -- b - A 0 is b
  ///     exactly for a finite A -- so its sparse product is skipped, and x = 0 + p is written as p
  ///     (identical, up to the sign of a zero entry);
  ///   * AMGCL's two vector updates, p <- alpha r + beta p and x <- 1 p + 1 x, run as one pass with
  ///     the same arithmetic per entry.
  void smooth( const double* b, double* x, bool xIsZero ) override
  {
    const ptrdiff_t n     = static_cast< ptrdiff_t >( r_.size() );
    double          alpha = 0.0, beta = 0.0;
    for ( unsigned k = 0; k < degree_; ++k ) {
      const bool    fromZero = xIsZero && k == 0;
      const double* residual = fromZero ? b : r_.data();
      if ( !fromZero )
        twoGridFineResidual( *A_, b, x, r_.data() );
      if ( k == 0 ) {
        alpha = 1.0 / d_;
        beta  = 0.0;
      }
      else if ( k == 1 ) {
        alpha = 2 * d_ * ( 1.0 / ( 2 * d_ * d_ - c_ * c_ ) );
        beta  = alpha * d_ - 1.0;
      }
      else {
        alpha = 1.0 / ( d_ - 0.25 * alpha * c_ * c_ );
        beta  = alpha * d_ - 1.0;
      }
      double* p = p_.data();
      if ( fromZero ) {
#pragma omp parallel for
        for ( ptrdiff_t i = 0; i < n; ++i ) {
          p[i] = alpha * residual[i];
          x[i] = p[i];
        }
      }
      else if ( beta == 0.0 ) {
#pragma omp parallel for
        for ( ptrdiff_t i = 0; i < n; ++i ) {
          p[i] = alpha * residual[i];
          x[i] = 1.0 * p[i] + 1.0 * x[i];
        }
      }
      else {
#pragma omp parallel for
        for ( ptrdiff_t i = 0; i < n; ++i ) {
          p[i] = alpha * residual[i] + beta * p[i];
          x[i] = 1.0 * p[i] + 1.0 * x[i];
        }
      }
    }
  }

  /// @brief res <- b - A x
  void residual( const double* b, const double* x, double* res ) const override
  {
    twoGridFineResidual( *A_, b, x, res );
  }
};

/// @brief The fine level stored in float, made from the double one with its polynomial
/// (TwoGridFineLevelT::makeFloatCopy()): scalar entries and 3x3 blocks, whose float products accumulate in double (the
/// usual promotion of a float scalar times a double, and the mixed block product defined with builtin_float_matrices).
inline std::unique_ptr< TwoGridFineSmoother > floatStoredCopy( const TwoGridFineLevelT< double >& level )
{
  return level.makeFloatCopy< float >();
}

/// @copydoc floatStoredCopy(const TwoGridFineLevelT<double>&)
inline std::unique_ptr< TwoGridFineSmoother > floatStoredCopy(
  const TwoGridFineLevelT< amgcl::static_matrix< double, 3, 3 > >& level )
{
  return level.makeFloatCopy< amgcl::static_matrix< float, 3, 3 > >();
}

/// @brief 2x2 blocks are stored in double only (like the AMGCL backends: LinearSolverBlock2 has no float variant).
/// @throws std::runtime_error always
inline std::unique_ptr< TwoGridFineSmoother > floatStoredCopy(
  const TwoGridFineLevelT< amgcl::static_matrix< double, 2, 2 > >& )
{
  throw std::runtime_error( "two-grid cycle: 2x2 blocks are stored in double only" );
}

/// The two-grid cycle of one vector field's equilibrated diagonal block A, shared by the p-two-grid
/// (PTwoGridPreconditionerT) and the rigid-body coarse level (RigidBodyCoarseLevelPreconditionerT), see the comment
/// block above: Chebyshev smoothing on A and the coarse-grid correction
///
///   x <- x + P A_c^-1 P^T (r - A x),   A_c = P^T A P, the Galerkin coarse operator,
///
/// with A_c^-1 one V-cycle of an AMGCL hierarchy, nuPre smoothing sweeps before it and nuPost after. A derived class
/// builds the prolongation P, A_c and the near null-space of A_c, and applies P and P^T (prolongateCorrection(),
/// restrictResidual()); this class does everything else: the Dirichlet rows, the fine level and its spectral radius,
/// the coarse hierarchy and the cycle (apply()).
class TwoGridPreconditioner {
public:
  typedef amgcl::backend::builtin< double >                    CoarseBackend;
  typedef amgcl::runtime::preconditioner< CoarseBackend >      CoarseSolver;
  typedef amgcl::backend::builtin< float >                     CoarseBackendFloat;
  typedef amgcl::runtime::preconditioner< CoarseBackendFloat > CoarseSolverFloat;
  typedef amgcl::backend::crs< double, int, int >              ScalarMatrix;
  typedef std::chrono::steady_clock                            Clock;

  virtual ~TwoGridPreconditioner() {}

  /// @brief The parameter tree given to the constructor, where the derived class reads its parameters from.
  boost::property_tree::ptree prm;

  /// @brief The fine level, stored as fineValueType_ asks; a float one is made from the double one, which only lives
  /// during the build (for the spectral radius).
  std::unique_ptr< TwoGridFineSmoother > fine_;
  /// @brief Per DOF, whether its row is a Dirichlet row (dirichletRowMask()).
  std::vector< char > isDirichlet_;
  /// @brief The Dirichlet rows.
  std::vector< int > dirichletRows_;
  /// @brief The diagonal entry of each of dirichletRows_: its exact local solve is x_i = r_i / a_ii.
  std::vector< double > dirichletDiagonal_;

  /// @brief The coarse AMGCL hierarchy, in exactly one of the two value types (coarseValueType_); none without a
  /// coarse level.
  std::unique_ptr< CoarseSolver > coarse_;
  /// @copydoc coarse_
  std::unique_ptr< CoarseSolverFloat > coarseFloat_;
  /// @brief The coarse right-hand side and correction of a float hierarchy.
  std::vector< float > coarseRhsFloat_, coarseXFloat_;
  /// @brief The near null-space of the coarse hierarchy, rows x columns row-major, kept alive here: the parameter tree
  /// of the hierarchy holds a pointer to it.
  std::vector< double > coarseNullspace_;
  /// @brief The columns of coarseNullspace_.
  int coarseNullspaceColumns_ = 0;

  /// @brief The size of the field's block.
  int n_ = 0;
  /// @brief The number of coarse DOFs, the size of the Galerkin operator; 0 for no coarse level (apply() is then the
  /// smoother alone).
  int nCoarse_ = 0;
  /// @brief True after a build() of a field without a free DOF (empty, or entirely Dirichlet-constrained): apply() is
  /// then the exact (diagonal) inverse.
  bool trivial_ = false;
  /// @brief True once build() has completed; a build that throws leaves it false, so that apply() raises instead of
  /// using a half-built preconditioner.
  bool built_ = false;

  /// @brief The fine Chebyshev sweeps before and after the coarse-grid correction.
  int nuPre_ = 1, nuPost_ = 1;
  /// @brief The fine Chebyshev degree.
  unsigned fineDegree_ = 5;
  /// @brief The fine Chebyshev interval relative to the spectral radius, as AMGCL's chebyshev float parameters.
  float fineLower_ = 0.01f, fineHigher_ = 1.1f;
  /// @brief The value types of the fine level and of the coarse hierarchy: "double" or "float".
  std::string fineValueType_ = "double", coarseValueType_ = "double";
  /// @brief The power iterations of the full estimate of the fine spectral radius and of the re-check of a supplied
  /// one, and the relative rise the re-check tolerates.
  int    finePowerIters_ = 300, fineRecheckIters_ = 15;
  double fineRecheckTolerance_ = 0.01;

  /// @brief The fine spectral radius the Chebyshev smoother was built with (0 for a trivial field).
  double fineSpectralRadius_ = 0.0;
  /// @brief The last iterate of the power iteration: the estimate of the dominant eigenvector, the start of a later
  /// re-check (empty if the radius was used as supplied).
  std::vector< double > fineDominantVector_;
  /// @brief How the last build arrived at its fine spectral radius: 0 estimated in full, 1 supplied and kept, 2
  /// supplied but raised by the re-check, 3 supplied but re-estimated in full by the re-check.
  int fineSpectralRadiusSource_ = 0;
  /// @brief Work vectors of the cycle.
  std::vector< double > rhs_, res_, coarseRhs_, coarseX_;

  /// @brief The wall time [s] of every stage of the last build(), in order, read by Python for the performance report.
  std::vector< std::pair< std::string, double > > setupSeconds_;
  /// @brief The cumulative wall time [s] of apply(): fine smoothing (with the residual), and coarse-grid correction
  /// (with the restriction and the prolongation).
  double applyFineSeconds_ = 0.0, applyCoarseSeconds_ = 0.0;

  /// @brief Wall time [s] since start.
  static double secondsSince( const Clock::time_point& start )
  {
    return std::chrono::duration< double >( Clock::now() - start ).count();
  }

  /// Dirichlet rows: a nonzero diagonal and no nonzero off-diagonal entry -- explicitly stored zeros,
  /// which the Dirichlet elimination (solvers/base/dirichlet.pyx) keeps, do not count. The exact
  /// decoupling of the cycle relies on nothing but that (a stored zero contributes 0 * x).
  static void dirichletRowMask( int n, const int* ptr, const int* col, const double* val, char* isDirichlet )
  {
#pragma omp parallel for
    for ( ptrdiff_t i = 0; i < n; ++i ) {
      bool diagonal = false, offDiagonal = false;
      for ( int j = ptr[i]; j < ptr[i + 1]; ++j ) {
        if ( val[j] == 0.0 )
          continue;
        if ( col[j] == i )
          diagonal = true;
        else
          offDiagonal = true;
      }
      isDirichlet[i] = diagonal && !offDiagonal;
    }
  }

  /// @brief x <- M^-1 r: one two-grid cycle on the free DOFs -- nuPre fine sweeps, coarse-grid
  /// correction, nuPost fine sweeps; Dirichlet DOFs are solved exactly, x_i = r_i / a_ii. The cycle's fine iterate
  /// is x itself: written from x = 0 without reading it (by the first pre-sweep, or, with no
  /// pre-sweeps, by the prolongated coarse correction itself), it stays zero on Dirichlet DOFs (zero
  /// right-hand side there, nothing of P) until they are set to r_i / a_ii last. Without pre-sweeps the
  /// fine residual of x = 0 is the right-hand side itself, so no product is needed for it.
  void apply( int n, const double* r, double* x )
  {
    checkBuilt();
    if ( n != n_ ) {
      throw std::runtime_error( "apply(): n=" + std::to_string( n ) + " does not match the built size " +
                                std::to_string( n_ ) );
    }
    if ( trivial_ ) {
      solveDirichletRows( r, x );
      return;
    }
    Clock::time_point start = Clock::now();
#pragma omp parallel for
    for ( ptrdiff_t i = 0; i < n; ++i )
      rhs_[i] = isDirichlet_[i] ? 0.0 : r[i];
    // whether x still holds nothing of this cycle: its next write then overwrites it instead of adding to it
    bool xIsZero = true;
    if ( nuPre_ > 0 ) {
      smoothFine( x, nuPre_, /*xIsZero=*/true );
      fine_->residual( rhs_.data(), x, res_.data() );
      xIsZero = false;
    }
    const std::vector< double >& fineResidual = nuPre_ > 0 ? res_ : rhs_;
    applyFineSeconds_ += secondsSince( start );

    if ( nCoarse_ > 0 ) {
      // the coarse AMG cycle overwrites its x itself (amgcl::amg::apply clears it, or copies rhs)
      start = Clock::now();
      restrictResidual( fineResidual.data(), coarseRhs_.data() );
      if ( coarseFloat_ ) {
        const ptrdiff_t nc = nCoarse_;
#pragma omp parallel for
        for ( ptrdiff_t i = 0; i < nc; ++i )
          coarseRhsFloat_[i] = static_cast< float >( coarseRhs_[i] );
        auto rhsFloat = amgcl::make_iterator_range( coarseRhsFloat_.data(), coarseRhsFloat_.data() + nc );
        auto xFloat   = amgcl::make_iterator_range( coarseXFloat_.data(), coarseXFloat_.data() + nc );
        coarseFloat_->apply( rhsFloat, xFloat );
#pragma omp parallel for
        for ( ptrdiff_t i = 0; i < nc; ++i )
          coarseX_[i] = coarseXFloat_[i];
      }
      else {
        auto coarseRhsRange = amgcl::make_iterator_range( coarseRhs_.data(), coarseRhs_.data() + nCoarse_ );
        auto coarseXRange   = amgcl::make_iterator_range( coarseX_.data(), coarseX_.data() + nCoarse_ );
        coarse_->apply( coarseRhsRange, coarseXRange );
      }
      // x <- x + P e after pre-sweeps, x <- P e (overwriting) without
      prolongateCorrection( coarseX_.data(), x, /*addToFine=*/!xIsZero );
      xIsZero = false;
      applyCoarseSeconds_ += secondsSince( start );
    }

    start = Clock::now();
    if ( xIsZero && nuPost_ == 0 ) // no sweep and no coarse level wrote x: the cycle is zero on the free DOFs
      std::fill( x, x + n, 0.0 );
    smoothFine( x, nuPost_, xIsZero );
    solveDirichletRows( r, x );
    applyFineSeconds_ += secondsSince( start );
  }

  /// @brief x_i <- r_i / a_ii on the Dirichlet rows, their exact local solve (bitwise r_i for the unit
  /// diagonal of an equilibrated block).
  void solveDirichletRows( const double* r, double* x ) const
  {
    for ( size_t k = 0; k < dirichletRows_.size(); ++k )
      x[dirichletRows_[k]] = r[dirichletRows_[k]] / dirichletDiagonal_[k];
  }

  /// @brief Throw unless the last build() has completed.
  void checkBuilt() const
  {
    if ( !built_ ) {
      throw std::runtime_error( name_ + ": no preconditioner built -- call build() first (the last one failed?)" );
    }
  }

  /// @brief The fine spectral radius the Chebyshev smoother was built with (0 for a trivial field).
  double fineSpectralRadius() const { return fineSpectralRadius_; }
  /// @brief How the last build arrived at it, see fineSpectralRadiusSource_.
  int fineSpectralRadiusSource() const { return fineSpectralRadiusSource_; }
  /// @brief The length of fineDominantVector_ (0 if the radius was used as supplied).
  int fineDominantVectorSize() const { return static_cast< int >( fineDominantVector_.size() ); }
  /// @brief Copy fineDominantVector_ into out (fineDominantVectorSize() entries).
  void copyFineDominantVector( double* out ) const
  {
    std::copy( fineDominantVector_.begin(), fineDominantVector_.end(), out );
  }
  /// @brief The number of coarse DOFs, the size of the Galerkin operator.
  int nCoarse() const { return nCoarse_; }

  /// @brief The coarse level's AMGCL hierarchy report.
  virtual std::string report() const
  {
    checkBuilt();
    if ( trivial_ )
      return name_ + ": no free DOF, the preconditioner is the identity\n";
    return coarseHierarchyReport();
  }

protected:
  /// @brief The preconditioner's name in messages, e.g. "ptwogrid".
  std::string name_;

  /// @param jsonParameters the parameters, kept as prm for the derived class
  /// @param name the preconditioner's name in messages
  TwoGridPreconditioner( const char* jsonParameters, const std::string& name )
    : prm( parseJsonParameters( jsonParameters ) ), name_( name )
  {
  }

  /// @brief How the derived class's parameters name the parameters of the cycle, for the messages of
  /// checkCycleParameters().
  struct CycleParameterNames {
    std::string nuPre, nuPost, fineDegree, fineLower, fineHigher, finePowerIters, fineValueType, coarseValueType;
  };

  /// @brief Check the parameters of the cycle, which the derived class's constructor has set -- and fineDegree, which
  /// it has read as a signed number -- so that a bad value fails before build() has spent anything on the
  /// prolongation, the Galerkin product or the coarse hierarchy.
  /// @throws std::runtime_error naming the parameter as `names` spell it
  void checkCycleParameters( int fineDegree, const CycleParameterNames& names )
  {
    if ( nuPre_ < 0 || nuPost_ < 0 ) {
      throw std::runtime_error( name_ + ": " + names.nuPre + " and " + names.nuPost + " must not be negative, got " +
                                std::to_string( nuPre_ ) + " and " + std::to_string( nuPost_ ) );
    }
    if ( fineDegree < 1 ) {
      throw std::runtime_error( name_ + ": " + names.fineDegree + " must be at least 1, got " +
                                std::to_string( fineDegree ) );
    }
    fineDegree_ = static_cast< unsigned >( fineDegree );
    if ( fineValueType_ != "double" && fineValueType_ != "float" ) {
      throw std::runtime_error( name_ + ": " + names.fineValueType + " must be double or float, got " +
                                fineValueType_ );
    }
    if ( coarseValueType_ != "double" && coarseValueType_ != "float" ) {
      throw std::runtime_error( name_ + ": " + names.coarseValueType + " must be double or float, got " +
                                coarseValueType_ );
    }
    // the Chebyshev interval is [fineLower, fineHigher] times the spectral radius: it must start above zero
    // and reach at least the radius itself, or the smoother amplifies the highest modes
    if ( !( fineLower_ > 0.0f && fineLower_ < 1.0f ) || !( fineHigher_ >= 1.0f ) ) {
      throw std::runtime_error( name_ + ": needs 0 < " + names.fineLower + " < 1 <= " + names.fineHigher + ", got " +
                                std::to_string( fineLower_ ) + " and " + std::to_string( fineHigher_ ) );
    }
    if ( finePowerIters_ < 1 ) {
      throw std::runtime_error( name_ + ": " + names.finePowerIters +
                                " must be at least 1 (the fine smoother estimates the spectral radius by power "
                                "iteration), got " +
                                std::to_string( finePowerIters_ ) );
    }
    if ( fineRecheckIters_ < 0 || !( fineRecheckTolerance_ >= 0.0 ) ) {
      throw std::runtime_error( name_ + ": needs fineRecheckIters >= 0 and fineRecheckTolerance >= 0" );
    }
  }

  /// @brief Start a build for the field's block (n, ptr, col, val): find its Dirichlet rows and reset everything a
  /// previous build left. A field without a free DOF -- empty, or every DOF Dirichlet-constrained -- has nothing to
  /// smooth or to coarsen (and no coarse operator to build a hierarchy on): its exact inverse is diagonal, which
  /// apply() then is, and its build is complete.
  /// @return whether the field has no free DOF
  bool beginBuild( int n, const int* ptr, const int* col, const double* val )
  {
    built_ = false;
    n_     = n;
    setupSeconds_.clear();

    isDirichlet_.assign( n, 0 );
    if ( n > 0 )
      dirichletRowMask( n, ptr, col, val, isDirichlet_.data() );
    dirichletRows_.clear();
    dirichletDiagonal_.clear();
    for ( int i = 0; i < n; ++i ) {
      if ( isDirichlet_[i] ) {
        dirichletRows_.push_back( i );
        dirichletDiagonal_.push_back( rowDiagonal( ptr, col, val, i ) );
      }
    }

    trivial_ = std::find( isDirichlet_.begin(), isDirichlet_.end(), 0 ) == isDirichlet_.end();
    coarse_.reset();
    coarseFloat_.reset();
    coarseNullspace_.clear();
    coarseNullspaceColumns_ = 0;
    fine_.reset();
    fineSpectralRadius_       = 0.0;
    fineSpectralRadiusSource_ = 0;
    fineDominantVector_.clear();
    nCoarse_ = 0;
    if ( trivial_ )
      built_ = true;
    return trivial_;
  }

  /// @brief Every free row needs a nonzero diagonal (blockamg's equilibration makes it 1): without one -- a zero block
  /// -- the coarse operator is singular and the Chebyshev bounds, relative to the fine spectral radius, divide by
  /// zero. Checked before anything is built.
  /// @throws std::runtime_error naming a free row without a nonzero diagonal entry
  void checkFreeRowsHaveDiagonal( int n, const int* ptr, const int* col, const double* val ) const
  {
    ptrdiff_t rowWithoutDiagonal = -1;
#pragma omp parallel for
    for ( ptrdiff_t i = 0; i < n; ++i ) {
      if ( isDirichlet_[i] )
        continue;
      if ( rowDiagonal( ptr, col, val, i ) == 0.0 ) {
#pragma omp critical
        rowWithoutDiagonal = i;
      }
    }
    if ( rowWithoutDiagonal >= 0 ) {
      throw std::runtime_error( name_ + ": free row " + std::to_string( rowWithoutDiagonal ) +
                                " (one of) has no nonzero diagonal entry -- is the field's block zero?" );
    }
  }

  /// @brief Add seconds to the build stage name (appended if it is new).
  void addSetupSeconds( const std::string& name, double seconds )
  {
    for ( auto& stage : setupSeconds_ ) {
      if ( stage.first == name ) {
        stage.second += seconds;
        return;
      }
    }
    setupSeconds_.push_back( std::make_pair( name, seconds ) );
  }

  /// @brief Build the coarse AMGCL hierarchy on the Galerkin coarse operator, in coarseValueType_: float builds and
  /// runs the whole coarse hierarchy -- including its own Chebyshev smoother's power iteration -- in single
  /// precision, AMGCL's own backendPrecision float; the coarse right-hand side and correction are converted at the
  /// boundary.
  /// @param Ac the Galerkin coarse operator, nCoarse_ rows
  /// @param coarsePrm the hierarchy's AMGCL preconditioner tree
  /// @param nullspace, nullspaceRows, nullspaceCols its near null-space, nullspaceRows (= nCoarse_) x nullspaceCols,
  /// row-major; nullspaceCols == 0 for none
  void buildCoarseHierarchy( const ScalarMatrix&         Ac,
                             boost::property_tree::ptree coarsePrm,
                             const double*               nullspace,
                             int                         nullspaceRows,
                             int                         nullspaceCols )
  {
    // the near null-space as a raw pointer in the tree, as for LinearSolverT::set_nullspace
    if ( nullspaceCols > 0 ) {
      if ( nullspaceRows != nCoarse_ ) {
        throw std::runtime_error( name_ + ": the coarse near null-space has " + std::to_string( nullspaceRows ) +
                                  " rows, but the coarse level has " + std::to_string( nCoarse_ ) + " DOFs." );
      }
      coarseNullspace_.assign( nullspace, nullspace + static_cast< size_t >( nullspaceRows ) * nullspaceCols );
      coarseNullspaceColumns_ = nullspaceCols;
      coarsePrm.put( "coarsening.nullspace.cols", nullspaceCols );
      coarsePrm.put( "coarsening.nullspace.rows", nullspaceRows );
      coarsePrm.put( "coarsening.nullspace.B", coarseNullspace_.data() );
    }
    if ( coarseValueType_ == "float" ) {
      coarseFloat_.reset( new CoarseSolverFloat( Ac, coarsePrm ) );
      coarseRhsFloat_.assign( nCoarse_, 0.0f );
      coarseXFloat_.assign( nCoarse_, 0.0f );
    }
    else
      coarse_.reset( new CoarseSolver( Ac, coarsePrm ) );
  }

  /// @brief Set up the fine level on the operator A, of ValueType entries (double, or a double B x B block), with the
  /// spectral radius of build()'s `spectralRadius` (see chooseFineSpectralRadius()), stored in fineValueType_.
  template < typename ValueType >
  void buildFineLevel( std::shared_ptr< typename TwoGridFineLevelT< ValueType >::Matrix > A,
                       double                                                             suppliedRadius,
                       const double*                                                      dominantVector,
                       int                                                                dominantVectorSize )
  {
    std::unique_ptr< TwoGridFineLevelT< ValueType > > level( new TwoGridFineLevelT< ValueType >() );
    level->setOperator( A );
    fineSpectralRadius_ = chooseFineSpectralRadius( *level, suppliedRadius, dominantVector, dominantVectorSize );
    if ( !( fineSpectralRadius_ > 0.0 ) || !std::isfinite( fineSpectralRadius_ ) ) {
      // the Chebyshev bounds are relative to it: a zero (or non-finite) radius -- a zero block, which
      // blockamg's equilibration already refuses -- would divide by zero in the polynomial
      throw std::runtime_error( name_ + ": the fine operator has no positive, finite spectral radius (got " +
                                std::to_string( fineSpectralRadius_ ) + ") -- is the field's block zero?" );
    }
    level->setPolynomial( fineDegree_, fineLower_, fineHigher_, fineSpectralRadius_ );
    if ( fineValueType_ == "float" )
      fine_ = floatStoredCopy( *level );
    else
      fine_ = std::move( level );
  }

  /// @brief Complete a build: the work vectors of the cycle.
  void finishBuild()
  {
    rhs_.assign( n_, 0.0 );
    res_.assign( n_, 0.0 );
    coarseRhs_.assign( nCoarse_, 0.0 );
    coarseX_.assign( nCoarse_, 0.0 );
    built_ = true;
  }

  /// @brief The report of the coarse AMGCL hierarchy.
  std::string coarseHierarchyReport() const
  {
    std::ostringstream oss;
    if ( coarseFloat_ )
      oss << *coarseFloat_;
    else if ( coarse_ )
      oss << *coarse_;
    return oss.str();
  }

  /// @brief coarse <- P^T fine: the restriction of a fine residual.
  virtual void restrictResidual( const double* fine, double* coarse ) const = 0;

  /// @brief fine <- P coarse, or fine <- fine + P coarse with addToFine: the prolongation of a coarse correction.
  virtual void prolongateCorrection( const double* coarse, double* fine, bool addToFine ) const = 0;

private:
  /// @brief `sweeps` fine sweeps; the first one from x = 0 if xIsZero
  void smoothFine( double* x, int sweeps, bool xIsZero )
  {
    for ( int sweep = 0; sweep < sweeps; ++sweep )
      fine_->smooth( rhs_.data(), x, xIsZero && sweep == 0 );
  }

  /// @brief The full estimate of the spectral radius of the fine level, from the deterministic start vector: mt19937
  /// seeded 0, uniform on [-1, 1], one draw per free DOF in order (AMGCL's single-thread draw, restricted to the free
  /// DOFs); the final iterate goes to fineDominantVector_.
  template < class Level >
  double estimateFineSpectralRadiusInFull( const Level& level )
  {
    std::vector< double >                    startVector( n_, 0.0 );
    std::mt19937                             rng( 0 );
    std::uniform_real_distribution< double > rnd( -1, 1 );
    for ( int i = 0; i < n_; ++i )
      if ( !isDirichlet_[i] )
        startVector[i] = rnd( rng );
    return level.estimateSpectralRadius( finePowerIters_, startVector, fineDominantVector_ );
  }

  /// @brief The fine spectral radius for this build: estimated in full if suppliedRadius is not positive; else the
  /// supplied one, re-checked first, with a dominantVector (n entries, the previous build's fineDominantVector()), by
  /// fineRecheckIters power iterations from that vector, which cost a few sparse products instead of the full
  /// estimate's finePowerIters: a re-check estimate up to fineRecheckTolerance (1%) above the supplied radius keeps
  /// it, one above that but within the smoother's `fineHigher` margin replaces it, and one beyond that margin -- the
  /// operator stiffened more than the margin covers -- triggers the full estimate. Without a dominantVector the radius
  /// is used as supplied.
  template < class Level >
  double chooseFineSpectralRadius( const Level&  level,
                                   double        suppliedRadius,
                                   const double* dominantVector,
                                   int           dominantVectorSize )
  {
    if ( suppliedRadius <= 0.0 ) {
      fineSpectralRadiusSource_ = 0;
      return estimateFineSpectralRadiusInFull( level );
    }
    if ( dominantVector == nullptr || fineRecheckIters_ <= 0 ) {
      fineDominantVector_.clear();
      fineSpectralRadiusSource_ = 1;
      return suppliedRadius;
    }
    // a dominant vector of a different layout, or one living entirely on DOFs that are Dirichlet
    // now, cannot be continued from -- estimate in full
    std::vector< double > startVector( n_, 0.0 );
    double                startNormSquared = 0.0;
    if ( dominantVectorSize == n_ ) {
      for ( int i = 0; i < n_; ++i ) {
        startVector[i] = isDirichlet_[i] ? 0.0 : dominantVector[i];
        startNormSquared += startVector[i] * startVector[i];
      }
    }
    if ( startNormSquared == 0.0 ) {
      fineSpectralRadiusSource_ = 3;
      return estimateFineSpectralRadiusInFull( level );
    }
    const double recheck = level.estimateSpectralRadius( fineRecheckIters_, startVector, fineDominantVector_ );
    // The full estimate itself is converged only to some 1e-5 relative after its 300 iterations, so
    // continuing it on an unchanged operator still creeps up a little; a rise below
    // fineRecheckTolerance (1%, well inside the 10% margin of `higher`) keeps the supplied radius, so
    // that an unchanged operator gives the same smoother as before.
    if ( recheck <= suppliedRadius * ( 1.0 + fineRecheckTolerance_ ) ) {
      fineSpectralRadiusSource_ = 1;
      return suppliedRadius;
    }
    if ( recheck <= suppliedRadius * fineHigher_ ) {
      fineSpectralRadiusSource_ = 2;
      return recheck;
    }
    fineSpectralRadiusSource_ = 3;
    return estimateFineSpectralRadiusInFull( level );
  }
};

/// @brief The p-two-grid preconditioner for one vector field's diagonal block: the two-grid cycle of
/// TwoGridPreconditioner with the P1 corner nodes as its coarse level -- P the identity on the corners and the
/// average of the two edge-endpoint corners on every midside node. build() once per hierarchy refresh, apply() once
/// per outer Krylov iteration.
class PTwoGridPreconditionerT : public TwoGridPreconditioner {
public:
  typedef amgcl::static_matrix< double, 3, 3 > Block3;

  /// @brief The prolongation and the restriction P^T: scalar, on the free DOFs.
  std::shared_ptr< ScalarMatrix > P_, PT_;
  /// @brief The storage of the fine operator: 1 scalar entries, 3 node-wise 3x3 blocks.
  int fineBlockSize_ = 1;

  /// @brief Parse the flat parameter tree: nuPre, nuPost (nu for both), fineDegree, fineLower, fineHigher,
  /// finePowerIters, fineRecheckIters, fineRecheckTolerance, fineBlockSize, fineValueType, coarseValueType, and the
  /// coarse level's AMGCL preconditioner tree under "coarse". The parameters are validated here, so that a bad value
  /// fails before build() has spent anything on the prolongation, the Galerkin product or the coarse hierarchy.
  PTwoGridPreconditionerT( const char* json_params ) : TwoGridPreconditioner( json_params, "ptwogrid" )
  {
    nuPre_                = prm.get( "nuPre", prm.get( "nu", 1 ) );
    nuPost_               = prm.get( "nuPost", prm.get( "nu", 1 ) );
    const int fineDegree  = prm.get( "fineDegree", 5 );
    fineLower_            = prm.get( "fineLower", 0.01f );
    fineHigher_           = prm.get( "fineHigher", 1.1f );
    fineBlockSize_        = prm.get( "fineBlockSize", 1 );
    fineValueType_        = prm.get( "fineValueType", std::string( "double" ) );
    coarseValueType_      = prm.get( "coarseValueType", std::string( "double" ) );
    finePowerIters_       = prm.get( "finePowerIters", 300 );
    fineRecheckIters_     = prm.get( "fineRecheckIters", 15 );
    fineRecheckTolerance_ = prm.get( "fineRecheckTolerance", 0.01 );
    checkCycleParameters( fineDegree,
                          CycleParameterNames{ "nuPre",
                                               "nuPost",
                                               "fineDegree",
                                               "fineLower",
                                               "fineHigher",
                                               "finePowerIters",
                                               "fineValueType",
                                               "coarseValueType" } );
    if ( fineBlockSize_ != 1 && fineBlockSize_ != 3 ) {
      throw std::runtime_error( "ptwogrid: fineBlockSize must be 1 or 3, got " + std::to_string( fineBlockSize_ ) );
    }
  }

  /// @brief Build everything apply() needs.
  /// @param n, ptr, col, val the field's equilibrated diagonal block (scalar CSR, sorted columns,
  /// Dirichlet identity rows included). Copied; the caller's arrays are not referenced after build().
  /// @param nNodes, isCorner, edgeEndpoints the P1 topology map (edgeEndpoints: nNodes x 2,
  /// row-major; read for midside nodes only).
  /// @param nullspace, nullspaceCols the coarse near null-space, rows = free coarse DOFs in coarse
  /// order (corner order, then DOF), row-major; nullspaceCols == 0 for none.
  /// @param spectralRadius > 0: start from this fine spectral radius (typically a previous build's)
  /// instead of estimating it from scratch, re-checked with a dominantVector (n entries, the previous build's
  /// fineDominantVector()), see TwoGridPreconditioner::chooseFineSpectralRadius().
  void build( int                n,
              const int*         ptr,
              const int*         col,
              const double*      val,
              int                nNodes,
              const signed char* isCorner,
              const int*         edgeEndpoints,
              const double*      nullspace,
              int                nullspaceRows,
              int                nullspaceCols,
              double             spectralRadius,
              const double*      dominantVector,
              int                dominantVectorSize )
  {
    Clock::time_point start = Clock::now();
    if ( n < 0 || nNodes < 0 || ( nNodes == 0 ) != ( n == 0 ) || ( nNodes > 0 && n % nNodes != 0 ) ) {
      throw std::runtime_error( "ptwogrid: block size " + std::to_string( n ) +
                                " is not a multiple of the topology map's node count " + std::to_string( nNodes ) +
                                " -- the map does not match this field." );
    }
    const int nDim = nNodes > 0 ? n / nNodes : 0;
    if ( fineBlockSize_ == 3 && n > 0 && nDim != 3 ) {
      throw std::runtime_error( "ptwogrid: fineBlockSize 3 needs a 3-dimensional field, got " + std::to_string( nDim ) +
                                " DOFs per node." );
    }
    if ( beginBuild( n, ptr, col, val ) ) {
      addSetupSeconds( "dirichlet + P", secondsSince( start ) );
      return;
    }
    checkFreeRowsHaveDiagonal( n, ptr, col, val );
    buildProlongation( nNodes, nDim, isCorner, edgeEndpoints );
    addSetupSeconds( "dirichlet + P", secondsSince( start ) );

    // The fine operator, copied once with an OpenMP-parallel first touch (crs's own range
    // constructor). The input arrays come from a single-threaded SciPy allocation, i.e. from one NUMA
    // node, and every sparse product on them -- the power iteration, the Galerkin product, every
    // smoothing step -- would otherwise be bound by that one node's memory bandwidth: used in place,
    // a multi-solve replay sequence was measured noticeably slower on a two-socket machine.
    start                                 = Clock::now();
    const int                       nnz   = ptr[n];
    std::shared_ptr< ScalarMatrix > Afine = std::make_shared< ScalarMatrix >( n,
                                                                              n,
                                                                              amgcl::make_iterator_range( ptr,
                                                                                                          ptr + n + 1 ),
                                                                              amgcl::make_iterator_range( col,
                                                                                                          col + nnz ),
                                                                              amgcl::make_iterator_range( val,
                                                                                                          val + nnz ) );
    addSetupSeconds( "fine operator", secondsSince( start ) );

    // Galerkin coarse operator, with AMGCL's own threaded SpGEMM, in the order AMGCL's own coarsening
    // forms it (amgcl/coarsening/detail/galerkin.hpp): P^T (A P).
    start                              = Clock::now();
    std::shared_ptr< ScalarMatrix > A1 = amgcl::backend::product( *PT_, *amgcl::backend::product( *Afine, *P_ ), true );
    addSetupSeconds( "galerkin P^T A P", secondsSince( start ) );

    start = Clock::now();
    boost::property_tree::ptree                     coarsePrm;
    boost::optional< boost::property_tree::ptree& > coarseChild = prm.get_child_optional( "coarse" );
    if ( coarseChild )
      coarsePrm = *coarseChild;
    buildCoarseHierarchy( *A1, coarsePrm, nullspace, nullspaceRows, nullspaceCols );
    A1.reset();
    addSetupSeconds( "coarse hierarchy", secondsSince( start ) );

    // fine level, in node-wise 3x3 blocks or in scalar entries
    if ( fineBlockSize_ == 3 ) {
      start       = Clock::now();
      auto Ablock = std::make_shared< TwoGridFineLevelT< Block3 >::Matrix >(
        amgcl::adapter::block_matrix< Block3 >( *Afine ) );
      Afine.reset();
      addSetupSeconds( "fine operator", secondsSince( start ) );
      start = Clock::now();
      buildFineLevel< Block3 >( Ablock, spectralRadius, dominantVector, dominantVectorSize );
    }
    else {
      start = Clock::now();
      buildFineLevel< double >( Afine, spectralRadius, dominantVector, dominantVectorSize );
    }
    addSetupSeconds( "fine spectral radius", secondsSince( start ) );
    finishBuild();
  }

  /// @brief P on the free DOFs: identity on corners, 1/2, 1/2 on each midside from its two
  /// edge-endpoint corners, per DOF; no rows for Dirichlet fine DOFs, no columns for Dirichlet coarse
  /// DOFs, and the weight of a dropped column is *not* redistributed (a constrained corner contributes
  /// exactly zero to a homogeneous Newton correction). A free midside DOF whose both endpoint DOFs are
  /// constrained is an orphan and raises.
  void buildProlongation( int nNodes, int nDim, const signed char* isCorner, const int* edgeEndpoints )
  {
    // coarse numbering: corners in node order, nDim DOFs each; free coarse DOFs renumbered densely
    std::vector< int > cornerIndex( nNodes, -1 );
    int                nCorners = 0;
    for ( int node = 0; node < nNodes; ++node )
      if ( isCorner[node] )
        cornerIndex[node] = nCorners++;
    std::vector< int > coarseFreeIndex( static_cast< size_t >( nCorners ) * nDim, -1 );
    nCoarse_ = 0;
    for ( int node = 0; node < nNodes; ++node )
      if ( isCorner[node] )
        for ( int d = 0; d < nDim; ++d )
          if ( !isDirichlet_[static_cast< size_t >( node ) * nDim + d] )
            coarseFreeIndex[static_cast< size_t >( cornerIndex[node] ) * nDim + d] = nCoarse_++;

    for ( int node = 0; node < nNodes; ++node ) {
      if ( isCorner[node] )
        continue;
      for ( int k = 0; k < 2; ++k ) {
        const int endpoint = edgeEndpoints[2 * node + k];
        if ( endpoint < 0 || endpoint >= nNodes || !isCorner[endpoint] ) {
          throw std::runtime_error( "ptwogrid: midside node " + std::to_string( node ) +
                                    " has an edge endpoint that is not a corner node." );
        }
      }
    }

    const ptrdiff_t n = static_cast< ptrdiff_t >( nNodes ) * nDim;
    P_                = std::make_shared< ScalarMatrix >();
    P_->set_size( n, nCoarse_, true );
    int orphanRow = -1;
#pragma omp parallel for
    for ( ptrdiff_t i = 0; i < n; ++i ) {
      int width = 0;
      if ( !isDirichlet_[i] ) {
        const int node = static_cast< int >( i / nDim ), d = static_cast< int >( i % nDim );
        if ( isCorner[node] )
          width = 1;
        else {
          for ( int k = 0; k < 2; ++k )
            if ( coarseFreeIndex[static_cast< size_t >( cornerIndex[edgeEndpoints[2 * node + k]] ) * nDim + d] >= 0 )
              ++width;
          if ( width == 0 ) {
#pragma omp critical
            orphanRow = static_cast< int >( i );
          }
        }
      }
      P_->ptr[i + 1] = width;
    }
    if ( orphanRow >= 0 ) {
      throw std::runtime_error( "ptwogrid: a free midside row of the restricted P is entirely zero -- both "
                                "edge-endpoint corners are Dirichlet-constrained, an orphan the topology map "
                                "disagrees with the Dirichlet data on. First bad row (one of): " +
                                std::to_string( orphanRow ) + "." );
    }
    P_->scan_row_sizes();
    P_->set_nonzeros();
#pragma omp parallel for
    for ( ptrdiff_t i = 0; i < n; ++i ) {
      int head = P_->ptr[i];
      if ( P_->ptr[i + 1] == head )
        continue;
      const int node = static_cast< int >( i / nDim ), d = static_cast< int >( i % nDim );
      if ( isCorner[node] ) {
        P_->col[head] = coarseFreeIndex[static_cast< size_t >( cornerIndex[node] ) * nDim + d];
        P_->val[head] = 1.0;
        continue;
      }
      int cols[2], width = 0;
      for ( int k = 0; k < 2; ++k ) {
        const int c = coarseFreeIndex[static_cast< size_t >( cornerIndex[edgeEndpoints[2 * node + k]] ) * nDim + d];
        if ( c >= 0 )
          cols[width++] = c;
      }
      if ( width == 2 && cols[0] > cols[1] )
        std::swap( cols[0], cols[1] );
      for ( int k = 0; k < width; ++k ) {
        P_->col[head + k] = cols[k];
        P_->val[head + k] = 0.5;
      }
    }
    PT_ = amgcl::backend::transpose( *P_ );
  }

  /// @brief The number of free coarse DOFs, the size of the Galerkin operator.
  int nCoarseFree() const { return nCoarse_; }

protected:
  void restrictResidual( const double* fine, double* coarse ) const override
  {
    auto fineRange   = amgcl::make_iterator_range( fine, fine + n_ );
    auto coarseRange = amgcl::make_iterator_range( coarse, coarse + nCoarse_ );
    amgcl::backend::spmv( 1.0, *PT_, fineRange, 0.0, coarseRange );
  }

  void prolongateCorrection( const double* coarse, double* fine, bool addToFine ) const override
  {
    auto coarseRange = amgcl::make_iterator_range( coarse, coarse + nCoarse_ );
    auto fineRange   = amgcl::make_iterator_range( fine, fine + n_ );
    amgcl::backend::spmv( 1.0, *P_, coarseRange, addToFine ? 1.0 : 0.0, fineRange );
  }
};

/// The rigid-body coarse level preconditioner of one vector field's equilibrated diagonal block (blockamg's key
/// rigidBodyCoarseLevel, edelweissfe/linsolve/blockamg/rigidbodycoarselevel.py): the two-grid cycle
/// of TwoGridPreconditioner whose coarse level is a smoothed aggregation spanned by the rigid-body modes of the
/// field's aggregates (amgcl-rigid-body-coarse-level.hpp), below which an AMGCL hierarchy continues the field's own.
/// The interface Cython drives, whatever the number of components per node (RigidBodyCoarseLevelPreconditionerT).
///
/// The parameters are those of the field's tree as blockamg configures it: backendBlockSize (2 or 3, the components
/// per node), backendPrecision (the storage of the fine level and of P and P^T: double, or float with 3x3 blocks),
/// prolongation ("smoothed" or "tentative"), coarseHierarchyPrecision (double or float), and the field's AMGCL tree
/// under "amgcl", which means what it means to AMGCL's own hierarchy (with AMGCL's defaults for what it leaves out):
///   * relax, npre, npost: the fine smoother, a Chebyshev polynomial (relax.type chebyshev, relax.scale false, at least
///     one power iteration) with npre sweeps before and npost after the coarse-grid correction;
///   * coarsening: smoothed aggregation, its aggr.eps_strong the strength threshold of the aggregates and its relax,
///     estimate_spectral_radius and power_iters the damping weight of the prolongation smoothing;
///   * the whole tree, too, for the AMGCL hierarchy below, with eps_strong halved as AMGCL halves it from one level to
///     the next and the coarse DOFs of an aggregate's modes together as one point (aggr.block_size), with the coarse
///     near null-space.
/// "rotations" (true) false leaves the rotations out of the coarse space: AMGCL's own coarse space on the block
/// backend, the control for tests and measurements. It reuses this construction rather than AMGCL's
/// translation_tentative_prolongation (amgcl-block-translations.hpp), so that the control differs from the rigid-body
/// coarse level in the rotations alone; a test checks that it reproduces AMGCL's first coarse level to rounding.
class RigidBodyCoarseLevelPreconditioner : public TwoGridPreconditioner {
public:
  /// @brief Build everything apply() needs.
  /// @param n, ptr, col, val the field's equilibrated diagonal block (scalar CSR, sorted columns, node-major with B
  /// DOFs per node, Dirichlet identity rows included). Copied; not referenced after build().
  /// @param coordinates, nNodes the node coordinates, nNodes x B, row-major (nNodes B = n)
  /// @param scaling the field's equilibration, n entries: the block is D K D with D = diag(scaling)
  /// @param spectralRadius, dominantVector, dominantVectorSize a supplied fine spectral radius (> 0) and the dominant
  /// vector to re-check it from, see TwoGridPreconditioner::chooseFineSpectralRadius()
  virtual void build( int           n,
                      const int*    ptr,
                      const int*    col,
                      const double* val,
                      const double* coordinates,
                      int           nNodes,
                      const double* scaling,
                      double        spectralRadius,
                      const double* dominantVector,
                      int           dominantVectorSize ) = 0;

  /// @brief The scalar CSR arrays of the prolongation P as apply() uses it (its entries widened to double), for
  /// inspection.
  virtual void prolongationCsr( std::vector< int >&    ptr,
                                std::vector< int >&    col,
                                std::vector< double >& val ) const = 0;

  /// @brief The components per node, B.
  int blockSize() const { return blockSize_; }
  /// @brief The coarse DOFs per aggregate, M = B (B + 1) / 2: one per rigid-body mode.
  int modesPerAggregate() const { return modesPerAggregate_; }
  /// @brief The number of nodes of the last build.
  int nNodes() const { return static_cast< int >( aggregateOfNode_.size() ); }
  /// @brief The number of aggregates of the last build.
  int nAggregates() const { return static_cast< int >( nAggregates_ ); }
  /// @brief The number of modes the last build dropped as linearly dependent.
  int droppedModes() const { return static_cast< int >( droppedModes_ ); }
  /// @brief The damping weight omega of the prolongation smoothing of the last build (0 for a tentative prolongation).
  double prolongationSmoothingWeight() const { return prolongationSmoothingWeight_; }
  /// @brief The columns of the coarse near null-space: the modes in the coarse space.
  int coarseNullspaceColumns() const { return coarseNullspaceColumns_; }

  /// @brief Copy the aggregate of every node (< 0 for none) into out (nNodes() entries).
  void copyAggregateOfNode( long long* out ) const
  {
    std::copy( aggregateOfNode_.begin(), aggregateOfNode_.end(), out );
  }

  /// @brief Copy the coarse near null-space into out (nCoarse() x coarseNullspaceColumns(), row-major).
  void copyCoarseNullspace( double* out ) const { std::copy( coarseNullspace_.begin(), coarseNullspace_.end(), out ); }

  /// @brief The scalar CSR arrays of the Galerkin coarse operator as the coarse hierarchy holds it (with the padded
  /// diagonal entries of dropped modes), for inspection.
  void coarseOperatorCsr( std::vector< int >& ptr, std::vector< int >& col, std::vector< double >& val ) const
  {
    checkBuilt();
    if ( coarseFloat_ )
      scalarCsr( coarseFloat_->system_matrix(), ptr, col, val );
    else if ( coarse_ )
      scalarCsr( coarse_->system_matrix(), ptr, col, val );
    else {
      ptr.assign( 1, 0 );
      col.clear();
      val.clear();
    }
  }

  /// @brief A summary of the rigid-body coarse level and the report of the AMGCL hierarchy below.
  std::string report() const override
  {
    checkBuilt();
    std::ostringstream report;
    if ( trivial_ ) {
      report << name_ << ": no free DOF, the preconditioner is the identity\n";
      return report.str();
    }
    const ptrdiff_t nNodes  = n_ / blockSize_;
    ptrdiff_t       nInNone = 0;
    for ( ptrdiff_t aggregate : aggregateOfNode_ )
      if ( aggregate < 0 )
        ++nInNone;
    report << "rigid-body coarse level: " << nNodes << " nodes (" << blockSize_ << "x" << blockSize_ << " blocks, "
           << fineValueType_ << "), " << nAggregates_ << " aggregates";
    if ( nAggregates_ > 0 )
      report << " (" << static_cast< double >( nNodes - nInNone ) / static_cast< double >( nAggregates_ )
             << " nodes on average, " << nInNone << " nodes in none)";
    report << "\n  coarse level: " << nCoarse_ << " DOFs, " << coarseNullspaceColumns_ << " modes per aggregate"
           << ( rotations_ ? "" : " (translations only)" ) << ", " << droppedModes_
           << " dropped as linearly dependent; " << prolongation_ << " prolongation";
    if ( prolongation_ == "smoothed" && nAggregates_ > 0 )
      report << ", omega " << prolongationSmoothingWeight_;
    report << ", " << prolongationBlocks_ << " blocks\n";
    if ( nCoarse_ > 0 )
      report << "AMGCL hierarchy below:\n" << coarseHierarchyReport();
    else
      report << "  no aggregate (no strong connection anywhere): the cycle is the smoother alone\n";
    return report.str();
  }

protected:
  /// @brief The components per node, B, and the coarse DOFs per aggregate, M.
  int blockSize_ = 3, modesPerAggregate_ = 6;
  /// @brief "smoothed" or "tentative".
  std::string prolongation_;
  /// @brief Whether the rotations are in the coarse space (false: the translations alone, a control).
  bool rotations_ = true;
  /// @brief The strength threshold of the aggregates (coarsening.aggr.eps_strong).
  float epsStrong_ = 0.08f;
  /// @brief The damping weight of the prolongation smoothing, see amgcl::coarsening::prolongation_smoothing_weight().
  float prolongationRelax_      = 1.0f;
  bool  estimateSpectralRadius_ = false;
  int   prolongationPowerIters_ = 0;
  /// @brief The AMGCL tree of the hierarchy below.
  boost::property_tree::ptree hierarchyBelow_;

  /// @brief The aggregate of every node of the last build, < 0 for none.
  std::vector< ptrdiff_t > aggregateOfNode_;
  /// @brief The number of aggregates and of dropped modes of the last build.
  ptrdiff_t nAggregates_ = 0, droppedModes_ = 0;
  /// @brief omega of the last build.
  double prolongationSmoothingWeight_ = 0.0;
  /// @brief The blocks of P of the last build.
  size_t prolongationBlocks_ = 0;

  /// @brief Parse and check the parameters (see the class comment), so that a bad value fails before build() has
  /// spent anything.
  RigidBodyCoarseLevelPreconditioner( const char* json_params )
    : TwoGridPreconditioner( json_params, "rigidBodyCoarseLevel" )
  {
    blockSize_ = prm.get( "backendBlockSize", 3 );
    if ( blockSize_ != 2 && blockSize_ != 3 ) {
      throw std::runtime_error( name_ +
                                ": needs a block backend of a 2D or 3D vector field, backendBlockSize 2 or 3, got " +
                                std::to_string( blockSize_ ) );
    }
    modesPerAggregate_ = blockSize_ * ( blockSize_ + 1 ) / 2;
    fineValueType_     = prm.get( "backendPrecision", std::string( "double" ) );
    if ( blockSize_ == 2 && fineValueType_ == "float" ) {
      throw std::runtime_error( name_ + ": 2x2 blocks are stored in double only, got backendPrecision float" );
    }
    coarseValueType_ = prm.get( "coarseHierarchyPrecision", std::string( "double" ) );
    prolongation_    = prm.get( "prolongation", std::string( "smoothed" ) );
    if ( prolongation_ != "smoothed" && prolongation_ != "tentative" ) {
      throw std::runtime_error( name_ + ": prolongation must be smoothed or tentative, got " + prolongation_ );
    }
    rotations_            = prm.get( "rotations", true );
    fineRecheckIters_     = prm.get( "fineRecheckIters", 15 );
    fineRecheckTolerance_ = prm.get( "fineRecheckTolerance", 0.01 );

    boost::property_tree::ptree                     tree;
    boost::optional< boost::property_tree::ptree& > amgclTree = prm.get_child_optional( "amgcl" );
    if ( amgclTree )
      tree = *amgclTree;
    if ( tree.get( "class", std::string( "amg" ) ) != "amg" ) {
      throw std::runtime_error( name_ + ": needs an AMG hierarchy, class amg, got class " +
                                tree.get< std::string >( "class" ) );
    }

    // the fine smoother: AMGCL's Chebyshev parameters (its defaults for what the tree leaves out), npre and npost
    // AMGCL's defaults 1 and 1 (amg::params)
    boost::property_tree::ptree relax = childOrEmpty( tree, "relax" );
    if ( relax.get( "type", std::string( "spai0" ) ) != "chebyshev" ) {
      const boost::optional< std::string > relaxType = relax.get_optional< std::string >( "type" );
      throw std::runtime_error( name_ +
                                ": smooths with a Chebyshev polynomial (the p-two-grid's fine smoother), relax.type "
                                "must be chebyshev, got " +
                                ( relaxType ? *relaxType : std::string( "spai0 (AMGCL's default)" ) ) );
    }
    relax.erase( "type" );
    const amgcl::relaxation::chebyshev< CoarseBackend >::params chebyshev( relax );
    if ( chebyshev.scale ) {
      throw std::runtime_error( name_ + ": the fine Chebyshev smoother has no Jacobi scaling, relax.scale must be "
                                        "false" );
    }
    nuPre_          = tree.get( "npre", 1 );
    nuPost_         = tree.get( "npost", 1 );
    fineLower_      = chebyshev.lower;
    fineHigher_     = chebyshev.higher;
    finePowerIters_ = chebyshev.power_iters;
    checkCycleParameters( static_cast< int >( chebyshev.degree ),
                          CycleParameterNames{ "npre",
                                               "npost",
                                               "relax.degree",
                                               "relax.lower",
                                               "relax.higher",
                                               "relax.power_iters",
                                               "backendPrecision",
                                               "coarseHierarchyPrecision" } );

    // the coarsening of the rigid-body coarse level: AMGCL's smoothed-aggregation parameters (its defaults for what the
    // tree leaves out)
    boost::property_tree::ptree coarsening     = childOrEmpty( tree, "coarsening" );
    const std::string           coarseningType = coarsening.get( "type", std::string( "smoothed_aggregation" ) );
    if ( coarseningType != "smoothed_aggregation" ) {
      throw std::runtime_error( name_ +
                                ": is a smoothed aggregation, coarsening.type must be smoothed_aggregation, "
                                "got " +
                                coarseningType );
    }
    coarsening.erase( "type" );
    const amgcl::coarsening::smoothed_aggregation< CoarseBackend >::params aggregation( coarsening );
    if ( aggregation.aggr.block_size != 1 ) {
      throw std::runtime_error( name_ +
                                ": aggregates the nodes of the block backend, coarsening.aggr.block_size has "
                                "no meaning there, got " +
                                std::to_string( aggregation.aggr.block_size ) );
    }
    epsStrong_              = aggregation.aggr.eps_strong;
    prolongationRelax_      = aggregation.relax;
    estimateSpectralRadius_ = aggregation.estimate_spectral_radius;
    prolongationPowerIters_ = aggregation.power_iters;

    // the AMGCL hierarchy below: the field's tree one level further down. Its points are the coarse dofs of an
    // aggregate that hold modes (aggr.block_size): AMGCL gives every aggregate of a level as many coarse dofs as the
    // near null-space has columns and keeps aggr.block_size from level to level, so it is the number of modes in the
    // coarse space on every level below. (Without the rotations, the B translations of an aggregate are one point and
    // its padded rotation dofs another, which has no strong connection and is in no aggregate.)
    hierarchyBelow_ = tree;
    hierarchyBelow_.put( "coarsening.aggr.eps_strong", 0.5f * epsStrong_ );
    hierarchyBelow_.put( "coarsening.aggr.block_size", rotations_ ? modesPerAggregate_ : blockSize_ );
  }

  /// @brief The scalar CSR arrays of an AMGCL scalar matrix, its entries widened to double.
  template < class Matrix >
  static void scalarCsr( const Matrix& A, std::vector< int >& ptr, std::vector< int >& col, std::vector< double >& val )
  {
    ptr.assign( A.ptr, A.ptr + A.nrows + 1 );
    col.assign( A.col, A.col + A.ptr[A.nrows] );
    val.assign( A.val, A.val + A.ptr[A.nrows] );
  }
};

/// @brief The rigid-body coarse level preconditioner of a field of B components per node (B = 2 or 3), see
/// RigidBodyCoarseLevelPreconditioner.
template < int B >
class RigidBodyCoarseLevelPreconditionerT : public RigidBodyCoarseLevelPreconditioner {
public:
  /// @brief The coarse DOFs per aggregate, one per rigid-body mode.
  enum { M = RigidBodyModes< B >::count };
  /// @brief The blocks of the fine operator, of the prolongation, of the restriction and of the coarse operator.
  typedef amgcl::static_matrix< double, B, B > FineBlock;
  typedef amgcl::static_matrix< double, B, M > ProlongationBlock;
  typedef amgcl::static_matrix< double, M, B > RestrictionBlock;
  typedef amgcl::static_matrix< double, M, M > CoarseBlock;
  /// @brief The fine operator, as the fine level stores it in double.
  typedef typename TwoGridFineLevelT< FineBlock >::Matrix FineMatrix;

  /// @brief See RigidBodyCoarseLevelPreconditioner.
  RigidBodyCoarseLevelPreconditionerT( const char* json_params ) : RigidBodyCoarseLevelPreconditioner( json_params ) {}

  void build( int           n,
              const int*    ptr,
              const int*    col,
              const double* val,
              const double* coordinates,
              int           nNodes,
              const double* scaling,
              double        spectralRadius,
              const double* dominantVector,
              int           dominantVectorSize ) override
  {
    Clock::time_point start = Clock::now();
    if ( n < 0 || nNodes < 0 || static_cast< ptrdiff_t >( nNodes ) * B != n ) {
      throw std::runtime_error( name_ + ": the field has " + std::to_string( n ) +
                                " DOFs, but the coordinates are "
                                "given for " +
                                std::to_string( nNodes ) + " nodes of " + std::to_string( B ) +
                                " components -- they are not the field's nodes" );
    }
    // AMGCL's mark of a node in no aggregate, as a value (the static member has no out-of-class definition)
    const ptrdiff_t inNoAggregate = amgcl::coarsening::plain_aggregates::removed;
    aggregateOfNode_.assign( nNodes, inNoAggregate );
    nAggregates_ = droppedModes_ = 0;
    prolongationSmoothingWeight_ = 0.0;
    prolongationBlocks_          = 0;
    restrictionFloat_.reset();
    prolongationFloat_.reset();
    restrictionDouble_.reset();
    prolongationDouble_.reset();
    if ( beginBuild( n, ptr, col, val ) ) {
      addSetupSeconds( "dirichlet rows", secondsSince( start ) );
      return;
    }
    checkFreeRowsHaveDiagonal( n, ptr, col, val );
    addSetupSeconds( "dirichlet rows", secondsSince( start ) );

    // The fine operator in node-wise B x B blocks, double: the aggregates, the prolongation, the Galerkin product and
    // the fine spectral radius are computed on it, then the fine level keeps it (or its float copy). The copy is an
    // OpenMP-parallel first touch, which spreads it over the NUMA nodes (see PTwoGridPreconditionerT::build()).
    start         = Clock::now();
    const int nnz = ptr[n];
    auto      A   = std::make_shared< FineMatrix >(
      amgcl::adapter::block_matrix< FineBlock >( std::make_tuple( n,
                                                                  amgcl::make_iterator_range( ptr, ptr + n + 1 ),
                                                                  amgcl::make_iterator_range( col, col + nnz ),
                                                                  amgcl::make_iterator_range( val, val + nnz ) ) ) );
    addSetupSeconds( "fine operator", secondsSince( start ) );

    // 1. the aggregates: AMGCL's plain aggregation of the nodes, its strength of connection by block traces
    start = Clock::now();
    amgcl::coarsening::plain_aggregates::params aggregation;
    aggregation.eps_strong = epsStrong_;
    std::unique_ptr< amgcl::coarsening::plain_aggregates > aggregates;
    try {
      aggregates.reset( new amgcl::coarsening::plain_aggregates( *A, aggregation ) );
      aggregateOfNode_ = aggregates->id;
      nAggregates_     = static_cast< ptrdiff_t >( aggregates->count );
    }
    catch ( const amgcl::error::empty_level& ) {
      // no strong connection anywhere, no aggregate: no coarse level, the cycle is the smoother alone (AMGCL ends its
      // own hierarchy the same way)
      aggregateOfNode_.assign( nNodes, inNoAggregate );
    }
    addSetupSeconds( "aggregates", secondsSince( start ) );

    if ( nAggregates_ > 0 ) {
      if ( static_cast< ptrdiff_t >( M ) * nAggregates_ >
           static_cast< ptrdiff_t >( std::numeric_limits< int >::max() ) ) {
        throw std::overflow_error( name_ + ": " + std::to_string( nAggregates_ ) +
                                   " aggregates have more coarse DOFs than 32-bit indices hold" );
      }

      // 2. the tentative prolongation from the rigid-body modes of every aggregate
      start          = Clock::now();
      auto tentative = rigidBodyTentativeProlongation< B >( nNodes,
                                                            nAggregates_,
                                                            aggregateOfNode_,
                                                            coordinates,
                                                            scaling,
                                                            isDirichlet_.data(),
                                                            rotations_ ? M : B );
      droppedModes_  = tentative.droppedModes;
      addSetupSeconds( "tentative prolongation", secondsSince( start ) );

      // 3. the smoothed prolongation P = (I - omega D_F^-1 A_F) P_tent, AMGCL's, and the restriction P^T
      start  = Clock::now();
      auto P = tentative.P;
      if ( prolongation_ == "smoothed" ) {
        prolongationSmoothingWeight_ = amgcl::coarsening::prolongation_smoothing_weight( *A,
                                                                                         prolongationRelax_,
                                                                                         estimateSpectralRadius_,
                                                                                         prolongationPowerIters_ );
        P                            = amgcl::coarsening::smoothed_prolongation( *A,
                                                      aggregates->strong_connection,
                                                      *tentative.P,
                                                      prolongationSmoothingWeight_ );
      }
      tentative.P.reset();
      aggregates.reset();
      auto R              = transposedBlocks( *P );
      prolongationBlocks_ = P->nnz;
      addSetupSeconds( "prolongation", secondsSince( start ) );

      // 4. the Galerkin coarse operator A_c = P^T A P, with the padded diagonal entries of the dropped modes
      start   = Clock::now();
      auto Ac = paddedScalarCoarseOperator( *galerkinCoarseOperator< CoarseBlock >( *A, *P, *R ), tentative.keptMode );
      addSetupSeconds( "galerkin P^T A P", secondsSince( start ) );

      // the AMGCL hierarchy below, with the coefficients of the modes in the coarse space as its near null-space
      start                       = Clock::now();
      nCoarse_                    = static_cast< int >( M * nAggregates_ );
      const int             modes = rotations_ ? M : B;
      std::vector< double > nullspace( static_cast< size_t >( nCoarse_ ) * modes );
      for ( ptrdiff_t i = 0; i < nCoarse_; ++i )
        for ( int k = 0; k < modes; ++k )
          nullspace[i * modes + k] = tentative.coarseNullspace[i * M + k];
      buildCoarseHierarchy( *Ac, hierarchyBelow_, nullspace.data(), nCoarse_, modes );
      Ac.reset();
      addSetupSeconds( "coarse hierarchy", secondsSince( start ) );

      // P and P^T as the cycle reads them: in the precision of the fine level, with 32-bit indices
      start = Clock::now();
      if ( fineValueType_ == "float" ) {
        restrictionFloat_  = convertedCopy< amgcl::static_matrix< float, M, B > >( *R );
        prolongationFloat_ = convertedCopy< amgcl::static_matrix< float, B, M > >( *P );
      }
      else {
        restrictionDouble_  = convertedCopy< RestrictionBlock >( *R );
        prolongationDouble_ = convertedCopy< ProlongationBlock >( *P );
      }
      addSetupSeconds( "transfer operators", secondsSince( start ) );
    }

    // the fine level: the Chebyshev smoother on A
    start = Clock::now();
    buildFineLevel< FineBlock >( A, spectralRadius, dominantVector, dominantVectorSize );
    addSetupSeconds( "fine spectral radius", secondsSince( start ) );
    finishBuild();
  }

  void prolongationCsr( std::vector< int >& ptr, std::vector< int >& col, std::vector< double >& val ) const override
  {
    checkBuilt();
    if ( prolongationFloat_ )
      blockCsrToScalar( *prolongationFloat_, ptr, col, val );
    else if ( prolongationDouble_ )
      blockCsrToScalar( *prolongationDouble_, ptr, col, val );
    else {
      ptr.assign( n_ + 1, 0 );
      col.clear();
      val.clear();
    }
  }

protected:
  void restrictResidual( const double* fine, double* coarse ) const override
  {
    if ( restrictionFloat_ )
      blockMatrixVectorProduct( *restrictionFloat_, fine, coarse, false );
    else
      blockMatrixVectorProduct( *restrictionDouble_, fine, coarse, false );
  }

  void prolongateCorrection( const double* coarse, double* fine, bool addToFine ) const override
  {
    if ( prolongationFloat_ )
      blockMatrixVectorProduct( *prolongationFloat_, coarse, fine, addToFine );
    else
      blockMatrixVectorProduct( *prolongationDouble_, coarse, fine, addToFine );
  }

private:
  /// @brief P^T and P as the cycle reads them, in exactly one of the two value types (backendPrecision).
  std::shared_ptr< amgcl::backend::crs< amgcl::static_matrix< float, M, B >, int, int > > restrictionFloat_;
  std::shared_ptr< amgcl::backend::crs< amgcl::static_matrix< float, B, M >, int, int > > prolongationFloat_;
  std::shared_ptr< amgcl::backend::crs< RestrictionBlock, int, int > >                    restrictionDouble_;
  std::shared_ptr< amgcl::backend::crs< ProlongationBlock, int, int > >                   prolongationDouble_;

  /// @brief The scalar CSR arrays of a block matrix of N x K blocks, its entries widened to double.
  template < class Matrix >
  static void blockCsrToScalar( const Matrix&          A,
                                std::vector< int >&    ptr,
                                std::vector< int >&    col,
                                std::vector< double >& val )
  {
    typedef typename amgcl::backend::value_type< Matrix >::type Block;
    const int N = amgcl::math::static_rows< Block >::value, K = amgcl::math::static_cols< Block >::value;
    ptr.assign( 1, 0 );
    col.clear();
    val.clear();
    for ( size_t i = 0; i < A.nrows; ++i ) {
      for ( int r = 0; r < N; ++r ) {
        for ( ptrdiff_t j = A.ptr[i]; j < A.ptr[i + 1]; ++j ) {
          for ( int c = 0; c < K; ++c ) {
            col.push_back( static_cast< int >( K * A.col[j] + c ) );
            val.push_back( A.val[j]( r, c ) );
          }
        }
        ptr.push_back( static_cast< int >( col.size() ) );
      }
    }
  }
};

/// @brief The rigid-body coarse level preconditioner for the parameters (see RigidBodyCoarseLevelPreconditioner), of
/// the field's backendBlockSize, owned by the caller.
/// @throws std::runtime_error for parameters that cannot be used
inline RigidBodyCoarseLevelPreconditioner* makeRigidBodyCoarseLevelPreconditioner( const char* json_params )
{
  const int blockSize = parseJsonParameters( json_params ).get( "backendBlockSize", 3 );
  if ( blockSize == 2 )
    return new RigidBodyCoarseLevelPreconditionerT< 2 >( json_params );
  if ( blockSize == 3 )
    return new RigidBodyCoarseLevelPreconditionerT< 3 >( json_params );
  throw std::runtime_error( "rigidBodyCoarseLevel: needs a block backend of a 2D or 3D vector field, backendBlockSize "
                            "2 or 3, got " +
                            std::to_string( blockSize ) );
}

// The default double-precision wrapper (plain amgcl::backend::builtin<double>, unchanged from before
// this module existed), and a float32 one for a mixed-precision backend (plain builtin<float>: the
// whole hierarchy -- not just the level matrices -- is float, see the comment above LinearSolverT).
typedef LinearSolverT< double > LinearSolver;
typedef LinearSolverT< float >  LinearSolverFloat;

// Block-valued instantiations: 3×3 for a 3D displacement field, 2×2 for a 2D one -- one template
// parameter apart. Both plain amgcl::backend::builtin<BlockType>, so relax.type = "gauss_seidel" and
// the exact ILU0/ILUk/ILUT/ILUP triangular solve work exactly as they do for the scalar backends above.
// Every block instantiation, LinearSolverBlock3Float included, takes the translations as its near
// null-space (set_nullspace(), see LinearSolverBlockT).
typedef LinearSolverBlockT< amgcl::static_matrix< double, 2, 2 > > LinearSolverBlock2;
typedef LinearSolverBlockT< amgcl::static_matrix< double, 3, 3 > > LinearSolverBlock3;

// 3×3 blocks set up in double but stored -- and read in the cycle -- in float, with 32-bit level
// indices, see builtin_float_matrices above.
typedef LinearSolverBlockT< amgcl::static_matrix< double, 3, 3 >,
                            amgcl::backend::builtin_float_matrices< amgcl::static_matrix< double, 3, 3 > > >
  LinearSolverBlock3Float;

/// @brief The AMGCL solver for a backend block size (1, 2 or 3) and value type, owned by the caller.
/// Float selects builtin<float> for scalars and LinearSolverBlock3Float for 3x3 blocks; 2x2 blocks
/// have no float variant.
inline AMGCLSolverInterface* makeAMGCLSolver( const char* json_params, int blockSize, bool isFloat )
{
  if ( blockSize == 1 )
    return isFloat ? static_cast< AMGCLSolverInterface* >( new LinearSolverFloat( json_params ) )
                   : static_cast< AMGCLSolverInterface* >( new LinearSolver( json_params ) );
  if ( blockSize == 2 && !isFloat )
    return new LinearSolverBlock2( json_params );
  if ( blockSize == 3 )
    return isFloat ? static_cast< AMGCLSolverInterface* >( new LinearSolverBlock3Float( json_params ) )
                   : static_cast< AMGCLSolverInterface* >( new LinearSolverBlock3( json_params ) );
  throw std::runtime_error( "makeAMGCLSolver(): no backend for block size " + std::to_string( blockSize ) +
                            ( isFloat ? " in float" : " in double" ) );
}

// blockamg's equilibrated, field-split system, see EquilibratedSystemT above.
typedef EquilibratedSystemT EquilibratedSystem;

// AMGCL's own native outer Krylov solves (lgmres, fgmres) are created by makeOuterKrylovSolver(), see above.

// The p-two-grid (P1) preconditioner, see PTwoGridPreconditionerT above.
typedef PTwoGridPreconditionerT PTwoGridPreconditioner;

// The rigid-body coarse level preconditioner is RigidBodyCoarseLevelPreconditioner itself (created by
// makeRigidBodyCoarseLevelPreconditioner()), see there.
