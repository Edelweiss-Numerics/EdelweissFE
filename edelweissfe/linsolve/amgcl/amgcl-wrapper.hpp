#pragma once

#include <algorithm>
#include <amgcl/adapter/block_matrix.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/backend/builtin.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/preconditioner/runtime.hpp>
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

  /// @brief Supply near null-space vectors for smoothed-aggregation coarsening.
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
// values. That was measured to cost up to 30% more outer iterations. builtin_float_matrices keeps
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
// set_nullspace() is not supported here and always throws: AMGCL's own tentative-prolongation
// nullspace path self-flags as unimplemented for block value types (amgcl/coarsening/
// tentative_prolongation.hpp: "TODO: this is just a workaround to make non-scalar value types
// compile. Most probably this won't actually work.") -- not merely undocumented, upstream itself does
// not trust it. A block-aware smoother's own per-node B×B inversion is expected to substitute for
// some of what a near-null-space would otherwise buy the scalar backend, but this has not been
// re-measured against the current (Chebyshev, rigid-body-aware) scalar-backend default.
template < typename BlockType, typename BackendType = amgcl::backend::builtin< BlockType > >
class LinearSolverBlockT : public AMGCLSolverInterface {
public:
  typedef BackendType Backend;
  typedef amgcl::make_solver< amgcl::runtime::preconditioner< Backend >, amgcl::runtime::solver::wrapper< Backend > >
    Solver;

  static const int BlockSize = amgcl::math::static_rows< BlockType >::value;

  boost::property_tree::ptree prm;
  std::unique_ptr< Solver >   solver_;

  LinearSolverBlockT( const char* json_params ) : prm( parseJsonParameters( json_params ) ), solver_() {}

  void set_nullspace( const double*, int, int ) override
  {
    throw std::runtime_error(
      "set_nullspace() is not supported with a block-valued AMGCL backend -- AMGCL's own "
      "tentative-prolongation nullspace path is an unimplemented, self-flagged 'probably won't work' "
      "TODO for block value types, not a supported feature. Do not request a near null-space when "
      "backendBlockSize > 1." );
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
    solver_.reset( new Solver( amgcl::adapter::block_matrix< BlockType >( A ), prm ) );
  }

  void applyPreconditioner( int n, const double* rhs, double* x ) override
  {
    std::fill( x, x + n, 0.0 );
    std::vector< double > rhs_v( rhs, rhs + n );
    std::vector< double > x_v( x, x + n );
    auto                  rhs_rng = amgcl::backend::reinterpret_as_rhs< BlockType >( rhs_v );
    auto                  x_rng   = amgcl::backend::reinterpret_as_rhs< BlockType >( x_v );
    solver_->precond().apply( rhs_rng, x_rng );
    std::copy( x_v.begin(), x_v.end(), x );
  }

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
    solver_.reset( new Solver( amgcl::adapter::block_matrix< BlockType >( A ), prm ) );

    std::vector< double > rhs_v( rhs, rhs + n );
    std::vector< double > x_v( n, 0.0 );
    std::tie( iters, error ) = ( *solver_ )( amgcl::backend::reinterpret_as_rhs< BlockType >( rhs_v ),
                                             amgcl::backend::reinterpret_as_rhs< BlockType >( x_v ) );
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
///   * the equilibrated operator As = D A D, the outer Krylov operator: matvec(), residual(), and the
///     operator LGMRESOuterSolverT::solve() iterates on. Each entry is formed exactly as SciPy's
///     former `As.data *= scaling[As.indices]; As.data *= scaling[rows]` did, (a_ij * s_j) * s_i, s = scaling, so
///     the operator is bitwise the former one;
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

  std::shared_ptr< Matrix >                operator_;
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
  void build( int           n,
              const int*    ptr,
              const int*    col,
              const double* val,
              const double* scaling,
              int           nFields,
              const int*    fieldBounds )
  {
    EquilibratedSystemT assembled;
    assembled.assemble( n, ptr, col, val, scaling, nFields, fieldBounds );
    *this = std::move( assembled );
  }

  /// @brief The work of build(), on this (fresh) object.
  void assemble( int           n,
                 const int*    ptr,
                 const int*    col,
                 const double* val,
                 const double* scaling,
                 int           nFields,
                 const int*    fieldBounds )
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
#pragma omp parallel for
    for ( ptrdiff_t i = 0; i < n; ++i ) {
      int head = operator_->ptr[i];
      for ( int j = ptr[i]; j < ptr[i + 1]; ++j ) {
        if ( val[j] != 0.0 ) {
          operator_->col[head]   = col[j];
          operator_->val[head++] = ( val[j] * scaling[col[j]] ) * scaling[i];
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
          blockVal[head++] = operator_->val[j];
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

  /// @brief y <- As x
  void matvec( const double* x, double* y ) const
  {
    checkBuilt();
    auto xRange = amgcl::make_iterator_range( x, x + n_ );
    auto yRange = amgcl::make_iterator_range( y, y + n_ );
    amgcl::backend::spmv( 1.0, *operator_, xRange, 0.0, yRange );
  }

  /// @brief r <- rhs - As x
  void residual( const double* rhs, const double* x, double* r ) const
  {
    checkBuilt();
    auto rhsRange = amgcl::make_iterator_range( rhs, rhs + n_ );
    auto xRange   = amgcl::make_iterator_range( x, x + n_ );
    auto rRange   = amgcl::make_iterator_range( r, r + n_ );
    amgcl::backend::residual( rhsRange, *operator_, xRange, rRange );
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
          sum += operator_->val[j] * x[c - start];
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

// Exposes AMGCL's own OpenMP-threaded sparse-matrix-matrix product and
// sparse-matrix addition (amgcl::backend::builtin's product()/sum(), verified directly against the
// installed headers: spgemm_saad/spgemm_rmerge both carry `#pragma omp parallel`/`#pragma omp for`,
// and so does sum()). Wired into edelweissfe/numerics/mpctransformation.py's `T^T @ K @ T + C`
// condensation as an opt-in alternative (`useAmgclSpgemm`/`useAmgclMPCCondensation`, default False
// pending a live gate at more thread counts) to the plain expression's scipy CSR sparse routines --
// the same class of gap EquilibratedSystemT (above) closed for the outer GMRES operator. Deliberately generic (two raw
// CSR matrices in, one CSR matrix out) rather than specific to the T^T K T + C expression itself: the caller composes
// `product()`/`sum()` calls (e.g. `KT = product(K, T)`, `Kt_noC = product(T^T, KT)`, `Kt = sum(1, Kt_noC, 1, C)`),
// passing T^T as its own already-transposed CSR array (scipy's own `.T.tocsr()` is a cheap O(nnz) relayout, not a
// second SpGEMM, so there is nothing to gain by transposing on this side too).
//
// Holds at most one result at a time (the most recent product()/sum() call) -- this is a scoping
// probe measuring one call at a time, not a class meant to be composed into a multi-step pipeline
// internally; chaining happens in Python, one call per step, exactly like the comment above shows.
//
// Square matrices only (n x n) -- the only shape this class or its target expression (T^T K T + C,
// every factor nDof x nDof) ever needs. Uses the same 4-tuple adapter construction every other class
// in this header uses (n, ptr_range, col_range, val_range) -- verified directly against amgcl/
// adapter/crs_tuple.hpp that this adapter's own cols_impl returns the same `n` as rows_impl (i.e. it
// only ever represents a square matrix), so there is no separate ncols to plumb per matrix. The two
// operands still need equal size for product()/sum() to be dimensionally valid, though -- checked
// explicitly in both (see there), not left as a comment-only assumption.
class SpGEMMHelperT {
public:
  typedef amgcl::backend::builtin< double > Backend;
  typedef Backend::matrix                   BackendMatrix;

  std::shared_ptr< BackendMatrix > result_;

  static BackendMatrix makeMatrix( int n, const int* ptr, const int* col, const double* val )
  {
    int  nnz = ptr[n];
    auto A   = std::make_tuple( n,
                              amgcl::make_iterator_range( ptr, ptr + n + 1 ),
                              amgcl::make_iterator_range( col, col + nnz ),
                              amgcl::make_iterator_range( val, val + nnz ) );
    return BackendMatrix( A );
  }

  // result_ <- A * B (both n x n). aN == bN is required for a square A*B to even be dimensionally
  // valid (A's ncols must equal B's nrows) -- enforced here, not just documented, since amgcl::
  // backend::product() itself trusts its inputs and would otherwise read/write out of bounds on a
  // mismatched pair rather than fail cleanly.
  void product( int           aN,
                const int*    aPtr,
                const int*    aCol,
                const double* aVal,
                int           bN,
                const int*    bPtr,
                const int*    bCol,
                const double* bVal )
  {
    if ( aN != bN ) {
      throw std::runtime_error( "SpGEMMHelperT::product(): shape mismatch -- A is " + std::to_string( aN ) + "x" +
                                std::to_string( aN ) + ", B is " + std::to_string( bN ) + "x" + std::to_string( bN ) +
                                " (square matrices of equal size required)." );
    }
    BackendMatrix A = makeMatrix( aN, aPtr, aCol, aVal );
    BackendMatrix B = makeMatrix( bN, bPtr, bCol, bVal );
    result_         = amgcl::backend::product( A, B, /*sort=*/true );
  }

  // result_ <- alpha*A + beta*B (both n x n, same shape) -- see product()'s comment on why aN == bN
  // is enforced rather than only documented.
  void sum( double        alpha,
            int           aN,
            const int*    aPtr,
            const int*    aCol,
            const double* aVal,
            double        beta,
            int           bN,
            const int*    bPtr,
            const int*    bCol,
            const double* bVal )
  {
    if ( aN != bN ) {
      throw std::runtime_error( "SpGEMMHelperT::sum(): shape mismatch -- A is " + std::to_string( aN ) + "x" +
                                std::to_string( aN ) + ", B is " + std::to_string( bN ) + "x" + std::to_string( bN ) +
                                " (square matrices of equal size required)." );
    }
    BackendMatrix A = makeMatrix( aN, aPtr, aCol, aVal );
    BackendMatrix B = makeMatrix( bN, bPtr, bCol, bVal );
    result_         = amgcl::backend::sum( alpha, A, beta, B, /*sort=*/true );
  }

  int resultNRows() const { return result_ ? static_cast< int >( result_->nrows ) : 0; }
  int resultNCols() const { return result_ ? static_cast< int >( result_->ncols ) : 0; }
  int resultNnz() const { return result_ ? static_cast< int >( result_->nnz ) : 0; }

  // Copies the most recent result's CSR arrays into caller-owned buffers (sized resultNRows()+1,
  // resultNnz(), resultNnz() respectively) -- the result_ shared_ptr keeps owning its own storage
  // regardless, so this is a copy-out, not a transfer of ownership.
  void copyResult( int* ptrOut, int* colOut, double* valOut ) const
  {
    if ( !result_ ) {
      throw std::runtime_error( "SpGEMMHelperT::copyResult(): no result yet -- call product() or sum() first" );
    }
    std::copy( result_->ptr, result_->ptr + result_->nrows + 1, ptrOut );
    std::copy( result_->col, result_->col + result_->nnz, colOut );
    std::copy( result_->val, result_->val + result_->nnz, valOut );
  }
};

// Bridges AMGCL's own native amgcl::solver::lgmres (an outer Krylov solve running entirely on
// the OpenMP-threaded builtin backend) to blockamg.py's Python-level block Gauss-Seidel preconditioner
// (the same closure the shipped scipy.sparse.linalg.gmres path already uses), via a plain C
// function-pointer callback + opaque context set from Cython.
//
// Motivation: with the outer operator SpMV already threaded (EquilibratedSystemT above), the single
// largest remaining bucket in the shipped arm's own solve wall (measured at roughly 38% of total
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
// any C++-side state to learn it.
typedef void ( *PyPrecondApplyFn )( void* ctx, int n, const double* rhs, double* x );

// Adapts a PyPrecondApplyFn callback to amgcl::solver::lgmres's Precond interface: a single const
// apply(rhs, x) method (see amgcl/solver/precond_side.hpp's spmv() -- with the default
// preconditioning side `right` that this class assumes, apply() is the *only* thing lgmres ever calls
// on a Precond object; see LGMRESOuterSolverT's own comment for why `right` is not overridden here).
//
// rhs/x are written directly against amgcl::backend::builtin<double>::vector (numa_vector<double>),
// not kept as a generic template parameter: lgmres calls apply() exclusively on its own internal
// Arnoldi/augmentation-vector storage (vs[j], outer_v entries, the final correction dx), which is
// always exactly this concrete type for the Backend this class is built for -- there is no second
// vector type apply() is ever actually called with in this codebase's usage, so templating it, as
// amgcl::preconditioner::dummy does for genuine backend-portability, would cost readability for no
// real flexibility gained here.
class PyLGMRESPrecondT {
public:
  PyLGMRESPrecondT( PyPrecondApplyFn applyFn, void* ctx, int n ) : applyFn_( applyFn ), ctx_( ctx ), n_( n ) {}

  void apply( const amgcl::backend::builtin< double >::vector& rhs, amgcl::backend::builtin< double >::vector& x ) const
  {
    applyFn_( ctx_, n_, rhs.data(), x.data() );
  }

private:
  PyPrecondApplyFn applyFn_;
  void*            ctx_;
  int              n_;
};

/// Wraps amgcl::solver::lgmres<builtin<double>> as blockamg.py's outer Krylov solve, in place of
/// scipy.sparse.linalg.gmres, at both of that path's call sites (the main solve and the true-residual
/// continuation retry -- see blockamg.py's __call__).
///
/// One instance lives as long as the system size does: lgmres preallocates every scratch vector
/// (Arnoldi basis, augmentation vectors) for a fixed n in its own constructor, so a size change (AMR)
/// needs a fresh instance, and blockamg.py rebuilds it then -- and when its own settings change. Every
/// call starts from an empty augmentation subspace (lgmres's always_reset, forced here): carrying the
/// augmentation vectors across calls ("recycling") was measured to buy nothing on 9 representative
/// systems and, on a live trajectory, to compound a struggling solve's subspace into much more
/// expensive and once NaN subsequent ones, so it is not offered. The intra-call restart-cycle
/// augmentation (K) is unaffected.
///
/// tol/maxiter are set on every call on the solver's own public prm member -- blockamg's forcing and
/// continuation tolerances change per call -- and every call iterates on the EquilibratedSystemT it is
/// given, built fresh for that solve.
class LGMRESOuterSolverT {
public:
  typedef amgcl::backend::builtin< double > Backend;
  typedef amgcl::solver::lgmres< Backend >  Solver;

  std::unique_ptr< Solver > solver_;
  int                       n_;

  /// @brief Construct for a fixed problem size n.
  /// @param json_params lgmres::params' own sub-tree, e.g. {"M": 30, "K": 3}, forwarded verbatim;
  /// always_reset is forced to true (see the class comment).
  LGMRESOuterSolverT( int n, const char* json_params ) : n_( n )
  {
    boost::property_tree::ptree prm = parseJsonParameters( json_params );
    prm.put( "always_reset", true );
    // right preconditioning, whatever the caller's tree says: lgmres's own convergence test is then on the
    // residual of the operator it iterates on, which blockamg relies on (see its cap-hit and true-residual
    // logic); left preconditioning would test the preconditioned residual instead
    prm.put( "pside", "right" );
    Solver::params p( prm );
    solver_.reset( new Solver( static_cast< size_t >( n ), p ) );
  }

  /// @brief x <- A^-1 rhs, right-preconditioned by the callback (applyFn/ctx). x holds the initial
  /// guess on input (blockamg's warm-started continuation retry pre-fills it) and the solution on
  /// output; iters/error return lgmres's iteration count and its own scaled relative residual.
  void solve( const EquilibratedSystemT& system,
              const double*              rhs,
              double*                    x,
              double                     tol,
              int                        maxiter,
              PyPrecondApplyFn           applyFn,
              void*                      ctx,
              int&                       iters,
              double&                    error )
  {
    system.checkBuilt();
    const int n = system.n_;
    if ( n != n_ ) {
      throw std::runtime_error( "LGMRESOuterSolverT::solve(): n=" + std::to_string( n ) + " does not match the size " +
                                std::to_string( n_ ) + " this instance was constructed for." );
    }

    solver_->prm.tol     = tol;
    solver_->prm.maxiter = static_cast< size_t >( maxiter );

    PyLGMRESPrecondT precond( applyFn, ctx, n );

    auto rhs_rng = amgcl::make_iterator_range( rhs, rhs + n );
    auto x_rng   = amgcl::make_iterator_range( x, x + n );

    size_t itersOut;
    double errorOut;
    std::tie( itersOut, errorOut ) = ( *solver_ )( *system.operator_, precond, rhs_rng, x_rng );
    iters                          = static_cast< int >( itersOut );
    error                          = errorOut;
  }
};

// --- p-two-grid (P1) preconditioner ---------------------------------------------------------------
//
// The p-multigrid two-grid cycle of edelweissfe/linsolve/blockamg/ptwogrid.py, entirely in C++: nu
// fine Chebyshev sweeps, the fine residual, restriction through P^T, one AMGCL V-cycle on the
// Galerkin coarse operator A1 = P^T A P, prolongation through P, nu more fine Chebyshev sweeps. Every
// stage of build() and apply() is OpenMP-threaded; apply() runs without any Python in between.
//
// Dirichlet handling: the field's diagonal block arrives *with* its Dirichlet identity rows (a row
// whose only nonzero is its diagonal; stored zeros, which the Dirichlet elimination keeps, do not
// count). The multi-point-constraint condensation fills its slave rows and columns with the identity
// as well (the slaves are recovered from their masters after the solve); such a row is the same row
// to this class, so "Dirichlet" below means either.
// The former path extracted the free submatrix A_FF (Dirichlet rows *and* columns removed) and
// smoothed on that. This class keeps the full block layout instead and makes the Dirichlet DOFs
// decouple exactly:
//
//   * the fine right-hand side and the fine iterate are zero on Dirichlet DOFs, and a Dirichlet row
//     carries nothing but its diagonal, so every residual r_D = 0 - a_DD * 0 = 0, hence every
//     Chebyshev update leaves x_D = 0. On the free rows, A x = A_FF x_F + A_FD x_D = A_FF x_F,
//     whether or not the upstream Dirichlet elimination also cleared the columns A_FD;
//   * the power iteration starts from a vector that is zero on Dirichlet DOFs, which stays zero by
//     the same argument, so it estimates the spectral radius of A_FF, not of A;
//   * P has no rows for Dirichlet fine DOFs and no columns for Dirichlet coarse DOFs, so
//     P^T A P = P_F^T A_FF P_F exactly.
//
// This is what makes a node-wise 3x3 block storage of the fine operator possible at all: a node may
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
/// (scalar entries or node-wise 3x3 blocks) and value type (double or float). Vectors are always plain
/// double arrays of the field's scalar size.
class PTwoGridFineSmoother {
public:
  virtual ~PTwoGridFineSmoother() {}

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
inline void ptwogridFineResidual( const amgcl::backend::crs< MatrixValueType, int, int >& A,
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
/// double 3x3 block); MatrixValueType is how the operator is stored for the smoother and the residual:
/// ValueType itself, or its float counterpart -- half the bytes per entry, the dominant traffic of
/// the cycle. The spectral-radius estimate only ever runs on a double operator: a float level is made
/// from a double one (makeFloatCopy()), which has estimated or re-checked the radius before.
template < typename ValueType, typename MatrixValueType = ValueType >
class PTwoGridFineLevelT : public PTwoGridFineSmoother {
public:
  typedef amgcl::backend::crs< MatrixValueType, int, int >   Matrix;
  typedef typename amgcl::math::rhs_of< ValueType >::type    RhsType;
  typedef typename amgcl::math::scalar_of< ValueType >::type ScalarType;
  static const int                                           BlockSize = amgcl::math::static_rows< ValueType >::value;

  std::shared_ptr< Matrix > A_;
  std::vector< double >     r_, p_;
  unsigned                  degree_ = 0;
  double                    d_ = 0.0, c_ = 0.0;

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
  std::unique_ptr< PTwoGridFineLevelT< ValueType, FloatValueType > > makeFloatCopy() const
  {
    std::unique_ptr< PTwoGridFineLevelT< ValueType, FloatValueType > > level(
      new PTwoGridFineLevelT< ValueType, FloatValueType >() );
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
        ptwogridFineResidual( *A_, b, x, r_.data() );
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
    ptwogridFineResidual( *A_, b, x, res );
  }
};

/// @brief The p-two-grid preconditioner for one vector field's diagonal block: build() once per
/// hierarchy refresh, apply() once per outer Krylov iteration. See the comment block above for the
/// Dirichlet scheme and the spectral-radius reuse.
class PTwoGridPreconditionerT {
public:
  typedef amgcl::backend::builtin< double >                    CoarseBackend;
  typedef amgcl::runtime::preconditioner< CoarseBackend >      CoarseSolver;
  typedef amgcl::backend::builtin< float >                     CoarseBackendFloat;
  typedef amgcl::runtime::preconditioner< CoarseBackendFloat > CoarseSolverFloat;
  typedef amgcl::backend::crs< double, int, int >              ScalarMatrix;
  typedef amgcl::static_matrix< double, 3, 3 >                 Block3;
  typedef amgcl::static_matrix< float, 3, 3 >                  Block3Float;
  typedef std::chrono::steady_clock                            Clock;

  boost::property_tree::ptree prm;

  // fine level, in exactly one of the two storage formats; with fineValueType float, the double one
  // only lives during build() (for the spectral radius) and fineFloat_ replaces it
  std::unique_ptr< PTwoGridFineLevelT< double > > fineScalar_;
  std::unique_ptr< PTwoGridFineLevelT< Block3 > > fineBlock3_;
  std::unique_ptr< PTwoGridFineSmoother >         fineFloat_;
  PTwoGridFineSmoother*                           fine_ = nullptr;
  std::vector< int >                              dirichletRows_;
  /// the diagonal entry of each of dirichletRows_: its exact local solve is x_i = r_i / a_ii
  std::vector< double > dirichletDiagonal_;

  std::shared_ptr< ScalarMatrix > P_, PT_;
  // the coarse AMGCL hierarchy, in exactly one of the two value types (coarseValueType)
  std::unique_ptr< CoarseSolver >      coarse_;
  std::unique_ptr< CoarseSolverFloat > coarseFloat_;
  std::vector< float >                 coarseRhsFloat_, coarseXFloat_;
  std::vector< double >                coarseNullspace_;

  std::vector< char > isDirichlet_;
  int                 n_ = 0, nCoarseFree_ = 0;
  /// true after a build() of a field without a free DOF (empty, or entirely Dirichlet-constrained):
  /// apply() is then the exact (diagonal) inverse
  bool trivial_ = false;
  /// true once build() has completed; a build that throws leaves it false, so that apply() raises instead
  /// of using a half-built preconditioner
  bool built_ = false;

  // parameters, parsed and validated in the constructor (see there)
  int         nuPre_ = 1, nuPost_ = 1, fineBlockSize_ = 1;
  unsigned    fineDegree_ = 5;
  float       fineLower_ = 0.01f, fineHigher_ = 1.1f;
  std::string fineValueType_, coarseValueType_;
  int         finePowerIters_ = 300, fineRecheckIters_ = 15;
  double      fineRecheckTolerance_ = 0.01;

  double                fineSpectralRadius_ = 0.0;
  std::vector< double > fineDominantVector_;
  // how the last build arrived at its fine spectral radius: 0 estimated in full, 1 supplied and kept,
  // 2 supplied but raised by the re-check, 3 supplied but re-estimated in full by the re-check
  int                   fineSpectralRadiusSource_ = 0;
  std::vector< double > rhs_, res_, coarseRhs_, coarseX_;

  // stage timings [s], read by Python for the performance report
  double setupDirichletAndPSeconds_ = 0.0, setupGalerkinSeconds_ = 0.0, setupCoarseSeconds_ = 0.0,
         setupFineOperatorSeconds_ = 0.0, setupSpectralRadiusSeconds_ = 0.0;
  double applyFineSeconds_ = 0.0, applyCoarseSeconds_ = 0.0;

  /// @brief Parse the flat parameter tree: nuPre, nuPost, fineDegree, fineLower, fineHigher,
  /// finePowerIters, fineRecheckIters, fineRecheckTolerance, fineBlockSize, fineValueType,
  /// coarseValueType, and the coarse level's AMGCL preconditioner tree under "coarse". The sweep counts,
  /// the fine degree, the block size and the value types are validated here, so that a bad value fails
  /// before build() has spent anything on the prolongation, the Galerkin product or the coarse hierarchy.
  PTwoGridPreconditionerT( const char* json_params ) : prm( parseJsonParameters( json_params ) )
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
    if ( nuPre_ < 0 || nuPost_ < 0 ) {
      throw std::runtime_error( "ptwogrid: negative fine sweep counts" );
    }
    if ( fineDegree < 1 ) {
      throw std::runtime_error( "ptwogrid: fineDegree must be at least 1, got " + std::to_string( fineDegree ) );
    }
    fineDegree_ = static_cast< unsigned >( fineDegree );
    if ( fineBlockSize_ != 1 && fineBlockSize_ != 3 ) {
      throw std::runtime_error( "ptwogrid: fineBlockSize must be 1 or 3, got " + std::to_string( fineBlockSize_ ) );
    }
    if ( fineValueType_ != "double" && fineValueType_ != "float" ) {
      throw std::runtime_error( "ptwogrid: fineValueType must be double or float, got " + fineValueType_ );
    }
    if ( coarseValueType_ != "double" && coarseValueType_ != "float" ) {
      throw std::runtime_error( "ptwogrid: coarseValueType must be double or float, got " + coarseValueType_ );
    }
    // the Chebyshev interval is [fineLower, fineHigher] times the spectral radius: it must start above zero
    // and reach at least the radius itself, or the smoother amplifies the highest modes
    if ( !( fineLower_ > 0.0f && fineLower_ < 1.0f ) || !( fineHigher_ >= 1.0f ) ) {
      throw std::runtime_error( "ptwogrid: needs 0 < fineLower < 1 <= fineHigher, got fineLower " +
                                std::to_string( fineLower_ ) + ", fineHigher " + std::to_string( fineHigher_ ) );
    }
    if ( finePowerIters_ < 1 || fineRecheckIters_ < 0 || !( fineRecheckTolerance_ >= 0.0 ) ) {
      throw std::runtime_error( "ptwogrid: needs finePowerIters >= 1, fineRecheckIters >= 0 and "
                                "fineRecheckTolerance >= 0" );
    }
  }

  /// @brief Wall time [s] since start.
  static double secondsSince( const Clock::time_point& start )
  {
    return std::chrono::duration< double >( Clock::now() - start ).count();
  }

  /// @brief Build everything apply() needs.
  /// @param n, ptr, col, val the field's equilibrated diagonal block (scalar CSR, sorted columns,
  /// Dirichlet identity rows included). Copied; the caller's arrays are not referenced after build().
  /// @param nNodes, isCorner, edgeEndpoints the P1 topology map (edgeEndpoints: nNodes x 2,
  /// row-major; read for midside nodes only).
  /// @param nullspace, nullspaceCols the coarse near null-space, rows = free coarse DOFs in coarse
  /// order (corner order, then DOF), row-major; nullspaceCols == 0 for none.
  /// @param spectralRadius > 0: start from this fine spectral radius (typically a previous build's)
  /// instead of estimating it from scratch. With a dominantVector (n entries, the previous build's
  /// fineDominantVector()) it is re-checked first by fineRecheckIters power iterations from that
  /// vector, which cost a few sparse products instead of the full estimate's finePowerIters: a
  /// re-check estimate up to 1% above the supplied radius keeps it, one above that but within the smoother's
  /// `fineHigher` margin replaces it, and one beyond that margin -- the operator stiffened more than
  /// the margin covers -- triggers the full estimate. Without a dominantVector the radius is used as
  /// supplied.
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
    built_ = false;
    n_     = n;

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

    // A field without a free DOF -- empty, or every DOF Dirichlet-constrained -- has nothing to smooth
    // or to coarsen (and no coarse operator to build a hierarchy on): its exact inverse is diagonal,
    // which apply() then is.
    trivial_ = std::find( isDirichlet_.begin(), isDirichlet_.end(), 0 ) == isDirichlet_.end();
    coarse_.reset();
    coarseFloat_.reset();
    fineScalar_.reset();
    fineBlock3_.reset();
    fineFloat_.reset();
    fine_                     = nullptr;
    fineSpectralRadius_       = 0.0;
    fineSpectralRadiusSource_ = 0;
    fineDominantVector_.clear();
    nCoarseFree_ = 0;
    if ( trivial_ ) {
      setupDirichletAndPSeconds_ = secondsSince( start );
      setupGalerkinSeconds_ = setupCoarseSeconds_ = setupFineOperatorSeconds_ = setupSpectralRadiusSeconds_ = 0.0;
      built_                                                                                                = true;
      return;
    }

    // Every free row needs a nonzero diagonal (blockamg's equilibration makes it 1): without one -- a
    // zero block -- the coarse operator is singular and the Chebyshev bounds, relative to the fine
    // spectral radius, divide by zero. Checked before anything is built.
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
      throw std::runtime_error( "ptwogrid: free row " + std::to_string( rowWithoutDiagonal ) +
                                " (one of) has no nonzero diagonal entry -- is the field's block zero?" );
    }

    buildProlongation( nNodes, nDim, isCorner, edgeEndpoints );
    setupDirichletAndPSeconds_ = secondsSince( start );

    // The fine operator, copied once with an OpenMP-parallel first touch (crs's own range
    // constructor). The input arrays come from a single-threaded SciPy allocation, i.e. from one NUMA
    // node, and every sparse product on them -- the power iteration, the Galerkin product, every
    // smoothing step -- would otherwise be bound by that one node's memory bandwidth: used in place,
    // a 13-solve replay sequence took 339 s instead of 276 s on a two-socket machine (16 threads).
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
    setupFineOperatorSeconds_             = secondsSince( start );

    // Galerkin coarse operator, with AMGCL's own threaded SpGEMM, in the order AMGCL's own coarsening
    // forms it (amgcl/coarsening/detail/galerkin.hpp): P^T (A P).
    start                              = Clock::now();
    std::shared_ptr< ScalarMatrix > A1 = amgcl::backend::product( *PT_, *amgcl::backend::product( *Afine, *P_ ), true );
    setupGalerkinSeconds_              = secondsSince( start );

    // coarse AMGCL hierarchy, with the near null-space (a raw pointer in the tree, as for
    // LinearSolverT::set_nullspace)
    start = Clock::now();
    boost::property_tree::ptree                     coarsePrm;
    boost::optional< boost::property_tree::ptree& > coarseChild = prm.get_child_optional( "coarse" );
    if ( coarseChild )
      coarsePrm = *coarseChild;
    if ( nullspaceCols > 0 ) {
      if ( nullspaceRows != nCoarseFree_ ) {
        throw std::runtime_error( "ptwogrid: the coarse near null-space has " + std::to_string( nullspaceRows ) +
                                  " rows, but there are " + std::to_string( nCoarseFree_ ) + " free coarse DOFs." );
      }
      coarseNullspace_.assign( nullspace, nullspace + static_cast< size_t >( nullspaceRows ) * nullspaceCols );
      coarsePrm.put( "coarsening.nullspace.cols", nullspaceCols );
      coarsePrm.put( "coarsening.nullspace.rows", nullspaceRows );
      coarsePrm.put( "coarsening.nullspace.B", coarseNullspace_.data() );
    }
    // coarseValueType float builds and runs the whole coarse hierarchy -- including its own Chebyshev
    // smoother's power iteration -- in single precision, AMGCL's own backendPrecision float; the
    // coarse right-hand side and correction are converted at the boundary.
    if ( coarseValueType_ == "float" ) {
      coarseFloat_.reset( new CoarseSolverFloat( *A1, coarsePrm ) );
      coarseRhsFloat_.assign( nCoarseFree_, 0.0f );
      coarseXFloat_.assign( nCoarseFree_, 0.0f );
    }
    else
      coarse_.reset( new CoarseSolver( *A1, coarsePrm ) );
    A1.reset();
    setupCoarseSeconds_ = secondsSince( start );

    // fine level
    start = Clock::now();
    std::shared_ptr< PTwoGridFineLevelT< Block3 >::Matrix > Ablock;
    if ( fineBlockSize_ == 3 ) {
      Ablock = std::make_shared< PTwoGridFineLevelT< Block3 >::Matrix >(
        amgcl::adapter::block_matrix< Block3 >( *Afine ) );
      Afine.reset();
    }
    setupFineOperatorSeconds_ += secondsSince( start );

    start = Clock::now();
    if ( Ablock ) {
      fineBlock3_.reset( new PTwoGridFineLevelT< Block3 >() );
      fineBlock3_->setOperator( Ablock );
    }
    else {
      fineScalar_.reset( new PTwoGridFineLevelT< double >() );
      fineScalar_->setOperator( Afine );
    }
    fineSpectralRadius_ = chooseFineSpectralRadius( spectralRadius, dominantVector, dominantVectorSize, fineHigher_ );
    if ( !( fineSpectralRadius_ > 0.0 ) || !std::isfinite( fineSpectralRadius_ ) ) {
      // the Chebyshev bounds are relative to it: a zero (or non-finite) radius -- a zero block, which
      // blockamg's equilibration already refuses -- would divide by zero in the polynomial
      throw std::runtime_error( "ptwogrid: the fine operator has no positive, finite spectral radius (got " +
                                std::to_string( fineSpectralRadius_ ) + ") -- is the field's block zero?" );
    }
    if ( fineBlock3_ )
      fineBlock3_->setPolynomial( fineDegree_, fineLower_, fineHigher_, fineSpectralRadius_ );
    else
      fineScalar_->setPolynomial( fineDegree_, fineLower_, fineHigher_, fineSpectralRadius_ );
    if ( fineValueType_ == "float" ) {
      if ( fineBlock3_ )
        fineFloat_ = fineBlock3_->makeFloatCopy< Block3Float >();
      else
        fineFloat_ = fineScalar_->makeFloatCopy< float >();
      fineBlock3_.reset();
      fineScalar_.reset();
      fine_ = fineFloat_.get();
    }
    else if ( fineBlock3_ )
      fine_ = fineBlock3_.get();
    else
      fine_ = fineScalar_.get();
    setupSpectralRadiusSeconds_ = secondsSince( start );

    rhs_.assign( n, 0.0 );
    res_.assign( n, 0.0 );
    coarseRhs_.assign( nCoarseFree_, 0.0 );
    coarseX_.assign( nCoarseFree_, 0.0 );
    built_ = true;
  }

  /// @brief powerIters power iterations on the fine operator from startVector (see
  /// PTwoGridFineLevelT::estimateSpectralRadius); the final iterate goes to fineDominantVector_.
  double estimateFineSpectralRadius( int powerIters, const std::vector< double >& startVector )
  {
    if ( fineBlock3_ )
      return fineBlock3_->estimateSpectralRadius( powerIters, startVector, fineDominantVector_ );
    return fineScalar_->estimateSpectralRadius( powerIters, startVector, fineDominantVector_ );
  }

  /// @brief The full estimate, from the deterministic start vector: mt19937 seeded 0, uniform on [-1, 1], one
  /// draw per free DOF in order (AMGCL's single-thread draw, restricted to the free DOFs).
  double estimateFineSpectralRadiusInFull()
  {
    const int                                iters = finePowerIters_;
    std::vector< double >                    startVector( n_, 0.0 );
    std::mt19937                             rng( 0 );
    std::uniform_real_distribution< double > rnd( -1, 1 );
    for ( int i = 0; i < n_; ++i )
      if ( !isDirichlet_[i] )
        startVector[i] = rnd( rng );
    return estimateFineSpectralRadius( iters, startVector );
  }

  /// @brief The fine spectral radius for this build -- see build()'s `spectralRadius` parameter.
  double chooseFineSpectralRadius( double        suppliedRadius,
                                   const double* dominantVector,
                                   int           dominantVectorSize,
                                   float         higher )
  {
    if ( suppliedRadius <= 0.0 ) {
      fineSpectralRadiusSource_ = 0;
      return estimateFineSpectralRadiusInFull();
    }
    const int recheckIters = fineRecheckIters_;
    if ( dominantVector == nullptr || recheckIters <= 0 ) {
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
      return estimateFineSpectralRadiusInFull();
    }
    const double recheck = estimateFineSpectralRadius( recheckIters, startVector );
    // The full estimate itself is converged only to some 1e-5 relative after its 300 iterations, so
    // continuing it on an unchanged operator still creeps up a little; a rise below
    // fineRecheckTolerance (1%, well inside the 10% margin of `higher`) keeps the supplied radius, so
    // that an unchanged operator gives the same smoother as before.
    const double tolerance = fineRecheckTolerance_;
    if ( recheck <= suppliedRadius * ( 1.0 + tolerance ) ) {
      fineSpectralRadiusSource_ = 1;
      return suppliedRadius;
    }
    if ( recheck <= suppliedRadius * higher ) {
      fineSpectralRadiusSource_ = 2;
      return recheck;
    }
    fineSpectralRadiusSource_ = 3;
    return estimateFineSpectralRadiusInFull();
  }

  /// Dirichlet rows: a nonzero diagonal and no nonzero off-diagonal entry -- explicitly stored zeros,
  /// which the Dirichlet elimination (solvers/base/dirichlet.pyx) keeps, do not count. The exact
  /// decoupling of the class comment relies on nothing but that (a stored zero contributes 0 * x).
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
    nCoarseFree_ = 0;
    for ( int node = 0; node < nNodes; ++node )
      if ( isCorner[node] )
        for ( int d = 0; d < nDim; ++d )
          if ( !isDirichlet_[static_cast< size_t >( node ) * nDim + d] )
            coarseFreeIndex[static_cast< size_t >( cornerIndex[node] ) * nDim + d] = nCoarseFree_++;

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
    P_->set_size( n, nCoarseFree_, true );
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

  /// @brief `sweeps` fine sweeps; the first one from x = 0 if xIsZero
  void smoothFine( double* x, int sweeps, bool xIsZero )
  {
    for ( int sweep = 0; sweep < sweeps; ++sweep )
      fine_->smooth( rhs_.data(), x, xIsZero && sweep == 0 );
  }

  /// @brief x <- M^-1 r: one two-grid cycle on the free DOFs -- nuPre fine sweeps, coarse-grid
  /// correction, nuPost fine sweeps; Dirichlet DOFs are solved exactly, x_i = r_i / a_ii. The cycle's fine iterate
  /// is x itself: written from x = 0 without reading it (by the first pre-sweep, or, with no
  /// pre-sweeps, by the prolongated coarse correction itself), it stays zero on Dirichlet DOFs (zero
  /// right-hand side there, no rows of P) until they are set to r_i / a_ii last. Without pre-sweeps the
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
    if ( nuPre_ > 0 ) {
      smoothFine( x, nuPre_, /*xIsZero=*/true );
      fine_->residual( rhs_.data(), x, res_.data() );
    }
    const std::vector< double >& fineResidual = nuPre_ > 0 ? res_ : rhs_;
    applyFineSeconds_ += secondsSince( start );

    // the coarse AMG cycle overwrites its x itself (amgcl::amg::apply clears it, or copies rhs)
    start               = Clock::now();
    auto fineRange      = amgcl::make_iterator_range( x, x + n );
    auto coarseRhsRange = amgcl::make_iterator_range( coarseRhs_.data(), coarseRhs_.data() + nCoarseFree_ );
    auto coarseXRange   = amgcl::make_iterator_range( coarseX_.data(), coarseX_.data() + nCoarseFree_ );
    amgcl::backend::spmv( 1.0, *PT_, fineResidual, 0.0, coarseRhsRange );
    if ( coarseFloat_ ) {
      const ptrdiff_t nc = nCoarseFree_;
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
    else
      coarse_->apply( coarseRhsRange, coarseXRange );
    // x <- x + P e after pre-sweeps, x <- P e (overwriting) without
    amgcl::backend::spmv( 1.0, *P_, coarseXRange, nuPre_ > 0 ? 1.0 : 0.0, fineRange );
    applyCoarseSeconds_ += secondsSince( start );

    start = Clock::now();
    smoothFine( x, nuPost_, /*xIsZero=*/false );
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
      throw std::runtime_error( "ptwogrid: no preconditioner built -- call build() first (the last one failed?)" );
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
  /// @brief The number of free coarse DOFs, the size of the Galerkin operator.
  int nCoarseFree() const { return nCoarseFree_; }

  /// @brief the coarse level's AMGCL hierarchy report
  std::string report() const
  {
    checkBuilt();
    if ( trivial_ )
      return "ptwogrid: no free DOF, the preconditioner is the identity\n";
    std::ostringstream oss;
    if ( coarseFloat_ )
      oss << *coarseFloat_;
    else
      oss << *coarse_;
    return oss.str();
  }
};

// The default double-precision wrapper (plain amgcl::backend::builtin<double>, unchanged from before
// this module existed), and a float32 one for a mixed-precision backend (plain builtin<float>: the
// whole hierarchy -- not just the level matrices -- is float, see the comment above LinearSolverT).
typedef LinearSolverT< double > LinearSolver;
typedef LinearSolverT< float >  LinearSolverFloat;

// Block-valued instantiations: 3×3 for a 3D displacement field, 2×2 for a 2D one -- one template
// parameter apart. Both plain amgcl::backend::builtin<BlockType>, so relax.type = "gauss_seidel" and
// the exact ILU0/ILUk/ILUT/ILUP triangular solve work exactly as they do for the scalar backends above.
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

// AMGCL's own native outer Krylov solve (lgmres), see LGMRESOuterSolverT above.
typedef LGMRESOuterSolverT LGMRESOuterSolver;

// OpenMP-threaded SpGEMM/sparse-sum, see SpGEMMHelperT above.
typedef SpGEMMHelperT SpGEMMHelper;

// The p-two-grid (P1) preconditioner, see PTwoGridPreconditionerT above.
typedef PTwoGridPreconditionerT PTwoGridPreconditioner;
