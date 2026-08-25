#pragma once

#include <algorithm>
#include <amgcl/adapter/block_matrix.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/backend/builtin.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/preconditioner/runtime.hpp>
#include <amgcl/relaxation/runtime.hpp>
#include <amgcl/solver/lgmres.hpp>
#include <amgcl/solver/runtime.hpp>
#include <amgcl/value_type/static_matrix.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

// Templated on the AMGCL backend's value type, so the same wrapper serves both the default
// double-precision hierarchy and a float32 one (half the memory traffic in the smoother apply, the
// dominant cost on large coupled solves). The outer
// Krylov solve (blockamg's GMRES) always stays double; only the preconditioner's own storage and
// arithmetic narrow. rhs/x at the applyPreconditioner()/solve() boundary are always double -- the
// value-type-dependent scratch conversion happens inside this class, not at the Cython/Python
// boundary, since applyPreconditioner() is called once per outer Krylov iteration (hot path) while
// build() is not.
template < typename ValueType >
class LinearSolverT {
public:
  typedef amgcl::backend::builtin< ValueType > Backend;
  typedef amgcl::make_solver< amgcl::runtime::preconditioner< Backend >, amgcl::runtime::solver::wrapper< Backend > >
    Solver;

  boost::property_tree::ptree prm;

  // Cached solver and matrix structure information
  std::unique_ptr< Solver > solver_;
  int                       cached_n;
  int                       cached_nnz;
  std::vector< int >        cached_ptr_;
  std::vector< int >        cached_col_;

  // Near null-space vectors, kept alive here because the property tree only stores a raw pointer to
  // them (AMGCL copies from it when the hierarchy is built). Must outlive every solver construction.
  // Always double, independent of ValueType/Backend: AMGCL's coarsening::nullspace_params::B is
  // hardcoded std::vector<double> (amgcl/coarsening/tentative_prolongation.hpp) -- the tentative
  // prolongation's QR factorization stays double-precision regardless of backend, since it runs once
  // at hierarchy build time, not in the per-iteration smoother apply this backend split targets.
  std::vector< double > nullspace_;

  // Constructor: Just stores the parameters
  LinearSolverT( const char* json_params ) : solver_(), cached_n( -1 ), cached_nnz( -1 )
  {
    std::string json_str( json_params );
    if ( !json_str.empty() ) {
      std::stringstream ss( json_str );
      boost::property_tree::read_json( ss, prm );
    }
  }

  // Supply near null-space vectors for smoothed-aggregation coarsening. B is a rows-by-cols matrix in
  // row-major order (column j is the j-th near null-space vector). AMGCL takes these as a raw pointer
  // in its property tree -- they cannot travel through the JSON parameter string -- so this is a
  // separate entry point. Must be called before the first solve(); the pointer is read when the AMG
  // hierarchy is constructed. Passing cols == 0 clears any previously set null-space.
  void set_nullspace( const double* B, int rows, int cols )
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
  // current property tree (including any near null-space set via set_nullspace).
  void build( int n, const int* ptr, const int* col, const ValueType* val )
  {
    int  nnz = ptr[n];
    auto A   = std::make_tuple( n,
                              amgcl::make_iterator_range( ptr, ptr + n + 1 ),
                              amgcl::make_iterator_range( col, col + nnz ),
                              amgcl::make_iterator_range( val, val + nnz ) );
    solver_.reset( new Solver( A, prm ) );
    cached_n   = n;
    cached_nnz = nnz;
    cached_ptr_.assign( ptr, ptr + n + 1 );
    cached_col_.assign( col, col + nnz );
  }

  // Apply one preconditioner (AMG) cycle to rhs: x <- M^-1 rhs, where M is the hierarchy built by
  // :meth:`build`. This is the operation a block Gauss-Seidel / field-split preconditioner performs on
  // each field per outer iteration. Cheap relative to the build. rhs/x are double regardless of
  // ValueType -- for the float backend this narrows on the way in and widens on the way out; for the
  // (default) double backend the extra copy is a no-op-equivalent memcpy, not a behaviour change.
  void applyPreconditioner( int n, const double* rhs, double* x )
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
  std::string report() const
  {
    if ( !solver_ ) {
      throw std::runtime_error( "report(): no hierarchy built yet -- call build() or solve() first" );
    }
    std::ostringstream oss;
    oss << *solver_;
    return oss.str();
  }

  void solve( int              n,
              const int*       ptr,
              const int*       col,
              const ValueType* val,
              const double*    rhs,
              double*          x,
              int&             iters,
              double&          error )
  {

    int nnz = ptr[n];

    auto ptr_rng = amgcl::make_iterator_range( ptr, ptr + n + 1 );
    auto col_rng = amgcl::make_iterator_range( col, col + nnz );
    auto val_rng = amgcl::make_iterator_range( val, val + nnz );

    auto A = std::make_tuple( n, ptr_rng, col_rng, val_rng );

    // (Re)build or update the cached solver depending on matrix structure
    if ( !solver_ ) {
      // First call: construct the solver and cache matrix structure
      solver_.reset( new Solver( A, prm ) );
      cached_n   = n;
      cached_nnz = nnz;
      cached_ptr_.assign( ptr, ptr + n + 1 );
      cached_col_.assign( col, col + nnz );
    }
    else if ( n != cached_n || nnz != cached_nnz || !std::equal( ptr, ptr + n + 1, cached_ptr_.begin() ) ||
              !std::equal( col, col + nnz, cached_col_.begin() ) ) {
      // Matrix structure changed: rebuild solver to preserve behavior
      solver_.reset( new Solver( A, prm ) );
      cached_n   = n;
      cached_nnz = nnz;
      cached_ptr_.assign( ptr, ptr + n + 1 );
      cached_col_.assign( col, col + nnz );
    }
    else {
      solver_.reset( new Solver( A, prm ) );
    }

    std::vector< ValueType > rhs_v( rhs, rhs + n );
    std::vector< ValueType > x_v( n, ValueType( 0 ) );
    std::tie( iters, error ) = ( *solver_ )( amgcl::make_iterator_range( rhs_v.data(), rhs_v.data() + n ),
                                             amgcl::make_iterator_range( x_v.data(), x_v.data() + n ) );
    std::copy( x_v.begin(), x_v.end(), x );
  }
};

// Block-valued backend: the per-field hierarchy stores/operates on B×B nodal blocks
// (amgcl::static_matrix<double,B,B>) instead of scalar entries. Two motivations, in confidence order:
// (i) the CSR index arrays shrink by ~B² (one column index per block instead of per scalar entry --
// index traffic, not values, is typically the larger share of hierarchy bandwidth, which is exactly
// what this attacks); (ii) block-aware smoothers (block-ILU0, block-GS) invert each node's B×B
// coupling exactly, AMGCL's own canonical recipe for vector-PDE (elasticity) operators.
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
template < typename BlockType >
class LinearSolverBlockT {
public:
  typedef amgcl::backend::builtin< BlockType > Backend;
  typedef amgcl::make_solver< amgcl::runtime::preconditioner< Backend >, amgcl::runtime::solver::wrapper< Backend > >
    Solver;

  static const int BlockSize = amgcl::math::static_rows< BlockType >::value;

  boost::property_tree::ptree prm;
  std::unique_ptr< Solver >   solver_;
  int                         cached_n;
  int                         cached_nnz;
  std::vector< int >          cached_ptr_;
  std::vector< int >          cached_col_;

  LinearSolverBlockT( const char* json_params ) : solver_(), cached_n( -1 ), cached_nnz( -1 )
  {
    std::string json_str( json_params );
    if ( !json_str.empty() ) {
      std::stringstream ss( json_str );
      boost::property_tree::read_json( ss, prm );
    }
  }

  void set_nullspace( const double*, int, int )
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

  void build( int n, const int* ptr, const int* col, const double* val )
  {
    checkBlockDivisible( n );
    int  nnz = ptr[n];
    auto A   = std::make_tuple( n,
                              amgcl::make_iterator_range( ptr, ptr + n + 1 ),
                              amgcl::make_iterator_range( col, col + nnz ),
                              amgcl::make_iterator_range( val, val + nnz ) );
    solver_.reset( new Solver( amgcl::adapter::block_matrix< BlockType >( A ), prm ) );
    cached_n   = n;
    cached_nnz = nnz;
    cached_ptr_.assign( ptr, ptr + n + 1 );
    cached_col_.assign( col, col + nnz );
  }

  void applyPreconditioner( int n, const double* rhs, double* x )
  {
    std::fill( x, x + n, 0.0 );
    std::vector< double > rhs_v( rhs, rhs + n );
    std::vector< double > x_v( x, x + n );
    auto                  rhs_rng = amgcl::backend::reinterpret_as_rhs< BlockType >( rhs_v );
    auto                  x_rng   = amgcl::backend::reinterpret_as_rhs< BlockType >( x_v );
    solver_->precond().apply( rhs_rng, x_rng );
    std::copy( x_v.begin(), x_v.end(), x );
  }

  std::string report() const
  {
    if ( !solver_ ) {
      throw std::runtime_error( "report(): no hierarchy built yet -- call build() or solve() first" );
    }
    std::ostringstream oss;
    oss << *solver_;
    return oss.str();
  }

  void solve( int           n,
              const int*    ptr,
              const int*    col,
              const double* val,
              const double* rhs,
              double*       x,
              int&          iters,
              double&       error )
  {
    checkBlockDivisible( n );
    int  nnz = ptr[n];
    auto A   = std::make_tuple( n,
                              amgcl::make_iterator_range( ptr, ptr + n + 1 ),
                              amgcl::make_iterator_range( col, col + nnz ),
                              amgcl::make_iterator_range( val, val + nnz ) );

    if ( !solver_ || n != cached_n || nnz != cached_nnz || !std::equal( ptr, ptr + n + 1, cached_ptr_.begin() ) ||
         !std::equal( col, col + nnz, cached_col_.begin() ) ) {
      solver_.reset( new Solver( amgcl::adapter::block_matrix< BlockType >( A ), prm ) );
      cached_n   = n;
      cached_nnz = nnz;
      cached_ptr_.assign( ptr, ptr + n + 1 );
      cached_col_.assign( col, col + nnz );
    }
    else {
      solver_.reset( new Solver( amgcl::adapter::block_matrix< BlockType >( A ), prm ) );
    }

    std::vector< double > rhs_v( rhs, rhs + n );
    std::vector< double > x_v( n, 0.0 );
    std::tie( iters, error ) = ( *solver_ )( amgcl::backend::reinterpret_as_rhs< BlockType >( rhs_v ),
                                             amgcl::backend::reinterpret_as_rhs< BlockType >( x_v ) );
    std::copy( x_v.begin(), x_v.end(), x );
  }
};

// A standalone, runtime-selectable smoother -- e.g. the p-two-grid preconditioner's fine sweep
// (ptwogrid.py), which previously ran a hand-rolled serial scipy/numpy Chebyshev polynomial and was
// measured at 81%+ of the preconditioner's own apply time. amgcl::relaxation::
// as_preconditioner<Backend, Relax> cannot serve this directly: its Relax parameter is a
// *compile-time* template-template parameter, so a JSON "type" string (the convention every other
// method here follows, chosen via amgcl::runtime::preconditioner<Backend>) cannot select it. AMGCL
// ships exactly the type this needs though: amgcl::runtime::relaxation::wrapper<Backend> is itself a
// plain template<class Backend> class implementing the same apply/apply_pre/apply_post interface as
// any concrete relaxation, dispatching internally on its own "type" key (chebyshev, gauss_seidel,
// ilu0, spai0, ...) -- so it is used here directly as the smoother, one level below the full AMG
// hierarchy, rather than through as_preconditioner at all.
//
// as_preconditioner's own apply() always clears x first (a fresh, standalone preconditioner
// application), which does not fit ptwogrid.py's V-cycle: it needs a *from-zero* pre-smooth followed
// by a coarse-grid correction and then a *warm* post-smooth continuing from the corrected x, not two
// independent from-zero applications. Exposing apply_pre() directly (identical to apply_post() for
// chebyshev; both just continue the polynomial recursion from the given x) lets the Python side decide
// "start from zero" (zero x itself before the pre-smooth loop) and "how many sweeps" (call applyStep
// repeatedly) exactly as the old hand-rolled smooth(x, rhs, sweeps) closure did, so ptwogrid.py's
// build()/applyPreconditioner() need no structural change -- only the closure's implementation swaps
// from serial scipy/numpy to this OpenMP-threaded builtin backend.
//
// The relaxation object does not retain A (chebyshev's constructor only extracts the spectral radius
// and, if scale=true, the inverse diagonal); apply_pre() takes A again on every call. LinearSolverT's
// AMG hierarchy gets this for free -- amgcl::make_solver copies the input tuple into the hierarchy's
// own amgcl::backend::crs<...> once, and every level's relaxation object (including the coarse-level
// chebyshev used above) is then built and applied on *that*, never on the raw tuple. Bypassing the
// hierarchy (as this class does, deliberately) loses that conversion, and it turns out to be load-
// bearing, not incidental: several relaxation constructors (chebyshev included) call `rows(A)` /
// `diagonal(A, ...)` / `row_begin(A, i)` unqualified, relying on ADL to find amgcl::backend's
// overloads. ADL only reaches into a type's *own* namespace -- amgcl::backend::crs<...> lives there,
// so ADL finds them; the raw adapter tuple (amgcl::adapter::crs_tuple.hpp's std::tuple<int,
// iterator_range<...>, ...>) does not, even though the tuple adapter itself is genuinely sufficient
// for copying (crs's own generic-Matrix constructor calls the *qualified* backend::rows(A) etc., which
// does work on the tuple). Confirmed by reproducing exactly this failure first: passing the raw tuple
// straight to amgcl::runtime::relaxation::wrapper's constructor compiles the tuple-vs-crs mismatch
// into a wall of "'rows' was not declared in this scope" errors across every relaxation type the
// runtime wrapper's switch instantiates (chebyshev included, despite it being the only type actually
// selected at runtime -- the switch statement forces the compiler to instantiate all ten). The fix:
// convert once via Backend::matrix's own generic-Matrix constructor (the same conversion
// as_preconditioner.hpp's own init() performs) and keep the result alive for every applyStep() call.
class RelaxationSmootherT {
public:
  typedef amgcl::backend::builtin< double >              Backend;
  typedef amgcl::runtime::relaxation::wrapper< Backend > Relaxation;
  typedef Backend::matrix                                BackendMatrix;

  boost::property_tree::ptree      prm;
  std::shared_ptr< BackendMatrix > A_;
  std::unique_ptr< Relaxation >    relax_;
  std::vector< double >            tmp_;

  // Constructor: parses the relaxation's own parameter tree directly (a flat "type"-keyed tree, e.g.
  // {"type": "chebyshev", "degree": 5, "power_iters": 50, "lower": 0.01} -- not nested under
  // "precond.relax" like the full-hierarchy wrappers above, since there is no hierarchy here).
  RelaxationSmootherT( const char* json_params ) : relax_()
  {
    std::string json_str( json_params );
    if ( !json_str.empty() ) {
      std::stringstream ss( json_str );
      boost::property_tree::read_json( ss, prm );
    }
  }

  // Build the smoother for A once, so applyStep() can be called repeatedly without rebuilding
  // (mirrors LinearSolverT::build()'s build-once / apply-many split).
  void build( int n, const int* ptr, const int* col, const double* val )
  {
    int  nnz = ptr[n];
    auto A   = std::make_tuple( n,
                              amgcl::make_iterator_range( ptr, ptr + n + 1 ),
                              amgcl::make_iterator_range( col, col + nnz ),
                              amgcl::make_iterator_range( val, val + nnz ) );
    A_       = std::make_shared< BackendMatrix >( A );
    relax_.reset( new Relaxation( *A_, prm ) );
    tmp_.assign( n, 0.0 );
  }

  // One in-place smoothing step continuing from the given x (apply_pre; identical to apply_post for
  // chebyshev, the only relaxation type this has been exercised with so far -- see the class comment).
  // Call repeatedly for multiple sweeps; zero x beforehand for a from-zero (pre-smooth) application.
  void applyStep( int n, const double* rhs, double* x )
  {
    if ( !relax_ ) {
      throw std::runtime_error( "applyStep(): no smoother built yet -- call build() first" );
    }
    auto rhs_rng = amgcl::make_iterator_range( rhs, rhs + n );
    auto x_rng   = amgcl::make_iterator_range( x, x + n );
    auto tmp_rng = amgcl::make_iterator_range( tmp_.data(), tmp_.data() + n );
    relax_->apply_pre( *A_, rhs_rng, x_rng, tmp_rng );
  }

  // r <- rhs - A*x, computed on the same cached OpenMP-threaded backend matrix applyStep() already
  // built A_ from -- not otherwise part of "the smoother", but ptwogrid.py's V-cycle needs exactly
  // this fine-level residual to restrict through P before the coarse-grid correction, and it was
  // computing it via a plain scipy.sparse CSR matvec (A_free @ xFree), which is not OpenMP-threaded
  // (scipy's sparse matvec is single-threaded C code regardless of OMP_NUM_THREADS) and was found,
  // by comparing an isolated per-call timing against the coupled solve's own instrumentation, to be
  // silently costing ~30ms/call on a ~190k-DOF/139-nnz-per-row matrix -- charged to "coarseSeconds"
  // even though it has nothing to do with the coarse level, because it shares that timing block with
  // the actual coarse apply in applyPreconditioner(). Reusing A_ here (already built, already the
  // right backend matrix) avoids a second conversion and gets this SpMV onto the same threaded path
  // as everything else in this class, for free.
  void residual( int n, const double* rhs, const double* x, double* r )
  {
    if ( !A_ ) {
      throw std::runtime_error( "residual(): no smoother built yet -- call build() first" );
    }
    auto rhs_rng = amgcl::make_iterator_range( rhs, rhs + n );
    auto x_rng   = amgcl::make_iterator_range( x, x + n );
    auto r_rng   = amgcl::make_iterator_range( r, r + n );
    amgcl::backend::residual( rhs_rng, *A_, x_rng, r_rng );
  }
};

// A plain OpenMP-threaded matrix wrapper, no smoother attached: the shipped default's outer GMRES
// operator SpMV (`gmres(As, bs, ...)` in blockamg.py) is a scipy CSR matvec on the *full* coupled
// system, flat against `OMP_NUM_THREADS` for the same reason any raw scipy sparse matvec is (scipy
// sparse matvec is single-threaded C code). Measured at roughly 14.6% of the shipped arm's total
// wall-clock across a range of real reference systems -- large enough on its own to be worth
// threading independently of anything else.
//
// Deliberately a separate class from RelaxationSmootherT, not a reuse of it: RelaxationSmootherT's
// constructor runs a smoother build (chebyshev's power iteration for the spectral radius, etc.) --
// pure waste for something that only ever needs matvec()/residual(). This class never constructs a
// relaxation object, so RelaxationSmootherT's own ADL/crs_tuple pitfall (see its own comment) does
// not apply here; build() still converts through Backend::matrix regardless, matching the proven
// pattern rather than relying on the raw crs_tuple adapter's own (narrower) spmv/residual
// specializations.
class ThreadedMatrixT {
public:
  typedef amgcl::backend::builtin< double > Backend;
  typedef Backend::matrix                   BackendMatrix;

  std::shared_ptr< BackendMatrix > A_;

  // Build (convert) the matrix once per solve -- this pattern churns every solve on a model with
  // contact/tie constraints, so there is no build-once/apply-many amortization across solves the way
  // LinearSolverT's hierarchy gets; the conversion cost is paid fresh here every time and must be
  // measured, not assumed free.
  void build( int n, const int* ptr, const int* col, const double* val )
  {
    int  nnz = ptr[n];
    auto A   = std::make_tuple( n,
                              amgcl::make_iterator_range( ptr, ptr + n + 1 ),
                              amgcl::make_iterator_range( col, col + nnz ),
                              amgcl::make_iterator_range( val, val + nnz ) );
    A_       = std::make_shared< BackendMatrix >( A );
  }

  // y <- A*x
  void matvec( int n, const double* x, double* y )
  {
    if ( !A_ ) {
      throw std::runtime_error( "matvec(): no matrix built yet -- call build() first" );
    }
    auto x_rng = amgcl::make_iterator_range( x, x + n );
    auto y_rng = amgcl::make_iterator_range( y, y + n );
    amgcl::backend::spmv( 1.0, *A_, x_rng, 0.0, y_rng );
  }

  // r <- rhs - A*x
  void residual( int n, const double* rhs, const double* x, double* r )
  {
    if ( !A_ ) {
      throw std::runtime_error( "residual(): no matrix built yet -- call build() first" );
    }
    auto rhs_rng = amgcl::make_iterator_range( rhs, rhs + n );
    auto x_rng   = amgcl::make_iterator_range( x, x + n );
    auto r_rng   = amgcl::make_iterator_range( r, r + n );
    amgcl::backend::residual( rhs_rng, *A_, x_rng, r_rng );
  }

  // --- rectangular support -------------------------------------------------------------------
  // An off-diagonal field-coupling block (blockamg.py's block Gauss-Seidel sweep) is nrows x ncols
  // with nrows != ncols. build()/matvec() above size the input and output ranges by the same n,
  // which for a rectangular operator writes past the end of the output buffer, so rectangular
  // blocks carry their own dimension bookkeeping. Dimensions are taken from buildRect(), not from
  // the caller, so the output length can never be inferred from the input length again.
  int nrows_ = 0;
  int ncols_ = 0;

  void buildRect( int nrows, int ncols, const int* ptr, const int* col, const double* val )
  {
    int nnz = ptr[nrows];
    // crs has a dedicated rectangular constructor (nrows, ncols, ptr, col, val); the crs_tuple
    // adapter only understands the square 4-element form, so build the backend matrix directly.
    A_     = std::make_shared< BackendMatrix >( static_cast< size_t >( nrows ),
                                            static_cast< size_t >( ncols ),
                                            amgcl::make_iterator_range( ptr, ptr + nrows + 1 ),
                                            amgcl::make_iterator_range( col, col + nnz ),
                                            amgcl::make_iterator_range( val, val + nnz ) );
    nrows_ = nrows;
    ncols_ = ncols;
  }

  // y (length nrows_) <- A (nrows_ x ncols_) * x (length ncols_)
  void matvecRect( const double* x, double* y )
  {
    if ( !A_ ) {
      throw std::runtime_error( "matvecRect(): no matrix built yet -- call buildRect() first" );
    }
    auto x_rng = amgcl::make_iterator_range( x, x + ncols_ );
    auto y_rng = amgcl::make_iterator_range( y, y + nrows_ );
    amgcl::backend::spmv( 1.0, *A_, x_rng, 0.0, y_rng );
  }
};

// Exposes AMGCL's own OpenMP-threaded sparse-matrix-matrix product and
// sparse-matrix addition (amgcl::backend::builtin's product()/sum(), verified directly against the
// installed headers: spgemm_saad/spgemm_rmerge both carry `#pragma omp parallel`/`#pragma omp for`,
// and so does sum()). Wired into edelweissfe/numerics/mpctransformation.py's `T^T @ K @ T + C`
// condensation as an opt-in alternative (`useAmgclSpgemm`/`useAmgclMPCCondensation`, default False
// pending a live gate at more thread counts) to the plain expression's scipy CSR sparse routines --
// the same class of gap ThreadedMatrixT (above) closed for the outer GMRES matvec and
// RelaxationSmootherT closed for AMG relaxation. Deliberately generic (two raw CSR matrices in, one
// CSR matrix out) rather than
// specific to the T^T K T + C expression itself: the caller composes `product()`/`sum()` calls
// (e.g. `KT = product(K, T)`, `Kt_noC = product(T^T, KT)`, `Kt = sum(1, Kt_noC, 1, C)`), passing
// T^T as its own already-transposed CSR array (scipy's own `.T.tocsr()` is a cheap O(nnz)
// relayout, not a second SpGEMM, so there is nothing to gain by transposing on this side too).
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
// Motivation: with the outer operator SpMV already threaded (ThreadedMatrixT above), the single
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

// Wraps amgcl::solver::lgmres<builtin<double>> as blockamg.py's outer Krylov solve, in place of
// scipy.sparse.linalg.gmres, at both of that path's call sites (the main solve and the true-residual
// continuation retry -- see blockamg.py's __call__).
//
// One instance is meant to be constructed once per BlockAMGSolver and reused for that solver's entire
// lifetime, not rebuilt per solve like the per-field AMG hierarchies (see blockamg.py's own
// mustRefresh bookkeeping for those). This persistence is independent of `always_reset`: it is what
// lets tol/maxiter be mutated per call onto the same underlying object (see below) without paying
// construction cost every solve, and it is what `always_reset=false` would use *if* enabled -- but an
// attribution ablation found cross-call
// recycling (`outer_v` surviving across separate operator() calls, AMGCL's own doc comment on `K`
// naming "solving multiple similar problems" as the intended use) contributes nothing measurable on
// 9 representative systems and, on a live pryout trajectory, actively compounds a struggling solve's
// poorly-conditioned subspace into a much more expensive (and once, NaN) subsequent one. `blockamg.py`
// now defaults `always_reset=True` (AMGCL's own upstream default) accordingly -- this class still
// earns its keep over plain scipy GMRES from threading and lgmres's own intra-call restart-cycle
// augmentation alone (independent of `always_reset`, see the ablation), not from cross-call memory.
//
// The system matrix itself is *not* cached across calls the way the AMG hierarchies are -- it is
// rebuilt fresh in every solve() call from the raw CSR arrays, exactly like ThreadedMatrixT above
// (the pattern churns every solve regardless on a model with contact/tie constraints, so there is
// nothing to amortize). lgmres's
// own header comment explicitly names this usage pattern too: "The system matrix may differ from the
// matrix used during initialization[...] used for the solution of non-stationary problems with
// slowly changing coefficients."
//
// tol/maxiter are *not* baked into the params this instance was constructed with -- they are set on
// every solve() call by mutating the underlying amgcl::solver::lgmres object's own public `prm`
// member directly (see solve() below). This is what lets a single persistent instance still honour a
// different Eisenstat--Walker forcing tolerance and true-residual-continuation tolerance on every
// call (blockamg.py's `eta`/`continuationEta`) without reconstructing the object and losing the
// recycled Krylov vectors `always_reset=false` exists to keep -- rebuild-per-call and reuse-across-
// calls are mutually exclusive if tol/maxiter could only be set at construction, so mutating `prm` in
// place is what reconciles them.
//
// The problem size n is fixed at construction (lgmres preallocates every scratch vector -- Arnoldi
// basis, augmentation vectors -- for a fixed n in its own constructor); a field-structure or dof-count
// change (AMR, a new increment with a resized system) needs a fresh instance. blockamg.py's Python
// side already tracks exactly this condition for the AMG hierarchies (mustRefresh's `n != self._n`)
// and reuses that same signal to decide when to rebuild this object too.
class LGMRESOuterSolverT {
public:
  typedef amgcl::backend::builtin< double > Backend;
  typedef amgcl::solver::lgmres< Backend >  Solver;
  typedef Backend::matrix                   BackendMatrix;

  std::unique_ptr< Solver > solver_;
  int                       n_;

  // Constructed once for a fixed problem size n. json_params is lgmres::params' own sub-tree, e.g.
  // {"M": 30, "K": 3, "always_reset": true} -- forwarded 1:1 to amgcl::solver::lgmres::params'
  // ptree constructor with no translation layer, matching every field name AMGCL itself uses (M, K,
  // always_reset, pside, maxiter, tol, abstol, ns_search, verbose). tol/maxiter passed here only set
  // the *initial* defaults; solve() overrides both on every call (see the class comment above) --
  // included anyway so a caller that never overrides them still gets sane, explicit values rather
  // than silently depending on AMGCL's own defaults.
  LGMRESOuterSolverT( int n, const char* json_params ) : n_( n )
  {
    boost::property_tree::ptree prm;
    std::string                 json_str( json_params );
    if ( !json_str.empty() ) {
      std::stringstream ss( json_str );
      boost::property_tree::read_json( ss, prm );
    }
    Solver::params p( prm );
    solver_.reset( new Solver( static_cast< size_t >( n ), p ) );
  }

  // x <- A^-1 rhs, right-preconditioned by the callback (applyFn/ctx), continuing this instance's own
  // recycled/augmented Krylov vectors from any previous solve() call unless always_reset was
  // requested at construction. x provides the initial guess on input (matching amgcl::solver::
  // lgmres::operator()'s own x0-in/solution-out contract -- callers doing a warm-started continuation
  // retry, as blockamg.py's true-residual continuation loop does, must pre-fill x themselves) and
  // holds the solution on output. tol/maxiter are applied to this call (and every subsequent one,
  // until next changed) by mutating the underlying solver's own public `prm` member in place -- see
  // the class comment above for why this, rather than reconstructing the object, is required to keep
  // both per-call tolerances and cross-call Krylov-vector recycling at the same time.
  void solve( int              n,
              const int*       ptr,
              const int*       col,
              const double*    val,
              const double*    rhs,
              double*          x,
              double           tol,
              int              maxiter,
              PyPrecondApplyFn applyFn,
              void*            ctx,
              int&             iters,
              double&          error,
              bool             resetOnce = false )
  {
    if ( n != n_ ) {
      throw std::runtime_error( "LGMRESOuterSolverT::solve(): n=" + std::to_string( n ) + " does not match the size " +
                                std::to_string( n_ ) +
                                " this instance was constructed for -- construct a fresh instance on "
                                "any field-structure/size change (blockamg.py's own mustRefresh "
                                "condition already detects this for the AMG hierarchies; the same "
                                "signal is reused here)." );
    }

    solver_->prm.tol     = tol;
    solver_->prm.maxiter = static_cast< size_t >( maxiter );

    // A caller-requested one-shot reset
    // of the recycled/augmented Krylov vectors (`outer_v`), without discarding the persistent solver_
    // object or reconstructing it. AMGCL's own operator() clears outer_v itself, unconditionally,
    // whenever prm.always_reset is true (verified directly in lgmres.hpp: the very first statement in
    // operator()) -- so flipping it true for exactly this one call and restoring it immediately after
    // reuses that already-correct mechanism instead of reaching into outer_v/outer_v_data directly.
    // blockamg.py wires this to its own `newIncrement` signal: reset across increment/cutback
    // boundaries (where a stale recycled subspace is hypothesized to cost pure overhead), keep
    // recycling within an increment's own Newton sequence (AMGCL's own
    // intended use for K, and where every other ord showed recycling helping, not hurting).
    // RAII, not a plain save/mutate/restore: `(*solver_)(...)` below can throw (AMGCL internals,
    // e.g. an allocation failure -- not a Python exception from the preconditioner callback, which
    // never becomes a C++ exception here, see PyLGMRESPrecondT/the trampoline). A bare save-then-
    // restore-at-the-end would leave `always_reset` permanently flipped on the persistent, reused
    // `solver_` object if that call threw, silently changing every subsequent solve()'s recycling
    // behaviour. This guard restores unconditionally, on every exit path.
    struct AlwaysResetGuard {
      Solver::params& prm_;
      bool            saved_;
      AlwaysResetGuard( Solver::params& prm, bool resetOnce ) : prm_( prm ), saved_( prm.always_reset )
      {
        if ( resetOnce ) {
          prm_.always_reset = true;
        }
      }
      ~AlwaysResetGuard() { prm_.always_reset = saved_; }
    } alwaysResetGuard( solver_->prm, resetOnce );

    int  nnz = ptr[n];
    auto A   = std::make_tuple( n,
                              amgcl::make_iterator_range( ptr, ptr + n + 1 ),
                              amgcl::make_iterator_range( col, col + nnz ),
                              amgcl::make_iterator_range( val, val + nnz ) );
    // Converted through Backend::matrix, not passed as the raw adapter tuple -- the same ADL trap
    // RelaxationSmootherT and ThreadedMatrixT above already found and documented: amgcl::solver::
    // lgmres's operator() calls
    // backend::residual()/backend::spmv() on A via ADL-found amgcl::backend overloads that only
    // resolve against amgcl::backend::crs<...>, not the raw std::tuple adapter (amgcl/adapter/
    // crs_tuple.hpp).
    BackendMatrix Amat( A );

    PyLGMRESPrecondT precond( applyFn, ctx, n );

    auto rhs_rng = amgcl::make_iterator_range( rhs, rhs + n );
    auto x_rng   = amgcl::make_iterator_range( x, x + n );

    size_t itersOut;
    double errorOut;
    std::tie( itersOut, errorOut ) = ( *solver_ )( Amat, precond, rhs_rng, x_rng );
    iters                          = static_cast< int >( itersOut );
    error                          = errorOut;
  }
};

// The default, unchanged double-precision wrapper, and a float32 one for a mixed-precision backend.
typedef LinearSolverT< double > LinearSolver;
typedef LinearSolverT< float >  LinearSolverFloat;

// Block-valued instantiations: 3×3 for a 3D displacement field, 2×2 for a 2D one -- one template
// parameter apart.
typedef LinearSolverBlockT< amgcl::static_matrix< double, 2, 2 > > LinearSolverBlock2;
typedef LinearSolverBlockT< amgcl::static_matrix< double, 3, 3 > > LinearSolverBlock3;

// Standalone OpenMP-threaded relaxation smoother, see RelaxationSmootherT above.
typedef RelaxationSmootherT RelaxationSmoother;

// Standalone OpenMP-threaded matvec/residual, see ThreadedMatrixT above.
typedef ThreadedMatrixT ThreadedMatrix;

// AMGCL's own native outer Krylov solve (lgmres), see LGMRESOuterSolverT above.
typedef LGMRESOuterSolverT LGMRESOuterSolver;

// OpenMP-threaded SpGEMM/sparse-sum, see SpGEMMHelperT above.
typedef SpGEMMHelperT SpGEMMHelper;
