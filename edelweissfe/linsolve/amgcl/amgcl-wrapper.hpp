#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/backend/builtin.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/preconditioner/runtime.hpp>
#include <amgcl/solver/runtime.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <sstream>
#include <string>
#include <tuple>
#include <memory>

typedef amgcl::backend::builtin< double > Backend;

typedef amgcl::make_solver< amgcl::runtime::preconditioner< Backend >, amgcl::runtime::solver::wrapper< Backend > >
  Solver;

class LinearSolver {
public:
  boost::property_tree::ptree prm;

  // Cached solver and matrix structure information
  std::unique_ptr< Solver > solver_;
  int                       cached_n;
  int                       cached_nnz;

  // Constructor: Just stores the parameters
  LinearSolver( const char* json_params )
    : solver_(), cached_n( -1 ), cached_nnz( -1 )
  {
    std::string json_str( json_params );
    if ( !json_str.empty() ) {
      std::stringstream ss( json_str );
      boost::property_tree::read_json( ss, prm );
    }
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
    } else if ( n != cached_n || nnz != cached_nnz ) {
      // Matrix structure changed: rebuild solver to preserve behavior
      solver_.reset( new Solver( A, prm ) );
      cached_n   = n;
      cached_nnz = nnz;
    } else {
      // Same structure: update preconditioner with new matrix values
      solver_->precond().update( A );
    }

    std::tie( iters, error ) =
      ( *solver_ )( amgcl::make_iterator_range( rhs, rhs + n ),
                   amgcl::make_iterator_range( x,   x   + n ) );
  }
};
