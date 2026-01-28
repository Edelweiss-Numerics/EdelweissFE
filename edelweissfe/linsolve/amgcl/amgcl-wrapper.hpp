#ifndef LIB_AMGCL_HPP
#define LIB_AMGCL_HPP

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

// OpenMP Backend
typedef amgcl::backend::builtin< double > Backend;

// Runtime Solver (Configurable via JSON)
typedef amgcl::make_solver< amgcl::runtime::preconditioner< Backend >, amgcl::runtime::solver::wrapper< Backend > >
  Solver;

class LinearSolver {
public:
  boost::property_tree::ptree prm;

  // Constructor: Just stores the parameters
  LinearSolver( const char* json_params )
  {
    std::string json_str( json_params );
    if ( !json_str.empty() ) {
      std::stringstream ss( json_str );
      boost::property_tree::read_json( ss, prm );
    }
  }

  // Solve: Takes raw CRS pointers, builds solver, and solves
  void solve( int           n,
              const int*    ptr,
              const int*    col,
              const double* val,
              const double* rhs,
              double*       x,
              int&          iters,
              double&       error )
  {

    // 1. Calculate Number of Non-Zeros (nnz)
    // ptr has size n+1, so the last element ptr[n] is the total nnz
    int nnz = ptr[n];

    // 2. Create Iterator Ranges for the raw pointers
    // AMGCL needs ranges (begin, end) to know the bounds
    auto ptr_rng = amgcl::make_iterator_range( ptr, ptr + n + 1 );
    auto col_rng = amgcl::make_iterator_range( col, col + nnz );
    auto val_rng = amgcl::make_iterator_range( val, val + nnz );

    // 3. Pack into std::tuple
    // Order: (Rows, RowPointers, ColumnIndices, Values)
    auto A = std::make_tuple( n, ptr_rng, col_rng, val_rng );

    // 2. Instantiate Solver (Setup Phase - Heavy computation)
    //    This builds the AMG hierarchy using the stored 'prm'.
    Solver S( A, prm );

    // 3. Solve (Solve Phase - Thread parallel)
    std::tie( iters, error ) = S( amgcl::make_iterator_range( rhs, rhs + n ), amgcl::make_iterator_range( x, x + n ) );
  }
};

#endif
