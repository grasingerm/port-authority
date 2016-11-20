#include <iostream>
#include <iomanip>
#include "metropolis_suite.hpp"
#include "pauth_debug.hpp"

using namespace std;

namespace pauth {

metropolis_suite::metropolis_suite(const metropolis &sim, seed_gen sg)
  : _nsamples_global(0) {

  _rc = MPI_Initialized(&_is_initialized); 
  except_mpi_rc(_rc);

  if (!(_is_initialized)) MPI_Init(nullptr, nullptr);
    
  MPI_Comm_size(MPI_COMM_WORLD, &_numtasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &_taskid);

  _nsamples = 0;
  _local_sim = sim;

  _local_sim.add_callback([&](const metropolis &sim) {
    for (auto &local_variable : _local_variables)
      local_variable.second += 
        _variables_to_average_map[local_variable.first](sim);

    ++_nsamples;
  });
}

void metropolis_suite::add_variable_to_average(const char *key, value_accessor acc) {
  _variables_to_average_map[key] = acc;
  _local_variables[key] = 0.0;
  _global_variables[key] = 0.0;
}

void metropolis_suite::simulate(const long unsigned nsteps) {
  _local_sim.simulate(nsteps / _numtasks);
  
  _rc = MPI_Reduce(&_nsamples, &_nsamples_global, 1, MPI_UNSIGNED_LONG, MPI_SUM, 
                   0, MPI_COMM_WORLD);
  except_mpi_rc(_rc);

  for (auto &local_variable : _local_variables) {
    const auto key = local_variable.first;
    _rc = MPI_Reduce(&(local_variable.second), &(_global_variables[key]), 1,
                     MPI_LONG_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); 
    except_mpi_rc(_rc);
  }
}

static size_t _w = 20;
static size_t _p = 12;

void metropolis_suite::report_averages(ostream &ostr) {
  if (_taskid == 0) {
    for (auto &global_variable : _global_variables) {
      ostr << setw(_w) << global_variable.first << " = " 
           << setprecision(_p) << global_variable.second / _nsamples_global
           << '\n';
    }
  }
}

}
