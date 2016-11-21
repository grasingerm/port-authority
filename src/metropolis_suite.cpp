#include <iostream>
#include <iomanip>
#include <chrono>
#include "metropolis_suite.hpp"
#include "pauth_debug.hpp"

using namespace std;

namespace pauth {

metropolis_suite::metropolis_suite(const metropolis &sim,
                                   const unsigned long equilibration_steps,
                                   const unsigned dstep,
                                   const info_lvl_flag ilf,
                                   ostream &ostr,
                                   seed_gen sg)
  : _nsamples_global(0), _info_lvl(ilf), _ostr(ostr) {

  _rc = MPI_Initialized(&_is_initialized); 
  except_mpi_rc(_rc);

  if (!(_is_initialized)) {
    MPI_Init(nullptr, nullptr);
    if (_info_lvl >= info_lvl_flag::DEBUG) _ostr << "MPI initialized.\n";
  }
    
  MPI_Comm_size(MPI_COMM_WORLD, &_numtasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &_taskid);

  if (_info_lvl >= info_lvl_flag::VERBOSE)
    _ostr << "MPI task " << _taskid << " started.\n";

  _nsamples = 0;
  _local_sim = sim;
  _local_sim.seed(sg() + _taskid);

  if (equilibration_steps) {
    _local_sim.simulate(equilibration_steps);
    _local_sim.reset_step();
  }

  _local_sim.add_callback([&, dstep](const metropolis &sim) {
    if (sim.step() % dstep == 0) {
      for (auto &local_variable : _local_variables)
        local_variable.second += 
          _variables_to_average_map[local_variable.first](sim);

      ++_nsamples;
    }
  });
}

void metropolis_suite::add_variable_to_average(const char *key, value_accessor acc) {
  _variables_to_average_map[key] = acc;
  _local_variables[key] = 0.0;
  _global_variables[key] = 0.0;
}

void metropolis_suite::simulate(const long unsigned nsteps) {
  if (_info_lvl >= info_lvl_flag::PROFILE && _taskid == 0) _ostr << "\n---\n";

  if (_info_lvl >= info_lvl_flag::VERBOSE) {
    _ostr << "Simulation started on task " << _taskid << ".\n";
  }
  double start_time = 0.0;
  if (_taskid == 0) start_time = MPI_Wtime();

  _local_sim.simulate(nsteps / _numtasks);

  if (_info_lvl >= info_lvl_flag::VERBOSE) {
    _ostr << "Simulation finished on task " << _taskid << ".\n";
  }

  _rc = MPI_Reduce(&_nsamples, &_nsamples_global, 1, MPI_UNSIGNED_LONG, MPI_SUM, 
                   0, MPI_COMM_WORLD);
  except_mpi_rc(_rc);

  for (auto &local_variable : _local_variables) {
    const auto key = local_variable.first;
    _rc = MPI_Reduce(&(local_variable.second), &(_global_variables[key]), 1,
                     MPI_LONG_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); 
    except_mpi_rc(_rc);
  }
  
  if (_info_lvl >= info_lvl_flag::PROFILE && _taskid == 0) { 
    const auto elapsed = MPI_Wtime() - start_time;
    _ostr << "Time elapsed: " << elapsed << " seconds\n";
  }
}

static size_t _w = 20;
static size_t _p = 12;

void metropolis_suite::report_averages() const {
  if (_taskid == 0) {
    _ostr << "\n---\n";
    if (_info_lvl >= info_lvl_flag::VERBOSE) {
      _ostr << setw(_w) << "Number of tasks" << " : " << _numtasks << '\n';
      _ostr << setw(_w) << "Samples per task" << " : " << _nsamples << '\n';
      _ostr << setw(_w) << "Total samples" << " : " << _nsamples_global << '\n';
    }
    for (auto &global_variable : _global_variables) {
      _ostr << setw(_w) << string("< ") + global_variable.first + string(" >") 
            << " : " << setprecision(_p) 
            << global_variable.second / _nsamples_global << '\n';
    }
  }
}

void metropolis_suite::tabulate_averages(const double x, const char delim) const {
  if (_taskid == 0) {
    _ostr << x;
    for (auto &global_variable : _global_variables)
      _ostr << delim << global_variable.second / _nsamples_global;
    _ostr << '\n';
  }
}

}
