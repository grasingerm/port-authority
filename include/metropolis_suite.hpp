#ifndef __METROPOLIS_SUITE_HPP__
#define __METROPOLIS_SUITE_HPP__

#include <map>
#include "metropolis.hpp"
#include "mpi.h"

namespace pauth {

class metropolis_suite {
public:
  /*! \brief Constructor for a suite of Markov chain Monte Carlo simulations
   *
   * \param     sim         Simulation object to copy
   * \param     sg          Function for generating a seed
   * \return                Markov chain Monte Carlo simulation suite
   */
  metropolis_suite(const metropolis &sim,
                   seed_gen sg = _default_seed_gen);

  ~metropolis_suite() {
    if (!(_is_initialized)) MPI_Finalize();
  }

  /*! \brief Variables to average map
   *
   * \return      Variables to average map
   */
  inline const auto &variables_to_average_map() const { 
    return _variables_to_average_map; 
  }

  /*! \brief Variables to average map
   *
   * \return      Variables to average map
   */
  void add_variable_to_average(const char *key, value_accessor acc);

  /*! \brief Run simulation
   *
   * \param     nsteps      Number of steps to simulate
   */
  void simulate(const long unsigned nsteps);

  /*! \brief Suite variable averages
   *
   * \return      Suite variable averages
   */
  inline auto averages() {
    if (_taskid == 0) {
      auto _averages(_global_variables);
      for (auto &average : _averages) average.second /= _nsamples_global;
      return _averages;
    }
  }

private:
  metropolis _local_sim;
  std::map<std::string, value_accessor> _variables_to_average_map;
  std::map<std::string, long double> _local_variables;
  std::map<std::string, long double> _global_variables;
  int _rc;
  int _is_initialized;
  int _numtasks;
  int _taskid;
  long unsigned _nsamples;
  long unsigned _nsamples_global;
};

} // namespace pauth

#endif
