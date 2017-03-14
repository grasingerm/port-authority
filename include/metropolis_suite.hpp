#ifndef __METROPOLIS_SUITE_HPP__
#define __METROPOLIS_SUITE_HPP__

#include <iosfwd>
#include <map>
#include "metropolis.hpp"
#include "mpi.h"

namespace pauth {

enum class info_lvl_flag { QUIET, PROFILE, VERBOSE, DEBUG }; 

class metropolis_suite {
public:
  /*! \brief Constructor for a suite of Markov chain Monte Carlo simulations
   *
   * Constructs and equilibrates a Markov chain Monte Carlo simulation suite.
   * This ``suite'' consists of a collection of MCMC simulations, each running
   * on its own separate MPI process.
   *
   * \param     sim                   Simulation object to copy
   * \param     equilibration_steps   Number of steps to run at initialization
   * \param     dstep                 Number of steps in between samples 
   * \param     ilf                   Flag for determing diagnostic info to print
   * \param     ostr                  Output stream to print to
   * \param     sg                    Function for generating a seed
   * \return                          Markov chain Monte Carlo simulation suite
   */
  metropolis_suite(const metropolis &sim,
                   const unsigned long equilibration_steps = 0,
                   const unsigned dstep = 1,
                   const info_lvl_flag ilf = info_lvl_flag::QUIET,
                   std::ostream &ostr = std::cout,
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
  inline auto averages() const {
    auto _averages(_global_variables);
    for (auto &average : _averages) average.second /= _nsamples_global;
    return _averages;
  }

  /*! \brief Suite variables
   *
   * \return      Suite variables
   */
  inline const auto& global_variables() const {
    return _global_variables;
  }

  /*! \brief Report suite variable averages
   */
  void report_averages() const;

  /*! \brief Report suite variable averages
   */
  void tabulate_averages(const double x, const char delim=',') const;

  /*! \brief Set info level
   *
   * \param   info_lvl    Info level to set
   */
  inline void set_info_lvl(const info_lvl_flag info_lvl) {
    _info_lvl = info_lvl;
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
  info_lvl_flag _info_lvl;
  std::ostream &_ostr;
};

} // namespace pauth

#endif
