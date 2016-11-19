#ifndef __METROPOLIS_SUITE_HPP__
#define __METROPOLIS_SUITE_HPP__

#include "metropolis.hpp"

namespace pauth {

class metropolis_suite {
public:
  /*! \brief Constructor for a suite of Markov chain Monte Carlo simulations
   *
   * \param     id          Molecular id of simulation molecules
   * \param     N           Number of molecules
   * \param     D           Number of dimensions 
   * \param     L           Simulation box edge length
   * \param     delta_max   Maximum move size
   * \param     pot         Molecular potential
   * \param     T           Simulation temperature
   * \param     kB          Boltzmann's constant
   * \param     m           Metric for measuring molecule-molecule distances
   * \param     bc          Boundary conditions
   * \param     acceptance  Determines whether a move is accepted for rejected
   * \param     seed        Seed for random number generator
   * \return                Markov chain Monte Carlo simulation object
   */
  metropolis_suite(const molecular_id id, const size_t N, const size_t D, 
                   const double L, const double delta_max, 
                   abstract_potential* pot, const double T, 
                   const double kB = _default_kB, 
                   metric m = _default_metric, bc boundary = _default_bc,
                   acc acceptance = _default_acc,
                   const unsigned seed = _default_seed); 

  /*! \brief Constructor for a Markov chain Monte Carlo simulation
   *
   * \param     fname       Filename of initial molecular positions
   * \param     id          Molecular id of simulation molecules
   * \param     N           Number of molecules
   * \param     D           Number of dimensions 
   * \param     L           Simulation box edge length
   * \param     delta_max   Maximum move size
   * \param     pot         Molecular potential
   * \param     T           Simulation temperature
   * \param     kB          Boltzmann's constant
   * \param     m           Metric for measuring molecule-molecule distances
   * \param     bc          Boundary conditions
   * \param     acceptance  Determines whether a move is accepted for rejected
   * \param     seed        Seed for random number generator
   * \return                Markov chain Monte Carlo simulation object
   */
  metropolis_suite(const char *fname, const molecular_id id, const size_t N, 
                   const size_t D, const double L, const double delta_max, 
                   abstract_potential* pot, const double T, 
                   const double kB = _default_kB, 
                   metric m = _default_metric, bc boundary = _default_bc,
                   acc acceptance = _default_acc,
                   const unsigned seed = _default_seed);

  /*! \brief Metropolis simulations
   *
   * \return      Collection of simulation objects
   */
  inline const auto &simulations() const { return _simulations; }
  
  /*! \brief Variables to average map
   *
   * \return      Variables to average map
   */
  inline const auto &variables_to_average_map() const { 
    return _variables_to_average_map; 
  }

private:
  std::vector<metropolis> _simulations;
  std::map<std::string, accessor> _variables_to_average_map;
};

} // namespace pauth

#endif
