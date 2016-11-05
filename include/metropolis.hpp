#ifndef __METROPOLIS_HPP__
#define __METROPOLIS_HPP__

#include <vector>
#include <array>
#include <random>
#include <chrono>
#include <functional>
#include <limits>
#include <armadillo>
#include "molecular.hpp"
#include "pauth_types.hpp"
#include "acceptance.hpp"

namespace pauth {

static const double _default_kB = 1.0;
static double (*_default_exp)(double) = std::exp;
static unsigned _default_seed;

// try using hardware entropy source, otherwise generator random seed with clock
try {
  _default_seed = std::random_device()(); // truly stochastic seed
}
catch (const std::exception &e) {
  _default_seed = std::chrono::high_resolution_clock::now().time_since_epoch() 
                  % std::numeric_limits<unsigned>::max();
}

static metric _default_metric = periodic_euclidean;
static bc _default_bc = periodic_bc;
static acc _default_acc = metropolis_acc;

class metropolis {
public:
  /*! \brief Constructor for a Markov chain Monte Carlo simulation
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
   * \param     seed        Seed for random number generator
   * \param     acceptance  Determines whether a move is accepted for rejected
   * \return                Markov chain Monte Carlo simulation object
   */
  metropolis(const molecular_id id, const size_t N, const size_t D, 
             const double L, const double delta_max, 
             const abstract_potential* pot, const double T, 
             const double kB = _default_kB, 
             metric m = _default_metric, bc boundary = _default_bc,
             const unsigned seed = _default_seed, 
             acc acceptance = _default_acc);

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
   * \param     seed        Seed for random number generator
   * \param     acceptance  Determines whether a move is accepted for rejected
   * \return                Markov chain Monte Carlo simulation object
   */
  metropolis(const char *fname, const molecular_id id, const size_t N, 
             const size_t D, const double L, const double delta_max, 
             const abstract_potential* pot, const double T, 
             const double kB = _default_kB, 
             metric m = _default_metric, bc boundary = _default_bc,
             const unsigned seed = _default_seed, 
             acc acceptance = _default_acc);

  /*! \brief Get number of molecules
   *
   * \return      Number of molecules
   */
  inline auto N() { return _molecular_ids.size(); }

  /*! \brief Get number of dimensions
   *
   * \return      Number of dimensions
   */
  inline auto D() { return _positions.n_rows; }

  /*! \brief Get molecular ids
   *
   * \return      Container of molecular ids
   */
  inline auto molecular_ids() { return _molecular_ids; }

  /*! \brief Get molecular positions
   *
   * \return      Molecular positions
   */
  inline auto &positions() { return _positions; }
  
  /*! \brief Get molecular positions
   *
   * \return      Molecular positions
   */
  inline const auto &positions() const { return _positions; }

  /*! \brief Get Boltzmann's constant
   *
   * \return      Boltzmann's constant
   */
  inline auto kB() const { return _kB; }

  /*! \brief Get temperature
   *
   * \return      Temperature
   */
  inline auto T() const { return _T; }

  /*! \brief Get beta
   *
   * \return      1 / kT
   */
  inline auto beta() const { return _beta; }

  /*! \brief Get edge lengths of simulation box
   *
   * \return      Edge lengths
   */
  inline const auto &edge_lengths() const { return _edge_lengths; }

  /*! \brief Get volume of simulation box
   *
   * \return      Volume
   */
  inline auto V() const { return _V; }

  /*! \brief Add a callback function
   *
   * \param   cb    Callback function to add
   */
  inline void add_parallel_callback(const callback cb) { 
    _parallel_callbacks.push_back(cb); 
  }

  /*! \brief Add a callback function
   *
   * \param   cb    Callback function to add
   */
  inline void add_sequential_callback(const callback cb) { 
    _sequential_callbacks.push_back(cb); 
  }

  /*! \brief Add a callback function to default container
   *
   * \param   cb    Callback function to add
   */
  inline void add_callback(const callback cb) { 
    add_sequential_callback(cb); 
  }

  /*! \brief Get current step
   *
   * \return    Current step
   */
  inline auto step() { return _step; }

  /*! \brief Run metropolis simulation
   *
   * \param   nsteps    Number of steps to simulate
   */
  void simulate(const long unsigned nsteps);

private:
  std::vector<molecular_id> molecular_ids;
  arma::mat _positions;
  arma::vec _edge_lengths;
  double _V;
  double _delta_max;
  std::vector<abstract_potential*> _potentials;
  double _T;
  double _kB;
  double _beta;
  metric _dist;
  bc _bc;
  std::default_random_engine _rng;
  std::uniform_real_distribution<double> _delta_dist;
  std::uniform_real_distribution<double> _eps_dist;
  std::uniform_int_distribution<size_t> _choice_dist;
  std::function<bool(const double, const double)> _acc;
  std::vector<callback> _parallel_callbacks;
  std::vector<callback> _sequential_callbacks;
  long unsigned _step;
};

} // namespace pauth

#endif
