#ifndef __METROPOLIS_HPP__
#define __METROPOLIS_HPP__

#include <vector>
#include <array>
#include <random>
#include <chrono>
#include <functional>
#include <limits>
#include <armadillo>
#include <exception>
#include "pauth_types.hpp"
#include "molecular.hpp"
#include "potentials.hpp"
#include "acceptance.hpp"
#include "boundary.hpp"
#include "distance.hpp"

namespace pauth {

static const double _default_kB = 1.0;

// try using hardware entropy source, otherwise generator random seed with clock
static unsigned _default_seed = std::random_device()();
//_default_seed = std::chrono::high_resolution_clock::now().time_since_epoch() 
//                % std::numeric_limits<unsigned>::max();

static metric _default_metric = periodic_euclidean;
static bc _default_bc = periodic_bc;
static acc _default_acc = metropolis_acc;

class metropolis {
public:
  static metric DEFAULT_METRIC;
  static bc DEFAULT_BC;
  static acc DEFAULT_ACC;

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
   * \param     acceptance  Determines whether a move is accepted for rejected
   * \param     seed        Seed for random number generator
   * \return                Markov chain Monte Carlo simulation object
   */
  metropolis(const molecular_id id, const size_t N, const size_t D, 
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
  metropolis(const char *fname, const molecular_id id, const size_t N, 
             const size_t D, const double L, const double delta_max, 
             abstract_potential* pot, const double T, 
             const double kB = _default_kB, 
             metric m = _default_metric, bc boundary = _default_bc,
             acc acceptance = _default_acc,
             const unsigned seed = _default_seed); 

  /*! \brief Get number of molecules
   *
   * \return      Number of molecules
   */
  inline auto N() const { return _molecular_ids.size(); }

  /*! \brief Get number of dimensions
   *
   * \return      Number of dimensions
   */
  inline auto D() const { return _positions.n_rows; }

  /*! \brief Get molecular ids
   *
   * \return      Container of molecular ids
   */
  inline const auto &molecular_ids() const { return _molecular_ids; }

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

  /*! \brief Get metric
   *
   * \return      Metropolis metric
   */
  inline auto m(const arma::vec &r_i, const arma::vec &r_j, 
                const arma::vec &edge_lengths) const { 
    return _m(r_i, r_j, edge_lengths);
  }

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

  /*! \brief Get maximum step size
   *
   * \return      Max step size
   */
  inline auto delta_max() const { return _delta_max; }

  /*! \brief Add a callback function
   *
   * \param   cb    Callback function to add
   */
  inline void add_parallel_callback(const callback &cb) const { 
    _parallel_callbacks.push_back(cb); 
  }

  /*! \brief Add a callback function
   *
   * \param   cb    Callback function to add
   */
  inline void add_sequential_callback(const callback &cb) const { 
    _sequential_callbacks.push_back(cb); 
  }

  /*! \brief Add a callback function to default container
   *
   * \param   cb    Callback function to add
   */
  inline void add_callback(const callback &cb) const { 
    add_sequential_callback(cb); 
  }

  /*! \brief Get current step
   *
   * \return    Current step
   */
  inline auto step() const { return _step; }

  /*! \brief Molecular potentials accessor
   *
   * \return    Molecular potentials
   */
  inline const auto &potentials() const { return _potentials; }

  /*! \brief Accessor for trial move
   *
   * \return    Change in x for trial move
   */
  inline const auto &dx() const { return _dx; }

  /*! \brief Index of molecule that has been chosen to move
   *
   * \return      Molecule number of ``chosen'' molecule
   */
  inline auto choice() const { return _choice; }

  /*! \brief Accessor for change in energy that results from the move
   *
   * \return      Change in energy as a result of trial move
   */
  inline const auto &dU() const { return _dU; }

  /*! \brief Accessor for random number used to determine acceptance/rejection
   *
   * \return      Epsilon such that epsilon < B results in accepting the move
   */
  inline const auto &eps() const { return _eps; }
  
  /*! \brief Accessor for whether or not the trial move was accepted
   *
   * \return      Answers: ``was the trial move accepted?''
   */
  inline const auto &accepted() const { return _accepted; }

  /*! \brief Run metropolis simulation
   *
   * \param   nsteps    Number of steps to simulate
   */
  void simulate(const long unsigned nsteps);

private:
  std::vector<molecular_id> _molecular_ids;
  arma::mat _positions;
  arma::vec _edge_lengths;
  double _V;
  double _delta_max;
  std::vector<abstract_potential*> _potentials;
  double _T;
  double _kB;
  double _beta;
  metric _m;
  bc _bc;
  std::default_random_engine _rng;
  std::uniform_real_distribution<double> _delta_dist;
  std::uniform_real_distribution<double> _eps_dist;
  std::uniform_int_distribution<size_t> _choice_dist;
  acc _acc;
  mutable std::vector<callback> _parallel_callbacks;
  mutable std::vector<callback> _sequential_callbacks;
  long unsigned _step;

  arma::vec _dx;
  size_t _choice;
  double _dU;
  double _eps;
  bool _accepted;
};

} // namespace pauth

#endif
