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

class metropolis {
public:
  metropolis(const molecular_id id, const size_t N, const size_t D, 
             const double L, const double delta_max, 
             const abstract_potential* pot, const double T, 
             const double kB = _default_kB, 
             metric m = _default_metric, bc boundary = _default_bc,
             const unsigned seed = _default_seed, 
             function<double(double)> fexp = _default_exp)

  /*! \brief Get number of molecules
   *
   * \return      Number of molecules
   */
  inline auto N() { return _molecular_ids.size(); }

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
  inline void add_callback(const callback cb) { callbacks.push_back(cb); }
  
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
  std::function<double(double)> _exp; // in-case we want to use a look-up table
  std::vector<callback> _callbacks;
};

} // namespace pauth

#endif
