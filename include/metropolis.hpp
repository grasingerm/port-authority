#ifndef __METROPOLIS_HPP__
#define __METROPOLIS_HPP__

#include <vector>
#include <array>
#include <random>
#include <chrono>
#include <functional>
#include <limits>

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

template <size_t D> class metropolis {
public:
  metropolis<D>(const double delta_max, abstract_potential* pot, const double L,
                const double T, const double kB = _default_kB, 
                const unsigned seed = _default_seed, 
                std::function<double(double)> fexp = _default_exp); 

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

private:
  std::vector<std::array<double, D>> _positions;
  double _kB;
  double _T;
  double _beta;
  double _delta_max;
  std::uniform_real_distribution<double> _delta_dist;
  std::uniform_real_distribution<double> _eps_dist;
  std::uniform_int_distribution<size_t> _choice_dist;
  std::normal_distribution<double> _eps_dist;
  std::vector<abstract_potential*> _potentials;
  std::array<double, D> _edge_lengths;
  double _V;
  std::function<double(double)> _exp; // in-case we want to use a look-up table
};

} // namespace pauth

#endif
