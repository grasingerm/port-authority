#ifndef __METROPOLIS_HPP__
#define __METROPOLIS_HPP__

#include <vector>
#include <array>
#include <random>

namespace pauth {

template <size_t D> class metropolis {
public:
  metropolis<D>(const double delta_max, abstract_potential* pot, const double T,
                const double kB = 1.0); 

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
  std::normal_distribution<double> _delta_dist;
  std::normal_distribution<double> _eps_dist;
  std::vector<abstract_potential*> _potentials;
  std::<double, D> _edge_lengths;
  double _V;
};

} // namespace pauth

#endif
