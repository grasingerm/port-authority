#ifndef __ACCESSORS_HPP__
#define __ACCESSORS_HPP__

#include "metropolis.hpp"
#include <array>

namespace pauth {

namespace accessors {

/*! \brief Wrapper for accessing molecular positions
 *
 * \param   sim   metropolis object
 * \return        Molecular positions
 */
inline const arma::mat &positions(const metropolis &sim) {
  return sim.positions();
}

/*! \brief Wrapper for accessing current metropolis step
 *
 * \param   sim   metropolis object
 * \return        Current step
 */
inline auto step(const metropolis &sim) { return sim.step(); }

/*! \brief Get potential energy of simulation
 *
 * \param     sim     Metropolis simulation object
 * \return            Potential energy
 */
double U(const metropolis &sim);

/*! \brief Get ideal gas pressure of simulation
 *
 * \param     sim     Metropolis simulation object
 * \return            Ideal pressure
 */
inline double ideal_pressure(const metropolis &sim) {
  return sim.N() * sim.T() * sim.kB() / sim.V();
}

/*! \brief Get virial pressure of simulation
 *
 * \param     sim     Metropolis simulation object
 * \return            Virial pressure
 */
double virial_pressure(const metropolis &sim);

/*! \brief Get pressure of simulation
 *
 * \param     sim     Metropolis simulation object
 * \return            Pressure
 */
inline double P(const metropolis &sim) {
  return ideal_pressure(sim) + virial_pressure(sim);
}

/*! \brief Ideal pressure, virial pressure, and total pressure
 *
 * \param     sim     Simulation object
 * \return            Ideal pressure, virial pressure, and total pressure
 */
std::array<double, 3> ivp(const metropolis &sim) {
  std::array<double, 3> retval;
  retval[0] = ideal_pressure(sim);
  retval[1] = virial_pressure(sim);
  retval[2] = retval[0] + retval[1];
  return retval;
}

} // namespace acc
} // namespace pauth

#endif
