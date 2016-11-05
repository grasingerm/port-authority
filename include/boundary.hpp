#ifndef __BOUNDARY_HPP__
#define __BOUNDARY_HPP__

#include <armadillo>

namespace pauth {

class metropolis;

/*! Wall boundary condition
 *
 * \param     sim     Metropolis simulation
 * \param     j       Index of molecule to move
 * \param     dx      Distance to move
 */
void wall_bc(metropolis &sim, const size_t j, arma::vec &dx);

/*! Periodic boundary condition
 *
 * \param     sim     Metropolis simulation
 * \param     j       Index of molecule to move
 * \param     dx      Distance to move
 */
void periodic_bc(metropolis &sim, const size_t j, arma::vec &dx);

} // namespace pauth

#endif
