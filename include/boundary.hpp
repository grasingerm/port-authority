#ifndef __BOUNDARY_HPP__
#define __BOUNDARY_HPP__

#include <armadillo>
#include "pauth_types.hpp"

namespace pauth {

/*! Wall boundary condition
 *
 * \param     sim     Metropolis simulation
 * \param     j       Index of molecule to move
 * \param     dx      Distance to move
 */
bc_ret wall_bc(metropolis &sim, const size_t j, arma::vec &dx);

/*! Periodic boundary condition
 *
 * \param     sim     Metropolis simulation
 * \param     j       Index of molecule to move
 * \param     dx      Distance to move
 */
bc_ret periodic_bc(metropolis &sim, const size_t j, arma::vec &dx);

/*! No boundaries boundary condition
 *
 * \param     sim     Metropolis simulation
 * \param     j       Index of molecule to move
 * \param     dx      Distance to move
 */
bc_ret no_bc(metropolis &sim, const size_t j, arma::vec &dx);

} // namespace pauth

#endif
