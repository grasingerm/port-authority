#ifndef __DISTANCE_HPP__
#define __DISTANCE_HPP__

#include <armadillo>
#include "pauth_types.hpp"

namespace pauth {

/*! \brief Get the square of the distance between molecules i and j
 *
 * Calculates the square of the distance between molecules i and j
 * Guarantees: >= 0
 *
 * \param     r_i           Position of first molecule
 * \param     r_j           Position of second molecule
 * \param                   Placeholder for simulation box edge lengths
 * \return                  Distance between the molecules squared
 */
inline double euclidean(const arma::vec &r_i, const arma::vec &r_j,
                        const arma::vec &) {
  const auto r_ij = r_i - r_j;
  return arma::dot(r_ij, r_ij);
}

/*! \brief Get the square of the distance between molecules i and j
 *
 * Calculates the square of the distance between molecules i and j
 * Guarantees: >= 0
 *
 * \param     r_i           Position of first molecule
 * \param     r_j           Position of second molecule
 * \param     edge_lengths  Simulation box edge lengths
 * \return                  Distance between the molecules squared
 */
double periodic_euclidean(const arma::vec &r_i, const arma::vec &r_j,
                          const arma::vec &edge_lengths);

} // namespace mmd

#endif
