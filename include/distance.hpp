#ifndef __DISTANCE_HPP__
#define __DISTANCE_HPP__

#include <armadillo>
#include "pauth_types.hpp"

namespace pauth {

/*! \brief Get the relative position between molecules i and j
 *
 * \param     r_i           Position of first molecule
 * \param     r_j           Position of second molecule
 * \param                   Placeholder for simulation box edge lengths
 * \return                  Relative position between the molecules
 */
inline arma::vec euclidean_rij(const arma::vec &r_i, const arma::vec &r_j,
                               const arma::vec &) {
  return r_i - r_j;
}

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
inline double euclidean_m(const arma::vec &r_i, const arma::vec &r_j,
                          const arma::vec &edge_lengths) {
  const auto r_ij = euclidean_rij(r_i, r_j, edge_lengths);
  return arma::dot(r_ij, r_ij);
}

static metric euclidean(euclidean_rij, euclidean_m);

/*! \brief Get the relative position between molecules i and j
 *
 * \param     r_i           Position of first molecule
 * \param     r_j           Position of second molecule
 * \param                   Placeholder for simulation box edge lengths
 * \return                  Relative position between the molecules
 */
arma::vec periodic_euclidean_rij(const arma::vec &r_i, const arma::vec &r_j,
                                 const arma::vec &edge_lengths);

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
double periodic_euclidean_m(const arma::vec &r_i, const arma::vec &r_j,
                            const arma::vec &edge_lengths);

static metric periodic_euclidean(periodic_euclidean_rij, periodic_euclidean_m);

} // namespace pauth

#endif
