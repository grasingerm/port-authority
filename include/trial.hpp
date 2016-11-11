#ifndef __TRIAL_HPP__
#define __TRIAL_HPP__

#include <armadillo>
#include <random>
#include <cmath>

/*! Generators trial moves for a continuous phase space
 *
 * \param   positions   Matrix of positions
 * \param   j           Index of molecule to move
 * \param   delta_dist  Distribution to sample from
 * \param   rng         Random number generator
 * \return              Trial move
 */
inline arma::vec continuous_trial_move
  (const arma::mat& positions, const size_t, 
   std::uniform_real_distribution<double> &delta_dist, 
   std::default_random_engine &rng) {
  
  arma::vec dx(positions.n_rows);
  for (size_t i = 0; i < positions.n_rows; ++i) dx(i) = delta_dist(rng);
  return dx;

}

/*! Generates trial move for a two-state system
 *
 * \param   positions   Matrix of positions
 * \param   j           Index of molecule to move
 * \param   delta_dist  Distribution to sample from
 * \param   rng         Random number generator
 * \return              Trial move
 */
inline arma::vec twostate_trial_move_bf
  (const arma::mat& positions, const size_t j, 
   std::uniform_real_distribution<double> &delta_dist, 
   std::default_random_engine &rng) {
  
  arma::vec dx(1);
  dx(0) = std::round(delta_dist(rng));
  if (static_cast<unsigned>(positions(0, j)) == 0) {
    if (dx(0) < 0) dx(0) *= -1.0;
  }
  else {
    if (dx(0) > 0) dx(0) *= -1.0;
  }
  return dx;

}

/*! Generates trial move for a two-state system
 *
 * \param   positions   Matrix of positions
 * \param   j           Index of molecule to move
 * \param   delta_dist  Distribution to sample from
 * \param   rng         Random number generator
 * \return              Trial move
 */
inline arma::vec twostate_trial_move
  (const arma::mat&, const size_t, 
   std::uniform_real_distribution<double> &delta_dist, 
   std::default_random_engine &rng) {
  
  arma::vec dx(1);
  dx(0) = std::round(delta_dist(rng));
  return dx;

}

#endif
