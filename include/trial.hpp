#ifndef __TRIAL_HPP__
#define __TRIAL_HPP__

#include <armadillo>
#include <random>
#include <cmath>

class continuous_trial_move {
public:
  /*! \brief Trial move generator for a continuous monte carlo phase space
   *
   * \param     delta_max     Maximum step size
   * \return                  Trial move generator
   */
  continuous_trial_move(const double delta_max) 
    : _delta_dist(-delta_max, delta_max) {}

  /*! \brief Generators trial moves for a continuous phase space
   *
   * \param     positions     Molecular positions
   * \param     j             Index of molecule to move
   * \return                  Trial move
   */
  arma::vec operator()(const arma::mat &positions, const size_t) {
    arma::vec dx(positions.n_rows);
    for (size_t i = 0; i < positions.n_rows; ++i) dx(i) = _delta_dist(_rng);
    return dx;
  }

private:
  std::uniform_real_distribution<double> _delta_dist;
  std::default_random_engine _rng;
};

class state_trial_move {
public:
  state_trial_move(const unsigned num_states = 2) 
    : _state_dist(0, num_states-1) {}

  /*! \brief Generates trial move for a two-state system
   *
   * \param     positions     Molecular positions
   * \param     j             Index of molecule to move
   * \return                  Trial move
   */
  arma::vec operator()(const arma::mat &positions, const size_t j) {
    arma::vec dx(1);
    dx(0) = _state_dist(_rng);
    return dx - positions(0, j);
  }

private:
  std::uniform_int_distribution<unsigned> _state_dist;
  std::default_random_engine _rng;
};

#endif
