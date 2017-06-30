#ifndef __TRIAL_HPP__
#define __TRIAL_HPP__

#include <armadillo>
#include <random>
#include <cmath>
#include <vector>
#include <initializer_list>
#include "seed.hpp"

namespace pauth {

/*! \brief Trial move for continuous degrees of freedom e.g. positions in space
 */
class continuous_trial_move {
public:
  /*! \brief Trial move generator for a continuous monte carlo phase space
   *
   * \param     delta_max     Maximum step size
   * \param     sg            Seed generator
   * \return                  Trial move generator
   */
  continuous_trial_move(const double delta_max, seed_gen sg = _default_seed_gen) 
    : _delta_dist(-delta_max, delta_max) { _rng.seed(sg()); }

  /*! \brief Copy constructor for continuous trial move generator
   *
   * \param     ctm           Continuous trial move generator
   * \param     sg            Seed generator
   * \return                  Copy
   */
  continuous_trial_move(const continuous_trial_move &ctm, 
                        seed_gen sg = _default_seed_gen) 
    : _delta_dist(ctm._delta_dist.param()) { _rng.seed(sg()); }

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

/*! \brief Trial move for discrete states as opposed to continuous positions
 */
class state_trial_move {
public:
  /*! \brief Trial move generator for a phase space that consists of "states"
   *
   * \param     num_states     Number of available states
   * \param     sg             Seed generator
   * \return                   Trial move generator
   */
  state_trial_move(const unsigned num_states = 2,
                   seed_gen sg = _default_seed_gen) 
    : _state_dist(0, num_states-1) { _rng.seed(sg()); }

  /*! \brief Copy constructor for state trial move generator
   *
   * \param     stm           State trial move generator
   * \param     sg            Seed generator
   * \return                  Copy
   */
  state_trial_move(const state_trial_move &stm, seed_gen sg = _default_seed_gen) 
    : _state_dist(stm._state_dist.param()) { _rng.seed(sg()); }

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

/*! \brief Direct sampling trial move generator
 */
template <class T> class direct_sample {
public:
  /*! \brief Trial move generator for direct sampling by distributions
   *
   * \param     dists          Sampling distributions
   * \param     sg             Seed generator
   * \return                   Trial move generator
   */
  direct_sample<T>(std::initializer_list<T> dists, 
                   seed_gen sg = _default_seed_gen) : _dof_dists(dists),
    _choice_dist(0, _dof_dists.size()-1) {
    _rng.seed(sg());
  }

  /*! \brief Trial move generator for direct sampling by a distribution
   *
   * \param     dist           Sampling distribution
   * \param     N              Degrees of freedom per particle
   * \param     sg             Seed generator
   * \return                   Trial move generator
   */
  direct_sample<T>(T dist, size_t N, seed_gen sg = _default_seed_gen) : 
    _dof_dists(dist, N), _choice_dist(0, _dof_dists.size()-1) { 
    _rng.seed(sg()); 
  }

  /*! \brief Generates trial move for a two-state system
   *
   * \param     positions     Molecular positions
   * \param     j             Index of molecule to move
   * \return                  Trial move
   */
  arma::vec operator()(const arma::mat &positions, const size_t j) {
    auto dx = arma::zeros<arma::vec>(_dof_dists.size());
    auto choice = _rng(_choice_dist); // choose a degree of freedom
    // directly sample a degree of freedom from its associated distribution
    dx(choice) = _rng(_dof_dists(choice)) - positions(choice, j);
    return dx;
  }

private:
  std::vector<T> _dof_dists;
  std::uniform_int_distribution<size_t> _choice_dist;
  std::default_random_engine _rng;
};

}

#endif
