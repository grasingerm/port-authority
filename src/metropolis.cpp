#include "metropolis.hpp"
#include "potentials.hpp"
#include <cmath>

#include "_metropolis_helpers.hpp"

using namespace std;

namespace pauth {

metric metropolis::DEFAULT_METRIC = _default_metric;
bc metropolis::DEFAULT_BC = _default_bc;
trial_move_generator metropolis::DEFAULT_TMG = _default_tmg;
acc metropolis::DEFAULT_ACC = _default_acc;

metropolis::metropolis(const molecular_id id, const size_t N, const size_t D, 
                       const double L, const double delta_max, 
                       abstract_potential *pot, const double T, 
                       const double kB, metric m, bc boundary, 
                       trial_move_generator tmg,
                       acc acceptance, const unsigned seed, const bool init_zeros) 
    : _molecular_ids(N, id), _positions(N, D), _edge_lengths(D), _V(_pow(L, D)), 
      _delta_max(delta_max),  _potentials(1, pot), 
      _T(T), _kB(kB), _beta(1.0 / (kB * T)), _m(m), _bc(boundary),
      _delta_dist(-delta_max, delta_max), _eps_dist(0.0, 1.0), 
      _choice_dist(0, N-1), _tmg(tmg), _acc(acceptance), _step(0), _dx(D), 
      _choice(0), _dU(0.0), _eps(0.0), _accepted(false) {

  _edge_lengths.fill(L);
  _dx.zeros();
  _rng.seed(seed);
  _edge_lengths.fill(L);
  if (init_zeros)
    _positions.zeros();
  else
    _init_positions_lattice(_positions, N, D, _edge_lengths);

}

metropolis::metropolis(const char *fname, const molecular_id id, const size_t N,
                       const size_t D, const double L, const double delta_max, 
                       abstract_potential* pot, const double T, 
                       const double kB, metric m, bc boundary,
                       trial_move_generator tmg,
                       acc acceptance, const unsigned seed) 
    : _molecular_ids(N, id), _positions(N, D), _edge_lengths(D), _V(_pow(L, D)), 
      _delta_max(delta_max),  _potentials(1, pot), 
      _T(T), _kB(kB), _beta(1.0 / (kB * T)), _m(m), _bc(boundary),
      _delta_dist(-delta_max, delta_max), _eps_dist(0.0, 1.0), 
      _choice_dist(0, N-1), _tmg(tmg), _acc(acceptance), _step(0), _dx(D), 
      _choice(0), _dU(0.0), _eps(0.0), _accepted(false) {
 
  _edge_lengths.fill(L);
  _dx.zeros();
  _rng.seed(seed);
  _edge_lengths.fill(L); 
  _load_positions(fname, _positions, N, D);

}

void metropolis::simulate(const long unsigned nsteps) {

  // Process initial conditions
  #pragma omp parallel for schedule(dynamic)
  for (auto cb_iter = _parallel_callbacks.cbegin(); 
       cb_iter < _parallel_callbacks.cend(); ++cb_iter) (*cb_iter)(*this);
  
  for (const auto &cb : _sequential_callbacks) cb(*this);

  // Run simulation
  for (; _step <= nsteps; ++_step) {
    // generator trial move and molecule choice
    const auto _choice = _choice_dist(_rng);
    _dx = _tmg(_positions, _choice, _delta_dist, _rng);

    // implement boundary condition, check if move allowed
    bool move_allowed;
    arma::vec new_x;
    tie(move_allowed, new_x) = _bc(*this, _choice, _dx);

    if (move_allowed) {
      // calculate change in energy
      double dU = 0.0;
      #pragma omp parallel for reduction(+:dU) schedule(dynamic)
      for (auto pot_iter = _potentials.cbegin(); pot_iter < _potentials.cend();
           ++pot_iter)
        dU += (*pot_iter)->delta_U(*this, _choice, new_x);
      
      // after reduction, store in simulation object
      _dU = dU;

      // do we accept or reject this move?
      _accepted = _acc(*this, _dU, (_eps = _eps_dist(_rng)));
      // make move while simultaneously implementing boundary conditions
      if (_accepted) _positions.col(_choice) = new_x;
    }

    // post-processing and processing
    #pragma omp parallel for schedule(dynamic)
    for (auto cb_iter = _parallel_callbacks.cbegin(); 
         cb_iter < _parallel_callbacks.cend(); ++cb_iter) (*cb_iter)(*this);
    
    for (const auto &cb : _sequential_callbacks) cb(*this);
  }

}

} // namespace pauth
