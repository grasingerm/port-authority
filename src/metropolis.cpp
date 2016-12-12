#include "metropolis.hpp"
#include "potentials.hpp"
#include <cmath>

#include "_metropolis_helpers.hpp"

using namespace std;

namespace pauth {

metric metropolis::DEFAULT_METRIC = _default_metric;
bc metropolis::DEFAULT_BC = _default_bc;
acc metropolis::DEFAULT_ACC = _default_acc;

metropolis::metropolis(const molecular_id id, const size_t N, const size_t D, 
                       const double L,
                       trial_move_generator tmg,
                       abstract_potential *pot, const double T, 
                       const double kB, metric m, bc boundary, 
                       acc acceptance, seed_gen sg, const bool init_zeros) 
    : _molecular_ids(N, id), _positions(D, N), _edge_lengths(D), _V(_pow(L, D)), 
      _potentials(1, pot), _T(T), _kB(kB), _beta(1.0 / (kB * T)), _m(m), 
      _bc(boundary), _eps_dist(0.0, 1.0), _choice_dist(0, N-1), _tmg(tmg), 
      _acc(acceptance), _step(0), _dx(D), _choice(0), _dU(0.0), _eps(0.0), 
      _accepted(false) {

  _edge_lengths.fill(L);
  _dx.zeros();
  _rng.seed(sg());
  _edge_lengths.fill(L);
  if (init_zeros)
    _positions.zeros();
  else
    _init_positions_lattice(_positions, N, D, _edge_lengths);

}

metropolis::metropolis(const char *fname, const molecular_id id, const size_t N,
                       const size_t D, const double L,
                       trial_move_generator tmg,
                       abstract_potential* pot, const double T, 
                       const double kB, metric m, bc boundary,
                       acc acceptance, seed_gen sg) 
    : _molecular_ids(N, id), _positions(D, N), _edge_lengths(D), _V(_pow(L, D)), 
      _potentials(1, pot), _T(T), _kB(kB), _beta(1.0 / (kB * T)), _m(m), 
      _bc(boundary), _eps_dist(0.0, 1.0), _choice_dist(0, N-1), _tmg(tmg), 
      _acc(acceptance), _step(0), _dx(D), _choice(0), _dU(0.0), _eps(0.0), _accepted(false) {
 
  _edge_lengths.fill(L);
  _dx.zeros();
  _rng.seed(sg());
  _edge_lengths.fill(L); 
  _load_positions(fname, _positions, N, D);

}

metropolis::metropolis(const char *fname, const size_t N,
                       const size_t D, const double L,
                       trial_move_generator tmg,
                       abstract_potential* pot, const double T, 
                       const double kB, metric m, bc boundary,
                       acc acceptance, seed_gen sg) 
    : _molecular_ids(N), _positions(D, N), _edge_lengths(D), _V(_pow(L, D)), 
      _potentials(1, pot), _T(T), _kB(kB), _beta(1.0 / (kB * T)), _m(m), 
      _bc(boundary), _eps_dist(0.0, 1.0), _choice_dist(0, N-1), _tmg(tmg), 
      _acc(acceptance), _step(0), _dx(D), _choice(0), _dU(0.0), _eps(0.0), _accepted(false) {
 
  _edge_lengths.fill(L);
  _dx.zeros();
  _rng.seed(sg());
  _edge_lengths.fill(L); 
  _load_positions(fname, _positions, N, D, _molecular_ids);

}

metropolis metropolis::operator=(const metropolis &rhs) {
  if (this == &rhs) return *this;

  _molecular_ids = rhs._molecular_ids;
  _positions = rhs._positions;
  _edge_lengths = rhs._edge_lengths;
  _V = rhs._V;
  _potentials = rhs._potentials;
  _T = rhs._T;
  _kB = rhs._kB;
  _beta = rhs._beta;
  _m = rhs._m;
  _bc = rhs._bc;
  _eps_dist = uniform_real_distribution<double>(0.0, 1.0);
  _choice_dist = uniform_int_distribution<size_t>(0, N()-1);
  _tmg = rhs._tmg;
  _acc = rhs._acc;
  _parallel_callbacks = rhs._parallel_callbacks;
  _sequential_callbacks = rhs._sequential_callbacks;
  _step = rhs._step;

  _dx = rhs._dx;
  _choice = rhs._choice;
  _dU = rhs._dU;
  _eps = rhs._eps;
  _accepted = rhs._accepted;

  return *this;
}

long unsigned metropolis::simulate(const long unsigned nsteps) {

  // Process initial conditions
  #pragma omp parallel for schedule(dynamic)
  for (auto cb_iter = _parallel_callbacks.cbegin(); 
       cb_iter < _parallel_callbacks.cend(); ++cb_iter) (*cb_iter)(*this);
  
  for (const auto &cb : _sequential_callbacks) cb(*this);

  // Run simulation
  for (; _step < nsteps; ++_step) {
    // generator trial move and molecule choice
    _choice = _choice_dist(_rng);
    _dx = _tmg(_positions, _choice);

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

    for (const auto &sc : _stopping_criteria) if (sc(*this)) goto exit_loop;
  }

exit_loop:
  return _step;

}

} // namespace pauth
