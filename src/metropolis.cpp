#include "metropolis.hpp"
#include "array_helpers.hpp"
#include <cmath>

using namespace std;

namespace pauth {

#include "_metropolis_helpers.hpp"

metropolis::metropolis(const molecular_id id, const size_t N, const size_t D, 
                       const double L, const double delta_max, 
                       const abstract_potential* pot, const double T, 
                       const double kB, metric m, bc boundary, 
                       const unsigned seed, acc acceptance) 
    : _molecular_ids(N, id), _positions(N, D), _V(_pow(L, D), 
      _delta_max(delta_max),  _potentials(1, pot), 
      _T(T), _kB(kB), _beta(1.0 / (kB * T)), _dist(m), _bc(boundary),
      _delta_dist(-delta_max, delta_max), _eps_dist(0.0, 1.0), 
      _choice_dist(0, N-1), _acc(acceptance), _step(0) {
 
  _rng.seed(seed);
  _edge_lengths.fill(L); 
  _init_positions_lattice(_positions, N, D, _edge_lengths);

}

metropolis::metropolis(const char *fname, const molecular_id id, const size_t N,
                       const size_t D, const double L, const double delta_max, 
                       const abstract_potential* pot, const double T, 
                       const double kB, metric m, bc boundary, 
                       const unsigned seed, acc acceptance) 
    : _molecular_ids(N, id), _positions(N, D), _V(_pow(L, D), 
      _delta_max(delta_max),  _potentials(1, pot), 
      _T(T), _kB(kB), _beta(1.0 / (kB * T)), _dist(m), _bc(boundary),
      _delta_dist(-delta_max, delta_max), _eps_dist(0.0, 1.0), 
      _choice_dist(0, N-1), _acc(acceptance), _step(0) {
 
  _rng.seed(seed);
  _edge_lengths.fill(L); 
  _load_positions(fname, _positions, N, D);

}

void metropolis::simulate(const long unsigned nsteps) {

  // Process initial conditions
  #pragma omp parallel for
  for (const auto &cb : _parallel_callbacks) cb(*this);
  
  for (const auto &cb : _sequential_callbacks) cb(*this);

  // Run simulation
  for (; _step <= nsteps; ++_step) {
    // generator trial move and molecule choice
    arma::vec dx(sim.D());
    for (size_t i = 0; i < sim.D(); ++i) dx(i) = _rng(_delta_dist);
    const auto choice = _rng(_choice_dist);

    // calculate change in energy
    double dU = 0.0;
    #pragma omp parallel for reduction(+:dU)
    for (const auto &pot : _potentials)
      dU += pot.delta_U(*this, choice, dx);

    // do we accept or reject this move?
    const bool accept = _acc(*this, dU, _rng(_eps_dist));
    // make move while simultaneously implementing boundary conditions
    if (accept) _bc(*this, choice, dx);

    // post-processing and processing
    #pragma omp parallel for
    for (const auto &cb : _parallel_callbacks) cb(*this);
    
    for (const auto &cb : _sequential_callbacks) cb(*this);
  }

}

} // namespace pauth
