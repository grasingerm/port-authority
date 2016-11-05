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
                       const unsigned seed, function<double(double)> fexp) 
    : _molecular_ids(N, id), _positions(N, D), _V(_pow(L, D), 
      _delta_max(delta_max),  _potentials(1, pot), 
      _T(T), _kB(kB), _beta(1.0 / (kB * T)), _dist(m), _bc(boundary),
      _delta_dist(-delta_max, delta_max), _eps_dist(0.0, 1.0), 
      _choice_dist(0, N-1), _exp(fexp) {
 
  _rng.seed(seed);
  _edge_lengths.fill(L); 
  _init_positions_lattice(_positions, N, D, _edge_lengths);

}

} // namespace pauth
