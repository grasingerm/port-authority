#include "metropolis.hpp"
#include "array_helpers.hpp"
#include <cmath>

using namespace std;

namespace pauth {

#include "_metropolis_helpers.hpp"

template <size_t D> 
metropolis::metropolis<D>(const size_t N, const double delta_max, 
                          abstract_potential* pot, const double T, 
                          const double L, const double kB, 
                          function<double(double)> fexp) 
    : _positions(N), _kB(kB), _T(T), _beta(1.0 / (kB * T)), 
      _delta_max(delta_max), _delta_dist(-delta_max, delta_max), 
      _eps_dist(0.0, 1.0), _choice_dist(0, N-1), _potentials(1, pot), 
      _V(_pow(L, D)), _exp(fexp) {
  
  _edge_lengths.fill(L); 
  _init_positions_lattice(_positions, N, _edge_lengths);

}

} // namespace pauth
