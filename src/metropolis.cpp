#include "metropolis.hpp"

namespace pauth {

template <size_t D> 
metropolis::metropolis<D>(const double delta_max, abstract_potential* pot, 
                          const double T, const double kB = 1.0) 
    : _kB(kB), _T(T), _beta(1.0 / (kB * T)), _delta_max(delta_max), 
      _delta_dist(-delta_max, delta_max), _eps_dist(0.0, 1.0), 
      _potentials(1, pot) {
  
}

} // namespace pauth
