#include "accessors.hpp"
#include "metropolis.hpp"

namespace pauth { 
namespace accessors {

double virial_pressure(const metropolis &sim) {
  const auto n = sim.N();
  const auto &positions = sim.positions();
  double vp = 0.0;

  for (const auto potential : sim.potentials())
    for (auto i = size_t{0}; i < n - 1; ++i)
      for (auto j = i + 1; j < n; ++j)
        // rij dot Fij
        vp += dot(sim.rij(positions.col(i), positions.col(j), sim.edge_lengths()),
                  potential->forceij(sim, i, j));

  return vp / (3.0 * sim.V());
}

} // namepsace accessors
} // namespace pauth
