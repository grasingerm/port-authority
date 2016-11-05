#include "simulation.hpp"
#include <cmath>

namespace mmd {

mutator equilibrate_temperature(const double tstar, const double eps,
                                const unsigned nsteps_eq) {
  unsigned nsteps = 0;
  return [=](simulation &sim) mutable {
    if (nsteps < nsteps_eq) {
      const double tcurr = temperature(sim);
      if (std::abs((tcurr - tstar) / tstar) < eps) {
        ++nsteps;
      } else {
        const double sqrt_alpha = sqrt(tstar / tcurr);
        auto &velocities = sim.get_velocities();

        for (size_t j = 0; j < velocities.n_cols; ++j)
          velocities.col(j) *= sqrt_alpha;
      }
    }
  };
}

mutator raise_temperature(const double tstar, const unsigned nsteps_eq) {
  unsigned nsteps = 0;
  return [=](simulation &sim) mutable {
    if (nsteps < nsteps_eq) {
      const double tcurr = temperature(sim);
      if (tcurr >= tstar) {
        ++nsteps;
      } else {
        const double sqrt_alpha = sqrt(tstar / tcurr);
        auto &velocities = sim.get_velocities();

        for (size_t j = 0; j < velocities.n_cols; ++j)
          velocities.col(j) *= sqrt_alpha;
      }
    }
  };
}

} // namespace mmd
