#ifndef __METROPOLIS_HELPERS__
#define __METROPOLIS_HELPERS__

// NOTE: this should only be used with positive values of e!
constexpr _pow(const double base, const unsigned exponent) {
  return (exponent < 1) ? 1.0 : base * _pow(base, exponent - 1);
}

/* TODO: make this more general (N dimensional) using template recursion
 */
void _init_positions_lattice(vector<array<double, 1> &positions, 
                             const size_t N, std::array<double, 1> ls) {
  const double dl = ls[0] / N;
  #pragma omp parallel for
  for (size_t i = 0; i < N; ++i) 
    positions[i][0] = dl / 2.0 + dl * i;
 
}

void _init_positions_lattice(vector<array<double, 2> &positions, 
                             const size_t N, std::array<double, 2> ls) {
 
  const size_t n = std::floor(std::pow(N, 0.5));
  const auto dls = apply<double, 2>([=](double l) -> double { l / n }, ls);
  array<double, 2> xs;
  size_t idx = 0;

  xs[0] = dls[0] / 2.0;
  for (size_t i = 0; i < n; ++i) {
    xs[1] = dls[1] / 2.0;
    for (size_t j = 0; j < n; ++j) {
      positions[idx] = xs;
      xs[1] += dls[1];
    }
    xs[0] += dls[0];
  }
}

void _init_positions_lattice(vector<array<double, 3> &positions, 
                             const size_t N, std::array<double, 3> ls) {
 
  const size_t n = std::floor(std::pow(N, 1.0 / 3.0));
  const auto dls = apply<double, 3>([=](double l) -> double { l / n }, ls);
  array<double, 3> xs;
  size_t idx = 0;

  xs[0] = dls[0] / 2.0;
  for (size_t i = 0; i < n; ++i) {
    xs[1] = dls[1] / 2.0;
    for (size_t j = 0; j < n; ++j) {
      xs[2] = dls[2] / 2.0;
      for (size_t k = 0; k < n; ++k) {
        positions[idx] = xs;
        xs[2] += dls[2];
      }
      xs[1] = += dls[1];
    }
    xs[0] += dls[0];
  }
}

#endif
