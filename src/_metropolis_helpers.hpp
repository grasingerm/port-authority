#ifndef __METROPOLIS_HELPERS__
#define __METROPOLIS_HELPERS__

#include <stdexcept>

// NOTE: this should only be used with positive values of e!
constexpr _pow(const double base, const unsigned exponent) {
  return (exponent < 1) ? 1.0 : base * _pow(base, exponent - 1);
}

/* TODO: make this more general (N dimensional) using recursion
 */
void _init_positions_lattice(arma::mat &positions, const size_t N, 
                             const size_t D, const arma::vec &ls) {

  switch(D) {
    case 1: {
      const double dl = ls[0] / N;
      #pragma omp parallel for
      for (size_t i = 0; i < N; ++i) 
        positions(0, i) = dl / 2.0 + dl * i;

      break;
    }
    case 2: {
      const size_t n = std::floor(std::pow(N, 0.5));
      const auto dls = ls / n;
      arma::vec xs(2);
      size_t idx = 0;

      xs(0) = dls(0) / 2.0;
      for (size_t i = 0; i < n; ++i) {
        xs(1) = dls(1) / 2.0;
        for (size_t j = 0; j < n; ++j) {
          positions.col(idx) = xs;
          xs(1) += dls(1);
        }
        xs(0) += dls(0);
      }

      break;
    }
    case 3: {
      const size_t n = std::floor(std::pow(N, 1.0 / 3.0));
      const auto dls = ls / n;
      arma::vec xs(3);
      size_t idx = 0;

      xs(0) = dls(0) / 2.0;
      for (size_t i = 0; i < n; ++i) {
        xs(1) = dls(1) / 2.0;
        for (size_t j = 0; j < n; ++j) {
          xs(2) = dls(2) / 2.0;
          for (size_t k = 0; k < n; ++k) {
            positions.col(idx) = xs;
            xs(2) += dls(2);
          }
          xs(1) = += dls(1);
        }
        xs(0) += dls(0);
      }

      break;
    }
    default: {
      throw std::domain_error("Higher dimensional default metropolis "
          "initialization not yet implemented");
    }
  }
}
  
}

#endif
