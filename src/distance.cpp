#include "distance.hpp"

namespace pauth {

using namespace arma;

double periodic_euclidean(const arma::vec &r_i, const arma::vec &r_j,
                          const arma::vec &edge_lengths) {
  double rij2 = 0;

  #pragma omp parallel for reduction(+:rij2)
  for (auto k = size_t{0}; k < r_i.n_rows; ++k) {
    double rijk = r_i(k) - r_j(k);

    /* correct for periodic boundary conditions using nearest image convention
     */
    if (rijk > edge_lengths(k) / 2.0)
      rijk -= edge_lengths(k);
    else if (rijk < -edge_lengths(k) / 2.0)
      rijk += edge_lengths(k);

    rij2 += rijk * rijk;
  }
  return rij2;
}

} // namespace pauth
