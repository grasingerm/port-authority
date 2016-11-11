#include "boundary.hpp"
#include "metropolis.hpp"

namespace pauth {

using namespace arma;

bc_ret wall_bc(metropolis &sim, const size_t j, arma::vec &dx) {
  const auto &edge_lengths = sim.edge_lengths();
  const auto D = sim.D();
  bool wall_hit = false;
  for (size_t i = 0; i < D; ++i) {
    dx(i) += sim.positions()(i, j);
    if (dx(i) > edge_lengths(i) || dx(i) < 0.0) {
      wall_hit = true;
      break;
    }
  }
  return bc_ret(!wall_hit, dx);
}

bc_ret periodic_bc(metropolis &sim, const size_t j, arma::vec &dx) {
  const auto &edge_lengths = sim.edge_lengths();
  const auto dim = edge_lengths.n_rows;
  for (size_t i = 0; i < dim; ++i) {
    dx(i) += sim.positions()(i, j);
    if (dx(i) > edge_lengths(i)) dx(i) -= edge_lengths(i);
    else if(dx(i) < 0.0) dx(i) += edge_lengths(i);
  }
  return bc_ret(true, dx);
}

bc_ret no_bc(metropolis &sim, const size_t j, arma::vec &dx) {
  dx += sim.positions().col(j);
  return bc_ret(true, dx);
}

} // namespace mmd
