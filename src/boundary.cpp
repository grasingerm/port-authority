#include "boundary.hpp"
#include "metropolis.hpp"

namespace pauth {

using namespace arma;

void wall_bc(metropolis &sim, const size_t j, arma::vec &dx) {
  const auto &edge_lengths = sim.edge_lengths();
  const auto dim = edge_lengths.n_rows;
  bool wall_hit = false;
  for (size_t i = 0; i < dim; ++i) {
    dx(i) += sim.positions()(i, j);
    if (dx(i) >= edge_lengths(i)) {
      wall_hit = true;
      break;
    }
  }
  if(!wall_hit) sim.positions().col(j) = dx;
}

void periodic_bc(metropolis &sim, const size_t j, arma::vec &dx) {
  const auto &edge_lengths = sim.edge_lengths();
  const auto dim = edge_lengths.n_rows;
  for (size_t i = 0; i < dim; ++i) {
    dx(i) += sim.positions()(i, j);
    if (dx(i) > edge_lengths(i)) dx(i) -= edge_lengths(i);
    else if(dx(i) < 0.0) dx(i) += edge_lengths(i);
  }
  sim.positions().col(j) = dx;
}

void no_bc(metropolis &sim, const size_t j, arma::vec &dx) {
  sim.positions().col(j) += dx;
}

} // namespace mmd
