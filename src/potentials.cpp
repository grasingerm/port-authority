#include "potentials.hpp"
#include "distance.hpp"
#include <cmath>

namespace mmd {

using namespace arma;

double abstract_LJ_potential::_U(const metropolis &sim) const {

  const auto &molecular_ids = sim.molecular_ids();
  const auto &positions = sim.positions();
  const auto &edge_lengths = sim.edge_lengths();
  const auto N = sim.N();
  double potential = 0;

  #pragma omp parallel for reduction(+:potential)
  for (auto i = size_t{0}; i < N - 1; ++i) {
    for (auto j = i + 1; j < N; ++j) {

      const double rij2 = sim.m(positions.col(i), positions.col(j), 
                                edge_lengths);
      const double rzero = get_rzero(molecular_ids[i], molecular_ids[j]);
      const double rat2 = (rzero * rzero) / _rij2;
      const double rat6 = rat2 * rat2 * rat2;

      potential += 4.0 * get_well_depth(molecular_ids[i], molecular_ids[j]) *
                   (rat6 * rat6 - rat6);
    }
  }

  return potential;
}

double abstract_LJ_potential::_delta_U(const metropolis &sim, const size_t j,
                                       arma::vec &dx) const {
  
  const auto &molecular_ids = sim.molecular_ids();
  const auto &positions = sim.positions();
  const auto &ro_j = positions().col(j);
  const auto rn_j = ro_j + dx;
  const auto &edge_lengths = sim.edge_lengths();
  const auto N = sim.N();
  double dU = 0;

  #pragma omp parallel for reduction(+:dU)
  for (auto i = size_t{0}; i < N; ++i) {

      double rij2 = sim.m(positions.col(i), ro_j, edge_lengths);
      double rzero = get_rzero(molecular_ids[i], molecular_ids[j]);
      double rat2 = (rzero * rzero) / _rij2;
      double rat6 = rat2 * rat2 * rat2;

      const double Uo = 4.0 * get_well_depth(molecular_ids[i], molecular_ids[j]) *
                        (rat6 * rat6 - rat6);
      
      rij2 = sim.m(positions.col(i), rn_j, edge_lengths);
      rzero = get_rzero(molecular_ids[i], molecular_ids[j]);
      rat2 = (rzero * rzero) / _rij2;
      rat6 = rat2 * rat2 * rat2;

      const double Un = 4.0 * get_well_depth(molecular_ids[i], molecular_ids[j]) *
                        (rat6 * rat6 - rat6);
     
      dU += Un - Uo;
    }
  }

  return dU;
}

double abstract_LJ_cutoff_potential::_U(const metropolis &sim) const {

  const auto &molecular_ids = sim.molecular_ids();
  const auto &positions = sim.positions();
  const auto &edge_lengths = sim.edge_lengths();
  const auto N = sim.N();
  double potential = 0;

  #pragma omp parallel for reduction(+:potential)
  for (auto i = size_t{0}; i < N - 1; ++i) {
    for (auto j = i + 1; j < N; ++j) {

      const double rij2 = sim.m(positions.col(i), positions.col(j), 
                                edge_lengths);

      if (rij2 <= _rc2) {

        const double rzero = get_rzero(molecular_ids[i], molecular_ids[j]);
        const double rz2 = rzero * rzero;
        const double rz4 = rz2 * rz2;
        const double rz8 = rz4 * rz4;

        const double dudr_rc = 6.0 * (rz4 * rz2 * rzero) / (_rc7)-12.0 *
                               (rz8 * rz4 * rzero) / (_rc13);
        const double u_rc = ((rz8 * rz4) / (_rc12) - (rz4 * rz2) / (_rc6));

        const double rat2 = rz2 / rij2;
        const double rat6 = rat2 * rat2 * rat2;

        potential += 4.0 * get_well_depth(molecular_ids[i], molecular_ids[j]) *
                     ((rat6 * rat6 - rat6) - u_rc -
                      (std::sqrt(rij2) - _cutoff) * dudr_rc);
      }
    }
  }

  return potential;
}

double abstract_LJ_cutoff_potential::_delta_U(const metropolis &sim, 
                                              const size_t j,
                                              arma::vec &dx) const {
  
  const auto &molecular_ids = sim.molecular_ids();
  const auto &positions = sim.positions();
  const auto &ro_j = positions().col(j);
  const auto rn_j = ro_j + dx;
  const auto &edge_lengths = sim.edge_lengths();
  const auto N = sim.N();
  double dU = 0;

  #pragma omp parallel for reduction(+:dU)
  for (auto i = size_t{0}; i < N; ++i) {
      const double rzero = get_rzero(molecular_ids[i], molecular_ids[j]);
      const double rz2 = rzero * rzero;
      const double rz4 = rz2 * rz2;
      const double rz8 = rz4 * rz4;

      const double dudr_rc = 6.0 * (rz4 * rz2 * rzero) / (_rc7)-12.0 *
                             (rz8 * rz4 * rzero) / (_rc13);
      const double u_rc = ((rz8 * rz4) / (_rc12) - (rz4 * rz2) / (_rc6));

      double Uo = 0.0;
      double rij2 = sim.m(positions.col(i), ro_j, edge_lengths);
      
      if (rij2 <= _rc2) {
        const double rat2 = rz2 / rij2;
        const double rat6 = rat2 * rat2 * rat2;

        Uo = 4.0 * get_well_depth(molecular_ids[i], molecular_ids[j]) *
             ((rat6 * rat6 - rat6) - u_rc - 
              (std::sqrt(rij2) - _cutoff) * dudr_rc);
      }
      
      rij2 = sim.m(positions.col(i), rn_j, edge_lengths);
      
      if (rij2 <= _rc2) {
        const double rat2 = rz2 / rij2;
        const double rat6 = rat2 * rat2 * rat2;

        Un = 4.0 * get_well_depth(molecular_ids[i], molecular_ids[j]) *
             ((rat6 * rat6 - rat6) - u_rc - 
              (std::sqrt(rij2) - _cutoff) * dudr_rc);
      }
     
      dU += Un - Uo;
    }
  }

  return dU;
}

double abstract_spring_potential::_U(const metropolis &sim) const {

  const auto N = sim.N();
  const auto &molecular_ids = sim.molecular_ids();
  const auto& positions = sim.positions();
  double potential = 0;

  #pragma parallel for reduction(+:potential)
  for (auto i = size_t{0}; i < N; ++i) {
    const double r2 = dot(positions.col(i), positions.col(i));
    potential += 0.5 * get_k(molecular_ids[i]) * r2;
  }

  return potential;
}

double abstract_spring_potential::_delta_U(const metropolis &sim,
                                           const size_t j,
                                           arma::vec &dx) const {
  const arma::vec &ro_j = sim.positions.col(j);
  const auto rn_j = ro_j + dx;
  return 0.5 * get_k(sim.molecular_ids()[j]) * 
         (dot(rn_j, rn_j) - dot(ro_j, ro_j));
}

const_poly_spring_potential::const_poly_spring_potential(
    const std::initializer_list<double> &coeffs)
    : pcoeffs(coeffs) {

  for (size_t i = 1; i < pcoeffs.size(); ++i)
    fcoeffs.push_back(pcoeffs[i] * 2 * i);
}

double const_poly_spring_potential::_U(const metropolis &sim) const {

  const auto N = sim.N();
  const auto &molecular_ids = sim.molecular_ids();
  const auto& positions = sim.positions();
  double potential = 0;

  #pragma parallel for reduction(+:potential)
  for (auto i = size_t{0}; i < n; ++i) {
    double Ui = 0.0
    const double r2 = dot(positions.col(i), positions.col(i));
    for (auto p = size_t{0}; p < pcoeffs.size(); ++p) {
      double x = 1;
      for (auto k = size_t{0}; k < p; ++k)
        x *= r2;
      Ui += pcoeffs[p] * x;
    }
    potential += Ui;
  }

  return potential;
}

double const_poly_spring_potential::_delta_U(const metropolis &sim,
                                             const size_t j,
                                             arma::vec &dx) const {
  const arma::vec &ro_j = sim.positions.col(j);
  const auto rn_j = ro_j + dx;

  double Uo = 0.0;
  double r2 = dot(ro_j, ro_j);
  for (auto p = size_t{0}; p < pcoeffs.size(); ++p) {
    double x = 1;
    for (auto k = size_t{0}; k < p; ++k)
      x *= r2;
    Uo += pcoeffs[p] * x;
  }

  double Un = 0.0;
  r2 = dot(ro_j, ro_j);
  for (auto p = size_t{0}; p < pcoeffs.size(); ++p) {
    double x = 1;
    for (auto k = size_t{0}; k < p; ++k)
      x *= r2;
    Un += pcoeffs[p] * x;
  }

  return Un - Uo;
}

double const_quad_spring_potential::_U(const metropolis &sim) const {

  const auto N = sim.N();
  const auto &molecular_ids = sim.molecular_ids();
  const auto& positions = sim.positions();
  double potential = 0;

  #pragma parallel for reduction(+:potential)
  for (auto i = size_t{0}; i < N; ++i) {
    const double r2 = dot(positions.col(i), positions.col(i));
    potential += a * r2 * r2 + b * r2 + c;
  }

  return potential;
}

double const_quad_spring_potential::_delta_U(const metropolis &sim,
                                             const size_t j,
                                             arma::vec &dx) const {
  const arma::vec &ro_j = sim.positions.col(j);
  const auto rn_j = ro_j + dx;
  const double r2o = dot(ro_j, ro_j);
  const double r2n = dot(rn_j, rn_j);
  const double Uo = a * r2o * r2o + b * r2o + c;
  const double Un = a * r2n * r2n + b * r2o + c;

  return Un - Uo;
}

// definitions for pure virtual destructors
abstract_potential::~abstract_potential() {}
abstract_LJ_potential::~abstract_LJ_potential() {}
abstract_LJ_cutoff_potential::~abstract_LJ_cutoff_potential() {}
abstract_spring_potential::~abstract_spring_potential() {}

} // namespace mmd
