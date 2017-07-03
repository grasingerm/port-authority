#include "potentials.hpp"
#include "metropolis.hpp"
#include "distance.hpp"
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <boost/algorithm/string.hpp>
#include "yaml-cpp/yaml.h"

namespace pauth {

using namespace arma;

double abstract_LJ_full_potential::_U(const metropolis &sim) const {

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
      const double rat2 = (rzero * rzero) / rij2;
      const double rat6 = rat2 * rat2 * rat2;

      potential += 4.0 * get_well_depth(molecular_ids[i], molecular_ids[j]) *
                   (rat6 * rat6 - rat6);
    }
  }

  return potential;
}

double abstract_LJ_full_potential::_delta_U(const metropolis &sim, const size_t j,
                                       arma::vec &rn_j) const {
  
  const auto &molecular_ids = sim.molecular_ids();
  const auto &positions = sim.positions();
  const auto &ro_j = positions.col(j);
  const auto &edge_lengths = sim.edge_lengths();
  const auto N = sim.N();
  double dU = 0;

  #pragma omp parallel for reduction(+:dU)
  for (auto i = size_t{0}; i < N; ++i) {
    if (i == j) continue;

    const auto well_depth = get_well_depth(molecular_ids[i], molecular_ids[j]);
    double rij2 = sim.m(positions.col(i), ro_j, edge_lengths);
    double rzero = get_rzero(molecular_ids[i], molecular_ids[j]);
    double rat2 = (rzero * rzero) / rij2;
    double rat6 = rat2 * rat2 * rat2;

    const double Uo = 4.0 * well_depth * (rat6 * rat6 - rat6);
    
    rij2 = sim.m(positions.col(i), rn_j, edge_lengths);
    rzero = get_rzero(molecular_ids[i], molecular_ids[j]);
    rat2 = (rzero * rzero) / rij2;
    rat6 = rat2 * rat2 * rat2;

    const double Un = 4.0 * well_depth * (rat6 * rat6 - rat6);
   
    dU += Un - Uo;
  }

  return dU;
}

arma::vec abstract_LJ_full_potential::_forceij(const metropolis &sim, const size_t i,
                                          const size_t j) const {
  const auto &molecular_ids = sim.molecular_ids();
  const auto &positions = sim.positions();
  const auto &edge_lengths = sim.edge_lengths();
  const auto rij = sim.rij(positions.col(i), positions.col(j), edge_lengths);
  const double _rij2 = dot(rij, rij);
  const double rzero = get_rzero(molecular_ids[i], molecular_ids[j]);
  const double rat2 = (rzero * rzero) / _rij2;
  const double rat6 = rat2 * rat2 * rat2;
  const double rat8 = rat6 * rat2;
  const double well_depth = get_well_depth(molecular_ids[i], molecular_ids[j]);

  return rij / rzero * well_depth * (48.0 * rat8 * rat6 - 24.0 * rat8);
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
                                              arma::vec &rn_j) const {
  
  const auto &molecular_ids = sim.molecular_ids();
  const auto &positions = sim.positions();
  const auto &ro_j = positions.col(j);
  const auto &edge_lengths = sim.edge_lengths();
  const auto N = sim.N();
  double dU = 0;

  #pragma omp parallel for reduction(+:dU)
  for (auto i = size_t{0}; i < N; ++i) {
    if (i == j) continue;

    const auto rzero = get_rzero(molecular_ids[i], molecular_ids[j]);
    const auto well_depth = get_well_depth(molecular_ids[i], molecular_ids[j]);
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

      Uo = 4.0 * well_depth *
           ((rat6 * rat6 - rat6) - u_rc - 
            (std::sqrt(rij2) - _cutoff) * dudr_rc);
    }
    
    double Un = 0.0;
    rij2 = sim.m(positions.col(i), rn_j, edge_lengths);
    
    if (rij2 <= _rc2) {
      const double rat2 = rz2 / rij2;
      const double rat6 = rat2 * rat2 * rat2;

      Un = 4.0 * well_depth *
           ((rat6 * rat6 - rat6) - u_rc - 
            (std::sqrt(rij2) - _cutoff) * dudr_rc);
    }
   
    dU += Un - Uo;
  }

  return dU;
}

arma::vec abstract_LJ_cutoff_potential::_forceij(const metropolis &sim, 
                                                 const size_t i,
                                                 const size_t j) const {

  const auto &molecular_ids = sim.molecular_ids();
  const auto &positions = sim.positions();
  const auto &edge_lengths = sim.edge_lengths();
  const auto _rij = sim.rij(positions.col(i), positions.col(j), edge_lengths);
  const double _rij2 = dot(_rij, _rij);

  if (_rij2 <= _rc2) {

    const double rzero = get_rzero(molecular_ids[i], molecular_ids[j]);
    const double rz2 = rzero * rzero;
    const double rz4 = rz2 * rz2;
    const double rz8 = rz4 * rz4;

    const double dudr_rc = 24.0 * (rz4 * rz2 * rzero) / (_rc7)-48.0 *
                           (rz8 * rz4 * rzero) / (_rc13);

    const double rat2 = rz2 / _rij2;
    const double rat6 = rat2 * rat2 * rat2;
    const double rat8 = rat6 * rat2;
    const double well_depth =
        get_well_depth(molecular_ids[i], molecular_ids[j]);

    return _rij / rzero * well_depth *
           (48.0 * rat8 * rat6 - 24.0 * rat8 + dudr_rc / std::sqrt(_rij2));
  }

  return arma::zeros(sim.D());
}

double abstract_spring_potential::_U(const metropolis &sim) const {

  const auto N = sim.N();
  const auto &molecular_ids = sim.molecular_ids();
  const auto& positions = sim.positions();
  double potential = 0;

  #pragma omp parallel for reduction(+:potential)
  for (auto i = size_t{0}; i < N; ++i) {
    const double r2 = dot(positions.col(i), positions.col(i));
    potential += 0.5 * get_k(molecular_ids[i]) * r2;
  }

  return potential;
}

double abstract_spring_potential::_delta_U(const metropolis &sim,
                                           const size_t j,
                                           arma::vec &rn_j) const {
  const arma::vec &ro_j = sim.positions().col(j);
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
  const auto& positions = sim.positions();
  double potential = 0;

  #pragma omp parallel for reduction(+:potential)
  for (auto i = size_t{0}; i < N; ++i) {
    double Ui = 0.0;
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
                                             arma::vec &rn_j) const {
  const arma::vec &ro_j = sim.positions().col(j);

  double Uo = 0.0;
  double r2 = dot(ro_j, ro_j);
  for (auto p = size_t{0}; p < pcoeffs.size(); ++p) {
    double x = 1;
    for (auto k = size_t{0}; k < p; ++k)
      x *= r2;
    Uo += pcoeffs[p] * x;
  }

  double Un = 0.0;
  r2 = dot(rn_j, rn_j);
  for (auto p = size_t{0}; p < pcoeffs.size(); ++p) {
    double x = 1;
    for (auto k = size_t{0}; k < p; ++k)
      x *= r2;
    Un += pcoeffs[p] * x;
  }

  return Un - Uo;
}

arma::vec const_poly_spring_potential::_forceij(const metropolis &sim, const size_t, 
                                                const size_t) const {
  return arma::zeros(sim.D());
}

double const_quad_spring_potential::_U(const metropolis &sim) const {

  const auto N = sim.N();
  const auto &positions = sim.positions();
  double potential = 0;

  #pragma omp parallel for reduction(+:potential)
  for (auto i = size_t{0}; i < N; ++i) {
    const double r2 = dot(positions.col(i), positions.col(i));
    potential += a * r2 * r2 + b * r2 + c;
  }

  return potential;
}

double const_quad_spring_potential::_delta_U(const metropolis &sim,
                                             const size_t j,
                                             arma::vec &rn_j) const {
  const arma::vec &ro_j = sim.positions().col(j);
  const double r2o = dot(ro_j, ro_j);
  const double r2n = dot(rn_j, rn_j);

  return a * (r2n*r2n - r2o*r2o) + b * (r2n - r2o);
}

arma::vec abstract_spring_potential::_forceij(const metropolis &sim, 
                                                      const size_t, 
                                                      const size_t) const {
  return arma::zeros(sim.D());
}

arma::vec const_quad_spring_potential::_forceij(const metropolis &sim, const size_t, 
                                                const size_t) const {
  return arma::zeros(sim.D());
}

double twostate_int_potential::_U(const metropolis &sim) const {

  const auto N = sim.N();
  const auto &positions = sim.positions();

  double potential = 0.0;

  #pragma omp parallel for reduction(+:potential)
  for (auto i = size_t{0}; i < N; ++i) {
    assert(_check_x(positions.col(i)));
    potential += _Ui(positions.col(i));
  }

  #pragma omp parallel for reduction(+:potential)
  for (auto i = size_t{0}; i < N; ++i) {
    for (auto j = size_t{i+1}; j < N; ++j) {
      potential += _Ui(positions.col(i)) * _Ui(positions.col(j));
    }
  }

  return potential;
}

double twostate_int_potential::_delta_U(const metropolis &sim, const size_t j,
                                        arma::vec &rn_j) const {

  const auto N = sim.N();
  const auto &positions = sim.positions();

  double dUi = _Ui(rn_j) - _Ui(positions.col(j));
  
  double U2 = 0.0;
  #pragma omp parallel for reduction(+:U2)
  for (auto i = size_t{0}; i < N; ++i) {
    if (i == j) continue;
    U2 += _Ui(positions.col(i));
  }
  
  return dUi + dUi * U2;
  
}

arma::vec twostate_int_potential::_forceij(const metropolis &sim, const size_t, 
                                           const size_t) const {
  return arma::zeros(sim.D());
}

abstract_LJ_lookup_potential::abstract_LJ_lookup_potential(const char *fname) {
  vector<string> molecular_strs;
  const auto well_params = YAML::LoadFile(fname);

  for (const auto &well_param : well_params) {
    auto molecular_pair_str = well_param.first.as<string>();
    boost::split(molecular_strs, molecular_pair_str, boost::is_any_of(","));
    for (auto &molecular_str : molecular_strs) boost::trim(molecular_str);
    if (molecular_names_to_ids.count(molecular_strs[0]) &&
        molecular_names_to_ids.count(molecular_strs[1])) {
      unordered_pair<molecular_id> key { 
        molecular_names_to_ids.at(molecular_strs[0]),
        molecular_names_to_ids.at(molecular_strs[1]) 
      };
      _well_depth_map[key] = well_param.second["epsilon"].as<double>();
      _rzero_map[key] = well_param.second["sigma"].as<double>();
    }
    else throw runtime_error(string("Molecular pair ") + 
                             well_param.first.as<string>() + 
                             string(" not found."));
  }
}

double abstract_LJ_lookup_potential::_get_well_depth(const molecular_id id1,
                                                     const molecular_id id2) const {
  unordered_pair<molecular_id> key { id1, id2 };
  return _well_depth_map.at(key);
}

double abstract_LJ_lookup_potential::_get_rzero(const molecular_id id1,
                                                const molecular_id id2) const {
  unordered_pair<molecular_id> key { id1, id2 };
  return _rzero_map.at(key);
}

double dipole_electric_potential::_U(const metropolis &sim) const {

  const auto N = sim.N();
  const auto &positions = sim.positions();
  double sum = 0.0;

  #pragma omp parallel for reduction(+:sum)
  for (auto i = size_t{0}; i < N; ++i) {
    auto& positionsi = positions.col(i);
    for (auto& kv : _efield)
      sum += -(positionsi(kv.first) * kv.second(positionsi));
  }

  return sum;

}

double dipole_electric_potential::_delta_U(const metropolis &sim, 
  const size_t j, arma::vec &dx) const {

  auto& positionsj = sim.positions().col(j);

  double sum = 0.0;
  for (auto& kv : _efield)
    sum += -(dx(kv.first) * kv.second(positionsj));

  return sum;
  
}

arma::vec dipole_electric_potential::_forceij(const metropolis &sim, 
    const size_t, const size_t) const { return arma::zeros(sim.D()); }

double abstract_dipole_strain_potential::_U(const metropolis &sim) const {
  
  const auto N = sim.N();
  const auto &positions = sim.positions();
  const auto &ids = sim.molecular_ids();
  double sum = 0.0;

  #pragma omp parallel for reduction(+:sum)
  for (auto i = size_t{0}; i < N; ++i) {

    auto& xsi = positions.col(i);
    auto pi = p(xsi);

    // Ui = 1/2 * p . X p
    sum += arma::dot(pi, inv_chi(xsi, ids[i]) * pi);

  }

  return 0.5 * sum;

}

double abstract_dipole_strain_potential::_delta_U(const metropolis &sim, 
    const size_t j, arma::vec &dx) const {

  auto& xs = sim.positions().col(j);
  const auto id = sim.molecular_ids()[j];
  auto xs_new = xs + dx;

  auto pi = p(xs);
  auto pnew = p(xs_new);

  return 0.5 * (arma::dot(pi, inv_chi(xs, id) * pi) - 
                arma::dot(pnew, inv_chi(xs_new, id) * pnew));

}

arma::vec abstract_dipole_strain_potential::_forceij
  (const metropolis &sim, const size_t, const size_t) const {
  return arma::zeros(sim.D()); 
}

// definitions for pure virtual destructors
abstract_potential::~abstract_potential() {}
abstract_LJ_potential::~abstract_LJ_potential() {}
abstract_LJ_full_potential::~abstract_LJ_full_potential() {}
abstract_LJ_lookup_potential::~abstract_LJ_lookup_potential() {}
abstract_LJ_cutoff_potential::~abstract_LJ_cutoff_potential() {}
abstract_spring_potential::~abstract_spring_potential() {}
abstract_dipole_strain_potential::~abstract_dipole_strain_potential() {}
abstract_dipole_strain_2d_potential::~abstract_dipole_strain_2d_potential() {}
abstract_dipole_strain_3d_potential::~abstract_dipole_strain_3d_potential() {}
abstract_dipole_strain_linear_potential::~abstract_dipole_strain_linear_potential() {}

} // namespace pauth
