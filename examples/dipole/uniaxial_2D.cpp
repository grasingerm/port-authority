#include <iostream>
#include <array>
#include <string>
#include <cstring>
#include <fstream>
#include "port_authority.hpp"
#include "mpi.h"

#define _USE_MATH_DEFINES
#include <cmath>

using namespace std;
using namespace pauth;

int main() {

  const double T = 1.0;
  const double kB = 1.0;
  const double delta_max = M_PI / 8;

  const molecular_id id = molecular_id::Test1;
  const size_t N = 1;
  const size_t D = 2;
  const metric m = euclidean;
  const bc boundary = periodic_bc;
  const unsigned long nsteps = 1000000;
  auto pot = const_E_uniaxial_2D_dipole_potential({0.0, 0.0, 2.0}, 1.5);
  string fname("dipole_2D_1.xyz");
  
  metropolis sim(fname.c_str(), id, N, D, {2*M_PI, M_PI}, continuous_trial_move(delta_max), 
                 &pot, T, kB, m, boundary,
                 metropolis_acc);

  cout << "Expected <pz> = " << 0 << '\n';

  metropolis_suite msuite(sim, 0, 1, info_lvl_flag::VERBOSE);

  msuite.add_variable_to_average("px", [&](const metropolis &sim) {
    return pot.dipole(id, sim.positions().col(0))(0);
  });
  msuite.add_variable_to_average("py", [&](const metropolis &sim) {
    return pot.dipole(id, sim.positions().col(0))(1);
  });
  msuite.add_variable_to_average("pz", [&](const metropolis &sim) {
    return pot.dipole(id, sim.positions().col(0))(2);
  });
  msuite.add_variable_to_average("U", accessors::U);

  msuite.simulate(nsteps);
  msuite.report_averages();

  return 0;
}
