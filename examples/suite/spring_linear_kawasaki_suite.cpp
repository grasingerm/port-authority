#include <iostream>
#include <array>
#include <string>
#include <cstring>
#include <fstream>
#include "port_authority.hpp"
#include "mpi.h"

using namespace std;
using namespace pauth;

int main() {

  const double T = 1.0;
  const double kB = 1.0;
  const double delta_max = 1.25;

  const molecular_id id = molecular_id::Test1;
  const size_t N = 1;
  const size_t D = 1;
  const double L = 1.0;
  const metric m = euclidean;
  const bc boundary = no_bc;
  const unsigned long nsteps = 10000000;
  const_k_spring_potential pot(1.0);
  string fname("spring_linear_1.xyz");
  
  metropolis sim(fname.c_str(), id, N, D, L, continuous_trial_move(delta_max), 
                 &pot, T, kB, m, boundary,
                 kawasaki_acc);

  metropolis_suite msuite(sim, 0, 1, info_lvl_flag::VERBOSE);

  msuite.add_variable_to_average("x", [](const metropolis &sim) {
    return sim.positions()(0, 0);
  });
  msuite.add_variable_to_average("x^2", [](const metropolis &sim) {
    return sim.positions()(0, 0) * sim.positions()(0, 0);
  });
  msuite.add_variable_to_average("U", accessors::U);

  msuite.simulate(nsteps);
  msuite.report_averages();

  return 0;
}
