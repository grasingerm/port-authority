#include <iostream>
#include <array>
#include <string>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <cmath>
#include "port_authority.hpp"
#include "mpi.h"

using namespace std;
using namespace pauth;

int main(int argc, char* argv[]) {

  unsigned long nsteps, equilibration_steps, dsteps;
  switch(argc) {
    case 1: {
      nsteps = 100000;
      equilibration_steps = 10000;
      dsteps = 10;
      break;
    }

    case 2: {
      nsteps = strtoul(argv[1], nullptr, 10);
      equilibration_steps = 10000;
      dsteps = 10;
      break;
    }

    case 3: {
      nsteps = strtoul(argv[1], nullptr, 10);
      equilibration_steps = strtoul(argv[2], nullptr, 10);
      dsteps = 10;
      break;
    }

    case 4: {
      nsteps = strtoul(argv[1], nullptr, 10);
      equilibration_steps = strtoul(argv[2], nullptr, 10);
      dsteps = strtoul(argv[3], nullptr, 10);
      break;
    }

    default: {
      cout << "usage: " << argv[0] << " [nsteps = 100000] "
              "[equilibration steps = 10000] [dsteps = 10]\n";
      return 1;
    }
  }

  const double T = 100.0 / 121.0; /* Ar temperature scale is 121 K */
  //const double density_scale = 1686.85;
  const double kB = 1.0;

  const molecular_id id = molecular_id::Test1;
  const size_t N = 256;
  const size_t D = 3;
  const double L = 6.8;
  const double a = L / pow(N, 1.0 / 3.0);
  const double delta_max = 0.2 * a;
  const metric m = periodic_euclidean;
  const bc boundary = periodic_bc;
  const_well_params_LJ_cutoff_potential pot(1.0, 1.0, 2.5);
  const double density_scale = 1686.85;
  const array<double, 2> density_bounds { { 950.0 / density_scale, 
                                            1150.0 / density_scale } };
  const size_t num_sims = 25;
  const double drho = (density_bounds[1] - density_bounds[0]) / num_sims;
  ofstream ofs("pressure_liquid_Ar.csv", ofstream::out);
 
  for (unsigned k = 0; i < num_sims; ++k) {
    const double density = k * drho + density_bounds[0];
    const double L = pow(static_cast<double>(N) / density, 1.0 / 3.0);

    metropolis sim("liquid256_init.xyz", id, N, D, L, continuous_trial_move(delta_max), 
                   &pot, T, kB, m, boundary,
                   metropolis_acc);

    metropolis_suite msuite(sim, equilibration_steps, dsteps, 
                            info_lvl_flag::QUIET, ofs);

    msuite.add_variable_to_average("P", accessors::P);

    msuite.simulate(nsteps);
    msuite.tabulate_averages(density);
  }

  return 0;
}
