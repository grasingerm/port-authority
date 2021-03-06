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

  const size_t N = 256;
  const size_t D = 3;
  const double L = 6.8;
  const double a = L / pow(N, 1.0 / 3.0);
  const double delta_max = 0.2 * a;
  const metric m = periodic_euclidean;
  const bc boundary = periodic_bc;
  LJ_cutoff_potential pot("lj-params.yaml", 2.5);
  
  metropolis sim("liquid256_init.xyz", N, D, L, continuous_trial_move(delta_max), 
                 &pot, T, kB, m, boundary,
                 metropolis_acc);

  metropolis_suite msuite(sim, equilibration_steps, dsteps, info_lvl_flag::VERBOSE);

  msuite.add_variable_to_average("U", accessors::U);
  msuite.add_variable_to_average("P", accessors::P);

  msuite.simulate(nsteps);
  msuite.report_averages();

  return 0;
}
