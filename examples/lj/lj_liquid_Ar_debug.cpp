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

  unsigned long nsteps;
  switch(argc) {
    case 1: {
      nsteps = 10000;
      break;
    }

    case 2: {
      nsteps = strtoul(argv[1], nullptr, 10);
      break;
    }

    default: {
      cout << "usage: " << argv[0] << " [nsteps = 10000]\n";
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
  
  cout << "T      =   " << T << '\n';
  cout << "N      =   " << N << '\n';
  cout << "L      =   " << L << '\n';
  cout << "a      =   " << a << '\n';
  cout << "delta  =   " << delta_max << '\n';
  
  metropolis sim("liquid256_init.xyz", id, N, D, L, continuous_trial_move(delta_max), 
                 &pot, T, kB, m, boundary,
                 metropolis_acc);

  //sim.add_callback(print_trajectory_callback(nsteps / 25));
  sim.add_callback(print_vector_with_step_callback<3>(cout, nsteps / 25, 
                   accessors::ivp, ' '));

  cout << "step P_ideal P_virial P\n";
  cout << "=======================\n";
  sim.simulate(nsteps);

  return 0;
}
