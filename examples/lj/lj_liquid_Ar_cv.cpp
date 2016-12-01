#include <iostream>
#include <array>
#include <string>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <cmath>
#include <fstream>
#include "port_authority.hpp"

using namespace std;
using namespace pauth;

void print_help(char *program_name) {
  cout << "usage: " << program_name << " [nsteps = 100000] "
          "[equilibration steps = 10000] [dsteps = 10] [c = 0.2]\n";
  return;
}

int main(int argc, char* argv[]) {

  unsigned long nsteps, equilibration_steps, dsteps;
  double c;

  switch(argc) {
    case 1: {
      nsteps = 100000;
      equilibration_steps = 10000;
      dsteps = 10;
      c = 0.2;
      break;
    }

    case 2: {
      if (argv[1][0] == 'h') {
        print_help(argv[0]);
        return 1;
      }

      nsteps = strtoul(argv[1], nullptr, 10);
      equilibration_steps = 10000;
      dsteps = 10;
      c = 0.2;
      break;
    }

    case 3: {
      nsteps = strtoul(argv[1], nullptr, 10);
      equilibration_steps = strtoul(argv[2], nullptr, 10);
      dsteps = 10;
      c = 0.2;
      break;
    }

    case 4: {
      nsteps = strtoul(argv[1], nullptr, 10);
      equilibration_steps = strtoul(argv[2], nullptr, 10);
      dsteps = strtoul(argv[3], nullptr, 10);
      c = 0.2;
      break;
    }

    case 5: {
      nsteps = strtoul(argv[1], nullptr, 10);
      equilibration_steps = strtoul(argv[2], nullptr, 10);
      dsteps = strtoul(argv[3], nullptr, 10);
      c = strtod(argv[4], nullptr);
      break;
    }

    default: {
      print_help(argv[0]);
      return 1;
    }
  }

  const double T = 100.0 / 121.0; /* Ar temperature scale is 121 K */
  //const double density_scale = 1686.85;
  const double kB = 1.0;

  const molecular_id id = molecular_id::Test1;
  const size_t N = 256;
  const size_t D = 3;
  const double L = 7.4;
  const double a = L / pow(N, 1.0 / 3.0);
  const double delta_max = c * a;
  const metric m = periodic_euclidean;
  const bc boundary = periodic_bc;
  const_well_params_LJ_cutoff_potential pot(1.0, 1.0, 2.5);
  
  metropolis sim("liquid256_init.xyz", id, N, D, L, continuous_trial_move(delta_max), 
                 &pot, T, kB, m, boundary,
                 metropolis_acc);
  sim.simulate(equilibration_steps);
  sim.reset_step();

  ofstream ofs("lj_liquid_U-fluctuations.csv", ofstream::out);
  ofs << "step,U_k,U_mean,mds_k,mds_mean\n";
  unsigned long nsamples = 0;
  long double U_sum = 0.0;
  long double mds_sum = 0.0;
  double U_k = 0.0;
  bool first_call_flag = true;

  sim.add_callback([&](const metropolis &sim) {
    if (sim.step() % dsteps == 0) {
      if (first_call_flag) {
        U_k = accessors::U(sim);
        first_call_flag = false;
      }

      U_k += (sim.accepted()) ? sim.dU() : 0.0; // change in energy is cached
      U_sum += U_k;
      ++nsamples;
      double U_mean = U_sum / nsamples;
      double mds_k = (U_k - U_mean) * (U_k - U_mean);
      mds_sum += mds_k;
      double mds_mean = mds_sum / nsamples;

      ofs << sim.step() << ','
          << U_k << ','
          << U_mean << ','
          << mds_k << ','
          << mds_mean << '\n';
    }
  });
  sim.add_callback([=](const metropolis &sim) {
    if ((sim.step() % (nsteps / 20)) == 0)
      cout << 100 * static_cast<double>(sim.step()) / static_cast<double>(nsteps)
           << "%\n";
  });
  mprof::tic();
  sim.simulate(nsteps);
  mprof::toc();

  double U_mean = U_sum / nsamples;
  double mds_mean = mds_sum / nsamples;
  cout << "            < U >   =   " << U_mean << '\n';
  cout << "< (U - < U >)^2 >   =   " << mds_mean << '\n';
  const double cvU = T*T * 3 * (N - 1) * kB / mds_mean;
  cout << "            3cv,U   =   " << cvU << '\n';
  cout << "            3cv     =   " << cvU + 1.5 << '\n';
  cout << "       3cv (J/kg-K) =   " << (cvU + 1.5) * 1.38064852 / (6.6e-3) << '\n';
  cout << "        cv (J/kg-K) =   " << (cvU + 1.5) * 1.38064852 / (6.6e-3) / 3.0 << '\n';

  return 0;
}
