#include <iostream>
#include <array>
#include <string>
#include <cstring>
#include <fstream>
#include "port_authority.hpp"
#include "mpi.h"

using namespace std;
using namespace pauth;
using namespace arma;

const unsigned MASTER = 0;

int main(int argc, char* argv[]) {

  long unsigned nsamples_global = 0;
  long double U_sum_global = 0.0;
  long double x_sum_global = 0.0;
  long double xsq_sum_global = 0.0;
  std::array<double, 5> Ts = { 1.0 / 0.1, 1.0, 1.0 / 5.0, 1.0 / 10.0, 1.0 / 100.0};
  const double kB = 1.0;

  std::array<mat, 4> ics;
  size_t idx = 0;
  for (unsigned i = 1; i < 3; ++i) {
    for (unsigned j = 1; j < 3; ++j) {
      ics[idx].resize(1, 2);
      ics[idx](0, 0) = static_cast<double>(i);
      ics[idx](0, 1) = static_cast<double>(j);
      ++idx;
    }
  }

  int taskid, numtasks, rc;
  
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
  cout << "MPI task " << taskid << " has started...\n";

  for (auto T : Ts) {
    for (const auto &ic : ics) {

      long unsigned nsamples = 0;
      long double U_sum = 0.0;
      long double x_sum = 0.0;
      long double xsq_sum = 0.0;

      const molecular_id id = molecular_id::Test1;
      const size_t N = 2;
      const size_t D = 1;
      const double L = 1.0;
      const metric m = euclidean;
      const unsigned long nsteps = 1000000;
      twostate_int_potential pot(0.0, 1.0);

      metropolis sim(id, N, D, L, state_trial_move(2), 
                     &pot, T, kB, m, no_bc,
                     metropolis::DEFAULT_ACC,
                     hardware_entropy_seed_gen, true);

      sim.set_positions(ic);

      sim.add_callback([&](const metropolis &sim) {
        for (const auto &potential : sim.potentials())
          U_sum += potential->U(sim);
        double x = sim.positions()(0, 0);
        x_sum += x;
        xsq_sum += x * x;
        ++nsamples;
      });
      
      sim.simulate(nsteps / numtasks);
      
      rc = MPI_Reduce(&nsamples, &nsamples_global, 1, MPI_UNSIGNED_LONG, MPI_SUM,
                        MASTER, MPI_COMM_WORLD);
      if (rc != MPI_SUCCESS) printf("%d: failure on mpc_reduce\n", taskid);

      rc = MPI_Reduce(&U_sum, &U_sum_global, 1, MPI_LONG_DOUBLE, MPI_SUM,
                      MASTER, MPI_COMM_WORLD);
      if (rc != MPI_SUCCESS) printf("%d: failure on mpc_reduce\n", taskid);

      rc = MPI_Reduce(&x_sum, &x_sum_global, 1, MPI_LONG_DOUBLE, MPI_SUM,
                      MASTER, MPI_COMM_WORLD);
      if (rc != MPI_SUCCESS) printf("%d: failure on mpc_reduce\n", taskid);

      rc = MPI_Reduce(&xsq_sum, &xsq_sum_global, 1, MPI_LONG_DOUBLE, MPI_SUM,
                      MASTER, MPI_COMM_WORLD);
      if (rc != MPI_SUCCESS) printf("%d: failure on mpc_reduce\n", taskid);

      if (taskid == MASTER) {
        cout << "beta   =    " << 1 / (T * kB) << '\n';
        cout << "ic     =    " << ic << '\n';
        cout << "<U>    =    " << U_sum_global / nsamples_global << '\n';
        cout << "<x>    =    " << x_sum_global / nsamples_global << '\n';
        cout << "<x^2>  =    " << xsq_sum_global / nsamples_global << '\n';
      }
    }
  }

  cout << "MPI task " << taskid << " finished.\n";

  MPI_Finalize();

  return 0;
}
