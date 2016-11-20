#include <iostream>
#include <array>
#include <string>
#include <cstring>
#include <fstream>
#include "port_authority.hpp"
#include "mpi.h"

using namespace std;
using namespace pauth;

const unsigned MASTER = 0;
static const unsigned BUFFER_SIZE = 256;

int main(int argc, char* argv[]) {

  const char* basename = "quad-spring";
  char fname_buffer[BUFFER_SIZE];
  const unsigned npts = 100;

  long unsigned nsamples_global = 0;
  long double U_sum_global = 0.0;
  long double x_sum_global = 0.0;
  long double xsq_sum_global = 0.0;
  std::array<double, 4> delta_maxs = { 0.05, 0.25, 1.25, 4.0 };
  std::array<double, 4> Ts = { 1.0 / 0.1, 1.0, 1.0 / 5.0, 1.0 / 10.0 };
  const double kB = 1.0;

  int taskid, numtasks, rc;
  
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
  cout << "MPI task " << taskid << " has started...\n";

  for (auto T : Ts) {

    for (auto delta_max : delta_maxs) {
      if (taskid == MASTER)
        cout << "T = " << T << ", delta = " << delta_max << '\n';

      snprintf(fname_buffer, BUFFER_SIZE, "%s_beta-%05u_delta-%05u.csv", 
               basename, static_cast<unsigned>(1 / (kB * T) * 1e3),
               static_cast<unsigned>(delta_max * 1e3));
      ofstream ostr(fname_buffer, ofstream::out);

      if (taskid == MASTER) ostr << "nsteps,U,x,x^2,time\n";

      double start_time = 0;
      long unsigned nsamples = 0;
      long unsigned steps_run = 0;
      long double U_sum = 0.0;
      long double x_sum = 0.0;
      long double xsq_sum = 0.0;

      if (taskid == MASTER) start_time = MPI_Wtime();

      for (unsigned i = 1; i <= 3; ++i) {
        const molecular_id id = molecular_id::Test1;
        const size_t N = 1;
        const size_t D = 1;
        const double L = 1.0;
        const metric m = euclidean;
        const bc boundary = no_bc;
        const unsigned long nsteps = 1000000000;
        const_quad_spring_potential pot(1.0, -2.0, 1.0);
    
        string fname = string("spring_linear_") + to_string(i) + string(".xyz");
        metropolis sim(fname.c_str(), id, N, D, L, continuous_trial_move(delta_max), 
                       &pot, T, kB, m, boundary,
                       metropolis::DEFAULT_ACC);

        sim.add_callback([&](const metropolis &sim) {
          for (const auto &potential : sim.potentials())
            U_sum += potential->U(sim);
          double x = sim.positions()(0, 0);
          x_sum += x;
          xsq_sum += x * x;
          ++nsamples;
        });
       
        for (unsigned k = 1; k <= npts; ++k) {
          sim.simulate(nsteps / numtasks / npts);
          steps_run += nsteps / npts;
          
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
            auto elapsed = MPI_Wtime() - start_time;
            ostr << steps_run << ','
                 << U_sum_global / nsamples_global << ','
                 << x_sum_global / nsamples_global << ','
                 << xsq_sum_global / nsamples_global << ','
                 << elapsed << '\n';
          }
        }
      }
    }
  }

  cout << "MPI task " << taskid << " finished.\n";

  MPI_Finalize();

  return 0;
}
