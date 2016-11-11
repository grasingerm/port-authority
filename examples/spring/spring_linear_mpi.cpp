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

  long unsigned nsamples_global = 0;
  long double U_sum_global = 0.0;
  long double x_sum_global = 0.0;
  long double xsq_sum_global = 0.0;
  std::array<double, 4> Ts = { 1.0 / 0.1, 1.0, 1.0 / 5.0, 1.0 / 10.0 };
  const double kB = 1.0;
  const double delta_max = 1.25;

  int taskid, numtasks, rc;
  
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
  cout << "MPI task " << taskid << " has started...\n";

  for (auto T : Ts) {

    long unsigned nsamples = 0;
    long double U_sum = 0.0;
    long double x_sum = 0.0;
    long double xsq_sum = 0.0;

    for (unsigned i = 1; i <= 3; ++i) {
      const molecular_id id = molecular_id::Test1;
      const size_t N = 1;
      const size_t D = 1;
      const double L = 1.0;
      const metric m = euclidean;
      const bc boundary = no_bc;
      const unsigned long nsteps = 10000000;
      const_k_spring_potential pot(1.0);
  
      string fname = string("spring_linear_") + to_string(i) + string(".xyz");
      metropolis sim(fname.c_str(), id, N, D, L, delta_max, 
                     &pot, T, kB, m, boundary, metropolis::DEFAULT_TMG,
                     metropolis::DEFAULT_ACC,
                     random_device()());

      sim.add_callback([&](const metropolis &sim) {
        for (const auto &potential : sim.potentials())
          U_sum += potential->U(sim);
        double x = sim.positions()(0, 0);
        x_sum += x;
        xsq_sum += x * x;
        ++nsamples;
      });
      
      sim.simulate(nsteps / numtasks);
    }
    
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
      cout << "<U>    =    " << U_sum_global / nsamples_global << '\n';
      cout << "<x>    =    " << x_sum_global / nsamples_global << '\n';
      cout << "<x^2>  =    " << xsq_sum_global / nsamples_global << '\n';
    }
  }

  cout << "MPI task " << taskid << " finished.\n";

  MPI_Finalize();

  return 0;
}
