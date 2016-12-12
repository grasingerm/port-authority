#include <iostream>
#include <array>
#include <string>
#include <cstring>
#include <fstream>
#include <random>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/tokenizer.hpp>
#include <boost/token_functions.hpp>
#include <exception>
#define _USE_MATH_DEFINES
#include <cmath>
#include "port_authority.hpp"
#include "mpi.h"

using namespace std;
using namespace pauth;
using namespace boost;
using namespace boost::program_options;

const unsigned MASTER = 0;

int main(int argc, char* argv[]) {

  unsigned numruns;
  unsigned long numsteps;
  unsigned numinit;

  options_description desc("\nTransition state theory assignment.\n\nAllowed arguments");

  desc.add_options()
    ("help,h", "Produce this help message.")
    ("numruns,n", value<unsigned>(&numruns)->default_value(25), 
     "Number of runs for each beta.")
    ("numsteps,s", value<unsigned long>(&numsteps)->default_value(100000), 
     "Number of steps for each run.")
    ("numinit,i", value<unsigned>(&numinit)->default_value(100), 
     "Number of steps to run in A well for getting random initial conditions.");

  variables_map vm;
  try {
    store(command_line_parser(argc, argv).options(desc).run(), vm);
    notify(vm);
  } catch(std::exception &e) {
    cout << endl << e.what() << endl;
    cout << desc << endl;
  }

  if (vm.count("help")) {
    cout << "--help specified" << endl;
    cout << desc << endl;
    return 0;
  }

  ofstream ostr("tst.csv", ofstream::out);
  std::array<double, 5> Ts = { { 
    1.0 / 0.01, 1.0 / 0.1, 1.0, 1.0 / 10.0, 1.0 / 100.0
  } };
  const double kB = 1.0;
  const double delta_max = 1.25;
  const double eps = 0.2;
  const molecular_id id = molecular_id::Test1;
  const metric m = euclidean;
  const bc boundary = no_bc;
  const_quad_spring_potential pot(1.0, -2.0, 1.0);

  int taskid, numtasks, rc;
  
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
  cout << "MPI task " << taskid << " has started...\n";

  if (taskid == MASTER) ostr << "beta,M,N,<d(x-q)>,ktst\n";

  default_random_engine rng(random_device()() + taskid);
  uniform_real_distribution<double> dist(0.0, 1.0);

  for (auto T : Ts) {
    long unsigned N_beta = 0;
    long unsigned M_beta = 0;

    for (unsigned k = 0; k < numruns / numtasks; ++k) {
      long unsigned N = 0, N_run = 0;
      long unsigned M = 0, M_run = 0;

      // initialize
      metropolis sim("ktst.xyz", id, 1, 1, 1.0, continuous_trial_move(delta_max), 
                     &pot, T, kB, m, boundary, 
                     metropolis_acc);

      // initialize to random point in well A
      sim.add_stopping_criterion([&](const metropolis &sim) -> bool {
        const auto x0 = sim.positions()(0, 0);
        if (sim.step() % 10 == 0 && x0 < (eps / 2.0 - 1.0) 
            && x0 > (-eps / 2.0 - 1.0)) {
          if (dist(rng) < 0.7) return true;
        }
        return false;
      });
     
      sim.simulate(numsteps / numtasks);
      sim.clear_stopping_criteria();
      sim.reset_step();
      cout << "(task, x0) = (" << taskid << ", " << sim.positions()(0, 0)
           << ")\n";

      // measure
      sim.add_callback([&](const metropolis &sim) {
        const auto x0 = sim.positions()(0, 0);
        if (x0 < eps / 2.0 && x0 > -eps / 2.0) ++M;

        ++N;
      });

      // run main simulation
      sim.simulate(numsteps);
     
      // post-process and record data
      rc = MPI_Reduce(&N, &N_run, 1, MPI_UNSIGNED_LONG, MPI_SUM,
                      MASTER, MPI_COMM_WORLD);
      if (rc != MPI_SUCCESS) printf("%d: failure on mpc_reduce\n", taskid);

      rc = MPI_Reduce(&M, &M_run, 1, MPI_UNSIGNED_LONG, MPI_SUM,
                      MASTER, MPI_COMM_WORLD);
      if (rc != MPI_SUCCESS) printf("%d: failure on mpc_reduce\n", taskid);

      N_beta += N_run;
      M_beta += M_run;
    }

    if (taskid == MASTER) {
      const double beta = 1 / (T * kB);
      const double dxq = M_beta / (N_beta * eps); 
      const double ktst = 0.5 * sqrt(2 / (M_PI * beta * 1.0)) * dxq;
      ostr << beta << ',' << M_beta << ',' << N_beta 
           << ',' << dxq << ',' << ktst << '\n';
      cout << "beta       =    " << beta << '\n';
      cout << "M          =    " << M_beta << '\n';
      cout << "N          =    " << N_beta << '\n';
      cout << "<d(x-q)>   =    " << dxq << '\n';
      cout << "ktst       =    " << ktst << '\n';
      cout << '\n';
    }
  }

  MPI_Finalize();

  return 0;
}
