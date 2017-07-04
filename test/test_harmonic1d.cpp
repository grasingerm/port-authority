#include <iostream>
#include <cassert>
#include <cstdlib>
#include <ctime>
#include <random>
#include "port_authority.hpp"
#include "mpi.h"

using namespace std;
using namespace pauth;

int main() {
    
  int taskid, numtasks;
  MPI_Init(nullptr, nullptr);
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &taskid);

  srand(time(NULL));
  const auto delta_max = static_cast<double>( (rand() % 200 + 50) / 100.0 );
  const auto spring_const = static_cast<double>( (rand() % 1990 + 10) / 500.0 );
  const auto T = static_cast<double>( (rand() % 200000 + 10) / 1000.0 );
  const auto x0 = static_cast<double>( (rand() % 100 - 200) / 50.0 );
  const unsigned nsteps = 10000000;

  /*
  if (taskid == 0) {
    cout << '\n';
    cout << "delta_max = " << delta_max << '\n';
    cout << "k         = " << spring_const << '\n';
    cout << "T         = " << T << '\n';
    cout << "x0        = " << x0 << '\n';
    cout << '\n';
  }
  */
  
  const double dx = 0.1;
  const double kB = 1.0;
  const molecular_id id = molecular_id::Test1;
  const size_t N = 1;
  const size_t D = 1;
  const double L = 1.0;
  const metric m = euclidean;
  const bc boundary = no_bc;
  const_k_spring_potential pot(spring_const);

  metropolis sim(id, N, D, L, continuous_trial_move(delta_max), 
                 &pot, T, kB, m, boundary, metropolis_acc, 
                 hardware_entropy_seed_gen, true);
  sim.positions()(0, 0) = x0;
  sim.update_U();
  const double u0 = accessors::U(sim);
  
  metropolis_suite msuite(sim, 0, 1, info_lvl_flag::QUIET);
  
  msuite.add_variable_to_average("x", [](const metropolis &sim) {
    return sim.positions()(0, 0);
  });
  msuite.add_variable_to_average("x^2", [](const metropolis &sim) {
    return sim.positions()(0, 0) * sim.positions()(0, 0);
  });
  msuite.add_variable_to_average("U", accessors::U);
  msuite.add_variable_to_average("delta(x - x0)", [=](const metropolis &sim) {
    const double x = sim.positions()(0, 0);
    return (x < x0 + dx && x > x0 - dx) ? 1 : 0;
  });

  msuite.simulate(nsteps);

  if(taskid == 0) {

    auto averages = msuite.averages();
    const double exp_x = averages["x"];
    const double exp_xsq = averages["x^2"];
    const double exp_E = averages["U"];
    const auto exp_xsq_an = (kB * T / spring_const);
    const auto exp_E_an = kB * T / 2.0;
    const auto Z_an = sqrt(2.0 * M_PI * kB * T / spring_const);

    //cout << "exp_x    =   " << exp_x << '\n';
    assert(abs(exp_x) < 1e-2);
    //cout << "exp_xsq  =   " << exp_xsq << " ?= " << exp_xsq_an << '\n';
    assert(abs((exp_xsq - exp_xsq_an) / exp_xsq_an) < 1e-2);
    //cout << "exp_E    =   " << exp_E << " ?= " << exp_E_an << '\n';
    assert(abs((exp_E - exp_E_an) / exp_E_an) < 1e-2);
    //cout << "Z        =   " 
    //     << (2.0 * dx * exp(-u0 / (kB * T)) / averages["delta(x - x0)"])
    //     << " ?= " << Z_an << '\n';
    assert(abs( ((2.0 * dx * exp(-u0 / (kB * T)) / averages["delta(x - x0)"]) -
                Z_an) / Z_an ) < 1e-2);
    //cout << "---------------------------------------------\n";

  }

  if(taskid == 0) cout << "1D Harmonic test passed.\n";

  MPI_Finalize();

  return 0;

}
