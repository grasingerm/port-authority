#include <iostream>
#include <cassert>
#include <cstdlib>
#include <ctime>
#include <random>
#include <armadillo>
#include "port_authority.hpp"
#include "mpi.h"

using namespace std;
using namespace pauth;
using namespace arma;

int main() {
    
  int taskid, numtasks;
  MPI_Init(nullptr, nullptr);
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &taskid);

  srand(time(NULL));
  const auto delta_max = static_cast<double>( (rand() % 200 + 50) / 100.0 );
  const auto spring_const = static_cast<double>( (rand() % 1990 + 10) / 500.0 );
  const auto T = static_cast<double>( (rand() % 200000 + 10) / 1000.0 );
  const auto p0x = static_cast<double>( (rand() % 100 - 200) / 50.0 );
  const auto p0y = static_cast<double>( (rand() % 100 - 200) / 50.0 );
  const auto E0x = static_cast<double>( (rand() % 100 - 200) / 20.0 );
  const auto E0y = static_cast<double>( (rand() % 100 - 200) / 20.0 );
  const auto E0z = static_cast<double>( (rand() % 100 - 200) / 20.0 );
  const unsigned nsteps = 20000000;

  if (taskid == 0) {
    cout << "2D linear dipole test started.\n";
    cout << '\n';
    cout << "delta_max = " << delta_max << '\n';
    cout << "k         = " << spring_const << '\n';
    cout << "T         = " << T << '\n';
    cout << "p0x       = " << p0x << '\n';
    cout << "p0y       = " << p0y << '\n';
    cout << "E0x       = " << E0x << '\n';
    cout << "E0y       = " << E0y << '\n';
    cout << "E0z       = " << E0z << '\n';
    cout << '\n';
  }
  
  const double dx = 0.1;
  const double kB = 1.0;
  const molecular_id id = molecular_id::Test1;
  const size_t N = 1;
  const size_t D = 3;
  const double L = 1.0;
  const metric m = euclidean;
  const bc boundary = no_bc;
  const auto inv_chi = eye(3, 3);
  dipole_strain_linear_2d_potential strain_pot(inv_chi);
  dipole_electric_potential electric_pot({E0x, E0y, E0z}, {0, 1, 2});

  metropolis sim(id, N, D, L, continuous_trial_move(delta_max), 
                 {&strain_pot, &electric_pot}, T, kB, m, boundary, 
                 metropolis_acc, hardware_entropy_seed_gen, true);
  sim.set_positions({p0x, p0y}, 0);
  const double u0 = accessors::U(sim);
  
  metropolis_suite msuite(sim, 0, 1, info_lvl_flag::QUIET);
  
  msuite.add_variable_to_average("px", [](const metropolis &sim) {
    return sim.positions()(0, 0);
  });
  msuite.add_variable_to_average("py", [](const metropolis &sim) {
    return sim.positions()(1, 0) * sim.positions()(0, 0);
  });
  msuite.add_variable_to_average("pz", [](const metropolis &sim) {
    return sim.positions()(2, 0) * sim.positions()(0, 0);
  });
  msuite.add_variable_to_average("px^2", [](const metropolis &sim) {
    auto x = sim.positions()(0, 0);
    return x*x;
  });
  msuite.add_variable_to_average("py^2", [](const metropolis &sim) {
    auto y = sim.positions()(1, 0);
    return y*y;
  });
  msuite.add_variable_to_average("pz^2", [](const metropolis &sim) {
    auto z = sim.positions()(2, 0);
    return z*z;
  });
  msuite.add_variable_to_average("U", accessors::U);
  msuite.add_variable_to_average("delta(x - x0)", [=](const metropolis &sim) {
    const double px = sim.positions()(0, 0);
    const double py = sim.positions()(1, 0);
    return (px < p0x + dx && px > p0x - dx && py < p0y + dx && py > p0y - dx) ? 1 : 0;
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

    cout << "exp_x    =   " << exp_x << '\n';
    assert(abs(exp_x) < 5e-2);
    cout << "exp_xsq  =   " << exp_xsq << " ?= " << exp_xsq_an << '\n';
    assert(abs((exp_xsq - exp_xsq_an) / exp_xsq_an) < 1e-2);
    cout << "exp_E    =   " << exp_E << " ?= " << exp_E_an << '\n';
    assert(abs((exp_E - exp_E_an) / exp_E_an) < 1e-2);
    cout << "Z        =   " 
         << (2.0 * dx * exp(-u0 / (kB * T)) / averages["delta(x - x0)"])
         << " ?= " << Z_an << '\n';
    assert(abs( ((2.0 * dx * exp(-u0 / (kB * T)) / averages["delta(x - x0)"]) -
                Z_an) / Z_an ) < 1e-2);
    cout << "---------------------------------------------\n";

  }

  if(taskid == 0) cout << "1D Harmonic test passed.\n";

  MPI_Finalize();

  return 0;

}
