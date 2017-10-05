#include <iostream>
#include <array>
#include <string>
#include <cstring>
#include <fstream>
#include "port_authority.hpp"

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/tokenizer.hpp>
#include <boost/token_functions.hpp>

using namespace boost;
using namespace boost::program_options;

#define _USE_MATH_DEFINES
#include <cmath>

#include "mpi.h"

using namespace std;
using namespace pauth;

int main(int argc, char* argv[]) {

  int taskid, numtasks;
  MPI_Init(nullptr, nullptr);
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &taskid);

  options_description desc("\nMetropolis simulation of a rotating, fixed dipole."
                           "\n\nAllowed arguments");

  double delta_max, mu, E0;
  unsigned long nsteps;
  string fname, acc_criterion_str;

  desc.add_options()
    ("help,h", "Produce this help message.")
    ("x0-file,x", value<string>(&fname)->default_value("dipole_2D_1.xyz"), 
                  "XYZ file")
    ("mu,k", value<double>(&mu)->default_value(1.0), "Dipole magnitude")
    ("E0,E", value<double>(&E0)->default_value(1.0), "Electric field magnitude")
    ("beta,b", value<double>()->default_value(1.0), "1 / kT")
    ("delta,d", value<double>(&delta_max)->default_value(M_PI / 8), "Maximum step size.")
    ("num-steps,n", value<size_t>(&nsteps)->default_value(100000), "Number of steps.")
    ("acception-criterion,a", 
     value<string>(&acc_criterion_str)->default_value("metropolis"), 
     "Acceptance criterion (metropolis|kawasaki)");

  variables_map vm;
  try {
    store(command_line_parser(argc, argv).options(desc).run(), vm);
    notify(vm);
  } catch(std::exception &e) {
    cout << endl << e.what() << endl;
    cout << desc << endl;
  }

  if (vm.count("help")) {
    if(taskid == 0) {
      cout << "--help specified" << endl;
      cout << desc << endl;
    }
    MPI_Finalize();
    return EXIT_SUCCESS;
  }

  acc acc_criterion_funct;
  if (acc_criterion_str == "metropolis") acc_criterion_funct = metropolis_acc;
  else if (acc_criterion_str == "kawasaki") 
    acc_criterion_funct = kawasaki_acc;
  else {
    cerr << "Error: acceptance-criterion \"" << acc_criterion_str << "\" is "
            "not understood.\n";
    return -1;
  }

  const double T = 1.0;
  const double kB = 1.0 / vm["beta"].as<double>();

  const molecular_id id = molecular_id::Test1;
  const size_t N = 1;
  const size_t D = 2;
  const metric m = euclidean;
  const bc boundary = periodic_bc;
  auto pot = const_E_fixed_2D_dipole_potential({0.0, 0.0, E0}, mu);
  continuous_trial_move tmove(delta_max);
 
  metropolis sim(fname.c_str(), id, N, D, {2*M_PI, M_PI}, tmove, 
                 &pot, T, kB, m, boundary,
                 metropolis_acc);

  metropolis_suite msuite(sim, 0, 1, info_lvl_flag::VERBOSE);

  msuite.add_variable_to_average("px", [&](const metropolis &sim) {
    return pot.dipole(id, sim.positions().col(0))(0);
  });
  msuite.add_variable_to_average("py", [&](const metropolis &sim) {
    return pot.dipole(id, sim.positions().col(0))(1);
  });
  msuite.add_variable_to_average("pz", [&](const metropolis &sim) {
    return pot.dipole(id, sim.positions().col(0))(2);
  });
  msuite.add_variable_to_average("U", accessors::U);

  msuite.simulate(nsteps);
  msuite.report_averages();
  
  if(taskid == 0) {
    const auto z = mu * E0 / (T * kB);
    cout << '\n';
    cout << "Expected <pz> = " << mu * (1.0/tanh(z) - 1.0 / z) << '\n';
    cout << "Low T limit <pz> = " << mu << '\n';
    cout << "High T limit <pz> = " << mu*mu * E0 / (3.0 * T * kB) << '\n';
  }

  MPI_Finalize();

  return 0;
}
