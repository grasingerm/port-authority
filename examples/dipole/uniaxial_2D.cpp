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

  double delta_max, kappa, E0;
  size_t nterms;
  unsigned long nsteps;
  string fname, acc_criterion_str;

  desc.add_options()
    ("help,h", "Produce this help message.")
    ("x0-file,x", value<string>(&fname)->default_value("dipole_2D_1.xyz"), 
                  "XYZ file")
    ("kappa,k", value<double>(&kappa)->default_value(1.0), "Susceptibility param")
    ("E0,E", value<double>(&E0)->default_value(1.0), "Electric field magnitude")
    ("beta,b", value<double>()->default_value(1.0), "1 / kT")
    ("delta,d", value<double>(&delta_max)->default_value(M_PI / 8), "Maximum step size.")
    ("num-steps,n", value<size_t>(&nsteps)->default_value(100000), "Number of steps.")
    ("num-terms,t", value<size_t>(&nterms)->default_value(10), 
      "Number of terms in expansion of analytical solution.")
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
  auto pot = const_E_uniaxial_2D_dipole_potential({0.0, 0.0, E0}, kappa);
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
    unsigned long long fact = 1;
    double num = std::pow(E0*E0 * kappa / (T * kB), 0) / (fact * (0 + 0.5)), 
           denom = 0.0;
    for (unsigned n = 1; n < nterms; ++n) {
      fact *= n;
      num += std::pow(E0*E0 * kappa / (T * kB), n) / (fact * (n + 0.5));
      denom += (2 * n * std::pow(kappa / (T * kB), n) * std::pow(E0, 2*n-1)) 
        / (fact * (n + 0.5));
    }
    cout << '\n';
    cout << "Expected <pz> = " << (num * T * kB) / denom << '\n';
    cout << "Low T limit <pz> = " << kappa * E0 << '\n';
    cout << "High T limit <pz> = " << kappa * E0 / 3.0 << '\n';
  }

  MPI_Finalize();

  return 0;
}
