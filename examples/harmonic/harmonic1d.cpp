#include <iostream>
#include <array>
#include <string>
#include <cstring>
#include <fstream>
#include <exception>
#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>
#include "port_authority.hpp"

using namespace std;
using namespace pauth;

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/tokenizer.hpp>
#include <boost/token_functions.hpp>

using namespace boost;
using namespace boost::program_options;

#include "mpi.h"

int main(int argc, char* argv[]) {

  int taskid, numtasks;
  MPI_Init(nullptr, nullptr);
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &taskid);

  options_description desc("\nMetropolis simulation of a harmonic oscillator."
                           "\n\nAllowed arguments");

  double delta_max, x0, dx, sk;
  unsigned long nsteps;

  desc.add_options()
    ("help,h", "Produce this help message.")
    ("x0,x", value<double>(&x0)->default_value(0.0), "Initial position.")
    ("spring-const,k", value<double>(&sk)->default_value(1.0), "Spring constant.")
    ("beta,b", value<double>()->default_value(1.0), "1 / kT")
    ("delta,d", value<double>(&delta_max)->default_value(2.0), "Maximum step size.")
    ("num-steps,n", value<size_t>(&nsteps)->default_value(100000), "Number of steps.")
    ("dx,e", value<double>(&dx)->default_value(0.1), "Threshold size for "
      "approximating <dirac_delta(x - x0)> which is used to approximate Z.")
    ("plot-histogram,p", "Plot histogram.");

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

  const double T = 1.0;
  const double kB = 1.0 / vm["beta"].as<double>();

  const molecular_id id = molecular_id::Test1;
  const size_t N = 1;
  const size_t D = 1;
  const double L = 1.0;
  const metric m = euclidean;
  const bc boundary = no_bc;
  const_k_spring_potential pot(sk);
  vector<double> xs;
  
  metropolis sim(id, N, D, L, continuous_trial_move(delta_max), 
                 &pot, T, kB, m, boundary, metropolis_acc, 
                 hardware_entropy_seed_gen, true);
  sim.positions()(0, 0) = x0;
  const double u0 = accessors::U(sim);

  if (vm.count("plot-histogram")) {
    sim.add_callback([&](const metropolis &sim) -> void {
      xs.push_back(sim.positions()(0, 0));
    });
  }

  metropolis_suite msuite(sim, 0, 1, info_lvl_flag::VERBOSE);

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
    cout << "<x>      =     " << exp_x << '\n';
    cout << "<x^2>    =     " << exp_xsq << '\n';
    cout << "kT/k     =     " << kB * T / sk << '\n';
    cout << "Delta x  =     " << sqrt(exp_xsq - exp_x*exp_x) << '\n';
    cout << "<E>      =     " << averages["U"] << '\n';
    cout << "kT / 2   =     " << kB * T / 2.0 << '\n';
    cout << "Z        =     " << (2.0 * dx * exp(-u0 / (kB * T)) 
                                  / averages["delta(x - x0)"]) << '\n';
    cout << "Z (an)   =     " << sqrt(2.0 * M_PI * kB * T / sk) << '\n';

    if (vm.count("plot-histogram")) {
      throw "Not yet implemented.";
    }
  }

  MPI_Finalize();

  return 0;
}
