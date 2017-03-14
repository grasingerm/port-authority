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

  double delta_max, dx, sk;
  unsigned long nsteps;
  string fname, acc_criterion_str;

  desc.add_options()
    ("help,h", "Produce this help message.")
    ("x0-file,x", value<string>(&fname)->default_value("harmonic6d.xyz"), 
                  "Spring constant.")
    ("spring-const,k", value<double>(&sk)->default_value(1.0), "Spring constant.")
    ("beta,b", value<double>()->default_value(1.0), "1 / kT")
    ("delta,d", value<double>(&delta_max)->default_value(2.0), "Maximum step size.")
    ("num-steps,n", value<size_t>(&nsteps)->default_value(100000), "Number of steps.")
    ("dx,e", value<double>(&dx)->default_value(0.1), "Threshold size for "
      "approximating <dirac_delta(x - x0)> which is used to approximate Z.")
    ("acception-criterion,a", 
     value<string>(&acc_criterion_str)->default_value("metropolis"), 
     "Acceptance criterion (metropolis|kawasaki)")
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
  const size_t N = 2;
  const size_t D = 3;
  const double L = 1.0;
  const metric m = euclidean;
  const bc boundary = no_bc;
  const_k_spring_potential pot(sk);
  vector<double> xs;
  
  metropolis sim(fname.c_str(), id, N, D, L, continuous_trial_move(delta_max), 
                 &pot, T, kB, m, boundary, acc_criterion_funct);
  const double u0 = accessors::U(sim); // store initial energy
  const auto x0 = sim.positions(); // store initial positions

  if (vm.count("plot-histogram")) {
    sim.add_callback([&](const metropolis &sim) -> void {
      xs.push_back(sim.positions()(0, 0));
    });
  }

  metropolis_suite msuite(sim, 0, 1, info_lvl_flag::PROFILE);

  msuite.add_variable_to_average("x1", [](const metropolis &sim) {
    return sim.positions()(0, 0);
  });
  msuite.add_variable_to_average("x1^2", [](const metropolis &sim) {
    return sim.positions()(0, 0) * sim.positions()(0, 0);
  });
  msuite.add_variable_to_average("y2", [](const metropolis &sim) {
    return sim.positions()(1, 1);
  });
  msuite.add_variable_to_average("y2^2", [](const metropolis &sim) {
    return sim.positions()(1, 1) * sim.positions()(1, 1);
  });
  msuite.add_variable_to_average("U", accessors::U);
  msuite.add_variable_to_average("delta(x - x0)", [=](const metropolis &sim) {
      const auto& current_positions = sim.positions();
      for (size_t i = 0; i < N; ++i) {
        const auto& x_i = current_positions.col(i);
        const auto& x0_i = x0.col(i);

        for (unsigned j = 0; j < D; ++j)
          //if (abs(x_i(j) - x0_i(j)) > dx) 
          //  return 0;
          if (x_i(j) < x0_i(j) - dx || x_i(j) > x0_i(j) + dx)
            return 0;
      }
      return 1;
  });

  msuite.simulate(nsteps);

  if(taskid == 0) {
    auto averages = msuite.averages();

    const double exp_x = averages["x1"];
    const double exp_xsq = averages["x1^2"];
    const double exp_y = averages["y2"];
    const double exp_ysq = averages["y2^2"];
    cout << "<x1>     =     " << exp_x << '\n';
    cout << "<x1^2>   =     " << exp_xsq << '\n';
    cout << "Delta x1 =     " << sqrt(exp_xsq - exp_x*exp_x) << '\n' << '\n';
    cout << "<y2>     =     " << exp_y << '\n';
    cout << "<y2^2>   =     " << exp_ysq << '\n';
    cout << "Delta y2 =     " << sqrt(exp_ysq - exp_y*exp_y) << '\n' << '\n';
    cout << "<E>      =     " << averages["U"] << '\n';
    cout << "3 kT     =     " << N * D * kB * T / 2.0 << '\n' << '\n';

    const double dirac = averages["delta(x - x0)"] / pow(2 * dx, N * D);
    cout << "dirac_delta(x - x0) = " << dirac << '\n';
    cout << "Z        =     " << exp(-u0 / (kB * T)) / dirac << '\n';
    cout << "Z (an)   =     " << pow(sqrt(2.0 * M_PI * kB * T / sk), N * D) 
                              << '\n';

    if (vm.count("plot-histogram")) {
      throw "Not yet implemented.";
    }
  }

  MPI_Finalize();

  return 0;
}
