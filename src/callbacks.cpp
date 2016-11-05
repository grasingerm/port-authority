#include "callbacks.hpp"
#include <cmath>
#include <iostream>

namespace mmd {

using namespace std;

inline void _check_dt(const double dt) {
  if (dt < 0)
    throw std::invalid_argument("Time between callbacks, dt, must be positive");
}

inline bool _time_to_exec(const simulation &sim, const double dt) {
  return (fmod(sim.get_time(), dt) < sim.get_dt());
}

void output_xyz(ostream &ostr, data_accessor da, const simulation &sim,
                const bool prepend_id) {

  ostr << sim.get_molecular_ids().size() << '\n';
  ostr << "time: " << sim.get_time() << '\n';

  const auto &data = da(sim);
  for (size_t j = 0; j < data.n_cols; ++j) {
    unsigned start_idx = 0;

    if (prepend_id)
      ostr << as_integer(sim.get_molecular_ids()[j]);
    else {
      start_idx = 1;
      ostr << data(1, j);
    }

    for (unsigned i = start_idx; i < 3; ++i)
      ostr << ' ' << data(i, j);

    ostr << '\n';
  }
}

void save_xyz_callback::operator()(const simulation &sim) {
  if (_time_to_exec(sim, dt)) output_xyz(outfile, da, sim, prepend_id);
}

void save_values_with_time_callback::operator()(const simulation &sim) {
  if (_time_to_exec(sim, dt)) {
    outfile << sim.get_time();
    for (const auto &va : vas)
      outfile << delim << va(sim);
    outfile << '\n';
  }
}

void print_values_with_time_callback::operator()(const simulation &sim) {
  if (_time_to_exec(sim, dt)) {
    ostr << sim.get_time();
    for (const auto &va : vas)
      ostr << delim << va(sim);
    ostr << '\n';
  }
}

// TODO: maybe initialize energy from t = 0, pass simulation object as arg?
callback check_energy(const double dt, const double eps) {

  _check_dt(dt);

  double init_energy = 0.0;
  double divisor = 1.0;
  bool init_call = true;

  return [=](const simulation &sim) mutable {
    if (init_call) {
      init_energy = total_energy(sim);
      divisor = (init_energy != 0.0) ? init_energy : 1.0;
      init_call = false;
    } else if (_time_to_exec(sim, dt)) {

      if (abs((init_energy - total_energy(sim)) / divisor) > eps) {
        cerr << "Energy is diverging outside the tolerance limit\n";
        cerr << "Initial energy: " << init_energy
             << ", current energy: " << total_energy(sim)
             << ", tolerance: " << eps << '\n';
        throw runtime_error("Energy is not being conserved");
      }
    }
  };
}

// TODO: maybe initialize momentum from t = 0, pass simulation object as arg?
callback check_momentum(const double dt, const double eps) {

  _check_dt(dt);

  arma::vec init_momentum;
  double divisor = 1.0;
  bool init_call = true;

  return [=](const simulation &sim) mutable {
    if (init_call) {
      init_momentum = momentum(sim);
      divisor = (norm(init_momentum) > eps) ? norm(init_momentum) : 1.0;
      init_call = false;
    } else if (_time_to_exec(sim, dt)) {

      if (norm(init_momentum - momentum(sim)) / divisor > eps) {
        cerr << "System momentum is diverging outside the tolerance limit\n";
        cerr << "Initial momentum:\n"
             << init_momentum << ", current momentum:\n"
             << momentum(sim) << ", tolerance: " << eps << '\n';
        throw runtime_error("Momentum is not being conserved");
      }
    }
  };
}

callback print_time(const double dt, const char *msg) {

  _check_dt(dt);

  return [=](const simulation &sim) {
    if (_time_to_exec(sim, dt)) {
      std::cout << msg << "; time: " << sim.get_time();
    }
  };
}

callback print_profile(const double dt) {

  _check_dt(dt);

  bool init_call = true;

  return [=](const simulation &sim) mutable {
    if (init_call) {
      mprof::tic();
      init_call = false;
    } else if (_time_to_exec(sim, dt)) {

      std::cout << "Time simulated: ";
      mprof::toc();
      mprof::tic();
    }
  };
}

}
