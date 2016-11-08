#include "callbacks.hpp"
#include <cmath>
#include <iostream>

namespace pauth {

using namespace std;

inline bool _step_to_exec(const metropolis &sim, const unsigned dstep) {
  return (sim.step() % dstep == 0);
}

void output_xyz(ostream &ostr, data_accessor da, const metropolis &sim,
                const bool prepend_id) {

  ostr << sim.N() << '\n';
  ostr << "step: " << sim.step() << '\n';

  const auto &data = da(sim);
  for (size_t j = 0; j < data.n_cols; ++j) {
    unsigned start_idx = 0;

    if (prepend_id)
      ostr << as_integer(sim.molecular_ids()[j]);
    else {
      start_idx = 1;
      ostr << data(1, j);
    }

    for (unsigned i = start_idx; i < data.n_rows; ++i)
      ostr << ' ' << data(i, j);

    ostr << '\n';
  }
}

void save_xyz_callback::operator()(const metropolis &sim) {
  if (_step_to_exec(sim, _dstep)) output_xyz(_outfile, _da, sim, _prepend_id);
}

void save_values_with_step_callback::operator()(const metropolis &sim) {
  if (_step_to_exec(sim, _dstep)) {
    _outfile << sim.step();
    for (const auto &va : _vas)
      _outfile << _delim << va(sim);
    _outfile << '\n';
  }
}

void print_values_with_step_callback::operator()(const metropolis &sim) {
  if (_step_to_exec(sim, _dstep)) {
    _ostr << sim.step();
    for (const auto &va : _vas)
      _ostr << _delim << va(sim);
    _ostr << '\n';
  }
}

callback print_step(const unsigned dstep, const char *msg) {
  return [=](const metropolis &sim) {
    if (_step_to_exec(sim, dstep)) {
      std::cout << msg << "; step: " << sim.step() << '\n';
    }
  };
}

callback print_profile(const unsigned dstep) {
  bool init_call = true;
  std::chrono::high_resolution_clock::time_point __tic_start__;

  return [=](const metropolis &sim) mutable {
    if (init_call) {
      __tic_start__ = std::chrono::high_resolution_clock::now();
      init_call = false;
    } else if (_step_to_exec(sim, dstep)) {

      std::cout << "Time to simulated: " << (chrono::high_resolution_clock::now() -
        __tic_start__).count() << " seconds.\n";
      __tic_start__ = std::chrono::high_resolution_clock::now();
    }
  };
}

}
