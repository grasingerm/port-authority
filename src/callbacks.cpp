#include "callbacks.hpp"
#include <cmath>
#include <iostream>

namespace pauth {

using namespace std;

inline void _check_dstep(const unsigned dstep) {
  if (dstep < 0)
    throw std::invalid_argument("Steps between callbacks, dstep, must be positive");
}

inline bool _step_to_exec(const simulation &sim, const unsigned dstep) {
  return (sim.step() % dstep == 0);
}

void output_xyz(ostream &ostr, data_accessor da, const simulation &sim,
                const bool prepend_id) {

  ostr << sim.N() << '\n';
  ostr << "step: " << sim.step() << '\n';

  const auto &data = _da(sim);
  for (size_t j = 0; j < data.n_cols; ++j) {
    unsigned start_idx = 0;

    if (_prepend_id)
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

void save_xyz_callback::operator()(const simulation &sim) {
  if (_step_to_exec(sim, _dstep)) output_xyz(_outfile, _da, sim, _prepend_id);
}

void save_values_with_step_callback::operator()(const simulation &sim) {
  if (_step_to_exec(sim, _dstep)) {
    outfile << sim.step();
    for (const auto &va : _vas)
      outfile << delim << va(sim);
    outfile << '\n';
  }
}

void print_values_with_step_callback::operator()(const simulation &sim) {
  if (_step_to_exec(sim, _dstep)) {
    ostr << sim.step();
    for (const auto &va : _vas)
      ostr << delim << va(sim);
    ostr << '\n';
  }
}

callback print_step(const unsigned dstep, const char *msg) {

  _check_dstep(dstep);

  return [=](const simulation &sim) {
    if (_step_to_exec(sim, dstep)) {
      std::cout << msg << "; step: " << sim.step();
    }
  };
}

callback print_profile(const unsigned dstep) {

  _check_dstep(dstep);

  bool init_call = true;

  return [=](const simulation &sim) mutable {
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
