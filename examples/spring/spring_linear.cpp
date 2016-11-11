#include <iostream>
#include "port_authority.hpp"

using namespace std;
using namespace pauth;

int main() {

  const molecular_id id = molecular_id::Test1;
  const size_t N = 1;
  const size_t D = 1;
  const double L = 1.0;
  const double delta_max = 1.0;
  const double T = 1.0 / 1.0;
  const double kB = 1.0;
  const metric m = euclidean;
  const bc boundary = no_bc;
  const unsigned long nsteps = 10000000;
  const_k_spring_potential pot(1.0);
 
  metropolis sim("spring_linear.xyz", id, N, D, L, delta_max, 
                 &pot, T, kB, m, boundary, metropolis::DEFAULT_TMG, 
                 metropolis::DEFAULT_ACC,
                 random_device()());

  long unsigned nsamples = 0;
  long double U_sum = 0.0;
  long double x_sum = 0.0;
  long double xsq_sum = 0.0;

  sim.add_callback([&](const metropolis &sim) {
    for (const auto &potential : sim.potentials())
      U_sum += potential->U(sim);
    double x = sim.positions()(0, 0);
    x_sum += x;
    xsq_sum += x * x;
    ++nsamples;
  });

  //sim.add_callback(print_step(nsteps / 20, "linear spring"));
  sim.add_callback(print_trajectory_callback(nsteps / 20));

  mprof::tic();
  sim.simulate(nsteps);
  mprof::toc();

  cout << '\n';
  cout << "<U>    =  " << U_sum / nsamples << '\n';
  cout << "<x>    =  " << x_sum / nsamples << '\n';
  cout << "<x^2>  =  " << xsq_sum / nsamples << '\n';

  return 0;
}
