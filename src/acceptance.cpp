#include "acceptance.hpp"
#include "metropolis.hpp"

namespace pauth {

bool metropolis_acc(const metropolis &sim, const double dU, const double eps) {
  if (dU <= 0) return true;
  else {
    return (eps <= exp(-sim.beta() * dU));
  }
}

bool kawasaki_acc(const metropolis &sim, const double dU, 
                  const double eps) {
  const double enbdu = exp(-sim.beta() * dU / 2);
  const double epbdu = exp(sim.beta() * dU / 2);
  return ( eps <= (enbdu / (enbdu + epbdu)) );
}

}
