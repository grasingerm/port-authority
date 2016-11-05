#ifndef __ACCEPTANCE_HPP__
#define __ACCEPTANCE_HPP__

#include <cmath>
#include "pauth_types.hpp"

using namespace std;

namespace pauth {

/*! \brief Decide to accept or reject move using Metropolis acceptance criteria
 *
 * \param     sim     Metropolis simulation object
 * \param     dU      Change of energy that results from move
 * \param     eps     Random number to determine acceptance
 * \return            Answer to: do we accept this move?
 */
bool metropolis_acc(const metropolis &sim, const double dU, const double eps) {
  if (dU <= 0) return true;
  else {
    return (eps <= exp(-sim.beta() * dU));
  }
}

/*! \brief Decide to accept or reject move using Kawasaki acceptance criteria
 *
 * \param     sim     Metropolis simulation object
 * \param     dU      Change of energy that results from move
 * \param     eps     Random number to determine acceptance
 * \return            Answer to: do we accept this move?
 */
inline bool kawasaki_acc(const metropolis &sim, const double dU, 
                         const double eps) {
  const double enbdu = exp(-sim.beta() * dU / 2);
  const double epbdu = exp(sim.beta() * dU / 2);
  return ( eps <= (enbdu / (enbdu + epbdu)) );
}

}

#endif
