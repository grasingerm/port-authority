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
bool metropolis_acc(const metropolis &sim, const double dU, const double eps);

/*! \brief Decide to accept or reject move using Kawasaki acceptance criteria
 *
 * \param     sim     Metropolis simulation object
 * \param     dU      Change of energy that results from move
 * \param     eps     Random number to determine acceptance
 * \return            Answer to: do we accept this move?
 */
bool kawasaki_acc(const metropolis &sim, const double dU, 
                  const double eps); 

}

#endif
