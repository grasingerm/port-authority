#ifndef __SEED_HPP__
#define __SEED_HPP__

#include <random>
#include <chrono>
#include "pauth_types.hpp"

namespace pauth {

/*! Generate seed using hardware entropy device
 *
 * \return      Random seed
 */
inline unsigned hardware_entropy_seed_gen() {
  return std::random_device()();
}

/*! Generate seed using current time
 *
 * \return      Time since epoch
 */
inline unsigned clock_seed_gen() {
  return std::chrono::high_resolution_clock::now().time_since_epoch().count(); 
}

static seed_gen _default_seed_gen = hardware_entropy_seed_gen;

}

#endif
