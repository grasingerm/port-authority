#ifndef __METROPOLIS_HPP__
#define __METROPOLIS_HPP__

#include <vector>
#include <array>
#include <random>

namespace pauth {

template <size_t D> class metropolis {
public:
private:
  std::vector<std::array<double, D>> positions;
  double delta_max;
};

} // namespace pauth

#endif
