#ifndef __PAUTH_TYPES_HPP__
#define __PAUTH_TYPES_HPP__

#include "molecular.hpp"
#include <armadillo>

namespace pauth {

class metropolis;
class abstract_potential;

using callback = std::function<void(const metropolis &)>;
using data_accessor = std::function<const arma::mat &(const metropolis &)>;
using value_accessor = std::function<double(const metropolis &)>;

using metric = std::function<double(const arma::vec&, const arma::vec&,
                                    const arma::vec&)>;
using bc = std::function<void(metropolis &, const size_t, const arma::vec&)>;
using acc = std::function<bool(const metropolis &, const double, const double)>;

} // namespace pauth

/*! \brief Helper function for printing the integer value of an enum class
 *
 * \param   value   Enumeration value
 * \return          Enumeration value converted to integer
 */
template <typename Enumeration>
auto as_integer(Enumeration const value) ->
    typename std::underlying_type<Enumeration>::type {

  return static_cast<typename std::underlying_type<Enumeration>::type>(value);
}

#endif
