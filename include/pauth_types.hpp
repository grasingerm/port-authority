#ifndef __PAUTH_TYPES_HPP__
#define __PAUTH_TYPES_HPP__

#include "molecular.hpp"
#include <armadillo>
#include <tuple>

namespace pauth {

class metropolis;
class abstract_potential;

using callback = std::function<void(const metropolis &)>;
using data_accessor = std::function<const arma::mat &(const metropolis &)>;
using value_accessor = std::function<double(const metropolis &)>;

using metric = std::tuple<
                std::function<arma::vec(const arma::vec&, const arma::vec&,
                                     const arma::vec&)>,
                std::function<double(const arma::vec&, const arma::vec&,
                                     const arma::vec&)>
               >;
using bc_ret = std::tuple<bool, arma::vec>;
using bc = std::function<bc_ret(metropolis &, const size_t, arma::vec&)>;
using trial_move_generator = std::function<arma::vec(const arma::mat&, const size_t)>;
using acc = std::function<bool(const metropolis &, const double, const double)>;
using seed_gen = std::function<unsigned()>;

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
