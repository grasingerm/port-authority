#ifndef __ARRAY_HELPERS_HPP__
#define __ARRAY_HELPERS_HPP__

#include <array>
#include <functional>
#include <iostream>

namespace {

using namespace std;

/*! \brief Print array
 *
 * \param os Output stream
 * \param a Array to print
 * \return Output stream
 */
template <typename T, size_t N>
ostream &operator<<(ostream &os, array<T, N> a) {
  auto aiter = a.cbegin();
  os << '[' << *aiter;
  for (++aiter; aiter != a.cend(); ++aiter)
    os << ' ' << *aiter;
  os << ']';
  return os;
}

/*! \brief Create a new array by applying a function to each element of an array
 * 
 * \param     f     Function to apply
 * \param     vals  Values for which to apply the function
 * \return          Mapped values
 */
template <typename T, size_t N> 
std::array<T, N> apply(std::function<T(T)> f, const std::array<T, N> &vals) {
  std::array<T, N> retval;
  #pragma omp parallel for
  for (size_t i = 0; i < N; ++i) retval[i] = f(vals[i]);
  return retval;
}

/*! \brief Subtract two arrays
 *
 * \param a1 Left hand side array
 * \param a2 Right hand side array
 * \return LHS array - RHS array
 */
template <typename T, size_t N>
auto operator-(const array<T, N> &a1, const array<T, N> &a2) {
  array<T, N> retval;
  #pragma omp parallel for
  for (size_t i = 0; i < N; ++i)
    retval[i] = a1[i] - a2[i];

  return retval;
}

/*! \brief Add two arrays
 *
 * \param a1 Left hand side array
 * \param a2 Right hand side array
 * \return LHS array + RHS array
 */
template <typename T, size_t N>
auto operator+(const array<T, N> &a1, const array<T, N> &a2) {
  array<T, N> retval;
  #pragma omp parallel for
  for (size_t i = 0; i < N; ++i)
    retval[i] = a1[i] + a2[i];

  return retval;
}

/*! \brief Inner product, or dot product, of two arrays
 * 
 * param a1 Left hand side array
 * \param a2 Right hand side array
 * \return Inner product of the arrays
 */
template <typename T, size_t N>
auto dot(const array<T, N> &a1, const array<T, N> &a2) {
  T retval;
  #pragma omp parallel for reduction(+:retval)
  for (int i = 0; i < N; ++i)
    retval += a1[i] * a2[i];

  return retval;
}

/*! \brief Euclidean norm, i.e. length, of a vector
 * 
 * param a Array of components
 * \return Eucliden length of vector
 */
template <typename T, size_t N> auto mag(const array<T, N> &a) {
  return sqrt(dot(a, a));
}

}

#endif
