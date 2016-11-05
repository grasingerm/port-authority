#ifndef __BALLINPROF__
#define __BALLINPROF__

#include <chrono>
#include <ctime>
#include <functional>
#include <iostream>
#include <tuple>

namespace mprof {

static std::chrono::time_point<std::chrono::high_resolution_clock> tic_start__ =
    std::chrono::time_point<std::chrono::high_resolution_clock>::min();

//! start stop watch
inline void tic() { tic_start__ = std::chrono::high_resolution_clock::now(); }

//! stop stop watch. tell time
void toc() {
  if (tic_start__ ==
      std::chrono::time_point<std::chrono::high_resolution_clock>::min())
    throw std::logic_error("toc() called before tic()");
  const std::chrono::duration<double> elapsed =
      std::chrono::high_resolution_clock::now() - tic_start__;
  std::cout << "Time elapsed: " << elapsed.count() << " seconds.\n";
  tic_start__ = std::chrono::high_resolution_clock::now();
}

//! Profile a simple function given a function pointer
//!
//! \param f Function pointer
//! \param args Function arguments
//! \return (function result, function duration)
template <typename Result, typename... Sig>
std::tuple<Result, std::chrono::duration<double>> profile(Result (*f)(Sig...),
                                                          Sig... args) {
  std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
  start = std::chrono::high_resolution_clock::now();
  Result result = f(args...);
  end = std::chrono::high_resolution_clock::now();

  return std::tuple<Result, std::chrono::duration<double>>(result, end - start);
}

//! Profile a simple function given a functor
//!
//! \param f Functor object
//! \param args Function arguments
//! \return (functor result, function duration)
template <typename Result, typename... Sig>
std::tuple<Result, std::chrono::duration<double>>
profile(const std::function<Result(Sig...)> &f, Sig... args) {
  std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
  start = std::chrono::high_resolution_clock::now();
  Result result = f(args...);
  end = std::chrono::high_resolution_clock::now();

  return std::tuple<Result, std::chrono::duration<double>>(result, end - start);
}

//! Profile a simple function given a function pointer
//!
//! \param f Function pointer
//! \param args Function arguments
//! \return function duration
template <typename... Sig>
std::chrono::duration<double> simple_profile(void (*f)(Sig...), Sig... args) {
  std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
  start = std::chrono::high_resolution_clock::now();
  f(args...);
  end = std::chrono::high_resolution_clock::now();

  return end - start;
}

//! Profile a simple function given a functor
//!
//! \param f Functor object
//! \param args Function arguments
//! \return function duration
template <typename... Sig>
std::chrono::duration<double>
simple_profile(const std::function<void(Sig...)> &f, Sig... args) {
  std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
  start = std::chrono::high_resolution_clock::now();
  f(args...);
  end = std::chrono::high_resolution_clock::now();

  return end - start;
}

} // namespace mprof

#endif // __BALLINPROF__
