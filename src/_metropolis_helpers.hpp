#ifndef __METROPOLIS_HELPERS__
#define __METROPOLIS_HELPERS__

#include "molecular.hpp"
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;

namespace pauth {

// NOTE: this should only be used with positive values of e!
constexpr double _pow(const double base, const unsigned exponent) {
  return (exponent < 1) ? 1.0 : base * _pow(base, exponent - 1);
}

/* TODO: make this more general (N dimensional) using recursion
 */
void _init_positions_lattice(arma::mat &positions, const size_t N, 
                             const size_t D, const arma::vec &ls) {

  switch(D) {
    case 1: {
      const double dl = ls(0) / N;
      #pragma omp parallel for
      for (size_t i = 0; i < N; ++i) 
        positions(0, i) = dl / 2.0 + dl * i;

      break;
    }
    case 2: {
      const size_t n = std::floor(std::pow(N, 0.5));
      const arma::vec dls = ls / n;
      arma::vec xs(2);
      size_t idx = 0;

      xs(0) = dls(0) / 2.0;
      for (size_t i = 0; i < n; ++i) {
        xs(1) = dls(1) / 2.0;
        for (size_t j = 0; j < n; ++j) {
          positions.col(idx) = xs;
          xs(1) += dls(1);
        }
        xs(0) += dls(0);
      }

      break;
    }
    case 3: {
      const size_t n = std::floor(std::pow(N, 1.0 / 3.0));
      const arma::vec dls = ls / n;
      arma::vec xs(3);
      size_t idx = 0;

      xs(0) = dls(0) / 2.0;
      for (size_t i = 0; i < n; ++i) {
        xs(1) = dls(1) / 2.0;
        for (size_t j = 0; j < n; ++j) {
          xs(2) = dls(2) / 2.0;
          for (size_t k = 0; k < n; ++k) {
            positions.col(idx) = xs;
            xs(2) += dls(2);
          }
          xs(1) += dls(1);
        }
        xs(0) += dls(0);
      }

      break;
    }
    default: {
      throw std::domain_error("Higher dimensional default metropolis "
          "initialization not yet implemented");
    }
  }
}

void _load_positions(const char *fname, arma::mat &positions, const size_t N,
                     const size_t D) {
  string line;
  ifstream infile(fname, ifstream::in);

  if (!infile.is_open())
    throw invalid_argument(string("Could not open data file: ") +
                           string(fname));

  size_t n;
  infile >> n;

  if (n < N)
    throw invalid_argument(
        string("Data file: ") + string(fname) +
        string("does not contain the enough molecules. "
               "Expected: ") + to_string(N) + string(", found: ") + 
               to_string(n)
    );

  getline(infile, line); // read-in newline
  getline(infile, line); // comment line

  size_t j = 0;

  // read positions from file
  while (!infile.eof() && j < n) {
    for (size_t i = 0; i < D; ++i)
      infile >> positions(i, j);
    
    ++j;
  }

  infile.close();
}

void _load_positions(const char *fname, arma::mat &positions, const size_t N,
                     const size_t D, vector<molecular_id> &molecular_ids) {
  string line;
  ifstream infile(fname, ifstream::in);

  if (!infile.is_open())
    throw invalid_argument(string("Could not open data file: ") +
                           string(fname));

  size_t n;
  infile >> n;

  if (n < N)
    throw invalid_argument(
        string("Data file: ") + string(fname) +
        string("does not contain the enough molecules. "
               "Expected: ") + to_string(N) + string(", found: ") + 
               to_string(n)
    );

  getline(infile, line); // read-in newline
  getline(infile, line); // comment line

  size_t j = 0;

  // read positions from file
  string molecular_str;
  while (!infile.eof() && j < n) {
    infile >> molecular_str;

    if (molecular_names_to_ids.count(molecular_str)) 
      molecular_ids[j] = molecular_names_to_ids.at(molecular_str);
    else
      throw runtime_error(string("Invalid molecule name, ") + molecular_str);

    for (size_t i = 0; i < D; ++i)
      infile >> positions(i, j);
    
    ++j;
  }

  infile.close();
}

} // namespace pauth

#endif
