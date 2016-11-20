#ifndef __PAUTH_DEBUG_HPP__
#define __PAUTH_DEBUG_HPP__

#include <iostream>
#include <stdexcept>
#include "mpi.h"

namespace pauth {

inline void check_mpi_rc(int rc, const char *msg = "MPI failure\n") {
  if (rc != MPI_SUCCESS) std::cerr << msg;
}

inline void except_mpi_rc(int rc, const char *msg = "MPI failure\n") {
  if (rc != MPI_SUCCESS) throw std::runtime_error(msg);
}

}

#endif
