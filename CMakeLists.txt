cmake_minimum_required(VERSION 2.8)

project(port_authority)

set(CMAKE_C_COMPILER mpicc)
set(CMAKE_CXX_COMPILER mpic++)

set(CMAKE_CXX_FLAGS_DEBUG "-Wall -Wextra -g -std=c++14 -pedantic -pedantic-errors")
set(CMAKE_CXX_FLAGS_RELEASE "-Wall -Wextra -std=c++14 -O3")

# use OpenMP if compiler supports it
find_package(OpenMP)
if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package(MPI REQUIRED)

set(CMAKE_BINARY_DIR build)
set(LIBRARY_OUTPUT_PATH lib)
set(EXECUTABLE_OUTPUT_PATH bin)

include_directories(include)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

file(GLOB libsources "src/*.cpp" "include/*.cpp")

add_library(port_authority SHARED ${libsources})
add_executable(spring_linear examples/spring/spring_linear.cpp)
add_executable(spring_linear_mpi examples/spring/spring_linear_mpi.cpp)
add_executable(spring_linear_mpi_study examples/spring/spring_linear_mpi_study.cpp)
add_executable(spring_quad_mpi examples/spring/spring_quad_mpi.cpp)
add_executable(spring_quad_mpi_study examples/spring/spring_quad_mpi_study.cpp)
add_executable(two_state examples/two-state/two_state.cpp)
add_executable(two_state_ic_study examples/two-state/two_state_ic_study.cpp)
add_executable(spring_linear_kawasaki_mpi examples/spring/spring_linear_kawasaki_mpi.cpp)
add_executable(spring_quad_kawasaki_mpi examples/spring/spring_quad_kawasaki_mpi.cpp)

if (UNIX)
    target_link_libraries(port_authority m)
endif (UNIX)

target_link_libraries(port_authority armadillo)
target_link_libraries(spring_linear armadillo port_authority)
target_link_libraries(spring_linear_mpi armadillo port_authority ${MPI_CXX_LIBRARIES})
target_link_libraries(spring_linear_mpi_study armadillo port_authority ${MPI_CXX_LIBRARIES})
target_link_libraries(spring_quad_mpi armadillo port_authority ${MPI_CXX_LIBRARIES})
target_link_libraries(spring_quad_mpi_study armadillo port_authority ${MPI_CXX_LIBRARIES})
target_link_libraries(two_state armadillo port_authority ${MPI_CXX_LIBRARIES})
target_link_libraries(two_state_ic_study armadillo port_authority ${MPI_CXX_LIBRARIES})
target_link_libraries(spring_linear_kawasaki_mpi armadillo port_authority ${MPI_CXX_LIBRARIES})
target_link_libraries(spring_quad_kawasaki_mpi armadillo port_authority ${MPI_CXX_LIBRARIES})
