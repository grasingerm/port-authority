cmake_minimum_required(VERSION 2.8)

project(port_authority)

set(CMAKE_CXX_FLAGS_DEBUG "-Wall -Wextra -g -std=c++14 -pedantic -pedantic-errors")
set(CMAKE_CXX_FLAGS_RELEASE "-Wall -Wextra -std=c++14 -O3")

# use OpenMP if compiler supports it
find_package(OpenMP)
if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(CMAKE_BINARY_DIR build)
set(LIBRARY_OUTPUT_PATH lib)
set(EXECUTABLE_OUTPUT_PATH bin)

include_directories(include)

file(GLOB libsources "src/*.cpp" "include/*.cpp")

add_library(port_authority SHARED ${libsources})
add_executable(spring_linear examples/spring/spring_linear.cpp)

if (UNIX)
    target_link_libraries(port_authority m)
endif (UNIX)

target_link_libraries(port_authority armadillo)
target_link_libraries(spring_linear armadillo port_authority)
