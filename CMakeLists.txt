cmake_minimum_required (VERSION 3.10)

IF(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.12.0")
  MESSAGE(STATUS "Setting policy CMP0074 to use <Package>_ROOT variables")
  CMAKE_POLICY(SET CMP0074 NEW)
ENDIF()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

project (rich3d C CXX CUDA)

#project (rich3d)

file(GLOB sources *.h *.hpp)

add_executable(rich3d ${sources} rich3d.cpp)

set(OPENMP_LIBRARIES "/usr/lib/gcc/x86_64-linux-gnu/11/include")
set(OPENMP_INCLUDES "/usr/lib/gcc/x86_64-linux-gnu/11/include")
#find_package(OpenMP REQUIRED)
#include_directories("${OPENMP_INCLUDES}")
#link_directories("${OPENMP_LIBRARIES}")

find_package(Kokkos REQUIRED)
target_link_libraries(rich3d PUBLIC Kokkos::kokkos)


# To execute, type cmake -DKokkos_ROOT=/home/dell/Codes/kokkos-build-omp/
