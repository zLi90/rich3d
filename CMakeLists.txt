cmake_minimum_required (VERSION 3.10)

IF(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.12.0")
  MESSAGE(STATUS "Setting policy CMP0074 to use <Package>_ROOT variables")
  CMAKE_POLICY(SET CMP0074 NEW)
ENDIF()

#----------------------------------------------------
#CMakeLists.txt:
#----------------------------------------------------
project (rich3d)
IF(Enable_GPU)
	if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  		set(CMAKE_CUDA_ARCHITECTURES 80)
	endif()
ELSE()
    set(CMAKE_C_COMPILER "/usr/local/bin/gcc-11")
    set(CMAKE_CXX_COMPILER "/usr/local/bin/g++-11")
    set(OPENMP_LIBRARIES "/usr/local/Cellar/gcc/11.2.0/lib")
    set(OPENMP_INCLUDES "/usr/local/Cellar/gcc/11.2.0/include")
	find_package(OpenMP REQUIRED)
	include_directories("${OPENMP_INCLUDES}")
	link_directories("${OPENMP_LIBRARIES}")
ENDIF()

file(GLOB sources *.h)
add_executable(rich3d ${sources} rich3d.cpp)

find_package(Kokkos REQUIRED)
target_link_libraries(rich3d PUBLIC Kokkos::kokkos)
#----------------------------------------------------
# Execution with OpenMP:
# cmake -DEnable_GPU=False -DKokkos_ROOT=${path-to-kokkos-build(OpenMP)}
#----------------------------------------------------
# Execution with CUDA:
# cmake -DEnable_GPU=True -DKokkos_ROOT=${path-to-kokkos-build(CUDA)}
#----------------------------------------------------