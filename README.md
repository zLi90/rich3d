# rich3d

A 3D Richards solver based on Kokkos.

# Requirements

 - C/C++ Compiler. The default compiler is gcc-11.
 - cmake (version 3.10 and above)
 - Kokkos (version 3.0)

 NOTE: If using Kokkos 4.0, the user needs to replace `Kokkos::InitArguments` in `driver.cpp` by the new conventions.

# Build and Execute

Using OpenMP:
`cmake -DEnable_GPU=OFF -DKokkos_ROOT=${path-to-kokkos-build(OpenMP)}`

Using CUDA:
`cmake -DEnable_GPU=ON -DKokkos_ROOT=${path-to-kokkos-build(CUDA)}`

Then type `make` to compile.

Type `./rich3d input_folder/ output_folder/ number_of_threads` to execute.

The `input_folder` should contain the input file (named "input")

# Input File
 - scheme = 1 : Use fully explicit scheme.
 - scheme = 2 : Use Predictor-Corrector scheme.
 - scheme = 3 : Use Modified Picard scheme.
 - scheme = others : Use Newton-Raphson scheme.
 - pcg_solve = 1 : Use Conjugate Gradient solver, otherwise use GMRES solver.
 - precondition = 1 : Use Jacobi preconditioner, otherwise no preconditioner is used.
 - iter_max, eps_min : Stopping criteria for Newton-Raphson and Picard schemes.

# Special Options in config.h
 - TEST_TRACY = 1 : Use the exponential soil model of Tracy2006. Otherwise the van Genuchten function is used. Note that when TEST_TRACY = 1, the top boundary condition is read from a file named `head_bczm`.
