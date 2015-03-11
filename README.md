Matrix Multiplication in C
==========================

This repository contains multiple sequential and parallel implementations of matrix multiplications (C = A * B + C) written in C.

Implemented Versions
--------------------

* Sequential:
  * Basic
  * With single optimization:
    * pointer instead of index
    * restrict
    * loop reorder
    * index precalculation
    * loop unrolling (4, 8, 16)
    * blocking
  * With multiple optimizations:
    * restrict + loop reorder + blocking
    * restrict + loop reorder
  * Strassen algorithm
  
* Parallel:
  * OpenMP:
    * guided schedule
    * static schedule
    * static schedule + loop collapsing
    * offload to Intel XeonPhi
  * Strassen algorithm
  * OpenACC
  * CUDA
  * BLAS: 
    * MKL
    * MKL offload to Intel XeonPhi
    * cuBLAS2
  * MPI:
    * Cannon algorithm:
      * optimized sequential per node
      * MKL per node
      * cuBLAS2 per node
    * DNS 

Building
--------

1. Chose the tests to execute and their parameters in "src/config.hpp"

2. Compile:
	- Windows + VS2013 (2010/2012 should equally work if the version number in the solution and project is changed):
		Solution under "build/vs2013"
	- Others: use the makefiles in "build/make/" like
		make -f makefile_gcc matmul
	  If you have selected the cuda tests you have to give matmulcu as target
		make -f makefile_icc matmulcu
	  If you have selected a mpi test you have to give matmulmpi as target
		make -f makefile_icc matmulmpi