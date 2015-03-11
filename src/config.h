#pragma once

//-----------------------------------------------------------------------------
//! Copyright (c) 2014-2015, Benjamin Worpitz
//! All rights reserved.
//! 
//! Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met :
//! * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
//! * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
//! * Neither the name of the TU Dresden nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
//! 
//! THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
//! IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) 
//! HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Choose the tests to perform.
// If MPI tests are executed, no other tests can be done by the same executable.
//-----------------------------------------------------------------------------
//#define MATMUL_MPI
#ifdef MATMUL_MPI
	//#define MATMUL_TEST_PAR_MPI_CANNON_STD						// Distributed matmul using the cannon algorithm, mpi and the optimized sequential implementation on each node.
	//#define MATMUL_TEST_PAR_MPI_CANNON_MKL						// Distributed matmul using the cannon algorithm, mpi and the intel mkl on each node.
	#define MATMUL_TEST_PAR_MPI_CANNON_CUBLAS						// Distributed matmul using the cannon algorithm, mpi and the cublas on each node.
	//#define MATMUL_TEST_PAR_MPI_DNS								// Distributed matmul using the dns algorithm, mpi and the optimized sequential implementation on each node.
#else
	#define MATMUL_TEST_SEQ									// The basic sequential matmul algorithm.
	#define MATMUL_TEST_SEQ_BASIC_OPT							// Optimized versions of the sequential algorithm each with only one optimization.
	#define MATMUL_TEST_SEQ_COMPL_OPT							// Optimized versions of the sequential algorithm with multiple optimizations at once.
	#define MATMUL_TEST_SEQ_STRASSEN							// The basic sequential strassen algorithm.
	#define MATMUL_TEST_PAR_OPENMP							// The optimized but not blocked algorithm with OpenMP annotations.
	//#define MATMUL_TEST_PAR_STRASSEN_OMP						// The strassen algorithm using OpenMP methods for the matmul base case and matadd and matsub.
	//#define MATMUL_TEST_PAR_OPENACC							// The optimized but not blocked algorithm with OpenACC annotations.
	//#define MATMUL_TEST_PAR_CUDA								// The matmul algorithm from the cuda developers guide.
	//#define MATMUL_TEST_PAR_BLAS_MKL							// matmul using the intel mkl blas implementation.
	//#define MATMUL_TEST_PAR_BLAS_CUBLAS						// matmul using the nvidia cublas2 implementation.
	//#define MATMUL_TEST_PAR_PHI_OFF_OPENMP					// Offloading the matmul onto the xeon phi using OpenMP.
	//#define MATMUL_TEST_PAR_PHI_OFF_BLAS_MKL					// Offloading the matmul onto the xeon phi using the intel mkl blas implementation.
#endif

//-----------------------------------------------------------------------------
// Result checking.
//-----------------------------------------------------------------------------
//#define MATMUL_VERIFY_RESULT								// The result of a computation will be compared with the result of the standard sequential algorithm.

//-----------------------------------------------------------------------------
// Select the element data type
//-----------------------------------------------------------------------------
#define MATMUL_ELEMENT_TYPE_DOUBLE							// If this is defined, double precision data elements are used, else single precision.

//-----------------------------------------------------------------------------
// Include definitions based on the settings made until here. Do not change!
//-----------------------------------------------------------------------------
#include "config_build.h"

//-----------------------------------------------------------------------------
// Allocation.
//-----------------------------------------------------------------------------
#define MATMUL_ALIGNED_MALLOC								// The matrices will be allocated in aligned storage if this is defined.

//-----------------------------------------------------------------------------
// Set the test matrix sizes.
//-----------------------------------------------------------------------------
#define MATMUL_MIN_N 4									// The minimal matrix size. This has to be greater or equal to 1.
#define MATMUL_STEP_N 16										// If MATMUL_STEP_N == 0 the size is doubled continuously.
#define MATMUL_MAX_N 512									// The maximal size.

//-----------------------------------------------------------------------------
// Measurment settings.
//-----------------------------------------------------------------------------
#define MATMUL_REPEAT_COUNT 2
#define MATMUL_REPEAT_TAKE_MINIMUM							// If this is defined the minimum of all repetitions is returned instead of the average.

//-----------------------------------------------------------------------------
// Sequential matmul settings.
//-----------------------------------------------------------------------------
#if (defined MATMUL_BUILD_SEQ_BASIC_OPT) || (defined MATMUL_BUILD_SEQ_COMPL_OPT)
	#define MATMUL_SEQ_BLOCK_FACTOR 128						// The block factor used.
#endif
#ifdef MATMUL_BUILD_SEQ_COMPL_OPT
	#define MATMUL_SEQ_COMPLETE_OPT_NO_BLOCK_CUT_OFF 1280	// The cut-off at which blocking is disabled for the sequential optimized algorithm.
#endif

//-----------------------------------------------------------------------------
// Strassen settings.
//-----------------------------------------------------------------------------
#ifdef MATMUL_BUILD_SEQ_STRASSEN
	#define MATMUL_STRASSEN_CUT_OFF 128						// The cut-off at which the standard algorithm is used instead of further reucrsive calculation.
#endif
#ifdef MATMUL_BUILD_PAR_STRASSEN_OMP
	#define MATMUL_STRASSEN_OMP_CUT_OFF 512						// The cut-off at which the standard algorithm is used instead of further reucrsive calculation.
#endif

//-----------------------------------------------------------------------------
// OpenMP Settings.
//-----------------------------------------------------------------------------
//#define MATMUL_OPENMP_PRINT_NUM_CORES						// If this is defined, each call to a matmul function will print out the number of cores used currently. This can have a huge performance impact especially for the recursive Strassen Method.

//-----------------------------------------------------------------------------
// OpenACC settings.
//-----------------------------------------------------------------------------
#ifdef MATMUL_BUILD_PAR_OPENACC
	#define MATMUL_OPENACC_GANG_SIZE 32
	#define MATMUL_OPENACC_VECTOR_SIZE 64
#endif
//-----------------------------------------------------------------------------
// CUDA Settings.
//-----------------------------------------------------------------------------
#ifdef MATMUL_BUILD_PAR_CUDA
	#define MATMUL_CUDA_BLOCKSIZE 32						// The block size used on the gpu.
#endif

#ifdef MATMUL_BUILD_PAR_PHI_OFF_BLAS_MKL
	#define MATMUL_PHI_OFF_BLAS_MKL_AUTO_WORKDIVISION		// If this is not set, the GEMM will be fully computed on the phi.
#endif
