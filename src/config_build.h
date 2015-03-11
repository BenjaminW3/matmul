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
// !!! Do not change anything in this file !!!
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// The tests to be executed define the parts to be build. 
//-----------------------------------------------------------------------------
#ifdef MATMUL_TEST_SEQ
	#define MATMUL_BUILD_SEQ
#endif
#ifdef MATMUL_TEST_SEQ_BASIC_OPT
	#define MATMUL_BUILD_SEQ_BASIC_OPT
#endif
#ifdef MATMUL_TEST_SEQ_COMPL_OPT
	#define MATMUL_BUILD_SEQ_COMPL_OPT
#endif
#ifdef MATMUL_TEST_SEQ_STRASSEN
	#define MATMUL_BUILD_SEQ_STRASSEN
	#define MATMUL_BUILD_SEQ_COMPL_OPT
#endif
#ifdef MATMUL_TEST_PAR_OPENMP
	#define MATMUL_BUILD_PAR_OPENMP
#endif
#ifdef MATMUL_TEST_PAR_STRASSEN_OMP
	#define MATMUL_BUILD_PAR_STRASSEN_OMP
	#define MATMUL_BUILD_PAR_OPENMP
#endif
#ifdef MATMUL_TEST_PAR_OPENACC
	#define MATMUL_BUILD_PAR_OPENACC
#endif
#ifdef MATMUL_TEST_PAR_CUDA
	#define MATMUL_BUILD_PAR_CUDA
#endif
#ifdef MATMUL_TEST_PAR_MPI_CANNON_STD
	#define MATMUL_BUILD_PAR_MPI_CANNON
	#define MATMUL_BUILD_PAR_MPI_CANNON_STD
	#define MATMUL_BUILD_SEQ_COMPL_OPT
#endif
#ifdef MATMUL_TEST_PAR_MPI_CANNON_MKL
	#define MATMUL_BUILD_PAR_MPI_CANNON
	#define MATMUL_BUILD_PAR_MPI_CANNON_MKL
	#define MATMUL_BUILD_PAR_BLAS_MKL
#endif
#ifdef MATMUL_TEST_PAR_MPI_CANNON_CUBLAS
	#define MATMUL_BUILD_PAR_MPI_CANNON
	#define MATMUL_BUILD_PAR_MPI_CANNON_CUBLAS
	#define MATMUL_BUILD_PAR_BLAS_CUBLAS
#endif
#ifdef MATMUL_TEST_PAR_MPI_DNS
	#define MATMUL_BUILD_PAR_MPI_DNS
	#define MATMUL_BUILD_SEQ_COMPL_OPT
#endif
#ifdef MATMUL_TEST_PAR_BLAS_MKL
	#define MATMUL_BUILD_PAR_BLAS_MKL
#endif
#ifdef MATMUL_TEST_PAR_BLAS_CUBLAS
	#define MATMUL_BUILD_PAR_BLAS_CUBLAS
#endif
#ifdef MATMUL_TEST_PAR_PHI_OFF_OPENMP
	#define MATMUL_BUILD_PAR_PHI_OFF_OPENMP
#endif
#ifdef MATMUL_TEST_PAR_PHI_OFF_BLAS_MKL
	#define MATMUL_BUILD_PAR_PHI_OFF_BLAS_MKL
#endif

//-----------------------------------------------------------------------------
// If the result is to be checked, the basic sequential algorithm is required.
//-----------------------------------------------------------------------------
#ifdef MATMUL_VERIFY_RESULT
	#define MATMUL_BUILD_SEQ
#endif

//-----------------------------------------------------------------------------
// If MPI tests are to be build, set some additional definitions.
//-----------------------------------------------------------------------------
#ifdef MATMUL_MPI
	// Do not change these definitions!
	#include <mpi.h>
	#define MATMUL_MPI_COMM MPI_COMM_WORLD
	#define MATMUL_MPI_ROOT 0
#endif

//-----------------------------------------------------------------------------
// Data type depending definitions
//-----------------------------------------------------------------------------
#ifdef MATMUL_ELEMENT_TYPE_DOUBLE
	typedef double TElement;
	#define MATMUL_EPSILON DBL_EPSILON	//!< This is used to calculate wheter a result value is within a matrix size dependant error range.
	#ifdef MATMUL_MPI
		#define MATMUL_MPI_ELEMENT_TYPE MPI_DOUBLE
	#endif
#else
	typedef float TElement;
	#define MATMUL_EPSILON FLT_EPSILON	//!< This is used to calculate wheter a result value is within a matrix size dependant error range.
	#ifdef MATMUL_MPI
		#define MATMUL_MPI_ELEMENT_TYPE MPI_FLOAT
	#endif
#endif

//-----------------------------------------------------------------------------
// Compiler Settings. Do not change anything in here.
//-----------------------------------------------------------------------------
#if defined __INTEL_COMPILER				// ICC additiionally defines _MSC_VER if used in VS so this has to come first
	#define MATMUL_ICC
	#define OPEN_MP_3
	#ifdef _MSC_VER
		#define PRINTF_SIZE_T "Iu"
	#else
		#define PRINTF_SIZE_T "zu"
	#endif

#elif defined __clang__
	#define MATMUL_CLANG
	//#define OPEN_MP_3						// Clang 3.4 does not support OpenMP. The OpenMP support will be integrated in a future version.
	#ifdef _MSC_VER
		#define PRINTF_SIZE_T "Iu"
	#else
		#define PRINTF_SIZE_T "zu"
	#endif

#elif (defined _MSC_VER) && (_MSC_VER<=1800)
	#define MATMUL_MSVC
	#define restrict __restrict 			// Visual C++ 2013 and below do not define C99 restrict keyword under its supposed name. (And its not fully standard conformant)
	#define OPEN_MP_2						// Visual C++ 2013 and below only support OpenMP up to version 2.5.
	#define PRINTF_SIZE_T "Iu"				// Visual C++ 2013 and below do not support C99 printf specifiers.

#elif defined __CUDACC__
	#define restrict __restrict__

#elif defined __PGI
	#define OPEN_MP_3
	#define PRINTF_SIZE_T "zu"

#else
	#define OPEN_MP_3						// TODO: There might be other restrictions on other compilers
	#define PRINTF_SIZE_T "zu"
#endif
