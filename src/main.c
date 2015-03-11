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

#include "config.h"

#ifdef MATMUL_BUILD_SEQ
	#include "matmul_seq.h"
#endif
#ifdef MATMUL_BUILD_SEQ_BASIC_OPT
	#include "matmul_seq_basic_opt.h"
#endif
#ifdef MATMUL_BUILD_SEQ_COMPL_OPT
	#include "matmul_seq_compl_opt.h"
#endif
#ifdef MATMUL_BUILD_SEQ_STRASSEN
	#include "matmul_seq_strassen.h"
#endif
#ifdef MATMUL_BUILD_PAR_OPENMP
	#include "matmul_par_openmp.h"
#endif
#ifdef MATMUL_BUILD_PAR_STRASSEN_OMP
	#include "matmul_par_strassen_omp.h"
#endif
#ifdef MATMUL_BUILD_PAR_OPENACC
	#include "matmul_par_openacc.h"
#endif
#ifdef MATMUL_BUILD_PAR_CUDA
	#include "matmul_par_cuda.cuh"
#endif
#ifdef MATMUL_BUILD_PAR_MPI_CANNON
	#include "matmul_par_mpi_cannon.h"
#endif
#ifdef MATMUL_BUILD_PAR_MPI_DNS
	#include "matmul_par_mpi_dns.h"
#endif
#ifdef MATMUL_BUILD_PAR_BLAS_MKL
	#include "matmul_par_blas_mkl.h"
#endif
#ifdef MATMUL_BUILD_PAR_BLAS_CUBLAS
	#include "matmul_par_blas_cublas.cuh"
#endif
#ifdef MATMUL_BUILD_PAR_PHI_OFF_OPENMP
	#include "matmul_par_phi_off_openmp.h"
#endif
#ifdef MATMUL_BUILD_PAR_PHI_OFF_BLAS_MKL
	#include "matmul_par_phi_off_blas_mkl.h"
#endif

#include "array.h"
#include "malloc.h"
#include "mat_common.h"	// mat_cmp, mat_print

#include <stdio.h>		// printf
#include <assert.h>		// assert
#include <stdlib.h>		// malloc, free
#include <stdbool.h>	// bool, true, false
#include <time.h>		// time()
#include <float.h>		// DBL_MAX
#include <math.h>		// pow

#ifdef _MSC_VER
#include <Windows.h>
//-----------------------------------------------------------------------------
//! \return A monotonically increasing time value in seconds.
//-----------------------------------------------------------------------------
double getTimeSec()
{
	LARGE_INTEGER li, frequency;
	BOOL bSucQueryPerformanceCounter = QueryPerformanceCounter(&li);
	if(bSucQueryPerformanceCounter)
	{
		BOOL bSucQueryPerformanceFrequency = QueryPerformanceFrequency(&frequency);
		if(bSucQueryPerformanceFrequency)
		{
			return ((double)li.QuadPart)/((double)frequency.QuadPart);
		}
		else
		{
			// Throw assertion in debug mode, else return 0 time.
			assert(bSucQueryPerformanceFrequency);
			return 0.0;
		}
	}
	else
	{
		// Throw assertion in debug mode, else return 0 time.
		assert(bSucQueryPerformanceCounter);
		return 0.0;
	}
}
#else
#include <sys/time.h>
//-----------------------------------------------------------------------------
//! \return A monotonically increasing time value in seconds.
//-----------------------------------------------------------------------------
double getTimeSec()
{
	struct timeval act_time;
	gettimeofday(&act_time, NULL);
	return (double)act_time.tv_sec + (double)act_time.tv_usec / 1000000.0;
}
#endif


#ifndef MATMUL_MPI
//-----------------------------------------------------------------------------
//! \return The time in milliseconds needed to multiply 2 random matrices of the given type and size
//! \param n The matrix dimension.
//-----------------------------------------------------------------------------
void measureRandomMatMul(
	void(*pMatMul)(size_t, TElement const * const, TElement const * const, TElement * const),
	size_t const n,
	size_t const uiRepeatCount,
	bool const bPrintFLOPS,
	double const fExponentOmega,
	bool const bPrintTime,
	bool const bPrintMatrices)
{
	// Generate the matrices of the given size. 
	size_t const uiNumElements = n * n;
	TElement const * const A = mat_alloc_rand_fill(uiNumElements);
	TElement const * const B = mat_alloc_rand_fill(uiNumElements);
	// Because we calculate C *= A*B we need to initialize C. Even if we would not need this, we would have to initialize the C array with data before using it because else we would measure page table time on first write.
	TElement * const C = mat_alloc(uiNumElements);
#ifdef MATMUL_VERIFY_RESULT
	TElement * const D = mat_alloc(uiNumElements);
#endif

#ifdef MATMUL_REPEAT_TAKE_MINIMUM
	double fTimeMeasuredSec = DBL_MAX;
#else
	double fTimeMeasuredSec = 0.0;
#endif

	// Print the operation
	printf("%5"PRINTF_SIZE_T" * %5"PRINTF_SIZE_T" ", n, n);

	for(size_t i = 0; i < uiRepeatCount; ++i)
	{
		// We have to fill C with new data in subsequent iterations because else the values in C would get bigger and bigger in each iteration.
		mat_rand_fill(C, uiNumElements);
#ifdef MATMUL_VERIFY_RESULT
		mat_copy(C, D, n);
#endif

		// If there are multiple repetitions, print the iteration we are at now.
		if(uiRepeatCount!=1)
		{
			printf("%2"PRINTF_SIZE_T" ", i);
		}

		double const fTimeStart = getTimeSec();

		// Matrix multiplication.
		pMatMul(n, A, B, C);

		double const fTimeEnd = getTimeSec();
		double const fTimeElapsed = fTimeEnd - fTimeStart;

		if(bPrintMatrices)
		{
			mat_print(n, A);
			printf("*\n");
			mat_print(n, B);
			printf("=\n");
			mat_print(n, C);
			printf("\n");
		}

#ifdef MATMUL_VERIFY_RESULT
		matmul_seq(n, A, B, D);
		mat_cmp(n, C, D);
#endif

#ifdef MATMUL_REPEAT_TAKE_MINIMUM
		fTimeMeasuredSec = fTimeElapsed<fTimeMeasuredSec ? fTimeElapsed : fTimeMeasuredSec;
#else
		fTimeMeasuredSec += fTimeElapsed * (1.0/double(MATMUL_REPEAT_COUNT));
#endif

	}

	// Print the time needed for the calculation.
	if(bPrintTime)
	{
		printf("%12.8lf s ", fTimeMeasuredSec);
	}

	// Print the GFLOPS.
	if(bPrintFLOPS)
	{
		double const fOperations = 2.0*pow((double)n, fExponentOmega);
		double const fFLOPS = (fTimeMeasuredSec!=0) ? (fOperations/fTimeMeasuredSec) : 0.0;
		printf("%12.8lf GFLOPS ", fFLOPS*1.0e-9);
	}

	printf("\n");

	mat_free((TElement * const)A);
	mat_free((TElement * const)B);
	mat_free(C);
#ifdef MATMUL_VERIFY_RESULT
	mat_free(D);
#endif
}

#else

//-----------------------------------------------------------------------------
//! \return The time in milliseconds needed to multiply 2 random matrices of the given type and size
//! \param n The matrix dimension.
//-----------------------------------------------------------------------------
void measureRandomMatMul(
	void(*pMatMul)(size_t, TElement const * const, TElement const * const, TElement * const),
	size_t const n,
	size_t const uiRepeatCount,
	bool const bPrintFLOPS,
	double const fExponentOmega,
	bool const bPrintTime,
	bool const bPrintMatrices)
{
	// Generate the matrices of the given size.
	size_t const uiNumElements = n * n;
	TElement const * /*const*/ A = 0;
	TElement const * /*const*/ B = 0;
	TElement * /*const*/ C = 0;
#ifdef MATMUL_VERIFY_RESULT
	TElement * /*const*/ D = 0;
#endif

	double /*const*/ fTimeMeasuredSec;

	int iRank1D;
	MPI_Comm_rank(MATMUL_MPI_COMM, &iRank1D);
	if(iRank1D==MATMUL_MPI_ROOT)
	{
		A = mat_alloc_rand_fill(uiNumElements);
		B = mat_alloc_rand_fill(uiNumElements);
		C = mat_alloc(uiNumElements);
#ifdef MATMUL_VERIFY_RESULT
		D = mat_alloc(uiNumElements);
#endif

#ifdef MATMUL_REPEAT_TAKE_MINIMUM
		fTimeMeasuredSec = DBL_MAX;
#else
		fTimeMeasuredSec = 0.0;
#endif

		// Print the operation
		printf("%5"PRINTF_SIZE_T" * %5"PRINTF_SIZE_T" ", n, n);
	}

	for(size_t i = 0; i < uiRepeatCount; ++i)
	{
		if(iRank1D==MATMUL_MPI_ROOT)
		{
			// We have to fill C with new data in subsequent iterations because else the values in C would get bigger and bigger in each iteration.
			mat_rand_fill(C, uiNumElements);
#ifdef MATMUL_VERIFY_RESULT
			mat_copy(C, D, n);
#endif
		}

		double fTimeStart = 0;

		// Only the root process does the printing.
		if(iRank1D==MATMUL_MPI_ROOT)
		{
			// If there are multiple repetitions, print the iteration we are at now.
			if(uiRepeatCount!=1)
			{
				printf("%2"PRINTF_SIZE_T" ", i);
			}

			fTimeStart = getTimeSec();
		}

		// Matrix multiplication.
		pMatMul(n, A, B, C);

		// Only the root process does the printing.
		if(iRank1D==MATMUL_MPI_ROOT)
		{
			double const fTimeEnd = getTimeSec();
			double const fTimeElapsed = fTimeEnd - fTimeStart;

			if(bPrintMatrices)
			{
				mat_print(n, A);
				printf("*\n");
				mat_print(n, B);
				printf("=\n");
				mat_print(n, C);
				printf("\n");
			}

#ifdef MATMUL_VERIFY_RESULT
			matmul_seq(n, A, B, D);
			mat_cmp(n, C, D);
#endif

#ifdef MATMUL_REPEAT_TAKE_MINIMUM
			fTimeMeasuredSec = fTimeElapsed<fTimeMeasuredSec ? fTimeElapsed : fTimeMeasuredSec;
#else
			fTimeMeasuredSec += fTimeElapsed * (1.0/double(MATMUL_REPEAT_COUNT));
#endif
		}
	}

	// Only the root process does the printing.
	if(iRank1D==MATMUL_MPI_ROOT)
	{
		// Print the time needed for the calculation.
		if(bPrintTime)
		{
			printf("%12.8lf s ", fTimeMeasuredSec);
		}

		// Print the GFLOPS.
		if(bPrintFLOPS)
		{
			double const fOperations = 2.0*pow((double)n, fExponentOmega);
			double const fFLOPS = (fTimeMeasuredSec!=0) ? (fOperations/fTimeMeasuredSec) : 0.0;
			printf("%12.8lf GFLOPS ", fFLOPS*1.0e-9);
		}

		printf("\n");

		mat_free((TElement * const)A);
		mat_free((TElement * const)B);
		mat_free(C);
#ifdef MATMUL_VERIFY_RESULT
		mat_free(D);
#endif
	}
}

#endif

//-----------------------------------------------------------------------------
//! A struct containing an array of all matrix sizes to test.
//-----------------------------------------------------------------------------
typedef struct SMatMulSizes
{
	size_t uiNumSizes;
	size_t * puiSizes;
} SMatMulSizes;

//-----------------------------------------------------------------------------
//! Fills the matrix sizes struct.
//! \param uiNMin The start matrix dimension.
//! \param uiStepWidth The step width for each iteration. If set to 0 the size is doubled on each iteration.
//! \param uiNMax The macimum matrix dimension.
//-----------------------------------------------------------------------------
SMatMulSizes buildSizes(
	size_t const uiNMin,
	size_t const uiStepWidth,
	size_t const uiNMax)
{
	SMatMulSizes sizes;
	sizes.uiNumSizes = 0;
	sizes.puiSizes = 0;

	size_t uiN;
	for(uiN = uiNMin; uiN <= uiNMax; uiN += (uiStepWidth == 0) ? uiN : uiStepWidth)
	{
		++sizes.uiNumSizes;
	}

	sizes.puiSizes = (size_t *)malloc(sizes.uiNumSizes * sizeof(size_t));

	size_t uiIndex = 0;
	for(uiN = uiNMin; uiN <= uiNMax; uiN += (uiStepWidth == 0) ? uiN : uiStepWidth)
	{
		sizes.puiSizes[uiIndex] = uiN;
		++uiIndex;
	}

	return sizes;
}

//-----------------------------------------------------------------------------
//! Class template with static member templates because function templates do not allow partial specialisation.
//! \param uiNMin The start matrix dimension.
//! \param uiStepWidth The step width for each iteration. If set to 0 the size is doubled on each iteration.
//! \param uiNMax The macimum matrix dimension.
//-----------------------------------------------------------------------------
void measureRandomMatMuls(
	void(*pMatMul)(size_t const, TElement const * const, TElement const * const, TElement * const),
	SMatMulSizes const * const pSizes,
	size_t const uiRepeatCount,
	bool const bPrintFLOPS,
	double const fExponentOmega,
	bool const bPrintTime,
	bool const bPrintMatrices)
{
	if(pSizes)
	{
		for(size_t uiIndex = 0; uiIndex < pSizes->uiNumSizes; ++uiIndex)
		{
			// Do the calculation measurement.
			measureRandomMatMul(pMatMul, pSizes->puiSizes[uiIndex], uiRepeatCount, bPrintFLOPS, fExponentOmega, bPrintTime, bPrintMatrices);
		}
	}
	else
	{
		printf("Pointer to structure of test sizes 'pSizes' is not allowed to be nullptr!\n");
	}
}
//-----------------------------------------------------------------------------
//! Prints some startup informations.
//-----------------------------------------------------------------------------
void main_print_startup()
{
	printf("Copyright (c) 2014-2015, Benjamin Worpitz\n\n");

#ifdef NDEBUG
	printf("matmul release build\n\n");
#else
	printf("matmul debug build\n\n");
#endif
}

//-----------------------------------------------------------------------------
//! Main method initiating the measurements of all algorithms selected in config.h.
//-----------------------------------------------------------------------------
int main(
#ifdef MATMUL_MPI
	int argc, 
	char ** argv
#endif
	)
{
	// Disable buffering of fprintf. Always print it immediately.
	setvbuf(stdout, 0, _IONBF, 0);

#ifdef MATMUL_MPI
	// Initialize MPI before calling any mpi methods.
	int mpiStatus = MPI_Init(&argc, &argv);
	int iRank1D;
	if(mpiStatus != MPI_SUCCESS)
	{
		printf("Unable to initialize MPI. MPI_Init failed.\n");
	}
	else
	{
		MPI_Comm_rank(MATMUL_MPI_COMM, &iRank1D);
	
		if(iRank1D==MATMUL_MPI_ROOT)
		{
			main_print_startup();
		}
	}
#else
	main_print_startup();
#endif

	double const fTimeStart = getTimeSec();

	srand((unsigned int)time(0));

	size_t const uiNMin = MATMUL_MIN_N;
	size_t const uiStepWidth = MATMUL_STEP_N;
	size_t const uiNMax = MATMUL_MAX_N;
	size_t const uiRepeatCount = MATMUL_REPEAT_COUNT;

	SMatMulSizes const standardSizes = buildSizes(uiNMin, uiStepWidth, uiNMax);

	// 1. Sequential implementations
#ifdef MATMUL_TEST_SEQ
	// 1.1 Basic sequential
	printf("matmul_seq:\n");
	measureRandomMatMuls(matmul_seq, &standardSizes, uiRepeatCount, true, 3.0f, true, false);
	printf("\n");
#endif

#ifdef MATMUL_TEST_SEQ_BASIC_OPT
	// 1.2 Sequential optimizations
	printf("matmul_seq_index_pointer:\n");
	measureRandomMatMuls(matmul_seq_index_pointer, &standardSizes, uiRepeatCount, true, 3.0f, true, false);
	printf("\n");

	printf("matmul_seq_restrict:\n");
	measureRandomMatMuls(matmul_seq_restrict, &standardSizes, uiRepeatCount, true, 3.0f, true, false);
	printf("\n");

	printf("matmul_seq_loop_reorder:\n");
	measureRandomMatMuls(matmul_seq_loop_reorder, &standardSizes, uiRepeatCount, true, 3.0f, true, false);
	printf("\n");

	printf("matmul_seq_index_precalculate:\n");
	measureRandomMatMuls(matmul_seq_index_precalculate, &standardSizes, uiRepeatCount, true, 3.0f, true, false);
	printf("\n");

	printf("matmul_seq_loop_unroll_4:\n");
	SMatMulSizes const unroll4Sizes = buildSizes(uiNMin<4 ? 4 : uiNMin, uiStepWidth, uiNMax);
	measureRandomMatMuls(matmul_seq_loop_unroll_4, &unroll4Sizes, uiRepeatCount, true, 3.0f, true, false);
	free(unroll4Sizes.puiSizes);
	printf("\n");

	printf("matmul_seq_loop_unroll_8:\n");
	SMatMulSizes const unroll8Sizes = buildSizes(uiNMin<8 ? 8 : uiNMin, uiStepWidth, uiNMax);
	measureRandomMatMuls(matmul_seq_loop_unroll_8, &unroll8Sizes, uiRepeatCount, true, 3.0f, true, false);
	free(unroll8Sizes.puiSizes);
	printf("\n");

	printf("matmul_seq_loop_unroll_16:\n");
	SMatMulSizes const unroll16Sizes = buildSizes(uiNMin<16 ? 16 : uiNMin, uiStepWidth, uiNMax);
	measureRandomMatMuls(matmul_seq_loop_unroll_16, &unroll16Sizes, uiRepeatCount, true, 3.0f, true, false);
	free(unroll16Sizes.puiSizes);
	printf("\n");

	printf("matmul_seq_block:\n");
	measureRandomMatMuls(matmul_seq_block, &standardSizes, uiRepeatCount, true, 3.0f, true, false);
	printf("\n");
#endif

#ifdef MATMUL_TEST_SEQ_COMPL_OPT
	printf("matmul_seq_complete_opt_block:\n");
	measureRandomMatMuls(matmul_seq_complete_opt_block, &standardSizes, uiRepeatCount, true, 3.0f, true, false);
	printf("\n");

	printf("matmul_seq_complete_opt_no_block:\n");
	measureRandomMatMuls(matmul_seq_complete_opt_no_block, &standardSizes, uiRepeatCount, true, 3.0f, true, false);
	printf("\n");

	printf("matmul_seq_complete_opt:\n");
	measureRandomMatMuls(matmul_seq_complete_opt, &standardSizes, uiRepeatCount, true, 3.0f, true, false);
	printf("\n");
#endif

#ifdef MATMUL_TEST_SEQ_STRASSEN
	// 1.2 Sequential Strassen.
	printf("matmul_seq_strassen:\n");
	measureRandomMatMuls(matmul_seq_strassen, &standardSizes, uiRepeatCount, true, log(7.0) / log(2.0), true, false);
	printf("\n");
#endif

#ifdef MATMUL_TEST_PAR_OPENMP
	// 2. Parallel implementations
	// 2.1. OpenMP
	printf("matmul_par_openmp_guided_schedule:\n");
	measureRandomMatMuls(matmul_par_openmp_guided_schedule, &standardSizes, uiRepeatCount, true, 3.0f, true, false);
	printf("\n");

	printf("matmul_par_openmp_static_schedule:\n");
	measureRandomMatMuls(matmul_par_openmp_static_schedule, &standardSizes, uiRepeatCount, true, 3.0f, true, false);
	printf("\n");

	#ifdef OPEN_MP_3
		printf("matmul_par_openmp_static_schedule_collapse:\n");
		measureRandomMatMuls(matmul_par_openmp_static_schedule_collapse, &standardSizes, uiRepeatCount, true, 3.0f, true, false);
		printf("\n");
	#endif
#endif

#ifdef MATMUL_TEST_PAR_PHI_OFF_OPENMP
	printf("matmul_par_phi_offload_openmp_guided_schedule:\n");
	measureRandomMatMuls(matmul_par_phi_offload_openmp_guided_schedule, &standardSizes, uiRepeatCount, true, 3.0f, true, false);
	printf("\n");

	printf("matmul_par_phi_offload_openmp_static_schedule:\n");
	measureRandomMatMuls(matmul_par_phi_offload_openmp_static_schedule, &standardSizes, uiRepeatCount, true, 3.0f, true, false);
	printf("\n");

	printf("matmul_par_phi_offload_openmp_static_schedule_collapse:\n");
	measureRandomMatMuls(matmul_par_phi_offload_openmp_static_schedule_collapse, &standardSizes, uiRepeatCount, true, 3.0f, true, false);
	printf("\n");
#endif

#ifdef MATMUL_TEST_PAR_STRASSEN_OMP
	// 2.2. Strassen OpenMP
	printf("matmul_par_strassen_omp:\n");
	measureRandomMatMuls(matmul_par_strassen_omp, &standardSizes, uiRepeatCount, true, log(7.0) / log(2.0), true, false);
	printf("\n");
#endif

#ifdef MATMUL_TEST_PAR_OPENACC
	// 2.3. OpenACC
	printf("matmul_par_openacc_kernels:\n");
	measureRandomMatMuls(matmul_par_openacc_kernels, &standardSizes, uiRepeatCount, true, 3.0f, true, false);
	printf("\n");

	printf("matmul_par_openacc_parallel:\n");
	measureRandomMatMuls(matmul_par_openacc_parallel, &standardSizes, uiRepeatCount, true, 3.0f, true, false);
	printf("\n");
#endif

#ifdef MATMUL_TEST_PAR_CUDA
	// 2.4. CUDA
	printf("matmul_par_cuda:\n");
	measureRandomMatMuls(matmul_par_cuda, &standardSizes, uiRepeatCount, true, 3.0f, true, false);
	printf("\n");
#endif

#ifdef MATMUL_MPI
	// Initialize MPI before calling any mpi methods.
	if(mpiStatus == MPI_SUCCESS)
	{
#ifdef MATMUL_TEST_PAR_MPI_CANNON_STD
		if(iRank1D==MATMUL_MPI_ROOT)
		{
			printf("matmul_par_mpi_cannon_block:\n");
		}
		
		//measureRandomMatMuls(matmul_par_mpi_cannon_block, &standardSizes, uiRepeatCount, true, 3.0f, true, false);
			
		if(iRank1D==MATMUL_MPI_ROOT)
		{
			printf("\n");

			printf("matmul_par_mpi_cannon_nonblock:\n");
		}
			
		measureRandomMatMuls(matmul_par_mpi_cannon_nonblock, &standardSizes, uiRepeatCount, true, 3.0f, true, false);
		
		if(iRank1D==MATMUL_MPI_ROOT)
		{
			printf("\n");
		}
#endif
#ifdef MATMUL_TEST_PAR_MPI_CANNON_MKL
		if(iRank1D==MATMUL_MPI_ROOT)
		{
			printf("matmul_par_mpi_cannon_nonblock_blas_mkl:\n");
		}
			
		measureRandomMatMuls(matmul_par_mpi_cannon_nonblock_blas_mkl, &standardSizes, uiRepeatCount, true, 3.0f, true, false);
		
		if(iRank1D==MATMUL_MPI_ROOT)
		{
			printf("\n");
		}
#endif
#ifdef MATMUL_BUILD_PAR_MPI_CANNON_CUBLAS
		if(iRank1D==MATMUL_MPI_ROOT)
		{
			printf("matmul_par_mpi_cannon_nonblock_blas_cublas:\n");
		}

		measureRandomMatMuls(matmul_par_mpi_cannon_nonblock_blas_cublas, &standardSizes, uiRepeatCount, true, 3.0f, true, false);

		if(iRank1D==MATMUL_MPI_ROOT)
		{
			printf("\n");
		}
#endif
#ifdef MATMUL_TEST_PAR_MPI_DNS
		if(iRank1D==MATMUL_MPI_ROOT)
		{
			printf("matmul_par_mpi_dns:\n");
		}
			
		measureRandomMatMuls(matmul_par_mpi_dns, &standardSizes, uiRepeatCount, true, 3.0f, true, false);
		
		if(iRank1D==MATMUL_MPI_ROOT)
		{
			printf("\n");
		}
#endif
	}
#endif

#ifdef MATMUL_TEST_PAR_BLAS_MKL
	printf("matmul_par_blas_mkl:\n");
	measureRandomMatMuls(matmul_par_blas_mkl, &standardSizes, uiRepeatCount, true, 3.0f, true, false);
	printf("\n");
#endif

#ifdef MATMUL_TEST_PAR_PHI_OFF_BLAS_MKL
	printf("matmul_par_phi_off_blas_mkl:\n");
	measureRandomMatMuls(matmul_par_phi_off_blas_mkl, &standardSizes, uiRepeatCount, true, 3.0f, true, false);
	printf("\n");
#endif

#ifdef MATMUL_TEST_PAR_BLAS_CUBLAS
	printf("matmul_par_blas_cublas2:\n");
	measureRandomMatMuls(matmul_par_blas_cublas2, &standardSizes, uiRepeatCount, true, 3.0f, true, false);
	printf("\n");
#endif

	free(standardSizes.puiSizes);
	
#ifdef MATMUL_MPI
	if(mpiStatus == MPI_SUCCESS)
	{
		if(iRank1D==MATMUL_MPI_ROOT)
		{
			double const fTimeEnd = getTimeSec();
			double const fTimeElapsed = fTimeEnd - fTimeStart;
			printf("Total runtime: %12.6lf s ", fTimeElapsed);
		}
		MPI_Finalize();
	}
#else
	double const fTimeEnd = getTimeSec();
	double const fTimeElapsed = fTimeEnd - fTimeStart;
	printf("Total runtime: %12.6lf s ", fTimeElapsed);
#endif

	return 0;
}
