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

#include "matmul_par_xeonphi_openmp.h"

#ifdef MATMUL_BUILD_XEONPHI_OPENMP

	#include <omp.h>

	#include <stdio.h>		// printf

	#ifndef OPEN_MP_3
		typedef int TOpenMPForLoopIndex;
	#else
		typedef size_t TOpenMPForLoopIndex;
	#endif
	//-----------------------------------------------------------------------------
	// http://software.intel.com/sites/products/documentation/doclib/mkl_sa/11/mkl_userguide_lnx/GUID-8B7FF103-0319-4D33-B36F-503917E847B4.htm
	// http://www.cism.ucl.ac.be/Services/Formations/Accelerators.pdf !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	//-----------------------------------------------------------------------------
	void matmul_xeon_phi_offload_openmp_guided_schedule(
		size_t const n,
		TElement const * const restrict A,
		TElement const * const restrict B,
		TElement * const restrict C)
	{
		TOpenMPForLoopIndex const iN = (int)n;

#pragma offload target(mic) in(A:length(n*n)) in(B:length(n*n)) inout(C:length(n*n))
#pragma omp parallel//shared(A,B,C,iN)
		{
#ifdef MATMUL_OPENMP_PRINT_NUM_CORES
#pragma omp single
			{
				printf("p=%d threads ", omp_get_num_threads());
			}
#endif

			TOpenMPForLoopIndex i;	// For OpenMP < 3.0 you have to declare the loop index outside of the loop header.
#pragma omp for schedule(guided)
			for(i = 0; i < iN; ++i)
			{
				size_t const uiRowBeginIndexAC = i*n;
				for(size_t k = 0; k < n; ++k)
				{
					size_t const uiRowBeginIndexB = k*n;
					TElement const a = A[uiRowBeginIndexAC + k];
					for(size_t j = 0; j < n; ++j)
					{
						C[uiRowBeginIndexAC + j] += a * B[uiRowBeginIndexB + j];
					}
				}
			}
		}
	}

	//-----------------------------------------------------------------------------
	//
	//-----------------------------------------------------------------------------
	void matmul_xeon_phi_offload_openmp_static_schedule(
		size_t const n,
		TElement const * const restrict A,
		TElement const * const restrict B,
		TElement * const restrict C)
	{
		TOpenMPForLoopIndex const iN = (int)n;

#pragma offload target(mic) in(A:length(n*n)) in(B:length(n*n)) inout(C:length(n*n))
#pragma omp parallel //shared(A,B,C,iN)
		{
#ifdef MATMUL_OPENMP_PRINT_NUM_CORES
#pragma omp single
			{
				printf("p=%d threads ", omp_get_num_threads());
			}
#endif

			TOpenMPForLoopIndex i;	// For OpenMP < 3.0 you have to declare the loop index outside of the loop header.
#pragma omp for schedule(static)
			for(i = 0; i < iN; ++i)
			{
				size_t const uiRowBeginIndexAC = i*n;
				for(size_t k = 0; k < n; ++k)
				{
					size_t const uiRowBeginIndexB = k*n;
					TElement const a = A[uiRowBeginIndexAC + k];
					for(size_t j = 0; j < n; ++j)
					{
						C[uiRowBeginIndexAC + j] += a * B[uiRowBeginIndexB + j];
					}
				}
			}
		}
	}

	#ifdef OPEN_MP_3
		//-----------------------------------------------------------------------------
		//
		//-----------------------------------------------------------------------------
		void matmul_xeon_phi_offload_openmp_static_schedule_collapse(
			size_t const n,
			TElement const * const restrict A,
			TElement const * const restrict B,
			TElement * const restrict C)
		{
	#pragma offload target(mic) in(A:length(n*n)) in(B:length(n*n)) inout(C:length(n*n))
	#pragma omp parallel //shared(A,B,C,iN)
			{
#ifdef MATMUL_OPENMP_PRINT_NUM_CORES
		#pragma omp single
				{
					printf("p=%d threads ", omp_get_num_threads());
				}
#endif

	#pragma omp for collapse(2) schedule(static)	// http://software.intel.com/en-us/articles/openmp-loop-collapse-directive
				for(i = 0; i < iN; ++i)
				{
					for(size_t k = 0; k < n; ++k)
					{
						size_t const uiRowBeginIndexAC = i*n;
						size_t const uiRowBeginIndexB = k*n;
						TElement const a = A[uiRowBeginIndexAC + k];
						for(size_t j = 0; j < n; ++j)
						{
							C[uiRowBeginIndexAC + j] += a * B[uiRowBeginIndexB + j];
						}
					}
				}
			}
		}
	#endif

#endif
