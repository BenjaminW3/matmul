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

#include "matmul_par_openacc.h"

#ifdef MATMUL_BUILD_PAR_OPENACC

	#include <openacc.h>

	//-----------------------------------------------------------------------------
	//
	//-----------------------------------------------------------------------------
	void matmul_par_openacc_kernels(
		size_t const n,
		TElement const * const restrict A,
		TElement const * const restrict B,
		TElement * const restrict C)
	{
#pragma acc kernels copyin(A[0:(n*n)], B[0:(n*n)]) copy(C[0:(n*n)])
		{
#pragma acc loop independent gang(MATMUL_OPENACC_GANG_SIZE)
			for (size_t i = 0; i < n; ++i)
			{
#pragma acc loop independent vector(MATMUL_OPENACC_VECTOR_SIZE)
				for (size_t j = 0; j < n; ++j)
				{
					size_t const in = i*n;
					TElement ctmp = 0;
#pragma acc loop seq//reduction(+:ctmp) // Reduction here is much slower then sequential execution!
					for (size_t k = 0; k < n; ++k)
					{
						ctmp += A[in + k] * B[k*n +j];
					}
					C[in + j] += ctmp;
				}
			}
		}
	}

	//-----------------------------------------------------------------------------
	//
	//-----------------------------------------------------------------------------
	void matmul_par_openacc_parallel(
		size_t const n,
		TElement const * const restrict A,
		TElement const * const restrict B,
		TElement * const restrict C)
	{
#pragma acc parallel copyin(A[0:(n*n)], B[0:(n*n)]) copy(C[0:(n*n)])
		{
#pragma acc loop
			for(size_t i = 0; i < n; ++i)
			{
#pragma acc loop
				for(size_t j = 0; j < n; ++j)
				{
#pragma acc loop seq // Reduction here is much slower then sequential execution!
					for(size_t k = 0; k < n; ++k)
					{
						C[i*n + j] += A[i*n + k] * B[k*n + j];
					}
				}
			}
		}
	}

#endif
