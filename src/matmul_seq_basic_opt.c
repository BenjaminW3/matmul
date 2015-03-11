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

#include "matmul_seq_basic_opt.h"

#include <stdio.h>		// printf

#ifdef MATMUL_BUILD_SEQ_BASIC_OPT
	//-----------------------------------------------------------------------------
	// Use explicit pointer access instead of index access that requires multiplication.
	// This prohibits vectorization by the compiler because the pointers are not marked with restrict.
	//-----------------------------------------------------------------------------
	void matmul_seq_index_pointer(
		size_t const n,
		TElement const * const A,
		TElement const * const B,
		TElement * const C)
	{
		TElement * pC = C;
		TElement const * pARow = A;
		for(size_t i = 0; i < n; ++i, pARow += n)
		{
			TElement const * pBCol = B;
			for(size_t j = 0; j < n; ++j, ++pC, ++pBCol)
			{
				TElement const * pA = pARow;
				TElement const * pB = pBCol;
				for(size_t k = 0; k < n; ++k, ++pA, pB += n)
				{
					(*pC) += (*pA) * (*pB);
				}
			}
		}
	}
	//-----------------------------------------------------------------------------
	// This allows vectorization by the compiler because the pointers are marked with restrict.
	//-----------------------------------------------------------------------------
	/*void matmul_seq_index_pointer_restrict(
		size_t const n,
		TElement const * const restrict A,
		TElement const * const restrict B,
		TElement * const restrict C)
	{
		// Use explicit pointer access instead of index access that requires multiplication.
		TElement * restrict pC = C;
		TElement const * restrict pARow = A;
		for(size_t i = 0; i < n; ++i, pARow += n)
		{
			TElement const * restrict pBCol = B;
			for(size_t j = 0; j < n; ++j, ++pC, ++pBCol)
			{
				TElement const * restrict pA = pARow;
				TElement const * restrict pB = pBCol;
				for(size_t k = 0; k < n; ++k, ++pA, pB += n)
				{
					(*pC) += (*pA) * (*pB);
				}
			}
		}
	}*/

	//-----------------------------------------------------------------------------
	//
	//-----------------------------------------------------------------------------
	void matmul_seq_restrict(
		size_t const n,
		TElement const * const restrict A,
		TElement const * const restrict B,
		TElement * const restrict C)
	{
		for(size_t i = 0; i < n; ++i)
		{
			for(size_t j = 0; j < n; ++j)
			{
				for(size_t k = 0; k < n; ++k)
				{
					C[i*n + j] += A[i*n + k] * B[k*n + j];
				}
			}
		}
	}

	//-----------------------------------------------------------------------------
	//
	//-----------------------------------------------------------------------------
	void matmul_seq_loop_reorder(
		size_t const n,
		TElement const * const A,
		TElement const * const B,
		TElement * const C)
	{
		for(size_t i = 0; i < n; ++i)
		{
			for(size_t k = 0; k < n; ++k)
			{
				for(size_t j = 0; j < n; ++j)
				{
					// Cache efficiency inside the innermost loop:
					// In the original loop order C[i*n + j] is in the cache and A[i*n + k + 1] is likely to be in the cache. There would be a strided access to B so every access is likely a cache miss.
					// In the new order A[i*n + k] is always in the cache and C[i*n + j + 1] as well as B[k*n + j + 1] are likely to be in the cache. 
					// => This is one cache miss less per iteration.
					C[i*n + j] += A[i*n + k] * B[k*n + j];
				}
			}
		}
	}

	//-----------------------------------------------------------------------------
	//
	//-----------------------------------------------------------------------------
	void matmul_seq_index_precalculate(
		size_t const n,
		TElement const * const A,
		TElement const * const B,
		TElement * const C)
	{
		for(size_t i = 0; i < n; ++i)
		{
			size_t const uiRowBeginIndex = i*n;

			for(size_t j = 0; j < n; ++j)
			{
				TElement fSum = 0;
				for(size_t k = 0; k < n; ++k)
				{
					fSum += A[uiRowBeginIndex + k] * B[k*n + j];
				}

				C[uiRowBeginIndex + j] += fSum;
			}
		}
	}
	//-----------------------------------------------------------------------------
	//
	//-----------------------------------------------------------------------------
	void matmul_seq_loop_unroll_4(
		size_t const n,
		TElement const * const A,
		TElement const * const B,
		TElement * const C)
	{
		for(size_t i = 0; i < n; ++i)
		{
			for(size_t j = 0; j < n; ++j)
			{
				size_t k;
				size_t const uiIN = i*n;
				for(k = 0; k+3 < n; k += 4)
				{
					C[uiIN + j]
						+= A[uiIN + k] * B[k*n + j]
						+ A[uiIN + k+1] * B[(k+1)*n + j]
						+ A[uiIN + k+2] * B[(k+2)*n + j]
						+ A[uiIN + k+3] * B[(k+3)*n + j];
				}
				for(; k < n; ++k)
				{
					C[uiIN + j] += A[uiIN + k] * B[k*n + j];
				}
			}
		}
	}

	//-----------------------------------------------------------------------------
	//
	//-----------------------------------------------------------------------------
	void matmul_seq_loop_unroll_8(
		size_t const n,
		TElement const * const A,
		TElement const * const B,
		TElement * const C)
	{
		for(size_t i = 0; i < n; ++i)
		{
			for(size_t j = 0; j < n; ++j)
			{
				size_t k;
				size_t const uiIN = i*n;
				for(k = 0; k+7 < n; k += 8)
				{
					C[uiIN + j]
						+= A[uiIN + k] * B[k*n + j]
						+ A[uiIN + k+1] * B[(k+1)*n + j]
						+ A[uiIN + k+2] * B[(k+2)*n + j]
						+ A[uiIN + k+3] * B[(k+3)*n + j]
						+ A[uiIN + k+4] * B[(k+4)*n + j]
						+ A[uiIN + k+5] * B[(k+5)*n + j]
						+ A[uiIN + k+6] * B[(k+6)*n + j]
						+ A[uiIN + k+7] * B[(k+7)*n + j];
				}
				for(; k < n; ++k)
				{
					C[uiIN + j] += A[uiIN + k] * B[k*n + j];
				}
			}
		}
	}

	//-----------------------------------------------------------------------------
	//
	//-----------------------------------------------------------------------------
	void matmul_seq_loop_unroll_16(
		size_t const n,
		TElement const * const A,
		TElement const * const B,
		TElement * const C)
	{
		for(size_t i = 0; i < n; ++i)
		{
			for(size_t j = 0; j < n; ++j)
			{
				size_t k = 0;
				size_t const uiIN = i*n;
				for(; k+15 < n; k += 16)
				{
					C[i*n + j]
						+= A[uiIN + k] * B[k*n + j]
						+ A[uiIN + k+1] * B[(k+1)*n + j]
						+ A[uiIN + k+2] * B[(k+2)*n + j]
						+ A[uiIN + k+3] * B[(k+3)*n + j]
						+ A[uiIN + k+4] * B[(k+4)*n + j]
						+ A[uiIN + k+5] * B[(k+5)*n + j]
						+ A[uiIN + k+6] * B[(k+6)*n + j]
						+ A[uiIN + k+7] * B[(k+7)*n + j]
						+ A[uiIN + k+8] * B[(k+8)*n + j]
						+ A[uiIN + k+9] * B[(k+9)*n + j]
						+ A[uiIN + k+10] * B[(k+10)*n + j]
						+ A[uiIN + k+11] * B[(k+11)*n + j]
						+ A[uiIN + k+12] * B[(k+12)*n + j]
						+ A[uiIN + k+13] * B[(k+13)*n + j]
						+ A[uiIN + k+14] * B[(k+14)*n + j]
						+ A[uiIN + k+15] * B[(k+15)*n + j];
				}
				for(; k < n; ++k)
				{
					C[uiIN + j] += A[uiIN + k] * B[k*n + j];
				}
			}
		}
	}

	//-----------------------------------------------------------------------------
	//
	//-----------------------------------------------------------------------------
	void matmul_seq_block(
		size_t const n,
		TElement const * const A,
		TElement const * const B,
		TElement * const C)
	{
		size_t const S = MATMUL_SEQ_BLOCK_FACTOR;

		for(size_t ii = 0; ii<n; ii += S)
		{
			size_t const iiS = ii+S;
			for(size_t jj = 0; jj<n; jj += S)
			{
				size_t const jjS = jj+S;
				for(size_t kk = 0; kk<n; kk += S)
				{
					size_t const kkS = kk+S;
					size_t const uiUpperBoundi = (iiS>n ? n : iiS);
					for(size_t i = ii; i<uiUpperBoundi; ++i)
					{
						size_t const uiUpperBoundj = (jjS>n ? n : jjS);
						for(size_t j = jj; j<uiUpperBoundj; ++j)
						{
							size_t const uiUpperBoundk = (kkS>n ? n : kkS);
							for(size_t k = kk; k<uiUpperBoundk; ++k)
							{
								C[i*n + j] += A[i*n + k] * B[k*n + j];
							}
						}
					}
				}
			}
		}
	}
#endif
