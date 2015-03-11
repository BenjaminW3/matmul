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

#include "matmul_seq_compl_opt.h"

#include <stdio.h>		// printf

#ifdef MATMUL_BUILD_SEQ_COMPL_OPT
	//-----------------------------------------------------------------------------
	//
	//-----------------------------------------------------------------------------
	void matmul_pitch_seq_complete_opt_no_block(
		size_t const n,
		size_t const uiPitchA, TElement const * const restrict A,
		size_t const uiPitchB, TElement const * const restrict B,
		size_t const uiPitchC, TElement * const restrict C)
	{
#ifdef MATMUL_MSVC
		for(size_t i = 0; i<n; ++i)
		{
			size_t const uiRowBeginIndexC = i*uiPitchC;
			size_t const uiRowBeginIndexA = i*uiPitchA;
			for(size_t k = 0; k<n; ++k)
			{
				size_t const uiRowBeginIndexB = k*uiPitchB;
				TElement const a = A[uiRowBeginIndexA + k];
				for(size_t j = 0; j<n; ++j)
				{
					C[uiRowBeginIndexC + j] += a * B[uiRowBeginIndexB + j];
				}
				// Unrolling by hand often confuses compilers
				/*size_t j = 0;
				for(; j+7 < n; j += 8)
				{
				size_t const uiIndexB = uiRowBeginIndexB + j;
				size_t const uiIndexC = uiRowBeginIndexC + j;
				C[uiIndexC] += a * B[uiIndexB];
				C[uiIndexC+1] += a * B[uiIndexB+1];
				C[uiIndexC+2] += a * B[uiIndexB+2];
				C[uiIndexC+3] += a * B[uiIndexB+3];
				C[uiIndexC+4] += a * B[uiIndexB+4];
				C[uiIndexC+5] += a * B[uiIndexB+5];
				C[uiIndexC+6] += a * B[uiIndexB+6];
				C[uiIndexC+7] += a * B[uiIndexB+7];
				}
				for(; j < n; ++j)
				{
				C[uiRowBeginIndexC + j] += a * B[uiRowBeginIndexB + j];
				}*/
			}
		}
#else
		for(size_t i = 0; i < n; ++i)
		{
			for(size_t k = 0; k < n; ++k)
			{
				for(size_t j = 0; j < n; ++j)
				{
					C[i*uiPitchC + j] += A[i*uiPitchA + k] * B[k*uiPitchB + j];
				}
			}
		}
#endif
		// Version with pointers instead of array indices is slower for most compilers.
		/*TElement const * restrict pARow = A;
		TElement * restrict pCRow = C;
		for(size_t i = 0; i < n; ++i, pARow += uiPitchA, pCRow += uiPitchC)
		{
			TElement const * restrict pA = pARow;
			TElement const * restrict pBRow = B;
			for(size_t k = 0; k < n; ++k, ++pA, pBRow += uiPitchB)
			{
				TElement const * restrict pB = pBRow;
				TElement * restrict pC = pCRow;
				TElement const a = (*pA);
				for(size_t j = 0; j < n; ++j, ++pB, ++pC)
				{
					(*pC) += a * (*pB);
				}*/
				/*size_t j = 0;
				for(; j+7 < n; j += 8)
				{
					(*(pC+0)) += a * (*(pB+0));
					(*(pC+1)) += a * (*(pB+1));
					(*(pC+2)) += a * (*(pB+2));
					(*(pC+3)) += a * (*(pB+3));
					(*(pC+4)) += a * (*(pB+4));
					(*(pC+5)) += a * (*(pB+5));
					(*(pC+6)) += a * (*(pB+6));
					(*(pC+7)) += a * (*(pB+7));
				}
				for(; j < n; ++j)
				{
					(*pC) += a * (*pB);
				}*/
			/*}
		}*/
	}
	//-----------------------------------------------------------------------------
	//
	//-----------------------------------------------------------------------------
	void matmul_seq_complete_opt_no_block(
		size_t const n,
		TElement const * const restrict A,
		TElement const * const restrict B,
		TElement * const restrict C)
	{
		matmul_pitch_seq_complete_opt_no_block(
			n,
			n, A,
			n, B,
			n, C);
	}

	//-----------------------------------------------------------------------------
	//
	//-----------------------------------------------------------------------------
	void matmul_pitch_seq_complete_opt_block(
		size_t const n,
		size_t const uiPitchA, TElement const * const restrict A,
		size_t const uiPitchB, TElement const * const restrict B,
		size_t const uiPitchC, TElement * const restrict C)
	{
		size_t const S = MATMUL_SEQ_BLOCK_FACTOR;

		// Version with pointers instead of array indices is slower for most compilers.
		/*size_t const SPitchA = S*uiPitchA;
		size_t const SPitchB = S*uiPitchB;
		size_t const SPitchC = S*uiPitchC;

		TElement const * restrict pARowBlock = A;
		TElement * restrict pCRowBlock = C;
		for(size_t ii = 0; ii<n; ii += S, pARowBlock += SPitchA, pCRowBlock += SPitchC)
		{
			size_t const iiS = ii+S;
			size_t const uiUpperBoundi = (iiS>n ? n : iiS);

			TElement const * restrict pABlock = pARowBlock;
			TElement const * restrict pBRowBlock = B;
			for(size_t kk = 0; kk<n; kk += S, pABlock += S, pBRowBlock += SPitchB)
			{
				size_t const kkS = kk+S;
				size_t const uiUpperBoundk = (kkS>n ? n : kkS);

				TElement const * restrict pBBlock = pBRowBlock;
				TElement * restrict pCBlock = pCRowBlock;
				for(size_t jj = 0; jj<n; jj += S, pBBlock += S, pCBlock += S)
				{
					size_t const jjS = jj+S;
					size_t const uiUpperBoundj = (jjS>n ? n : jjS);

					TElement const * restrict pARow = pABlock;
					TElement * restrict pCRow = pCBlock;
					for(size_t i = ii; i < uiUpperBoundi; ++i, pARow += uiPitchA, pCRow += uiPitchC)
					{
						TElement const * restrict pA = pARow;
						TElement const * restrict pBRow = pBBlock;
						for(size_t k = kk; k < uiUpperBoundk; ++k, ++pA, pBRow += uiPitchB)
						{
							TElement const * restrict pB = pBRow;
							TElement * restrict pC = pCRow;
							TElement const a = (*pA);
							for(size_t j = jj; j < uiUpperBoundj; ++j, ++pB, ++pC)
							{
								(*pC) += a * (*pB);
							}
						}
					}
				}
			}
		}*/

		//for(size_t ii = 0; ii<n; ii += S)	// Blocking of outermost loop is not necessary, we only need blocks in 2 dimensions.
		{
			//size_t const iiS = ii+S;
			for(size_t kk = 0; kk<n; kk += S)
			{
				size_t const kkS = kk+S;
				for(size_t jj = 0; jj<n; jj += S)
				{
					size_t const jjS = jj+S;
					//size_t const uiUpperBoundi = (iiS>n ? n : iiS);
					//for(size_t i = ii; i < uiUpperBoundi; ++i)

					size_t uiRowBeginIndexC = 0;
					size_t uiRowBeginIndexA = 0;
					for(size_t i = 0; i<n; ++i)
					{
						size_t uiRowBeginIndexB = kk*uiPitchB;
						size_t const uiUpperBoundk = (kkS>n ? n : kkS);
						for(size_t k = kk; k<uiUpperBoundk; ++k)
						{
							TElement const a = A[uiRowBeginIndexA + k];
							size_t const uiUpperBoundj = (jjS>n ? n : jjS);
							for(size_t j = jj; j<uiUpperBoundj; ++j)
							{
								C[uiRowBeginIndexC + j] += a * B[uiRowBeginIndexB + j];
							}
							uiRowBeginIndexB += uiPitchB;
						}
						uiRowBeginIndexC += uiPitchC;
						uiRowBeginIndexA += uiPitchA;
					}
				}
			}
		}
	}
	//-----------------------------------------------------------------------------
	//
	//-----------------------------------------------------------------------------
	void matmul_seq_complete_opt_block(
		size_t const n,
		TElement const * const restrict A,
		TElement const * const restrict B,
		TElement * const restrict C)
	{
		matmul_pitch_seq_complete_opt_block(
			n,
			n, A,
			n, B,
			n, C);
	}

	//-----------------------------------------------------------------------------
	//
	//-----------------------------------------------------------------------------
	void matmul_pitch_seq_complete_opt(
		size_t const n,
		size_t const uiPitchA, TElement const * const restrict A,
		size_t const uiPitchB, TElement const * const restrict B,
		size_t const uiPitchC, TElement * const restrict C)
	{
#ifdef MATMUL_MSVC
		// MSVC-2013 is better with a handrolled transition between blocked and nonblocked version.
		if(n<=MATMUL_SEQ_COMPLETE_OPT_NO_BLOCK_CUT_OFF)
		{
			matmul_pitch_seq_complete_opt_no_block(
				n,
				uiPitchA, A,
				uiPitchB, B,
				uiPitchC, C);
		}
		else
		{
			matmul_pitch_seq_complete_opt_block(
				n,
				uiPitchA, A,
				uiPitchB, B,
				uiPitchC, C);
		}
#else
		// ICC-14 compiler automatically optimizes the matmul function better then we could reach with a handrolled one (blocked and nonblocked).
		matmul_pitch_seq_complete_opt_no_block(
			n,
			uiPitchA, A,
			uiPitchB, B,
			uiPitchC, C);
#endif
	}
	//-----------------------------------------------------------------------------
	//
	//-----------------------------------------------------------------------------
	void matmul_seq_complete_opt(
		size_t const n,
		TElement const * const restrict A,
		TElement const * const restrict B,
		TElement * const restrict C)
	{
		matmul_pitch_seq_complete_opt(
			n,
			n, A,
			n, B,
			n, C);
	}
#endif
