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

#include "matmul_seq_strassen.h"

#ifdef MATMUL_BUILD_SEQ_STRASSEN

	#include "matmul_seq_compl_opt.h"	// matmul_pitch_seq_complete_opt

	#include "malloc.h"		// mat_alloc
	#include "array.h"		// mat_alloc_zero_fill

	#include <assert.h>		// assert

	//-----------------------------------------------------------------------------
	//! Adapted from http://ezekiel.vancouver.wsu.edu/~cs330/lectures/linear_algebra/mm/mm.c W. Cochran  wcochran@vancouver.wsu.edu
	//-----------------------------------------------------------------------------

	//-----------------------------------------------------------------------------
	//
	//-----------------------------------------------------------------------------
	void matadd_pitch_seq(
		size_t const n,
		size_t const uiPitchA, TElement const * const restrict A,
		size_t const uiPitchB, TElement const * const restrict B,
		size_t const uiPitchC, TElement * const restrict C)
	{
#ifdef MATMUL_MSVC
		for(size_t i = 0; i < n; ++i)
		{
			size_t const uiRowBeginIndexA = i*uiPitchA;
			size_t const uiRowBeginIndexB = i*uiPitchB;
			size_t const uiRowBeginIndexC = i*uiPitchC;
			for(size_t j = 0; j < n; ++j)
			{
				C[uiRowBeginIndexC + j] = A[uiRowBeginIndexA + j] + B[uiRowBeginIndexB + j];
			}
		}
#else
		for(size_t i = 0; i < n; ++i)
		{
			for(size_t j = 0; j < n; ++j)
			{
				C[i*uiPitchC + j] = A[i*uiPitchA + j] + B[i*uiPitchB + j];
			}
		}
#endif
	}
	//-----------------------------------------------------------------------------
	//
	//-----------------------------------------------------------------------------
	void matadd2_pitch_seq(
		size_t const n,
		size_t const uiPitchA, TElement const * const restrict A,
		size_t const uiPitchC, TElement * const restrict C)
	{
#ifdef MATMUL_MSVC
		for(size_t i = 0; i < n; ++i)
		{
			size_t const uiRowBeginIndexA = i*uiPitchA;
			size_t const uiRowBeginIndexC = i*uiPitchC;
			for(size_t j = 0; j < n; ++j)
			{
				C[uiRowBeginIndexC + j] += A[uiRowBeginIndexA + j];
			}
		}
#else
		for(size_t i = 0; i < n; ++i)
		{
			for(size_t j = 0; j < n; ++j)
			{
				C[i*uiPitchC + j] += A[i*uiPitchA + j];
			}
		}
#endif
	}
	//-----------------------------------------------------------------------------
	//
	//-----------------------------------------------------------------------------
	void matsub_pitch_seq(
		size_t const n,
		size_t const uiPitchA, TElement const * const restrict A,
		size_t const uiPitchB, TElement const * const restrict B,
		size_t const uiPitchC, TElement * const restrict C)
	{
#ifdef MATMUL_MSVC
		for(size_t i = 0; i < n; ++i)
		{
			size_t const uiRowBeginIndexA = i*uiPitchA;
			size_t const uiRowBeginIndexB = i*uiPitchB;
			size_t const uiRowBeginIndexC = i*uiPitchC;
			for(size_t j = 0; j < n; ++j)
			{
				C[uiRowBeginIndexC + j] = A[uiRowBeginIndexA + j] - B[uiRowBeginIndexB + j];
			}
		}
#else
		for(size_t i = 0; i < n; ++i)
		{
			for(size_t j = 0; j < n; ++j)
			{
				C[i*uiPitchC + j] = A[i*uiPitchA + j] - B[i*uiPitchB + j];
			}
		}
#endif
	}

	//-----------------------------------------------------------------------------
	//
	//-----------------------------------------------------------------------------
	void matmul_pitch_seq_strassen(
		size_t const n,
		size_t const uiPitchA, TElement const * const restrict X,
		size_t const uiPitchB, TElement const * const restrict Y,
		size_t const uiPitchC, TElement * const restrict Z)
	{
		// Recursive base case.
		// If the matrices are smaller then the cutoff size we just use the conventional algorithm.
		if(n <= MATMUL_STRASSEN_CUT_OFF)
		{
			matmul_pitch_seq_complete_opt(n, uiPitchA, X, uiPitchB, Y, uiPitchC, Z);
		}
		else
		{
			assert(n%2==0);

			size_t const h = n/2;      // size of sub-matrices

			TElement const * const A = X;    // A-D matrices embedded in X
			TElement const * const B = X + h;
			TElement const * const C = X + h*uiPitchA;
			TElement const * const D = C + h;

			TElement const * const E = Y;    // E-H matrices embeded in Y
			TElement const * const F = Y + h;
			TElement const * const G = Y + h*uiPitchB;
			TElement const * const H = G + h;

			// Allocate temporary matrices.
			size_t const uiNumElements = h * h;
			TElement * P[7];
			for(size_t i = 0; i < 7; ++i)
			{
				P[i] = mat_alloc_zero_fill(uiNumElements);
			}
			TElement * const T = mat_alloc(uiNumElements);
			TElement * const U = mat_alloc(uiNumElements);

			// P0 = A*(F - H);
			matsub_pitch_seq(h, uiPitchB, F, uiPitchB, H, h, T);
			matmul_pitch_seq_strassen(h, uiPitchA, A, h, T, h, P[0]);

			// P1 = (A + B)*H
			matadd_pitch_seq(h, uiPitchA, A, uiPitchA, B, h, T);
			matmul_pitch_seq_strassen(h, h, T, uiPitchB, H, h, P[1]);

			// P2 = (C + D)*E
			matadd_pitch_seq(h, uiPitchA, C, uiPitchA, D, h, T);
			matmul_pitch_seq_strassen(h, h, T, uiPitchB, E, h, P[2]);

			// P3 = D*(G - E);
			matsub_pitch_seq(h, uiPitchB, G, uiPitchB, E, h, T);
			matmul_pitch_seq_strassen(h, uiPitchA, D, h, T, h, P[3]);

			// P4 = (A + D)*(E + H)
			matadd_pitch_seq(h, uiPitchA, A, uiPitchA, D, h, T);
			matadd_pitch_seq(h, uiPitchB, E, uiPitchB, H, h, U);
			matmul_pitch_seq_strassen(h, h, T, h, U, h, P[4]);

			// P5 = (B - D)*(G + H)
			matsub_pitch_seq(h, uiPitchA, B, uiPitchA, D, h, T);
			matadd_pitch_seq(h, uiPitchB, G, uiPitchB, H, h, U);
			matmul_pitch_seq_strassen(h, h, T, h, U, h, P[5]);

			// P6 = (A - C)*(E + F)
			matsub_pitch_seq(h, uiPitchA, A, uiPitchA, C, h, T);
			matadd_pitch_seq(h, uiPitchB, E, uiPitchB, F, h, U);
			matmul_pitch_seq_strassen(h, h, T, h, U, h, P[6]);

			// Z upper left = (P3 + P4) + (P5 - P1)
			matadd_pitch_seq(h, h, P[4], h, P[3], h, T);
			matsub_pitch_seq(h, h, P[5], h, P[1], h, U);
			TElement * const V = P[5];	// P[5] is only used once, so we reuse it as temporary buffer.
			matadd_pitch_seq(h, h, T, h, U, h, V);
			matadd2_pitch_seq(h, h, V, uiPitchC, Z);

			// Z lower left = P2 + P3
			matadd_pitch_seq(h, h, P[2], h, P[3], h, V);
			matadd2_pitch_seq(h, h, V, uiPitchC, Z + h*uiPitchC);

			// Z upper right = P0 + P1
			matadd_pitch_seq(h, h, P[0], h, P[1], h, V);
			matadd2_pitch_seq(h, h, V, uiPitchC, Z + h);

			// Z lower right = (P0 + P4) - (P2 + P6)
			matadd_pitch_seq(h, h, P[0], h, P[4], h, T);
			matadd_pitch_seq(h, h, P[2], h, P[6], h, U);
			matsub_pitch_seq(h, h, T, h, U, h, V);
			matadd2_pitch_seq(h, h, V, uiPitchC, Z + h*(uiPitchC + 1));

			// Deallocate temporary matrices.
			mat_free(U);
			mat_free(T);
			for(size_t i = 0; i < 7; ++i)
			{
				mat_free(P[i]);
			}
		}
	}
	//-----------------------------------------------------------------------------
	//
	//-----------------------------------------------------------------------------
	void matmul_seq_strassen(
		size_t const n,
		TElement const * const restrict A,
		TElement const * const restrict B,
		TElement * const restrict C)
	{
		matmul_pitch_seq_strassen(
			n, 
			n, A,
			n, B,
			n, C);
	}
#endif
