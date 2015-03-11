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

#include "matmul_par_strassen_omp.h"

#ifdef MATMUL_BUILD_PAR_STRASSEN_OMP

	#include "matmul_par_openmp.h"	// matmul_pitch_par_openmp_guided_schedule

	#include "malloc.h"		// mat_alloc
	#include "array.h"		// mat_alloc_zero_fill

	#include <assert.h>		// assert

	#include <omp.h>

	#ifndef OPEN_MP_3
		typedef int TOpenMPForLoopIndex;
	#else
		typedef size_t TOpenMPForLoopIndex;
	#endif

	//-----------------------------------------------------------------------------
	//
	//-----------------------------------------------------------------------------
	void matadd_pitch_par_omp(
		size_t const n,
		size_t const uiPitchA, TElement const * const restrict A,
		size_t const uiPitchB, TElement const * const restrict B,
		size_t const uiPitchC, TElement * const restrict C)
	{
		TOpenMPForLoopIndex iN = (int)n;

		TOpenMPForLoopIndex i;	// For OpenMP < 3.0 you have to declare the loop index outside of the loop header.
#ifdef MATMUL_MSVC
		#pragma omp parallel for
		for(i = 0; i < iN; ++i)
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
		#pragma omp parallel for
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
	void matadd2_pitch_omp(
		size_t const n,
		size_t const uiPitchA, TElement const * const restrict A,
		size_t const uiPitchC, TElement * const restrict C)
	{
		TOpenMPForLoopIndex iN = (int)n;

		TOpenMPForLoopIndex i;	// For OpenMP < 3.0 you have to declare the loop index outside of the loop header.

#ifdef MATMUL_MSVC
		#pragma omp parallel for
		for(i = 0; i < iN; ++i)
		{
			size_t const uiRowBeginIndexA = i*uiPitchA;
			size_t const uiRowBeginIndexC = i*uiPitchC;
			for(size_t j = 0; j < n; ++j)
			{
				C[uiRowBeginIndexC + j] += A[uiRowBeginIndexA + j];
			}
		}
#else
		#pragma omp parallel for
		for(i = 0; i < iN; ++i)
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
	void matsub_pitch_par_omp(
		size_t const n,
		size_t const uiPitchA, TElement const * const restrict A,
		size_t const uiPitchB, TElement const * const restrict B,
		size_t const uiPitchC, TElement * const restrict C)
	{
		TOpenMPForLoopIndex iN = (int)n;

		TOpenMPForLoopIndex i;	// For OpenMP < 3.0 you have to declare the loop index outside of the loop header.
#ifdef MATMUL_MSVC
		#pragma omp parallel for
		for(i = 0; i < iN; ++i)
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
		#pragma omp parallel for
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
	void matmul_pitch_par_strassen_omp(
		size_t const n,
		size_t const uiPitchA, TElement const * const restrict X,
		size_t const uiPitchB, TElement const * const restrict Y,
		size_t const uiPitchC, TElement * const restrict Z)
	{
		// Recursive base case.
		// If the matrices are smaller then the cutoff size we just use the conventional algorithm.
		if(n <= MATMUL_STRASSEN_OMP_CUT_OFF)
		{
			matmul_pitch_par_openmp_static_schedule(n, uiPitchA, X, uiPitchB, Y, uiPitchC, Z);
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

			//#pragma omp parallel sections	// Parallel sections decrease the performance!
			{
				//#pragma omp section
				{
					// P0 = A*(F - H);
					matsub_pitch_par_omp(h, uiPitchB, F, uiPitchB, H, h, T);
					matmul_pitch_par_strassen_omp(h, uiPitchA, A, h, T, h, P[0]);
				}
				//#pragma omp section
				{
					// P1 = (A + B)*H
					matadd_pitch_par_omp(h, uiPitchA, A, uiPitchA, B, h, T);
					matmul_pitch_par_strassen_omp(h, h, T, uiPitchB, H, h, P[1]);
				}
				//#pragma omp section
				{
					// P2 = (C + D)*E
					matadd_pitch_par_omp(h, uiPitchA, C, uiPitchA, D, h, T);
					matmul_pitch_par_strassen_omp(h, h, T, uiPitchB, E, h, P[2]);
				}
				//#pragma omp section
				{
					// P3 = D*(G - E);
					matsub_pitch_par_omp(h, uiPitchB, G, uiPitchB, E, h, T);
					matmul_pitch_par_strassen_omp(h, uiPitchA, D, h, T, h, P[3]);
				}
				//#pragma omp section
				{
					// P4 = (A + D)*(E + H)
					matadd_pitch_par_omp(h, uiPitchA, A, uiPitchA, D, h, T);
					matadd_pitch_par_omp(h, uiPitchB, E, uiPitchB, H, h, U);
					matmul_pitch_par_strassen_omp(h, h, T, h, U, h, P[4]);
				}
				//#pragma omp section
				{
					// P5 = (B - D)*(G + H)
					matsub_pitch_par_omp(h, uiPitchA, B, uiPitchA, D, h, T);
					matadd_pitch_par_omp(h, uiPitchB, G, uiPitchB, H, h, U);
					matmul_pitch_par_strassen_omp(h, h, T, h, U, h, P[5]);
				}
				//#pragma omp section
				{
					// P6 = (A - C)*(E + F)
					matsub_pitch_par_omp(h, uiPitchA, A, uiPitchA, C, h, T);
					matadd_pitch_par_omp(h, uiPitchB, E, uiPitchB, F, h, U);
					matmul_pitch_par_strassen_omp(h, h, T, h, U, h, P[6]);
				}
			}

			//#pragma omp parallel sections
			{
				//#pragma omp section
				// Z upper left = (P3 + P4) + (P5 - P1)
				matadd_pitch_par_omp(h, h, P[4], h, P[3], h, T);
				//#pragma omp section
				matsub_pitch_par_omp(h, h, P[5], h, P[1], h, U);
			}
			TElement * const V = P[5];	// P[5] is only used once, so we reuse it as temporary buffer.
			matadd_pitch_par_omp(h, h, T, h, U, h, V);
			matadd2_pitch_omp(h, h, V, uiPitchC, Z);

			// Z lower left = P2 + P3
			matadd_pitch_par_omp(h, h, P[2], h, P[3], h, V);
			matadd2_pitch_omp(h, h, V, uiPitchC, Z + h*uiPitchC);

			// Z upper right = P0 + P1
			matadd_pitch_par_omp(h, h, P[0], h, P[1], h, V);
			matadd2_pitch_omp(h, h, V, uiPitchC, Z + h);

			//#pragma omp parallel sections
			{
				//#pragma omp section
				// Z lower right = (P0 + P4) - (P2 + P6)
				matadd_pitch_par_omp(h, h, P[0], h, P[4], h, T);
				//#pragma omp section
				matadd_pitch_par_omp(h, h, P[2], h, P[6], h, U);
			}
			matadd_pitch_par_omp(h, h, T, h, U, h, V);
			matadd2_pitch_omp(h, h, V, uiPitchC, Z + h*(uiPitchC + 1));

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
	void matmul_par_strassen_omp(
		size_t const n,
		TElement const * const restrict A,
		TElement const * const restrict B,
		TElement * const restrict C)
	{
		matmul_pitch_par_strassen_omp(
			n,
			n, A,
			n, B,
			n, C);
	}
#endif
