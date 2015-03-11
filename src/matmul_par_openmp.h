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
 
#include "config.h"

#ifdef MATMUL_BUILD_PAR_OPENMP

	#include <stddef.h>	// size_t

	//-----------------------------------------------------------------------------
	//! OpenMP square matrix multiplication. C = A*B + C.
	//! \param n The matrix dimension.
	//! \param A The left input matrix.
	//! \param B The right input matrix.
	//! \param C The input and result matrix.
	//-----------------------------------------------------------------------------
	void matmul_pitch_par_openmp_guided_schedule(
		size_t const n,
		size_t const uiPitchA, TElement const * const restrict A,
		size_t const uiPitchB, TElement const * const restrict B,
		size_t const uiPitchC, TElement * const restrict C);
	//-----------------------------------------------------------------------------
	//! OpenMP square matrix multiplication. C = A*B + C.
	//! \param n The matrix dimension.
	//! \param A The left input matrix.
	//! \param B The right input matrix.
	//! \param C The input and result matrix.
	//-----------------------------------------------------------------------------
	void matmul_par_openmp_guided_schedule(
		size_t const n,
		TElement const * const restrict A,
		TElement const * const restrict B,
		TElement * const restrict C);


	//-----------------------------------------------------------------------------
	//! OpenMP square matrix multiplication with static schedule. C = A*B + C.
	//! \param n The matrix dimension.
	//! \param A The left input matrix.
	//! \param B The right input matrix.
	//! \param C The input and result matrix.
	//-----------------------------------------------------------------------------
	void matmul_pitch_par_openmp_static_schedule(
		size_t const n,
		size_t const uiPitchA, TElement const * const restrict A,
		size_t const uiPitchB, TElement const * const restrict B,
		size_t const uiPitchC, TElement * const restrict C);
	//-----------------------------------------------------------------------------
	//! OpenMP square matrix multiplication with static schedule. C = A*B + C.
	//! \param n The matrix dimension.
	//! \param A The left input matrix.
	//! \param B The right input matrix.
	//! \param C The input and result matrix.
	//-----------------------------------------------------------------------------
	void matmul_par_openmp_static_schedule(
		size_t const n,
		TElement const * const restrict A,
		TElement const * const restrict B,
		TElement * const restrict C);

#ifdef OPEN_MP_3
	//-----------------------------------------------------------------------------
	//! OpenMP square matrix multiplication with static schedule and nested parallel loops. C = A*B + C.
	//! \param n The matrix dimension.
	//! \param A The left input matrix.
	//! \param B The right input matrix.
	//! \param C The input and result matrix.
	//-----------------------------------------------------------------------------
	void matmul_par_openmp_static_schedule_collapse(
		size_t const n,
		TElement const * const restrict A,
		TElement const * const restrict B,
		TElement * const restrict C);
#endif

#endif
