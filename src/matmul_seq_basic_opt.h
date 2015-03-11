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

#ifdef MATMUL_BUILD_SEQ_BASIC_OPT

	#include <stddef.h>	// size_t

	//-----------------------------------------------------------------------------
	//! Standard square matrix multiplication O(n^3). C = A*B + C.
	//! \param n The matrix dimension.
	//! \param A The left input matrix.
	//! \param B The right input matrix.
	//! \param C The input and result matrix.
	//-----------------------------------------------------------------------------
	void matmul_seq_index_pointer(
		size_t const n,
		TElement const * const A,
		TElement const * const B,
		TElement * const C);

	//-----------------------------------------------------------------------------
	//! Square matrix multiplication with added restrict keyword O(n^3). C = A*B + C.
	//! \param n The matrix dimension.
	//! \param A The left input matrix.
	//! \param B The right input matrix.
	//! \param C The input and result matrix.
	//-----------------------------------------------------------------------------
	void matmul_seq_restrict(
		size_t const n,
		TElement const * const restrict A,
		TElement const * const restrict B,
		TElement * const restrict C);

	//-----------------------------------------------------------------------------
	//! Square matrix multiplication with loop order ikj O(n^3). C = A*B + C.
	//! \param n The matrix dimension.
	//! \param A The left input matrix.
	//! \param B The right input matrix.
	//! \param C The input and result matrix.
	//-----------------------------------------------------------------------------
	void matmul_seq_loop_reorder(
		size_t const n,
		TElement const * const A,
		TElement const * const B,
		TElement * const C);

	//-----------------------------------------------------------------------------
	//! Standard square matrix multiplication O(n^3). C = A*B + C.
	//! \param n The matrix dimension.
	//! \param A The left input matrix.
	//! \param B The right input matrix.
	//! \param C The input and result matrix.
	//-----------------------------------------------------------------------------
	void matmul_seq_index_precalculate(
		size_t const n,
		TElement const * const A,
		TElement const * const B,
		TElement * const C);

	//-----------------------------------------------------------------------------
	//! Standard square matrix multiplication O(n^3). C = A*B + C.
	//! \param n The matrix dimension.
	//! \param A The left input matrix.
	//! \param B The right input matrix.
	//! \param C The input and result matrix.
	//-----------------------------------------------------------------------------
	void matmul_seq_loop_unroll_4(
		size_t const n,
		TElement const * const A,
		TElement const * const B,
		TElement * const C);

	//-----------------------------------------------------------------------------
	//! Standard square matrix multiplication O(n^3). C = A*B + C.
	//! \param n The matrix dimension.
	//! \param A The left input matrix.
	//! \param B The right input matrix.
	//! \param C The input and result matrix.
	//-----------------------------------------------------------------------------
	void matmul_seq_loop_unroll_8(
		size_t const n,
		TElement const * const A,
		TElement const * const B,
		TElement * const C);

	//-----------------------------------------------------------------------------
	//! Standard square matrix multiplication O(n^3). C = A*B + C.
	//! \param n The matrix dimension.
	//! \param A The left input matrix.
	//! \param B The right input matrix.
	//! \param C The input and result matrix.
	//-----------------------------------------------------------------------------
	void matmul_seq_loop_unroll_16(
		size_t const n,
		TElement const * const A,
		TElement const * const B,
		TElement * const C);

	//-----------------------------------------------------------------------------
	//! Blocked square matrix multiplication O(n^3). C = A*B + C.
	//! \param n The matrix dimension.
	//! \param A The left input matrix.
	//! \param B The right input matrix.
	//! \param C The input and result matrix.
	//-----------------------------------------------------------------------------
	void matmul_seq_block(
		size_t const n,
		TElement const * const A,
		TElement const * const B,
		TElement * const C);
#endif
