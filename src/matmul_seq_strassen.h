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

#ifdef MATMUL_BUILD_SEQ_STRASSEN

	#include <stddef.h>	// size_t

	//-----------------------------------------------------------------------------
	//! C = A+B
	//! \param n The matrix dimension.
	//! \param A The left input matrix.
	//! \param B The right input matrix.
	//! \param C The input and result matrix.
	//-----------------------------------------------------------------------------
	void matadd_pitch_seq(
		size_t const n,
		size_t const uiPitchA, TElement const * const restrict A,
		size_t const uiPitchB, TElement const * const restrict B,
		size_t const uiPitchC, TElement * const restrict C);

	//-----------------------------------------------------------------------------
	//! C = A-B
	//! \param n The matrix dimension.
	//! \param A The left input matrix.
	//! \param B The right input matrix.
	//! \param C The input and result matrix.
	//-----------------------------------------------------------------------------
	void matsub_pitch_seq(
		size_t const n,
		size_t const uiPitchA, TElement const * const restrict A,
		size_t const uiPitchB, TElement const * const restrict B,
		size_t const uiPitchC, TElement * const restrict C);

	//-----------------------------------------------------------------------------
	//! Volker Strassen algorithm for matrix multiplication. Z = X*Y + Z.
	//! Theoretical Runtime is O(n^log2(7)) = O(n^2.807).
	//! Assume NxN matrices where n is a power of two.
	//! Algorithm:
	//!   Matrices X and Y are split into four smaller
	//!   (n/2)x(n/2) matrices as follows:
	//!          _    _          _   _
	//!     X = | A  B |    Y = | E F |
	//!         | C  D |        | G H |
	//!          -    -          -   -
	//!   Then we build the following 7 matrices (requiring seven (n/2)x(n/2) matrix multiplications -- this is where the 2.807 = log2(7) improvement comes from):
	//!     P0 = A*(F - H);
	//!     P1 = (A + B)*H
	//!     P2 = (C + D)*E
	//!     P3 = D*(G - E);
	//!     P4 = (A + D)*(E + H)
	//!     P5 = (B - D)*(G + H)
	//!     P6 = (A - C)*(E + F)
	//!   The final result is
	//!        _                                            _
	//!   Z = | (P3 + P4) + (P5 - P1)   P0 + P1              |
	//!       | P2 + P3                 (P0 + P4) - (P2 + P6)|
	//!        -                                            -
	//! 7*mul, 18*add
	//! http://mathworld.wolfram.com/StrassenFormulas.html
	//! http://en.wikipedia.org/wiki/Strassen_algorithm
	//! 
	//! \param n The matrix dimension.
	//! \param A The left input matrix.
	//! \param B The right input matrix.
	//! \param C The input and result matrix.
	//-----------------------------------------------------------------------------
	void matmul_pitch_seq_strassen(
		size_t const n,
		size_t const uiPitchA, TElement const * const restrict X,
		size_t const uiPitchB, TElement const * const restrict Y,
		size_t const uiPitchC, TElement * const restrict Z);

	//-----------------------------------------------------------------------------
	//! Initial strassen call. C = A*B + C.
	//! \param n The matrix dimension.
	//! \param A The left input matrix.
	//! \param B The right input matrix.
	//! \param C The input and result matrix.
	//-----------------------------------------------------------------------------
	void matmul_seq_strassen(
		size_t const n,
		TElement const * const restrict A,
		TElement const * const restrict B,
		TElement * const restrict C);


#endif
