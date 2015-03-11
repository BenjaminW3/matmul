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

#include <stddef.h>		// size_t
#include <stdbool.h>	// bool

//-----------------------------------------------------------------------------
//! Square matrix comparison.
//!
//! \param n The matrix size.
//! \param A The left input matrix.
//! \param B The right input matrix.
//! \param C The input and result matrix.
//-----------------------------------------------------------------------------
bool mat_cmp(
	size_t const n,
	TElement const * const restrict X,
	TElement const * const restrict Y);

//-----------------------------------------------------------------------------
//! Prints the matrix to the console.
//!
//! \param n The matrix size.
//! \param A The matrix to print.
//-----------------------------------------------------------------------------
void mat_print(
	size_t const n,
	TElement const * const restrict A);

//-----------------------------------------------------------------------------
//! Copy a block of the given size from the location given by sr and sc in pSrcMat to the location given by dr and dc in pDstMat.
//! 
//! \param b The block size to copy.
//! \param pSrcMat Row major source matrix as continous 1D array.
//! \param sn The size of the source matrix.
//! \param sr The row in the source matrix the block to copy begins.
//! \param sc The column in the source matrix the block to copy begins.
//! \param pDstMat Row major destination matrix as continous 1D array.
//! \param dn The size of the destination matrix.
//! \param dr The row in the destination matrix the block to copy begins.
//! \param dc The column in the destination matrix the block to copy begins.
//-----------------------------------------------------------------------------
void mat_copy_block(
	size_t const b,
	TElement const * const restrict pSrcMat,
	size_t const sn,
	size_t const sr,
	size_t const sc,
	TElement * const restrict pDstMat,
	size_t const dn,
	size_t const dr,
	size_t const dc);

//-----------------------------------------------------------------------------
//! Copy the matrix pSrcMat to the pDstMat.
//! 
//! \param n The matrix size.
//! \param pSrcMat Row major source matrix as continous 1D array.
//! \param pDstMat Row major destination matrix as continous 1D array.
//-----------------------------------------------------------------------------
void mat_copy(
	TElement const * const restrict pSrcMat,
	TElement * const restrict pDstMat,
	size_t const n);

//-----------------------------------------------------------------------------
//! \param pSrcMat The block to copy.
//! \param n The size of the source matrix (pSrcMat).
//! \param uiBlockIndexHorizontal The horizontal destination block index inside source matrix.
//! \param uiBlockIndexVertical The vertical destination block index inside source matrix.
//! \param pDstBlock The destination matrix to copy into.
//! \param b The block size. This is the size of pDstBlock.
//-----------------------------------------------------------------------------
void mat_get_block(
	TElement const * const restrict pSrcMat,
	size_t const n,
	size_t const uiBlockIndexHorizontal,
	size_t const uiBlockIndexVertical,
	TElement * const restrict pDstBlock,
	size_t const b);

//-----------------------------------------------------------------------------
//! \param pSrcBlock The block to copy.
//! \param b The block size. This is the size of pSrcBlock.
//! \param pDstBlock The destination matrix to copy into.
//! \param n The size of the destination matrix (pDstBlock).
//! \param uiBlockIndexHorizontal The horizontal destination block index inside destination matrix.
//! \param uiBlockIndexVertical The vertical destination block index inside destination matrix.
//-----------------------------------------------------------------------------
void mat_set_block(
	TElement const * const restrict pSrcBlock,
	size_t const b,
	TElement * const restrict pDstMat,
	size_t const n,
	size_t const uiBlockIndexHorizontal,
	size_t const uiBlockIndexVertical);

//-----------------------------------------------------------------------------
//! Rearrange the matrix so that blocks are continous for scatter.
//! 
//! \param pSrcMat Row major matrix as continous 1D array.
//!	Matrix:
//!		 1  2  3  4
//!		 5  6  7  8
//!		 9 10 11 12
//!		13 14 15 16
//!	is stored as:  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16
//! \param n The size of the destination matrix.
//! \param pBlockMajorMat 1D array containing the blocks of pSrcMat sequentialized row or column major depending on bColumnFirst.
//! \param b The block size.
//! \param bColumnFirst	If bColumnFirst is true the matrix is stored as:	1  2  5  6  3  4  7  8  9 10 13 14 11 12 15 16.
//!						If bColumnFirst is false the matrix is stored as:	1  2  5  6  9 10 13 14 3  4  7  8  11 12 15 16.
//-----------------------------------------------------------------------------
void mat_row_major_to_mat_x_block_major(
	TElement const * const restrict pSrcMat,
	size_t const n,
	TElement * restrict pBlockMajorMat,
	size_t const b,
	bool const bColumnFirst);

//-----------------------------------------------------------------------------
// Rearrange the matrix so that blocks are continous for scatter.
// 
//! \param pBlockMajorMatContinous 1D array containing continous blocks.
//! \param b The block size.
//! \param pSrcMat Row major matrix as continous 1D array.
//!	Matrix:
//!		 1  2  3  4
//!		 5  6  7  8
//!		 9 10 11 12
//!		13 14 15 16
//!	is stored as:  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16
//! \param n The size of the destination matrix.
//! \param bColumnFirst	If bColumnFirst is true the matrix is stored as:	1  2  5  6  3  4  7  8  9 10 13 14 11 12 15 16.
//!						If bColumnFirst is false the matrix is stored as:	1  2  5  6  9 10 13 14 3  4  7  8  11 12 15 16.
//-----------------------------------------------------------------------------
void mat_x_block_major_to_mat_row_major(
	TElement const * restrict pBlockMajorMat,
	size_t const b,
	TElement * const restrict pDstMat,
	size_t const n,
	bool const bColumnFirst);
