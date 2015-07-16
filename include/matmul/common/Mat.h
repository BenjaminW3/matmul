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

#include <matmul/common/Config.h>   // TElem, TIdx

#include <stdbool.h>                // bool

#ifdef __cplusplus
    extern "C"
    {
#endif
    //-----------------------------------------------------------------------------
    //! Square matrix comparison.
    //!
    //! \param m The number of rows.
    //! \param n The number of columns.
    //! \param A The left input matrix.
    //! \param lda Specifies the leading dimension of A.
    //! \param B The right input matrix.
    //! \param ldb Specifies the leading dimension of B.
    //! \return If the matrices compare equal (under the given threshold).
    //-----------------------------------------------------------------------------
    bool matmul_mat_cmp(
        TIdx const m, TIdx const n,
        TElem const * const MATMUL_RESTRICT A, TIdx const lda,
        TElem const * const MATMUL_RESTRICT B, TIdx const ldb,
        TElem const fErrorThreshold);

    //-----------------------------------------------------------------------------
    //! Prints the matrix to the console.
    //!
    //! \param m The number of rows.
    //! \param n The number of columns.
    //! \param A The matrix to print.
    //! \param lda Specifies the leading dimension of A.
    //-----------------------------------------------------------------------------
    void matmul_mat_print(
        TIdx const m, TIdx const n,
        TElem const * const MATMUL_RESTRICT A, TIdx const lda,
        char * const pElemSeperator, char * const pRowSeperator,
        char * const pDimBegin, char * const pDimEnd);
    //-----------------------------------------------------------------------------
    //! Prints the matrix to the console.
    //!
    //! \param m The number of rows.
    //! \param n The number of columns.
    //! \param A The matrix to print.
    //! \param lda Specifies the leading dimension of A.
    //-----------------------------------------------------------------------------
    void matmul_mat_print_simple(
        TIdx const m, TIdx const n,
        TElem const * const MATMUL_RESTRICT A, TIdx const lda);
    //-----------------------------------------------------------------------------
    //! Prints the matrix to the console.
    //!
    //! \param m The number of rows.
    //! \param n The number of columns.
    //! \param A The matrix to print.
    //! \param lda Specifies the leading dimension of A.
    //-----------------------------------------------------------------------------
    void matmul_mat_print_mathematica(
        TIdx const m, TIdx const n,
        TElem const * const MATMUL_RESTRICT A, TIdx const lda);

    //-----------------------------------------------------------------------------
    //! Get if the GEMM is allowed to return early.
    //!
    //! \param m Specifies the number of rows of the matrix A and of the matrix C.
    //! \param n Specifies the number of columns of the matrix B and the number of columns of the matrix C.
    //! \param k Specifies the number of columns of the matrix A and the number of rows of the matrix B.
    //! \param alpha Scalar value used to scale the product of matrices A and B.
    //! \param beta Scalar value used to scale matrix C.
    //! \return If the matrix multiplication is allowed to return without calculations.
    //-----------------------------------------------------------------------------
    bool matmul_mat_gemm_early_out(
        TIdx const m, TIdx const n, TIdx const k,
        TElem const alpha,
        TElem const beta);

    //-----------------------------------------------------------------------------
    //! Copy a block of the given size from the location given by sr and sc in pSrcMat to the location given by dr and dc in pDstMat.
    //!
    //! \param m The number of rows.
    //! \param n The number of columns.
    //! \param pSrcMat Row major source matrix.
    //! \param lds The leading dimension of the source matrix.
    //! \param sr The row in the source matrix the block to copy begins.
    //! \param sc The column in the source matrix the block to copy begins.
    //! \param pDstMat Row major destination matrix.
    //! \param ldd The leading dimension of the destination matrix.
    //! \param dr The row in the destination matrix the block to copy begins.
    //! \param dc The column in the destination matrix the block to copy begins.
    //-----------------------------------------------------------------------------
    void matmul_mat_copy_block(
        TIdx const m,
        TIdx const n,
        TElem const * const MATMUL_RESTRICT pSrcMat, TIdx const lds,
        TIdx const sr,
        TIdx const sc,
        TElem * const MATMUL_RESTRICT pDstMat, TIdx const ldd,
        TIdx const dr,
        TIdx const dc);

    //-----------------------------------------------------------------------------
    //! Copy the matrix pSrcMat to the pDstMat.
    //!
    //! \param m The number of rows.
    //! \param n The number of columns.
    //! \param pSrcMat Row major source matrix.
    //! \param lds The leading dimension of the source matrix.
    //! \param pDstMat Row major destination matrix.
    //! \param ldd The leading dimension of the destination matrix.
    //-----------------------------------------------------------------------------
    void matmul_mat_copy(
        TIdx const m,
        TIdx const n,
        TElem const * const MATMUL_RESTRICT pSrcMat, TIdx const lds,
        TElem * const MATMUL_RESTRICT pDstMat, TIdx const ldd);

    //-----------------------------------------------------------------------------
    //! Rearrange the matrix so that blocks are continous for scatter.
    //!
    //! \param pSrcMat Row major matrix as continous 1D array.
    //!    Matrix:
    //!         1  2  3  4
    //!         5  6  7  8
    //!         9 10 11 12
    //!        13 14 15 16
    //!    is stored as:  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16
    //! \param m The number of rows of the source matrix.
    //! \param n The number of columns of the source matrix.
    //! \param lds The leading dimension of the source matrix.
    //! \param pBlockMajorMat 1D array containing the blocks of pSrcMat sequentialized row or column major depending on bColumnFirst.
    //! \param b The block size.
    //! \param bColumnFirst    If bColumnFirst is true the matrix is stored as:    1  2  5  6  3  4  7  8  9 10 13 14 11 12 15 16.
    //!                        If bColumnFirst is false the matrix is stored as:    1  2  5  6  9 10 13 14 3  4  7  8  11 12 15 16.
    //-----------------------------------------------------------------------------
    void matmul_mat_row_major_to_mat_x_block_major(
        TElem const * const MATMUL_RESTRICT pSrcMat, TIdx const m, TIdx const n, TIdx const lds,
        TElem * MATMUL_RESTRICT pBlockMajorMat, TIdx const b,
        bool const bColumnFirst);

    //-----------------------------------------------------------------------------
    // Rearrange the matrix so that blocks are continous for scatter.
    //
    //! \param pBlockMajorMatContinous 1D array containing continous blocks.
    //! \param b The block size.
    //! \param pSrcMat Row major matrix as continous 1D array.
    //!    Matrix:
    //!         1  2  3  4
    //!         5  6  7  8
    //!         9 10 11 12
    //!        13 14 15 16
    //!    is stored as:  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16
    //! \param m The number of rows of the destination matrix.
    //! \param n The number of columns of the destination matrix.
    //! \param ldd The leading dimension of the destination matrix.
    //! \param bColumnFirst    If bColumnFirst is true the matrix is stored as:    1  2  5  6  3  4  7  8  9 10 13 14 11 12 15 16.
    //!                        If bColumnFirst is false the matrix is stored as:    1  2  5  6  9 10 13 14 3  4  7  8  11 12 15 16.
    //-----------------------------------------------------------------------------
    void matmul_mat_x_block_major_to_mat_row_major(
        TElem const * MATMUL_RESTRICT pBlockMajorMat, TIdx const b,
        TElem * const MATMUL_RESTRICT pDstMat, TIdx const m, TIdx const n, TIdx const ldd,
        bool const bColumnFirst);

    //-----------------------------------------------------------------------------
    //! \param pSrcMat The block to copy.
    //! \param lds The leading destination of the source matrix (pSrcMat).
    //! \param uiBlockIdxHorizontal The horizontal destination block index inside source matrix.
    //! \param uiBlockIdxVertical The vertical destination block index inside source matrix.
    //! \param pDstBlock The destination matrix to copy into.
    //! \param b The block size. This is the size of pDstBlock.
    //-----------------------------------------------------------------------------
    void matmul_mat_get_block(
        TElem const * const MATMUL_RESTRICT pSrcMat, TIdx const lds,
        TIdx const uiBlockIdxHorizontal,
        TIdx const uiBlockIdxVertical,
        TElem * const MATMUL_RESTRICT pDstBlock, TIdx const b);

    //-----------------------------------------------------------------------------
    //! \param pSrcBlock The block to copy.
    //! \param b The block size. This is the size of pSrcBlock.
    //! \param pDstBlock The destination matrix to copy into.
    //! \param ldd The leading destination of the destination matrix (pDstBlock).
    //! \param uiBlockIdxHorizontal The horizontal destination block index inside destination matrix.
    //! \param uiBlockIdxVertical The vertical destination block index inside destination matrix.
    //-----------------------------------------------------------------------------
    void matmul_mat_set_block(
        TElem const * const MATMUL_RESTRICT pSrcBlock, TIdx const b,
        TElem * const MATMUL_RESTRICT pDstMat, TIdx const ldd,
        TIdx const uiBlockIdxHorizontal,
        TIdx const uiBlockIdxVertical);
#ifdef __cplusplus
    }
#endif
