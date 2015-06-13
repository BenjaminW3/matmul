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

#ifdef MATMUL_BUILD_SEQ_STRASSEN

    #include <matmul/seq/Strassen.h>

    #include <matmul/seq/MultipleOpts.h>    // matmul_gemm_seq_multiple_opts

    #include <matmul/common/Alloc.h>        // matmul_arr_alloc
    #include <matmul/common/Array.h>        // matmul_arr_alloc_zero_fill
    #include <matmul/common/Mat.h>          // matmul_mat_gemm_early_out

    #include <assert.h>                     // assert
    #include <stdio.h>                      // printf

    //-----------------------------------------------------------------------------
    //! Adapted from http://ezekiel.vancouver.wsu.edu/~cs330/lectures/linear_algebra/mm/mm.c W. Cochran  wcochran@vancouver.wsu.edu
    //-----------------------------------------------------------------------------

    //-----------------------------------------------------------------------------
    //
    //-----------------------------------------------------------------------------
    void matmul_mat_add_pitch_seq(
        size_t const m, size_t const n,
        TElem const * const MATMUL_RESTRICT A, size_t const lda,
        TElem const * const MATMUL_RESTRICT B, size_t const ldb,
        TElem * const MATMUL_RESTRICT C, size_t const ldc)
    {
#ifdef MATMUL_MSVC
        for(size_t i = 0; i < m; ++i)
        {
            size_t const uiRowBeginIdxA = i*lda;
            size_t const uiRowBeginIdxB = i*ldb;
            size_t const uiRowBeginIdxC = i*ldc;
            for(size_t j = 0; j < n; ++j)
            {
                C[uiRowBeginIdxC + j] = A[uiRowBeginIdxA + j] + B[uiRowBeginIdxB + j];
            }
        }
#else
        for(size_t i = 0; i < m; ++i)
        {
            for(size_t j = 0; j < n; ++j)
            {
                C[i*ldc + j] = A[i*lda + j] + B[i*ldb + j];
            }
        }
#endif
    }
    //-----------------------------------------------------------------------------
    //
    //-----------------------------------------------------------------------------
    void matmul_mat_add2_pitch_seq(
        size_t const m, size_t const n,
        TElem const * const MATMUL_RESTRICT A, size_t const lda,
        TElem * const MATMUL_RESTRICT C, size_t const ldc)
    {
#ifdef MATMUL_MSVC
        for(size_t i = 0; i < m; ++i)
        {
            size_t const uiRowBeginIdxA = i*lda;
            size_t const uiRowBeginIdxC = i*ldc;
            for(size_t j = 0; j < n; ++j)
            {
                C[uiRowBeginIdxC + j] += A[uiRowBeginIdxA + j];
            }
        }
#else
        for(size_t i = 0; i < m; ++i)
        {
            for(size_t j = 0; j < n; ++j)
            {
                C[i*ldc + j] += A[i*lda + j];
            }
        }
#endif
    }
    //-----------------------------------------------------------------------------
    //
    //-----------------------------------------------------------------------------
    void matmul_mat_sub_pitch_seq(
        size_t const m, size_t const n,
        TElem const * const MATMUL_RESTRICT A, size_t const lda,
        TElem const * const MATMUL_RESTRICT B, size_t const ldb,
        TElem * const MATMUL_RESTRICT C, size_t const ldc)
    {
#ifdef MATMUL_MSVC
        for(size_t i = 0; i < m; ++i)
        {
            size_t const uiRowBeginIdxA = i*lda;
            size_t const uiRowBeginIdxB = i*ldb;
            size_t const uiRowBeginIdxC = i*ldc;
            for(size_t j = 0; j < n; ++j)
            {
                C[uiRowBeginIdxC + j] = A[uiRowBeginIdxA + j] - B[uiRowBeginIdxB + j];
            }
        }
#else
        for(size_t i = 0; i < m; ++i)
        {
            for(size_t j = 0; j < n; ++j)
            {
                C[i*ldc + j] = A[i*lda + j] - B[i*ldb + j];
            }
        }
#endif
    }

    //-----------------------------------------------------------------------------
    //
    //-----------------------------------------------------------------------------
    void matmul_gemm_seq_strassen(
        size_t const m, size_t const n, size_t const k,
        TElem const alpha,
        TElem const * const MATMUL_RESTRICT X, size_t const lda,
        TElem const * const MATMUL_RESTRICT Y, size_t const ldb,
        TElem const beta,
        TElem * const MATMUL_RESTRICT Z, size_t const ldc)
    {
        if(matmul_mat_gemm_early_out(m, n, k, alpha, beta))
        {
            return;
        }

        // Apply beta multiplication to C.
        if(beta != (TElem)1)
        {
            for(size_t i = 0; i < m; ++i)
            {
                for(size_t j = 0; j < n; ++j)
                {
                    Z[i*ldc + j] *= beta;
                }
            }
        }

        // Recursive base case.
        // If the matrices are smaller then the cutoff size we just use the conventional algorithm.
        if((n % 2 == 1) || (n <= MATMUL_STRASSEN_CUT_OFF))
        {
            matmul_gemm_seq_multiple_opts(m, n, k, alpha, X, lda, Y, ldb, (TElem)1, Z, ldc);
        }
        else
        {
            // \TODO: Implement for non square matrices?
            if(m!=n || m!=k)
            {
                printf("Invalid matrix size! The matrices have to be square for the Strassen GEMM.\n");
                return;
            }

            size_t const h = n/2;           // size of sub-matrices

            TElem const * const A = X;      // A-D matrices embedded in X
            TElem const * const B = X + h;
            TElem const * const C = X + h*lda;
            TElem const * const D = C + h;

            TElem const * const E = Y;      // E-H matrices embeded in Y
            TElem const * const F = Y + h;
            TElem const * const G = Y + h*ldb;
            TElem const * const H = G + h;

            // Allocate temporary matrices.
            size_t const uiNumElements = h * h;
            TElem * P[7];
            for(size_t i = 0; i < 7; ++i)
            {
                P[i] = matmul_arr_alloc_zero_fill(uiNumElements);
            }
            TElem * const T = matmul_arr_alloc(uiNumElements);
            TElem * const U = matmul_arr_alloc(uiNumElements);

            // P0 = A*(F - H);
            matmul_mat_sub_pitch_seq(h, h, F, ldb, H, ldb, T, h);
            matmul_gemm_seq_strassen(h, h, h, alpha, A, lda, T, h, (TElem)1, P[0], h);

            // P1 = (A + B)*H
            matmul_mat_add_pitch_seq(h, h, A, lda, B, lda, T, h);
            matmul_gemm_seq_strassen(h, h, h, alpha, T, h, H, ldb, (TElem)1, P[1], h);

            // P2 = (C + D)*E
            matmul_mat_add_pitch_seq(h, h, C, lda, D, lda, T, h);
            matmul_gemm_seq_strassen(h, h, h, alpha, T, h, E, ldb, (TElem)1, P[2], h);

            // P3 = D*(G - E);
            matmul_mat_sub_pitch_seq(h, h, G, ldb, E, ldb, T, h);
            matmul_gemm_seq_strassen(h, h, h, alpha, D, lda, T, h, (TElem)1, P[3], h);

            // P4 = (A + D)*(E + H)
            matmul_mat_add_pitch_seq(h, h, A, lda, D, lda, T, h);
            matmul_mat_add_pitch_seq(h, h, E, ldb, H, ldb, U, h);
            matmul_gemm_seq_strassen(h, h, h, alpha, T, h, U, h, (TElem)1, P[4], h);

            // P5 = (B - D)*(G + H)
            matmul_mat_sub_pitch_seq(h, h, B, lda, D, lda, T, h);
            matmul_mat_add_pitch_seq(h, h, G, ldb, H, ldb, U, h);
            matmul_gemm_seq_strassen(h, h, h, alpha, T, h, U, h, (TElem)1, P[5], h);

            // P6 = (A - C)*(E + F)
            matmul_mat_sub_pitch_seq(h, h, A, lda, C, lda, T, h);
            matmul_mat_add_pitch_seq(h, h, E, ldb, F, ldb, U, h);
            matmul_gemm_seq_strassen(h, h, h, alpha, T, h, U, h, (TElem)1, P[6], h);

            // Z upper left = (P3 + P4) + (P5 - P1)
            matmul_mat_add_pitch_seq(h, h, P[4], h, P[3], h, T, h);
            matmul_mat_sub_pitch_seq(h, h, P[5], h, P[1], h, U, h);
            TElem * const V = P[5];    // P[5] is only used once, so we reuse it as temporary buffer.
            matmul_mat_add_pitch_seq(h, h, T, h, U, h, V, h);
            matmul_mat_add2_pitch_seq(h, h, V, h, Z, ldc);

            // Z lower left = P2 + P3
            matmul_mat_add_pitch_seq(h, h, P[2], h, P[3], h, V, h);
            matmul_mat_add2_pitch_seq(h, h, V, h, Z + h*ldc, ldc);

            // Z upper right = P0 + P1
            matmul_mat_add_pitch_seq(h, h, P[0], h, P[1], h, V, h);
            matmul_mat_add2_pitch_seq(h, h, V, h, Z + h, ldc);

            // Z lower right = (P0 + P4) - (P2 + P6)
            matmul_mat_add_pitch_seq(h, h, P[0], h, P[4], h, T, h);
            matmul_mat_add_pitch_seq(h, h, P[2], h, P[6], h, U, h);
            matmul_mat_sub_pitch_seq(h, h, T, h, U, h, V, h);
            matmul_mat_add2_pitch_seq(h, h, V, h, Z + h*(ldc + 1), ldc);

            // Deallocate temporary matrices.
            matmul_arr_free(U);
            matmul_arr_free(T);
            for(size_t i = 0; i < 7; ++i)
            {
                matmul_arr_free(P[i]);
            }
        }
    }
#endif
