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

#ifdef MATMUL_BUILD_SEQ_SINGLE_OPTS

    #include <matmul/seq/SingleOpts.h>

    #include <matmul/common/Mat.h>  // matmul_mat_gemm_early_out

    //-----------------------------------------------------------------------------
    // Use explicit pointer access instead of index access that requires multiplication.
    // This prohibits vectorization by the compiler because the pointers are not marked with MATMUL_RESTRICT.
    //-----------------------------------------------------------------------------
    void matmul_gemm_seq_index_pointer(
        size_t const m, size_t const n, size_t const k,
        TElem const alpha,
        TElem const * const A, size_t const lda,
        TElem const * const B, size_t const ldb,
        TElem const beta,
        TElem * const C, size_t const ldc)
    {
        if(matmul_mat_gemm_early_out(m, n, k, alpha, beta))
        {
            return;
        }

        TElem * pCRow = C;
        TElem const * pARow = A;

        for(size_t i = 0; i < m; ++i, pARow += lda, pCRow += ldc)
        {
            TElem * pC = pCRow;
            TElem const * pBCol = B;

            for(size_t j = 0; j < n; ++j, ++pC, ++pBCol)
            {
                (*pC) *= beta;

                TElem const * pA = pARow;
                TElem const * pB = pBCol;

                for(size_t k2 = 0; k2 < k; ++k2, ++pA, pB += ldb)
                {
                    (*pC) += alpha * (*pA) * (*pB);
                }
            }
        }
    }

    //-----------------------------------------------------------------------------
    //
    //-----------------------------------------------------------------------------
    void matmul_gemm_seq_restrict(
        size_t const m, size_t const n, size_t const k,
        TElem const alpha,
        TElem const * const MATMUL_RESTRICT A, size_t const lda,
        TElem const * const MATMUL_RESTRICT B, size_t const ldb,
        TElem const beta,
        TElem * const MATMUL_RESTRICT C, size_t const ldc)
    {
        if(matmul_mat_gemm_early_out(m, n, k, alpha, beta))
        {
            return;
        }

        for(size_t i = 0; i < m; ++i)
        {
            for(size_t j = 0; j < n; ++j)
            {
                C[i*ldc + j] *= beta;

                for(size_t k2 = 0; k2 < k; ++k2)
                {
                    C[i*ldc + j] += alpha * A[i*lda + k2] * B[k2*ldb + j];
                }
            }
        }
    }

    //-----------------------------------------------------------------------------
    //
    //-----------------------------------------------------------------------------
    void matmul_gemm_seq_loop_reorder(
        size_t const m, size_t const n, size_t const k,
        TElem const alpha,
        TElem const * const A, size_t const lda,
        TElem const * const B, size_t const ldb,
        TElem const beta,
        TElem * const C, size_t const ldc)
    {
        if(matmul_mat_gemm_early_out(m, n, k, alpha, beta))
        {
            return;
        }

        for(size_t i = 0; i < m; ++i)
        {
            for(size_t j = 0; j < n; ++j)
            {
                C[i*ldc + j] *= beta;
            }
            for(size_t k2 = 0; k2 < k; ++k2)
            {
                for(size_t j = 0; j < n; ++j)
                {
                    // Cache efficiency inside the innermost loop:
                    // In the original loop order C[i*ldc + j] is in the cache and A[i*lda + k2 + 1] is likely to be in the cache. There would be a strided access to B so every access is likely a cache miss.
                    // In the new order A[i*lda + k2] is always in the cache and C[i*ldc + j + 1] as well as B[k2*ldb + j + 1] are likely to be in the cache.
                    // => This is one cache miss less per iteration.
                    C[i*ldc + j] += alpha * A[i*lda + k2] * B[k2*ldb + j];
                }
            }
        }
    }

    //-----------------------------------------------------------------------------
    //
    //-----------------------------------------------------------------------------
    void matmul_gemm_seq_index_precalculate(
        size_t const m, size_t const n, size_t const k,
        TElem const alpha,
        TElem const * const A, size_t const lda,
        TElem const * const B, size_t const ldb,
        TElem const beta,
        TElem * const C, size_t const ldc)
    {
        if(matmul_mat_gemm_early_out(m, n, k, alpha, beta))
        {
            return;
        }

        for(size_t i = 0; i < m; ++i)
        {
            size_t const uiRowBeginIdxA = i*lda;
            size_t const uiRowBeginIdxC = i*ldc;

            for(size_t j = 0; j < n; ++j)
            {
                C[i*ldc + j] *= beta;

                size_t const uiIdxC = uiRowBeginIdxC + j;

                for(size_t k2 = 0; k2 < k; ++k2)
                {
                    C[uiIdxC] += alpha * A[uiRowBeginIdxA + k2] * B[k2*ldb + j];
                }
            }
        }
    }

    //-----------------------------------------------------------------------------
    //
    //-----------------------------------------------------------------------------
    void matmul_gemm_seq_loop_unroll_4(
        size_t const m, size_t const n, size_t const k,
        TElem const alpha,
        TElem const * const A, size_t const lda,
        TElem const * const B, size_t const ldb,
        TElem const beta,
        TElem * const C, size_t const ldc)
    {
        if(matmul_mat_gemm_early_out(m, n, k, alpha, beta))
        {
            return;
        }

        for(size_t i = 0; i < m; ++i)
        {
            for(size_t j = 0; j < n; ++j)
            {
                C[i*ldc + j] *= beta;

                size_t k2;
                for(k2 = 0; k2+3 < k; k2 += 4)
                {
                    // Do not add the A[i,k2]*B[k2,j] results up before assigning to C[i,j] because this changes the numerical result.
                    C[i*ldc + j] += alpha * A[i*lda + k2] * B[k2*ldb + j];
                    C[i*ldc + j] += alpha * A[i*lda + k2+1] * B[(k2+1)*ldb + j];
                    C[i*ldc + j] += alpha * A[i*lda + k2+2] * B[(k2+2)*ldb + j];
                    C[i*ldc + j] += alpha * A[i*lda + k2+3] * B[(k2+3)*ldb + j];
                }
                for(; k2 < k; ++k2)
                {
                    C[i*ldc + j] += alpha * A[i*lda + k2] * B[k2*ldb + j];
                }
            }
        }
    }

    //-----------------------------------------------------------------------------
    //
    //-----------------------------------------------------------------------------
    void matmul_gemm_seq_loop_unroll_8(
        size_t const m, size_t const n, size_t const k,
        TElem const alpha,
        TElem const * const A, size_t const lda,
        TElem const * const B, size_t const ldb,
        TElem const beta,
        TElem * const C, size_t const ldc)
    {
        if(matmul_mat_gemm_early_out(m, n, k, alpha, beta))
        {
            return;
        }

        for(size_t i = 0; i < m; ++i)
        {
            for(size_t j = 0; j < n; ++j)
            {
                C[i*ldc + j] *= beta;

                size_t k2;
                for(k2 = 0; k2+7 < k; k2 += 8)
                {
                    // Do not add the A[i,k2]*B[k2,j] results up before assigning to C[i,j] because this changes the numerical result.
                    C[i*ldc + j] += alpha * A[i*lda + k2] * B[k2*ldb + j];
                    C[i*ldc + j] += alpha * A[i*lda + k2+1] * B[(k2+1)*ldb + j];
                    C[i*ldc + j] += alpha * A[i*lda + k2+2] * B[(k2+2)*ldb + j];
                    C[i*ldc + j] += alpha * A[i*lda + k2+3] * B[(k2+3)*ldb + j];
                    C[i*ldc + j] += alpha * A[i*lda + k2+4] * B[(k2+4)*ldb + j];
                    C[i*ldc + j] += alpha * A[i*lda + k2+5] * B[(k2+5)*ldb + j];
                    C[i*ldc + j] += alpha * A[i*lda + k2+6] * B[(k2+6)*ldb + j];
                    C[i*ldc + j] += alpha * A[i*lda + k2+7] * B[(k2+7)*ldb + j];
                }
                for(; k2 < k; ++k2)
                {
                    C[i*ldc + j] += alpha * A[i*lda + k2] * B[k2*ldb + j];
                }
            }
        }
    }

    //-----------------------------------------------------------------------------
    //
    //-----------------------------------------------------------------------------
    void matmul_gemm_seq_loop_unroll_16(
        size_t const m, size_t const n, size_t const k,
        TElem const alpha,
        TElem const * const A, size_t const lda,
        TElem const * const B, size_t const ldb,
        TElem const beta,
        TElem * const C, size_t const ldc)
    {
        if(matmul_mat_gemm_early_out(m, n, k, alpha, beta))
        {
            return;
        }

        for(size_t i = 0; i < m; ++i)
        {
            for(size_t j = 0; j < n; ++j)
            {
                C[i*ldc + j] *= beta;

                size_t k2 = 0;
                for(; k2+15 < k; k2 += 16)
                {
                    // Do not add the A[i,k2]*B[k2,j] results up before assigning to C[i,j] because this changes the numerical result.
                    C[i*ldc + j] += alpha * A[i*lda + k2] * B[k2*ldb + j];
                    C[i*ldc + j] += alpha * A[i*lda + k2+1] * B[(k2+1)*ldb + j];
                    C[i*ldc + j] += alpha * A[i*lda + k2+2] * B[(k2+2)*ldb + j];
                    C[i*ldc + j] += alpha * A[i*lda + k2+3] * B[(k2+3)*ldb + j];
                    C[i*ldc + j] += alpha * A[i*lda + k2+4] * B[(k2+4)*ldb + j];
                    C[i*ldc + j] += alpha * A[i*lda + k2+5] * B[(k2+5)*ldb + j];
                    C[i*ldc + j] += alpha * A[i*lda + k2+6] * B[(k2+6)*ldb + j];
                    C[i*ldc + j] += alpha * A[i*lda + k2+7] * B[(k2+7)*ldb + j];
                    C[i*ldc + j] += alpha * A[i*lda + k2+8] * B[(k2+8)*ldb + j];
                    C[i*ldc + j] += alpha * A[i*lda + k2+9] * B[(k2+9)*ldb + j];
                    C[i*ldc + j] += alpha * A[i*lda + k2+10] * B[(k2+10)*ldb + j];
                    C[i*ldc + j] += alpha * A[i*lda + k2+11] * B[(k2+11)*ldb + j];
                    C[i*ldc + j] += alpha * A[i*lda + k2+12] * B[(k2+12)*ldb + j];
                    C[i*ldc + j] += alpha * A[i*lda + k2+13] * B[(k2+13)*ldb + j];
                    C[i*ldc + j] += alpha * A[i*lda + k2+14] * B[(k2+14)*ldb + j];
                    C[i*ldc + j] += alpha * A[i*lda + k2+15] * B[(k2+15)*ldb + j];
                }
                for(; k2 < k; ++k2)
                {
                    C[i*ldc + j] += alpha * A[i*lda + k2] * B[k2*ldb + j];
                }
            }
        }
    }

    //-----------------------------------------------------------------------------
    //
    //-----------------------------------------------------------------------------
    void matmul_gemm_seq_block(
        size_t const m, size_t const n, size_t const k,
        TElem const alpha,
        TElem const * const A, size_t const lda,
        TElem const * const B, size_t const ldb,
        TElem const beta,
        TElem * const C, size_t const ldc)
    {
        if(matmul_mat_gemm_early_out(m, n, k, alpha, beta))
        {
            return;
        }

        for(size_t i = 0; i < m; ++i)
        {
            for(size_t j = 0; j < n; ++j)
            {
                C[i*ldc + j] *= beta;
            }
        }

        size_t const S = MATMUL_SEQ_BLOCK_FACTOR;

        for(size_t ii = 0; ii<m; ii += S)
        {
            size_t const iiS = ii+S;
            for(size_t jj = 0; jj<n; jj += S)
            {
                size_t const jjS = jj+S;
                for(size_t kk = 0; kk<k; kk += S)
                {
                    size_t const kkS = kk+S;
                    size_t const uiUpperBoundi = (iiS>m ? m : iiS);
                    for(size_t i = ii; i<uiUpperBoundi; ++i)
                    {
                        size_t const uiUpperBoundj = (jjS>n ? n : jjS);
                        for(size_t j = jj; j<uiUpperBoundj; ++j)
                        {
                            size_t const uiUpperBoundk = (kkS>k ? k : kkS);
                            for(size_t k2 = kk; k2<uiUpperBoundk; ++k2)
                            {
                                C[i*ldc + j] += alpha * A[i*lda + k2] * B[k2*ldb + j];
                            }
                        }
                    }
                }
            }
        }
    }
#endif
