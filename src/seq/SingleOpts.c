//-----------------------------------------------------------------------------
//! \file
//! Copyright 2013-2015 Benjamin Worpitz
//!
//! This file is part of matmul.
//!
//! matmul is free software: you can redistribute it and/or modify
//! it under the terms of the GNU Lesser General Public License as published by
//! the Free Software Foundation, either version 3 of the License, or
//! (at your option) any later version.
//!
//! matmul is distributed in the hope that it will be useful,
//! but WITHOUT ANY WARRANTY; without even the implied warranty of
//! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
//! GNU Lesser General Public License for more details.
//!
//! You should have received a copy of the GNU Lesser General Public License
//! along with matmul.
//! If not, see <http://www.gnu.org/licenses/>.
//-----------------------------------------------------------------------------

#ifdef MATMUL_BUILD_SEQ_SINGLE_OPTS

    #include <matmul/seq/SingleOpts.h>

    #include <matmul/common/Mat.h>  // matmul_mat_gemm_early_out

    //-----------------------------------------------------------------------------
    // Use explicit pointer access instead of index access that requires multiplication.
    // This prohibits vectorization by the compiler because the pointers are not marked with MATMUL_RESTRICT.
    //-----------------------------------------------------------------------------
    TReturn matmul_gemm_seq_index_pointer(
        TSize const m, TSize const n, TSize const k,
        TElem const alpha,
        TElem const * const A, TSize const lda,
        TElem const * const B, TSize const ldb,
        TElem const beta,
        TElem * const C, TSize const ldc)
    {
        double const timeStart = getTimeSec();

        if(matmul_mat_gemm_early_out(m, n, k, alpha, beta))
        {
            MATMUL_TIME_RETURN_EARLY_OUT;
        }

        MATMUL_TIME_START;

        TElem * pCRow = C;
        TElem const * pARow = A;

        for(TSize i = 0; i < m; ++i, pARow += lda, pCRow += ldc)
        {
            TElem * pC = pCRow;
            TElem const * pBCol = B;

            for(TSize j = 0; j < n; ++j, ++pC, ++pBCol)
            {
                (*pC) *= beta;

                TElem const * pA = pARow;
                TElem const * pB = pBCol;

                for(TSize k2 = 0; k2 < k; ++k2, ++pA, pB += ldb)
                {
                    (*pC) += alpha * (*pA) * (*pB);
                }
            }
        }

        MATMUL_TIME_END;
        MATMUL_TIME_RETURN;
    }

    //-----------------------------------------------------------------------------
    //
    //-----------------------------------------------------------------------------
    TReturn matmul_gemm_seq_restrict(
        TSize const m, TSize const n, TSize const k,
        TElem const alpha,
        TElem const * const MATMUL_RESTRICT A, TSize const lda,
        TElem const * const MATMUL_RESTRICT B, TSize const ldb,
        TElem const beta,
        TElem * const MATMUL_RESTRICT C, TSize const ldc)
    {
        double const timeStart = getTimeSec();

        if(matmul_mat_gemm_early_out(m, n, k, alpha, beta))
        {
            MATMUL_TIME_RETURN_EARLY_OUT;
        }

        MATMUL_TIME_START;

        for(TSize i = 0; i < m; ++i)
        {
            for(TSize j = 0; j < n; ++j)
            {
                C[i*ldc + j] *= beta;

                for(TSize k2 = 0; k2 < k; ++k2)
                {
                    C[i*ldc + j] += alpha * A[i*lda + k2] * B[k2*ldb + j];
                }
            }
        }

        MATMUL_TIME_END;
        MATMUL_TIME_RETURN;
    }

    //-----------------------------------------------------------------------------
    //
    //-----------------------------------------------------------------------------
    TReturn matmul_gemm_seq_loop_reorder(
        TSize const m, TSize const n, TSize const k,
        TElem const alpha,
        TElem const * const A, TSize const lda,
        TElem const * const B, TSize const ldb,
        TElem const beta,
        TElem * const C, TSize const ldc)
    {
        double const timeStart = getTimeSec();

        if(matmul_mat_gemm_early_out(m, n, k, alpha, beta))
        {
            MATMUL_TIME_RETURN_EARLY_OUT;
        }

        MATMUL_TIME_START;

        for(TSize i = 0; i < m; ++i)
        {
            for(TSize j = 0; j < n; ++j)
            {
                C[i*ldc + j] *= beta;
            }
            for(TSize k2 = 0; k2 < k; ++k2)
            {
                for(TSize j = 0; j < n; ++j)
                {
                    // Cache efficiency inside the innermost loop:
                    // In the original loop order C[i*ldc + j] is in the cache and A[i*lda + k2 + 1] is likely to be in the cache. There would be a strided access to B so every access is likely a cache miss.
                    // In the new order A[i*lda + k2] is always in the cache and C[i*ldc + j + 1] as well as B[k2*ldb + j + 1] are likely to be in the cache.
                    // => This is one cache miss less per iteration.
                    C[i*ldc + j] += alpha * A[i*lda + k2] * B[k2*ldb + j];
                }
            }
        }

        MATMUL_TIME_END;
        MATMUL_TIME_RETURN;
    }

    //-----------------------------------------------------------------------------
    //
    //-----------------------------------------------------------------------------
    TReturn matmul_gemm_seq_index_precalculate(
        TSize const m, TSize const n, TSize const k,
        TElem const alpha,
        TElem const * const A, TSize const lda,
        TElem const * const B, TSize const ldb,
        TElem const beta,
        TElem * const C, TSize const ldc)
    {
        double const timeStart = getTimeSec();

        if(matmul_mat_gemm_early_out(m, n, k, alpha, beta))
        {
            MATMUL_TIME_RETURN_EARLY_OUT;
        }

        MATMUL_TIME_START;

        for(TSize i = 0; i < m; ++i)
        {
            TSize const rowBeginIdxA = i*lda;
            TSize const rowBeginIdxC = i*ldc;

            for(TSize j = 0; j < n; ++j)
            {
                C[i*ldc + j] *= beta;

                TSize const idxC = rowBeginIdxC + j;

                for(TSize k2 = 0; k2 < k; ++k2)
                {
                    C[idxC] += alpha * A[rowBeginIdxA + k2] * B[k2*ldb + j];
                }
            }
        }

        MATMUL_TIME_END;
        MATMUL_TIME_RETURN;
    }

    //-----------------------------------------------------------------------------
    //
    //-----------------------------------------------------------------------------
    TReturn matmul_gemm_seq_loop_unroll_4(
        TSize const m, TSize const n, TSize const k,
        TElem const alpha,
        TElem const * const A, TSize const lda,
        TElem const * const B, TSize const ldb,
        TElem const beta,
        TElem * const C, TSize const ldc)
    {
        double const timeStart = getTimeSec();

        if(matmul_mat_gemm_early_out(m, n, k, alpha, beta))
        {
            MATMUL_TIME_RETURN_EARLY_OUT;
        }

        MATMUL_TIME_START;

        for(TSize i = 0; i < m; ++i)
        {
            for(TSize j = 0; j < n; ++j)
            {
                C[i*ldc + j] *= beta;

                TSize k2;
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

        MATMUL_TIME_END;
        MATMUL_TIME_RETURN;
    }

    //-----------------------------------------------------------------------------
    //
    //-----------------------------------------------------------------------------
    TReturn matmul_gemm_seq_loop_unroll_8(
        TSize const m, TSize const n, TSize const k,
        TElem const alpha,
        TElem const * const A, TSize const lda,
        TElem const * const B, TSize const ldb,
        TElem const beta,
        TElem * const C, TSize const ldc)
    {
        double const timeStart = getTimeSec();

        if(matmul_mat_gemm_early_out(m, n, k, alpha, beta))
        {
            MATMUL_TIME_RETURN_EARLY_OUT;
        }

        MATMUL_TIME_START;

        for(TSize i = 0; i < m; ++i)
        {
            for(TSize j = 0; j < n; ++j)
            {
                C[i*ldc + j] *= beta;

                TSize k2;
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

        MATMUL_TIME_END;
        MATMUL_TIME_RETURN;
    }

    //-----------------------------------------------------------------------------
    //
    //-----------------------------------------------------------------------------
    TReturn matmul_gemm_seq_loop_unroll_16(
        TSize const m, TSize const n, TSize const k,
        TElem const alpha,
        TElem const * const A, TSize const lda,
        TElem const * const B, TSize const ldb,
        TElem const beta,
        TElem * const C, TSize const ldc)
    {
        double const timeStart = getTimeSec();

        if(matmul_mat_gemm_early_out(m, n, k, alpha, beta))
        {
            MATMUL_TIME_RETURN_EARLY_OUT;
        }

        MATMUL_TIME_START;

        for(TSize i = 0; i < m; ++i)
        {
            for(TSize j = 0; j < n; ++j)
            {
                C[i*ldc + j] *= beta;

                TSize k2 = 0;
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

        MATMUL_TIME_END;
        MATMUL_TIME_RETURN;
    }

    //-----------------------------------------------------------------------------
    //
    //-----------------------------------------------------------------------------
    TReturn matmul_gemm_seq_block(
        TSize const m, TSize const n, TSize const k,
        TElem const alpha,
        TElem const * const A, TSize const lda,
        TElem const * const B, TSize const ldb,
        TElem const beta,
        TElem * const C, TSize const ldc)
    {
        if(matmul_mat_gemm_early_out(m, n, k, alpha, beta))
        {
            MATMUL_TIME_RETURN_EARLY_OUT;
        }

        MATMUL_TIME_START;

        for(TSize i = 0; i < m; ++i)
        {
            for(TSize j = 0; j < n; ++j)
            {
                C[i*ldc + j] *= beta;
            }
        }

        TSize const S = MATMUL_SEQ_BLOCK_FACTOR;

        for(TSize ii = 0; ii<m; ii += S)
        {
            TSize const iiS = ii+S;
            for(TSize jj = 0; jj<n; jj += S)
            {
                TSize const jjS = jj+S;
                for(TSize kk = 0; kk<k; kk += S)
                {
                    TSize const kkS = kk+S;
                    TSize const upperBoundi = (iiS>m ? m : iiS);
                    for(TSize i = ii; i<upperBoundi; ++i)
                    {
                        TSize const upperBoundj = (jjS>n ? n : jjS);
                        for(TSize j = jj; j<upperBoundj; ++j)
                        {
                            TSize const upperBoundk = (kkS>k ? k : kkS);
                            for(TSize k2 = kk; k2<upperBoundk; ++k2)
                            {
                                C[i*ldc + j] += alpha * A[i*lda + k2] * B[k2*ldb + j];
                            }
                        }
                    }
                }
            }
        }

        MATMUL_TIME_END;
        MATMUL_TIME_RETURN;
    }
#endif
