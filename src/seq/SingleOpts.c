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
    void matmul_gemm_seq_index_pointer(
        TIdx const m, TIdx const n, TIdx const k,
        TElem const alpha,
        TElem const * const A, TIdx const lda,
        TElem const * const B, TIdx const ldb,
        TElem const beta,
        TElem * const C, TIdx const ldc)
    {
        if(matmul_mat_gemm_early_out(m, n, k, alpha, beta))
        {
            return;
        }

        TElem * pCRow = C;
        TElem const * pARow = A;

        for(TIdx i = 0; i < m; ++i, pARow += lda, pCRow += ldc)
        {
            TElem * pC = pCRow;
            TElem const * pBCol = B;

            for(TIdx j = 0; j < n; ++j, ++pC, ++pBCol)
            {
                (*pC) *= beta;

                TElem const * pA = pARow;
                TElem const * pB = pBCol;

                for(TIdx k2 = 0; k2 < k; ++k2, ++pA, pB += ldb)
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
        TIdx const m, TIdx const n, TIdx const k,
        TElem const alpha,
        TElem const * const MATMUL_RESTRICT A, TIdx const lda,
        TElem const * const MATMUL_RESTRICT B, TIdx const ldb,
        TElem const beta,
        TElem * const MATMUL_RESTRICT C, TIdx const ldc)
    {
        if(matmul_mat_gemm_early_out(m, n, k, alpha, beta))
        {
            return;
        }

        for(TIdx i = 0; i < m; ++i)
        {
            for(TIdx j = 0; j < n; ++j)
            {
                C[i*ldc + j] *= beta;

                for(TIdx k2 = 0; k2 < k; ++k2)
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
        TIdx const m, TIdx const n, TIdx const k,
        TElem const alpha,
        TElem const * const A, TIdx const lda,
        TElem const * const B, TIdx const ldb,
        TElem const beta,
        TElem * const C, TIdx const ldc)
    {
        if(matmul_mat_gemm_early_out(m, n, k, alpha, beta))
        {
            return;
        }

        for(TIdx i = 0; i < m; ++i)
        {
            for(TIdx j = 0; j < n; ++j)
            {
                C[i*ldc + j] *= beta;
            }
            for(TIdx k2 = 0; k2 < k; ++k2)
            {
                for(TIdx j = 0; j < n; ++j)
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
        TIdx const m, TIdx const n, TIdx const k,
        TElem const alpha,
        TElem const * const A, TIdx const lda,
        TElem const * const B, TIdx const ldb,
        TElem const beta,
        TElem * const C, TIdx const ldc)
    {
        if(matmul_mat_gemm_early_out(m, n, k, alpha, beta))
        {
            return;
        }

        for(TIdx i = 0; i < m; ++i)
        {
            TIdx const rowBeginIdxA = i*lda;
            TIdx const rowBeginIdxC = i*ldc;

            for(TIdx j = 0; j < n; ++j)
            {
                C[i*ldc + j] *= beta;

                TIdx const idxC = rowBeginIdxC + j;

                for(TIdx k2 = 0; k2 < k; ++k2)
                {
                    C[idxC] += alpha * A[rowBeginIdxA + k2] * B[k2*ldb + j];
                }
            }
        }
    }

    //-----------------------------------------------------------------------------
    //
    //-----------------------------------------------------------------------------
    void matmul_gemm_seq_loop_unroll_4(
        TIdx const m, TIdx const n, TIdx const k,
        TElem const alpha,
        TElem const * const A, TIdx const lda,
        TElem const * const B, TIdx const ldb,
        TElem const beta,
        TElem * const C, TIdx const ldc)
    {
        if(matmul_mat_gemm_early_out(m, n, k, alpha, beta))
        {
            return;
        }

        for(TIdx i = 0; i < m; ++i)
        {
            for(TIdx j = 0; j < n; ++j)
            {
                C[i*ldc + j] *= beta;

                TIdx k2;
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
        TIdx const m, TIdx const n, TIdx const k,
        TElem const alpha,
        TElem const * const A, TIdx const lda,
        TElem const * const B, TIdx const ldb,
        TElem const beta,
        TElem * const C, TIdx const ldc)
    {
        if(matmul_mat_gemm_early_out(m, n, k, alpha, beta))
        {
            return;
        }

        for(TIdx i = 0; i < m; ++i)
        {
            for(TIdx j = 0; j < n; ++j)
            {
                C[i*ldc + j] *= beta;

                TIdx k2;
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
        TIdx const m, TIdx const n, TIdx const k,
        TElem const alpha,
        TElem const * const A, TIdx const lda,
        TElem const * const B, TIdx const ldb,
        TElem const beta,
        TElem * const C, TIdx const ldc)
    {
        if(matmul_mat_gemm_early_out(m, n, k, alpha, beta))
        {
            return;
        }

        for(TIdx i = 0; i < m; ++i)
        {
            for(TIdx j = 0; j < n; ++j)
            {
                C[i*ldc + j] *= beta;

                TIdx k2 = 0;
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
        TIdx const m, TIdx const n, TIdx const k,
        TElem const alpha,
        TElem const * const A, TIdx const lda,
        TElem const * const B, TIdx const ldb,
        TElem const beta,
        TElem * const C, TIdx const ldc)
    {
        if(matmul_mat_gemm_early_out(m, n, k, alpha, beta))
        {
            return;
        }

        for(TIdx i = 0; i < m; ++i)
        {
            for(TIdx j = 0; j < n; ++j)
            {
                C[i*ldc + j] *= beta;
            }
        }

        TIdx const S = MATMUL_SEQ_BLOCK_FACTOR;

        for(TIdx ii = 0; ii<m; ii += S)
        {
            TIdx const iiS = ii+S;
            for(TIdx jj = 0; jj<n; jj += S)
            {
                TIdx const jjS = jj+S;
                for(TIdx kk = 0; kk<k; kk += S)
                {
                    TIdx const kkS = kk+S;
                    TIdx const upperBoundi = (iiS>m ? m : iiS);
                    for(TIdx i = ii; i<upperBoundi; ++i)
                    {
                        TIdx const upperBoundj = (jjS>n ? n : jjS);
                        for(TIdx j = jj; j<upperBoundj; ++j)
                        {
                            TIdx const upperBoundk = (kkS>k ? k : kkS);
                            for(TIdx k2 = kk; k2<upperBoundk; ++k2)
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
