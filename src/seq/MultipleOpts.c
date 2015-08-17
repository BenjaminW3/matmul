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


#if defined(MATMUL_BUILD_SEQ_MULTIPLE_OPTS) || defined(MATMUL_BUILD_SEQ_MULTIPLE_OPTS_BLOCK)

    #include <matmul/seq/MultipleOpts.h>

    #include <matmul/common/Mat.h>  // matmul_mat_gemm_early_out

    #ifdef MATMUL_BUILD_SEQ_MULTIPLE_OPTS
        //-----------------------------------------------------------------------------
        //
        //-----------------------------------------------------------------------------
        TReturn matmul_gemm_seq_multiple_opts(
            TIdx const m, TIdx const n, TIdx const k,
            TElem const alpha,
            TElem const * const MATMUL_RESTRICT A, TIdx const lda,
            TElem const * const MATMUL_RESTRICT B, TIdx const ldb,
            TElem const beta,
            TElem * const MATMUL_RESTRICT C, TIdx const ldc)
        {
            if(matmul_mat_gemm_early_out(m, n, k, alpha, beta))
            {
                MATMUL_TIME_RETURN_EARLY_OUT;
            }

            MATMUL_TIME_START;

#ifdef MATMUL_MSVC
            for(TIdx i = 0; i<m; ++i)
            {
                for(TIdx j = 0; j<n; ++j)
                {
                    C[i*ldc + j] *= beta;
                }
                TIdx const rowBeginIdxC = i*ldc;
                TIdx const rowBeginIdxA = i*lda;

                for(TIdx k2 = 0; k2<k; ++k2)
                {
                    TIdx const rowBeginIdxB = k2*ldb;
                    TElem const a = A[rowBeginIdxA + k2];

                    for(TIdx j = 0; j<n; ++j)
                    {
                        TIdx idxC = rowBeginIdxC + j;

                        C[idxC] += alpha * a * B[rowBeginIdxB + j];
                    }
                }
            }
#else
            for(TIdx i = 0; i < m; ++i)
            {
                for(TIdx j = 0; j<n; ++j)
                {
                    C[i*ldc + j] *= beta;
                }
                for(TIdx k2 = 0; k2 < k; ++k2)
                {
                    for(TIdx j = 0; j < n; ++j)
                    {
                        C[i*ldc + j] += alpha * A[i*lda + k2] * B[k2*ldb + j];
                    }
                }
            }
#endif

            MATMUL_TIME_END;
            MATMUL_TIME_RETURN;
        }
    #endif
    #ifdef MATMUL_BUILD_SEQ_MULTIPLE_OPTS_BLOCK
        //-----------------------------------------------------------------------------
        //
        //-----------------------------------------------------------------------------
        TReturn matmul_gemm_seq_multiple_opts_block(
            TIdx const m, TIdx const n, TIdx const k,
            TElem const alpha,
            TElem const * const MATMUL_RESTRICT A, TIdx const lda,
            TElem const * const MATMUL_RESTRICT B, TIdx const ldb,
            TElem const beta,
            TElem * const MATMUL_RESTRICT C, TIdx const ldc)
        {
            if(matmul_mat_gemm_early_out(m, n, k, alpha, beta))
            {
                MATMUL_TIME_RETURN_EARLY_OUT;
            }

            MATMUL_TIME_START;

            for(TIdx i = 0; i < m; ++i)
            {
                for(TIdx j = 0; j < n; ++j)
                {
                    C[i*ldc + j] *= beta;
                }
            }

            TIdx const S = MATMUL_SEQ_BLOCK_FACTOR;

            //for(TIdx ii = 0; ii<m; ii += S)    // Blocking of outermost loop is not necessary, we only need blocks in 2 dimensions.
            {
                //TIdx const iiS = ii+S;
                for(TIdx kk = 0; kk<k; kk += S)
                {
                    TIdx const kkS = kk+S;
                    for(TIdx jj = 0; jj<n; jj += S)
                    {
                        TIdx const jjS = jj+S;
                        //TIdx const upperBoundi = (iiS>m ? m : iiS);
                        //for(TIdx i = ii; i < upperBoundi; ++i)

                        TIdx rowBeginIdxC = 0;
                        TIdx rowBeginIdxA = 0;
                        for(TIdx i = 0; i<m; ++i)
                        {
                            TIdx rowBeginIdxB = kk*ldb;
                            TIdx const upperBoundk = (kkS>k ? k : kkS);
                            for(TIdx k2 = kk; k2<upperBoundk; ++k2)
                            {
                                TElem const a = alpha * A[rowBeginIdxA + k2];
                                TIdx const upperBoundj = (jjS>n ? n : jjS);
                                for(TIdx j = jj; j<upperBoundj; ++j)
                                {
                                    TIdx idxC = rowBeginIdxC + j;
                                    C[idxC] += a * B[rowBeginIdxB + j];
                                }
                                rowBeginIdxB += ldb;
                            }
                            rowBeginIdxC += ldc;
                            rowBeginIdxA += lda;
                        }
                    }
                }
            }

            MATMUL_TIME_END;
            MATMUL_TIME_RETURN;
        }
    #endif
#endif
