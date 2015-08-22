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
            TSize const m, TSize const n, TSize const k,
            TElem const alpha,
            TElem const * const MATMUL_RESTRICT A, TSize const lda,
            TElem const * const MATMUL_RESTRICT B, TSize const ldb,
            TElem const beta,
            TElem * const MATMUL_RESTRICT C, TSize const ldc)
        {
            if(matmul_mat_gemm_early_out(m, n, k, alpha, beta))
            {
                MATMUL_TIME_RETURN_EARLY_OUT;
            }

            MATMUL_TIME_START;

#ifdef _MSC_VER
            for(TSize i = 0; i<m; ++i)
            {
                for(TSize j = 0; j<n; ++j)
                {
                    C[i*ldc + j] *= beta;
                }
                TSize const rowBeginIdxC = i*ldc;
                TSize const rowBeginIdxA = i*lda;

                for(TSize k2 = 0; k2<k; ++k2)
                {
                    TSize const rowBeginIdxB = k2*ldb;
                    TElem const a = A[rowBeginIdxA + k2];

                    for(TSize j = 0; j<n; ++j)
                    {
                        TSize idxC = rowBeginIdxC + j;

                        C[idxC] += alpha * a * B[rowBeginIdxB + j];
                    }
                }
            }
#else
            for(TSize i = 0; i < m; ++i)
            {
                for(TSize j = 0; j<n; ++j)
                {
                    C[i*ldc + j] *= beta;
                }
                for(TSize k2 = 0; k2 < k; ++k2)
                {
                    for(TSize j = 0; j < n; ++j)
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
            TSize const m, TSize const n, TSize const k,
            TElem const alpha,
            TElem const * const MATMUL_RESTRICT A, TSize const lda,
            TElem const * const MATMUL_RESTRICT B, TSize const ldb,
            TElem const beta,
            TElem * const MATMUL_RESTRICT C, TSize const ldc)
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

            //for(TSize ii = 0; ii<m; ii += S)    // Blocking of outermost loop is not necessary, we only need blocks in 2 dimensions.
            {
                //TSize const iiS = ii+S;
                for(TSize kk = 0; kk<k; kk += S)
                {
                    TSize const kkS = kk+S;
                    for(TSize jj = 0; jj<n; jj += S)
                    {
                        TSize const jjS = jj+S;
                        //TSize const upperBoundi = (iiS>m ? m : iiS);
                        //for(TSize i = ii; i < upperBoundi; ++i)

                        TSize rowBeginIdxC = 0;
                        TSize rowBeginIdxA = 0;
                        for(TSize i = 0; i<m; ++i)
                        {
                            TSize rowBeginIdxB = kk*ldb;
                            TSize const upperBoundk = (kkS>k ? k : kkS);
                            for(TSize k2 = kk; k2<upperBoundk; ++k2)
                            {
                                TElem const a = alpha * A[rowBeginIdxA + k2];
                                TSize const upperBoundj = (jjS>n ? n : jjS);
                                for(TSize j = jj; j<upperBoundj; ++j)
                                {
                                    TSize idxC = rowBeginIdxC + j;
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
