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


#if defined(MATMUL_BUILD_SEQ_MULTIPLE_OPTS) || defined(MATMUL_BUILD_SEQ_MULTIPLE_OPTS_BLOCK)

    #include <matmul/seq/MultipleOpts.h>

    #include <matmul/common/Mat.h>  // matmul_mat_gemm_early_out

    #ifdef MATMUL_BUILD_SEQ_MULTIPLE_OPTS
        //-----------------------------------------------------------------------------
        //
        //-----------------------------------------------------------------------------
        void matmul_gemm_seq_multiple_opts(
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

#ifdef MATMUL_MSVC
            for(TIdx i = 0; i<m; ++i)
            {
                for(TIdx j = 0; j<n; ++j)
                {
                    C[i*ldc + j] *= beta;
                }
                TIdx const uiRowBeginIdxC = i*ldc;
                TIdx const uiRowBeginIdxA = i*lda;

                for(TIdx k2 = 0; k2<k; ++k2)
                {
                    TIdx const uiRowBeginIdxB = k2*ldb;
                    TElem const a = A[uiRowBeginIdxA + k2];

                    for(TIdx j = 0; j<n; ++j)
                    {
                        TIdx uiIdxC = uiRowBeginIdxC + j;

                        C[uiIdxC] += alpha * a * B[uiRowBeginIdxB + j];
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
        }
    #endif
    #ifdef MATMUL_BUILD_SEQ_MULTIPLE_OPTS_BLOCK
        //-----------------------------------------------------------------------------
        //
        //-----------------------------------------------------------------------------
        void matmul_gemm_seq_multiple_opts_block(
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
                        //TIdx const uiUpperBoundi = (iiS>m ? m : iiS);
                        //for(TIdx i = ii; i < uiUpperBoundi; ++i)

                        TIdx uiRowBeginIdxC = 0;
                        TIdx uiRowBeginIdxA = 0;
                        for(TIdx i = 0; i<m; ++i)
                        {
                            TIdx uiRowBeginIdxB = kk*ldb;
                            TIdx const uiUpperBoundk = (kkS>k ? k : kkS);
                            for(TIdx k2 = kk; k2<uiUpperBoundk; ++k2)
                            {
                                TElem const a = alpha * A[uiRowBeginIdxA + k2];
                                TIdx const uiUpperBoundj = (jjS>n ? n : jjS);
                                for(TIdx j = jj; j<uiUpperBoundj; ++j)
                                {
                                    TIdx uiIdxC = uiRowBeginIdxC + j;
                                    C[uiIdxC] += a * B[uiRowBeginIdxB + j];
                                }
                                uiRowBeginIdxB += ldb;
                            }
                            uiRowBeginIdxC += ldc;
                            uiRowBeginIdxA += lda;
                        }
                    }
                }
            }
        }
    #endif
#endif
