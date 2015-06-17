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


#ifdef MATMUL_BUILD_SEQ_MULTIPLE_OPTS

    #include <matmul/seq/MultipleOpts.h>

    #include <matmul/common/Mat.h>  // matmul_mat_gemm_early_out

    //-----------------------------------------------------------------------------
    //
    //-----------------------------------------------------------------------------
    void matmul_gemm_seq_multiple_opts_no_block(
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

#ifdef MATMUL_MSVC
        for(size_t i = 0; i<m; ++i)
        {
            for(size_t j = 0; j<n; ++j)
            {
                C[i*ldc + j] *= beta;
            }
            size_t const uiRowBeginIdxC = i*ldc;
            size_t const uiRowBeginIdxA = i*lda;

            for(size_t k2 = 0; k2<k; ++k2)
            {
                size_t const uiRowBeginIdxB = k2*ldb;
                TElem const a = A[uiRowBeginIdxA + k2];

                for(size_t j = 0; j<n; ++j)
                {
                    size_t uiIdxC = uiRowBeginIdxC + j;

                    C[uiIdxC] += alpha * a * B[uiRowBeginIdxB + j];
                }
            }
        }
#else
        for(size_t i = 0; i < m; ++i)
        {
            for(size_t j = 0; j<n; ++j)
            {
                C[i*ldc + j] *= beta;
            }
            for(size_t k2 = 0; k2 < k; ++k2)
            {
                for(size_t j = 0; j < n; ++j)
                {
                    C[i*ldc + j] += alpha * A[i*lda + k2] * B[k2*ldb + j];
                }
            }
        }
#endif
    }

    //-----------------------------------------------------------------------------
    //
    //-----------------------------------------------------------------------------
    void matmul_gemm_seq_multiple_opts_block(
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
            }
        }

        size_t const S = MATMUL_SEQ_BLOCK_FACTOR;

        //for(size_t ii = 0; ii<m; ii += S)    // Blocking of outermost loop is not necessary, we only need blocks in 2 dimensions.
        {
            //size_t const iiS = ii+S;
            for(size_t kk = 0; kk<k; kk += S)
            {
                size_t const kkS = kk+S;
                for(size_t jj = 0; jj<n; jj += S)
                {
                    size_t const jjS = jj+S;
                    //size_t const uiUpperBoundi = (iiS>m ? m : iiS);
                    //for(size_t i = ii; i < uiUpperBoundi; ++i)

                    size_t uiRowBeginIdxC = 0;
                    size_t uiRowBeginIdxA = 0;
                    for(size_t i = 0; i<m; ++i)
                    {
                        size_t uiRowBeginIdxB = kk*ldb;
                        size_t const uiUpperBoundk = (kkS>k ? k : kkS);
                        for(size_t k2 = kk; k2<uiUpperBoundk; ++k2)
                        {
                            TElem const a = alpha * A[uiRowBeginIdxA + k2];
                            size_t const uiUpperBoundj = (jjS>n ? n : jjS);
                            for(size_t j = jj; j<uiUpperBoundj; ++j)
                            {
                                size_t uiIdxC = uiRowBeginIdxC + j;
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

    //-----------------------------------------------------------------------------
    //
    //-----------------------------------------------------------------------------
    void matmul_gemm_seq_multiple_opts(
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

#ifdef MATMUL_MSVC
        // MSVC-2013 is better with a handrolled transition between blocked and nonblocked version.
        if(n<=MATMUL_SEQ_COMPLETE_OPT_NO_BLOCK_CUT_OFF)
        {
            matmul_gemm_seq_multiple_opts_no_block(
                m, n, k,
                alpha,
                A, lda,
                B, ldb,
                beta,
                C, ldc);
        }
        else
        {
            matmul_gemm_seq_multiple_opts_block(
                m, n, k,
                alpha,
                A, lda,
                B, ldb,
                beta,
                C, ldc);
        }
#else
        // ICC-14 compiler automatically optimizes the matmul function better then we could reach with a handrolled one (blocked and nonblocked).
        matmul_gemm_seq_multiple_opts_no_block(
            m, n, k,
            alpha,
            A, lda,
            B, ldb,
            beta,
            C, ldc);
#endif
    }
#endif
