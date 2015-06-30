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

#ifdef MATMUL_BUILD_PAR_OPENACC

    #include <matmul/par/OpenAcc.h>

    #include <matmul/common/Mat.h>  // matmul_mat_gemm_early_out

    #include <openacc.h>

    //-----------------------------------------------------------------------------
    //
    //-----------------------------------------------------------------------------
    void matmul_gemm_par_openacc_kernels(
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

#pragma acc kernels copyin(A[0:(lda*m)], B[0:(ldb*k)]) copy(C[0:(ldc*m)])
        {
#pragma acc loop independent gang(MATMUL_OPENACC_GANG_SIZE)
            for(TIdx i = 0; i < m; ++i)
            {
#pragma acc loop independent vector(MATMUL_OPENACC_VECTOR_SIZE)
                for(TIdx j = 0; j < n; ++j)
                {
                    TElem ctmp = 0;
#pragma acc loop seq//reduction(+:ctmp) // Reduction here is much slower then sequential execution!
                    for(TIdx k2 = 0; k2 < k; ++k2)
                    {
                        ctmp += alpha * A[i*lda + k2] * B[k2*ldb +j];
                    }
                    C[i*ldc + j] += C[i*ldc + j] * beta + ctmp;
                }
            }
        }
    }

    //-----------------------------------------------------------------------------
    //
    //-----------------------------------------------------------------------------
    void matmul_gemm_par_openacc_parallel(
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

#pragma acc parallel copyin(A[0:(lda*m)], B[0:(ldb*k)]) copy(C[0:(ldc*m)])
        {
#pragma acc loop
            for(TIdx i = 0; i < m; ++i)
            {
#pragma acc loop
                for(TIdx j = 0; j < n; ++j)
                {
                    C[i*ldc + j] *= beta;
#pragma acc loop seq // Reduction here is much slower then sequential execution!
                    for(TIdx k2 = 0; k2 < k; ++k2)
                    {
                        C[i*ldc + j] += alpha * A[i*lda + k2] * B[k2*ldb + j];
                    }
                }
            }
        }
    }

#endif
