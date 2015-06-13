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

#ifdef MATMUL_BUILD_PAR_BLAS_MKL

    #include <matmul/par/BlasMkl.h>

    #include <matmul/common/Mat.h>                          // matmul_mat_gemm_early_out

    #define MKL_ILP64

    #include <mkl.h>                                        // mkl_mic_enable
    #include <mkl_types.h>
    #include <mkl_cblas.h>

    #ifdef _MSC_VER
        // When compiling with visual studio the msvc open mp libs are used by default. The mkl routines are linked with the intel openmp libs.
        #pragma comment(linker,"/NODEFAULTLIB:VCOMPD.lib" ) // So we have to remove the msv default ones ...
        #pragma comment(linker,"/NODEFAULTLIB:VCOMP.lib")
        #pragma comment(lib,"libiomp5md.lib")               // ... and add the intel libs.

        #pragma comment(lib,"mkl_blas95_ilp64.lib")
        #pragma comment(lib,"mkl_intel_ilp64.lib")
        #pragma comment(lib,"mkl_intel_thread.lib")
        #pragma comment(lib,"mkl_core.lib")
    #endif
    //-----------------------------------------------------------------------------
    //
    //-----------------------------------------------------------------------------
    void matmul_gemm_par_blas_mkl(
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

        CBLAS_ORDER const order = CblasRowMajor;
        CBLAS_TRANSPOSE const transA = CblasNoTrans;
        CBLAS_TRANSPOSE const transB = CblasNoTrans;
        MKL_INT const m_ = (MKL_INT)m;
        MKL_INT const n_ = (MKL_INT)n;
        MKL_INT const k_ = (MKL_INT)k;
        MKL_INT const lda_ = (MKL_INT)lda;
        MKL_INT const ldb_ = (MKL_INT)ldb;
        MKL_INT const ldc_ = (MKL_INT)ldc;

        // Disable automatic MKL offloading to Xeon Phi.
        mkl_mic_disable();

        #ifdef MATMUL_ELEMENT_TYPE_DOUBLE
            cblas_dgemm(
                order,
                transA, transB,
                m_, n_, k_,
                alpha, A, lda_, B, ldb_,    // C = alpha * A * B
                beta, C, ldc_);            // + beta * C
        #else
            cblas_sgemm(
                order,
                transA, transB,
                m_, n_, k_,
                alpha, A, lda_, B, ldb_,    // C = alpha * A * B
                beta, C, ldc_);            // + beta * C
        #endif
    }
#endif
