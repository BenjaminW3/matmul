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

#ifdef MATMUL_BUILD_PAR_BLAS_MKL

    #include <matmul/par/BlasMkl.h>

    #include <matmul/common/Mat.h>                          // matmul_mat_gemm_early_out

    #define MKL_ILP64

    #include <mkl.h>                                        // mkl_mic_enable
    #include <mkl_types.h>
    #include <mkl_cblas.h>

    #ifdef _MSC_VER
        // When compiling with visual studio the msvc open mp libs are used by default. The mkl routines are linked with the intel OpenMP libs.
        #pragma comment(linker,"/NODEFAULTLIB:VCOMPD.lib" ) // So we have to remove the msvc default ones ...
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
    TReturn matmul_gemm_par_blas_mkl(
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
        
        MATMUL_TIME_START;

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

        MATMUL_TIME_END;
        MATMUL_TIME_RETURN;
    }
#endif
