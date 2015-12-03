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

#ifdef MATMUL_BUILD_PAR_PHI_OFF_BLAS_MKL

    #include <matmul/par/PhiOffBlasMkl.h>

    #include <matmul/common/Mat.h>                          // matmul_mat_gemm_early_out

    #define MKL_ILP64

    #include <stdio.h>                                      // printf
    #include <mkl.h>                                        // mkl_mic_enable
    #include <mkl_types.h>
    #include <mkl_cblas.h>

    #ifdef _MSC_VER
        // When compiling with visual studio the msvc open mp libs are used by default. The mkl routines are linked with the intel OpenMP libs.
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
    TReturn matmul_gemm_par_phi_off_blas_mkl(
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

        CBLAS_ORDER const order = CblasRowMajor;
        CBLAS_TRANSPOSE const transA = CblasNoTrans;
        CBLAS_TRANSPOSE const transB = CblasNoTrans;
        MKL_INT const m_ = (MKL_INT)m;
        MKL_INT const n_ = (MKL_INT)n;
        MKL_INT const k_ = (MKL_INT)k;
        MKL_INT const lda_ = (MKL_INT)lda;
        MKL_INT const ldb_ = (MKL_INT)ldb;
        MKL_INT const ldc_ = (MKL_INT)ldc;

        // Enable automatic MKL offloading.
        int iError = mkl_mic_enable();
        if(iError==0)
        {
            #ifdef MATMUL_PHI_OFF_BLAS_MKL_AUTO_WORKDIVISION
                mkl_mic_set_workdivision(MKL_TARGET_HOST, 0, MKL_MIC_AUTO_WORKDIVISION);
                mkl_mic_set_workdivision(MKL_TARGET_MIC, 0, MKL_MIC_AUTO_WORKDIVISION);
            #else
                mkl_mic_set_workdivision(MKL_TARGET_HOST, 0, 0.0);
                mkl_mic_set_workdivision(MKL_TARGET_MIC, 0, 1.0);
            #endif

            MATMUL_TIME_START;

            #ifdef MATMUL_ELEMENT_TYPE_DOUBLE
                cblas_dgemm(
                    order,
                    transA, transB,
                    m_, n_, k_,
                    alpha, A, lda_, B, ldb_,    // C = alpha * A * B
                    beta, C, ldc_);             // + beta * C
            #else
                cblas_sgemm(
                    order,
                    transA, transB,
                    m_, n_, k_,
                    alpha, A, lda_, B, ldb_,    // C = alpha * A * B
                    beta, C, ldc_);             // + beta * C
            #endif

            MATMUL_TIME_END;
            MATMUL_TIME_RETURN;
        }
        else
        {
            printf("[GEMM Phi Off MKL] mkl_mic_enable() returned error value %d", iError);
            MATMUL_TIME_RETURN_EARLY_OUT;
        }
    }
#endif
