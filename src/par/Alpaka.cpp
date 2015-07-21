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

#if defined(MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_B_SEQ_T_SEQ) || defined(MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_B_OMP2_T_SEQ) || defined(MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_B_SEQ_T_OMP2) || defined(MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_BT_OMP4) || defined(MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_B_SEQ_T_THREADS) || defined(MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_B_SEQ_T_FIBERS)

    #include <matmul/par/Alpaka.h>

    #include <matmul/par/Alpaka.hpp>

    #ifdef MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_B_SEQ_T_SEQ
        //-----------------------------------------------------------------------------
        //
        //-----------------------------------------------------------------------------
        void matmul_gemm_par_alpaka_cpu_b_seq_t_seq(
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

            matmul_gemm_par_alpaka<alpaka::acc::AccCpuSerial<alpaka::dim::DimInt<2u>, TIdx>>(
                m, n, k,
                alpha,
                A, lda,
                B, ldb,
                beta,
                C, ldc);
        }
    #endif
    #ifdef MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_B_OMP2_T_SEQ
        //-----------------------------------------------------------------------------
        //
        //-----------------------------------------------------------------------------
        void matmul_gemm_par_alpaka_cpu_b_omp2_t_seq(
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

            matmul_gemm_par_alpaka<alpaka::acc::AccCpuOmp2Blocks<alpaka::dim::DimInt<2u>, TIdx>>(
                m, n, k,
                alpha,
                A, lda,
                B, ldb,
                beta,
                C, ldc);
        }
    #endif
    #ifdef MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_B_SEQ_T_OMP2
        //-----------------------------------------------------------------------------
        //
        //-----------------------------------------------------------------------------
        void matmul_gemm_par_alpaka_cpu_b_seq_t_omp2(
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

            matmul_gemm_par_alpaka<alpaka::acc::AccCpuOmp2Threads<alpaka::dim::DimInt<2u>, TIdx>>(
                m, n, k,
                alpha,
                A, lda,
                B, ldb,
                beta,
                C, ldc);
        }
    #endif
    #ifdef MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_BT_OMP4
        //-----------------------------------------------------------------------------
        //
        //-----------------------------------------------------------------------------
        void matmul_gemm_par_alpaka_cpu_bt_omp4(
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

            matmul_gemm_par_alpaka<alpaka::acc::AccCpuOmp4<alpaka::dim::DimInt<2u>, TIdx>>(
                m, n, k,
                alpha,
                A, lda,
                B, ldb,
                beta,
                C, ldc);
        }
    #endif
    #ifdef MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_B_SEQ_T_THREADS
        //-----------------------------------------------------------------------------
        //
        //-----------------------------------------------------------------------------
        void matmul_gemm_par_alpaka_cpu_b_seq_t_threads(
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

            matmul_gemm_par_alpaka<alpaka::acc::AccCpuThreads<alpaka::dim::DimInt<2u>, TIdx>>(
                m, n, k,
                alpha,
                A, lda,
                B, ldb,
                beta,
                C, ldc);
        }
    #endif
    #ifdef MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_B_SEQ_T_FIBERS
        //-----------------------------------------------------------------------------
        //
        //-----------------------------------------------------------------------------
        void matmul_gemm_par_alpaka_cpu_b_seq_t_fibers(
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

            matmul_gemm_par_alpaka<alpaka::acc::AccCpuFibers<alpaka::dim::DimInt<2u>, TIdx>>(
                m, n, k,
                alpha,
                A, lda,
                B, ldb,
                beta,
                C, ldc);
        }
    #endif
#endif
