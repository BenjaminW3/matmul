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

#if defined(MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_B_SEQ_T_SEQ) || defined(MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_B_OMP2_T_SEQ) || defined(MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_B_SEQ_T_OMP2) || defined(MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_BT_OMP4) || defined(MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_B_SEQ_T_THREADS) || defined(MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_B_SEQ_T_FIBERS)

    #include <matmul/par/Alpaka.h>

    #include <matmul/par/Alpaka.hpp>

    #ifdef MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_B_SEQ_T_SEQ
        //-----------------------------------------------------------------------------
        //
        //-----------------------------------------------------------------------------
        TReturn matmul_gemm_seq_alpaka_cpu_b_seq_t_seq(
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

            return
                matmul_gemm_par_alpaka<alpaka::acc::AccCpuSerial<alpaka::dim::DimInt<2u>, TSize>, GemmAlpakaNoShared2Kernel>(
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
        TReturn matmul_gemm_par_alpaka_cpu_b_omp2_t_seq(
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

            return
                matmul_gemm_par_alpaka<alpaka::acc::AccCpuOmp2Blocks<alpaka::dim::DimInt<2u>, TSize>, GemmAlpakaNoShared2Kernel>(
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
        TReturn matmul_gemm_par_alpaka_cpu_b_seq_t_omp2(
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

            return
                matmul_gemm_par_alpaka<alpaka::acc::AccCpuOmp2Threads<alpaka::dim::DimInt<2u>, TSize>, GemmAlpakaNoShared2Kernel>(
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
        TReturn matmul_gemm_par_alpaka_cpu_bt_omp4(
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

            return
                matmul_gemm_par_alpaka<alpaka::acc::AccCpuOmp4<alpaka::dim::DimInt<2u>, TSize>, GemmAlpakaNoShared2Kernel>(
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
        TReturn matmul_gemm_par_alpaka_cpu_b_seq_t_threads(
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

            return
                matmul_gemm_par_alpaka<alpaka::acc::AccCpuThreads<alpaka::dim::DimInt<2u>, TSize>, GemmAlpakaNoShared2Kernel>(
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
        TReturn matmul_gemm_seq_alpaka_cpu_b_seq_t_fibers(
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

            return
                matmul_gemm_par_alpaka<alpaka::acc::AccCpuFibers<alpaka::dim::DimInt<2u>, TSize>, GemmAlpakaNoShared2Kernel>(
                    m, n, k,
                    alpha,
                    A, lda,
                    B, ldb,
                    beta,
                    C, ldc);
        }
    #endif
#endif
