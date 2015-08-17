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

#pragma once

#if defined(MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_B_SEQ_T_SEQ) || defined(MATMUL_BUILD_PAR_ALPAKA_ACC_GPU_CUDA) || defined(MATMUL_BUILD_PAR_ALPAKA_ACC_GPU_CUDA_MEMCPY) || defined(MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_B_OMP2_T_SEQ) || defined(MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_B_SEQ_T_OMP2) || defined(MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_BT_OMP4) || defined(MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_B_SEQ_T_THREADS) || defined(MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_B_SEQ_T_FIBERS)

    #include <matmul/common/Config.h>   // TElem, TIdx, TReturn

    #ifdef __cplusplus
        extern "C"
        {
    #endif
    #ifdef MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_B_SEQ_T_SEQ
        //-----------------------------------------------------------------------------
        //! (S/D)GEMM matrix-matrix product C = alpha * A * B + beta * C using alpaka`s serial accelerator back-end.
        //!
        //! \param m Specifies the number of rows of the matrix A and of the matrix C.
        //! \param n Specifies the number of columns of the matrix B and the number of columns of the matrix C.
        //! \param k Specifies the number of columns of the matrix A and the number of rows of the matrix B.
        //! \param alpha Scalar value used to scale the product of matrices A and B.
        //! \param A Array, size lda-by-k. The leading m-by-k part of the array must contain the matrix A.
        //! \param lda Specifies the leading dimension of A.
        //! \param B Array, size ldb-by-n. The leading k-by-n part of the array must contain the matrix B.
        //! \param ldb Specifies the leading dimension of B.
        //! \param beta Scalar value used to scale matrix C.
        //! \param C Array, size ldc-by-n. The leading m-by-n part of the array must contain the matrix C.
        //! \param ldc Specifies the leading dimension of C.
        //-----------------------------------------------------------------------------
        TReturn matmul_gemm_seq_alpaka_cpu_b_seq_t_seq(
            TIdx const m, TIdx const n, TIdx const k,
            TElem const alpha,
            TElem const * const MATMUL_RESTRICT A, TIdx const lda,
            TElem const * const MATMUL_RESTRICT B, TIdx const ldb,
            TElem const beta,
            TElem * const MATMUL_RESTRICT C, TIdx const ldc);
    #endif
    #ifdef MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_B_OMP2_T_SEQ
        //-----------------------------------------------------------------------------
        //! (S/D)GEMM matrix-matrix product C = alpha * A * B + beta * C using alpaka`s OpenMP 2.0 block accelerator back-end.
        //!
        //! \param m Specifies the number of rows of the matrix A and of the matrix C.
        //! \param n Specifies the number of columns of the matrix B and the number of columns of the matrix C.
        //! \param k Specifies the number of columns of the matrix A and the number of rows of the matrix B.
        //! \param alpha Scalar value used to scale the product of matrices A and B.
        //! \param A Array, size lda-by-k. The leading m-by-k part of the array must contain the matrix A.
        //! \param lda Specifies the leading dimension of A.
        //! \param B Array, size ldb-by-n. The leading k-by-n part of the array must contain the matrix B.
        //! \param ldb Specifies the leading dimension of B.
        //! \param beta Scalar value used to scale matrix C.
        //! \param C Array, size ldc-by-n. The leading m-by-n part of the array must contain the matrix C.
        //! \param ldc Specifies the leading dimension of C.
        //-----------------------------------------------------------------------------
        TReturn matmul_gemm_par_alpaka_cpu_b_omp2_t_seq(
            TIdx const m, TIdx const n, TIdx const k,
            TElem const alpha,
            TElem const * const MATMUL_RESTRICT A, TIdx const lda,
            TElem const * const MATMUL_RESTRICT B, TIdx const ldb,
            TElem const beta,
            TElem * const MATMUL_RESTRICT C, TIdx const ldc);
    #endif
    #ifdef MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_B_SEQ_T_OMP2
        //-----------------------------------------------------------------------------
        //! (S/D)GEMM matrix-matrix product C = alpha * A * B + beta * C using alpaka`s OpenMP 2.0 thread accelerator back-end.
        //!
        //! \param m Specifies the number of rows of the matrix A and of the matrix C.
        //! \param n Specifies the number of columns of the matrix B and the number of columns of the matrix C.
        //! \param k Specifies the number of columns of the matrix A and the number of rows of the matrix B.
        //! \param alpha Scalar value used to scale the product of matrices A and B.
        //! \param A Array, size lda-by-k. The leading m-by-k part of the array must contain the matrix A.
        //! \param lda Specifies the leading dimension of A.
        //! \param B Array, size ldb-by-n. The leading k-by-n part of the array must contain the matrix B.
        //! \param ldb Specifies the leading dimension of B.
        //! \param beta Scalar value used to scale matrix C.
        //! \param C Array, size ldc-by-n. The leading m-by-n part of the array must contain the matrix C.
        //! \param ldc Specifies the leading dimension of C.
        //-----------------------------------------------------------------------------
        TReturn matmul_gemm_par_alpaka_cpu_b_seq_t_omp2(
            TIdx const m, TIdx const n, TIdx const k,
            TElem const alpha,
            TElem const * const MATMUL_RESTRICT A, TIdx const lda,
            TElem const * const MATMUL_RESTRICT B, TIdx const ldb,
            TElem const beta,
            TElem * const MATMUL_RESTRICT C, TIdx const ldc);
    #endif
    #ifdef MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_BT_OMP4
        //-----------------------------------------------------------------------------
        //! (S/D)GEMM matrix-matrix product C = alpha * A * B + beta * C using alpaka`s OpenMP 4.0 accelerator back-end.
        //!
        //! \param m Specifies the number of rows of the matrix A and of the matrix C.
        //! \param n Specifies the number of columns of the matrix B and the number of columns of the matrix C.
        //! \param k Specifies the number of columns of the matrix A and the number of rows of the matrix B.
        //! \param alpha Scalar value used to scale the product of matrices A and B.
        //! \param A Array, size lda-by-k. The leading m-by-k part of the array must contain the matrix A.
        //! \param lda Specifies the leading dimension of A.
        //! \param B Array, size ldb-by-n. The leading k-by-n part of the array must contain the matrix B.
        //! \param ldb Specifies the leading dimension of B.
        //! \param beta Scalar value used to scale matrix C.
        //! \param C Array, size ldc-by-n. The leading m-by-n part of the array must contain the matrix C.
        //! \param ldc Specifies the leading dimension of C.
        //-----------------------------------------------------------------------------
        TReturn matmul_gemm_par_alpaka_cpu_bt_omp4(
            TIdx const m, TIdx const n, TIdx const k,
            TElem const alpha,
            TElem const * const MATMUL_RESTRICT A, TIdx const lda,
            TElem const * const MATMUL_RESTRICT B, TIdx const ldb,
            TElem const beta,
            TElem * const MATMUL_RESTRICT C, TIdx const ldc);
    #endif
    #ifdef MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_B_SEQ_T_THREADS
        //-----------------------------------------------------------------------------
        //! (S/D)GEMM matrix-matrix product C = alpha * A * B + beta * C using alpaka`s std::thread accelerator back-end.
        //!
        //! \param m Specifies the number of rows of the matrix A and of the matrix C.
        //! \param n Specifies the number of columns of the matrix B and the number of columns of the matrix C.
        //! \param k Specifies the number of columns of the matrix A and the number of rows of the matrix B.
        //! \param alpha Scalar value used to scale the product of matrices A and B.
        //! \param A Array, size lda-by-k. The leading m-by-k part of the array must contain the matrix A.
        //! \param lda Specifies the leading dimension of A.
        //! \param B Array, size ldb-by-n. The leading k-by-n part of the array must contain the matrix B.
        //! \param ldb Specifies the leading dimension of B.
        //! \param beta Scalar value used to scale matrix C.
        //! \param C Array, size ldc-by-n. The leading m-by-n part of the array must contain the matrix C.
        //! \param ldc Specifies the leading dimension of C.
        //-----------------------------------------------------------------------------
        TReturn matmul_gemm_par_alpaka_cpu_b_seq_t_threads(
            TIdx const m, TIdx const n, TIdx const k,
            TElem const alpha,
            TElem const * const MATMUL_RESTRICT A, TIdx const lda,
            TElem const * const MATMUL_RESTRICT B, TIdx const ldb,
            TElem const beta,
            TElem * const MATMUL_RESTRICT C, TIdx const ldc);
    #endif
    #ifdef MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_B_SEQ_T_FIBERS
        //-----------------------------------------------------------------------------
        //! (S/D)GEMM matrix-matrix product C = alpha * A * B + beta * C using alpaka`s boost::fiber accelerator back-end.
        //!
        //! \param m Specifies the number of rows of the matrix A and of the matrix C.
        //! \param n Specifies the number of columns of the matrix B and the number of columns of the matrix C.
        //! \param k Specifies the number of columns of the matrix A and the number of rows of the matrix B.
        //! \param alpha Scalar value used to scale the product of matrices A and B.
        //! \param A Array, size lda-by-k. The leading m-by-k part of the array must contain the matrix A.
        //! \param lda Specifies the leading dimension of A.
        //! \param B Array, size ldb-by-n. The leading k-by-n part of the array must contain the matrix B.
        //! \param ldb Specifies the leading dimension of B.
        //! \param beta Scalar value used to scale matrix C.
        //! \param C Array, size ldc-by-n. The leading m-by-n part of the array must contain the matrix C.
        //! \param ldc Specifies the leading dimension of C.
        //-----------------------------------------------------------------------------
        TReturn matmul_gemm_seq_alpaka_cpu_b_seq_t_fibers(
            TIdx const m, TIdx const n, TIdx const k,
            TElem const alpha,
            TElem const * const MATMUL_RESTRICT A, TIdx const lda,
            TElem const * const MATMUL_RESTRICT B, TIdx const ldb,
            TElem const beta,
            TElem * const MATMUL_RESTRICT C, TIdx const ldc);
    #endif
    #ifdef MATMUL_BUILD_PAR_ALPAKA_ACC_GPU_CUDA
        //-----------------------------------------------------------------------------
        //! (S/D)GEMM matrix-matrix product C = alpha * A * B + beta * C using alpaka`s CUDA accelerator back-end.
        //!
        //! \param m Specifies the number of rows of the matrix A and of the matrix C.
        //! \param n Specifies the number of columns of the matrix B and the number of columns of the matrix C.
        //! \param k Specifies the number of columns of the matrix A and the number of rows of the matrix B.
        //! \param alpha Scalar value used to scale the product of matrices A and B.
        //! \param A Array, size lda-by-k. The leading m-by-k part of the array must contain the matrix A.
        //! \param lda Specifies the leading dimension of A.
        //! \param B Array, size ldb-by-n. The leading k-by-n part of the array must contain the matrix B.
        //! \param ldb Specifies the leading dimension of B.
        //! \param beta Scalar value used to scale matrix C.
        //! \param C Array, size ldc-by-n. The leading m-by-n part of the array must contain the matrix C.
        //! \param ldc Specifies the leading dimension of C.
        //-----------------------------------------------------------------------------
        TReturn matmul_gemm_par_alpaka_gpu_cuda(
            TIdx const m, TIdx const n, TIdx const k,
            TElem const alpha,
            TElem const * const MATMUL_RESTRICT A, TIdx const lda,
            TElem const * const MATMUL_RESTRICT B, TIdx const ldb,
            TElem const beta,
            TElem * const MATMUL_RESTRICT C, TIdx const ldc);
    #endif
    #ifdef MATMUL_BUILD_PAR_ALPAKA_ACC_GPU_CUDA_MEMCPY
        //-----------------------------------------------------------------------------
        //! (S/D)GEMM matrix-matrix product C = alpha * A * B + beta * C using alpaka`s CUDA accelerator back-end.
        //!
        //! \param m Specifies the number of rows of the matrix A and of the matrix C.
        //! \param n Specifies the number of columns of the matrix B and the number of columns of the matrix C.
        //! \param k Specifies the number of columns of the matrix A and the number of rows of the matrix B.
        //! \param alpha Scalar value used to scale the product of matrices A and B.
        //! \param A Array, size lda-by-k. The leading m-by-k part of the array must contain the matrix A.
        //! \param lda Specifies the leading dimension of A.
        //! \param B Array, size ldb-by-n. The leading k-by-n part of the array must contain the matrix B.
        //! \param ldb Specifies the leading dimension of B.
        //! \param beta Scalar value used to scale matrix C.
        //! \param C Array, size ldc-by-n. The leading m-by-n part of the array must contain the matrix C.
        //! \param ldc Specifies the leading dimension of C.
        //-----------------------------------------------------------------------------
        TReturn matmul_gemm_par_alpaka_gpu_cuda_memcpy(
            TIdx const m, TIdx const n, TIdx const k,
            TElem const alpha,
            TElem const * const MATMUL_RESTRICT A, TIdx const lda,
            TElem const * const MATMUL_RESTRICT B, TIdx const ldb,
            TElem const beta,
            TElem * const MATMUL_RESTRICT C, TIdx const ldc);
    #endif
    #ifdef __cplusplus
        }
    #endif
#endif
