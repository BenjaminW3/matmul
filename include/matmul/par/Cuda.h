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

#if defined(MATMUL_BUILD_PAR_CUDA_FIXED_BLOCK_SIZE) || defined(MATMUL_BUILD_PAR_CUDA_MEMCPY_FIXED_BLOCK_SIZE) || defined(MATMUL_BUILD_PAR_CUDA_DYN_BLOCK_SIZE) || defined(MATMUL_BUILD_PAR_CUDA_MEMCPY_DYN_BLOCK_SIZE)

    #include <matmul/common/Config.h>   // TElem, TIdx

    #ifdef __cplusplus
        extern "C"
        {
    #endif
    #ifdef MATMUL_BUILD_PAR_CUDA_FIXED_BLOCK_SIZE
        //-----------------------------------------------------------------------------
        //! (S/D)GEMM matrix-matrix product C = alpha * A * B + beta * C using CUDA with a fixed block size and 2-dimensional static shared memory.
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
        void matmul_gemm_par_cuda_fixed_block_size_2d_static_shared(
            TIdx const m, TIdx const n, TIdx const k,
            TElem const alpha,
            TElem const * const MATMUL_RESTRICT A, TIdx const lda,
            TElem const * const MATMUL_RESTRICT B, TIdx const ldb,
            TElem const beta,
            TElem * const MATMUL_RESTRICT C, TIdx const ldc);
    #endif
    #ifdef MATMUL_BUILD_PAR_CUDA_MEMCPY_FIXED_BLOCK_SIZE
        //-----------------------------------------------------------------------------
        //! (S/D)GEMM matrix-matrix product C = alpha * A * B + beta * C using CUDA with a fixed block size and 2-dimensional static shared memory.
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
        void matmul_gemm_par_cuda_memcpy_fixed_block_size_2d_static_shared(
            TIdx const m, TIdx const n, TIdx const k,
            TElem const alpha,
            TElem const * const MATMUL_RESTRICT A, TIdx const lda,
            TElem const * const MATMUL_RESTRICT B, TIdx const ldb,
            TElem const beta,
            TElem * const MATMUL_RESTRICT C, TIdx const ldc);
    #endif
    #ifdef MATMUL_BUILD_PAR_CUDA_FIXED_BLOCK_SIZE
        //-----------------------------------------------------------------------------
        //! (S/D)GEMM matrix-matrix product C = alpha * A * B + beta * C using CUDA with a fixed block size and 1-dimensional static shared memory.
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
        void matmul_gemm_par_cuda_fixed_block_size_1d_static_shared(
            TIdx const m, TIdx const n, TIdx const k,
            TElem const alpha,
            TElem const * const MATMUL_RESTRICT A, TIdx const lda,
            TElem const * const MATMUL_RESTRICT B, TIdx const ldb,
            TElem const beta,
            TElem * const MATMUL_RESTRICT C, TIdx const ldc);
    #endif
    #ifdef MATMUL_BUILD_PAR_CUDA_MEMCPY_FIXED_BLOCK_SIZE
        //-----------------------------------------------------------------------------
        //! (S/D)GEMM matrix-matrix product C = alpha * A * B + beta * C using CUDA with a fixed block size and 1-dimensional static shared memory.
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
        void matmul_gemm_par_cuda_memcpy_fixed_block_size_1d_static_shared(
            TIdx const m, TIdx const n, TIdx const k,
            TElem const alpha,
            TElem const * const MATMUL_RESTRICT A, TIdx const lda,
            TElem const * const MATMUL_RESTRICT B, TIdx const ldb,
            TElem const beta,
            TElem * const MATMUL_RESTRICT C, TIdx const ldc);
    #endif
    #ifdef MATMUL_BUILD_PAR_CUDA_FIXED_BLOCK_SIZE
        //-----------------------------------------------------------------------------
        //! (S/D)GEMM matrix-matrix product C = alpha * A * B + beta * C using CUDA with a fixed block size and extern shared memory.
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
        void matmul_gemm_par_cuda_fixed_block_size_1d_extern_shared(
            TIdx const m, TIdx const n, TIdx const k,
            TElem const alpha,
            TElem const * const MATMUL_RESTRICT A, TIdx const lda,
            TElem const * const MATMUL_RESTRICT B, TIdx const ldb,
            TElem const beta,
            TElem * const MATMUL_RESTRICT C, TIdx const ldc);
    #endif
    #ifdef MATMUL_BUILD_PAR_CUDA_MEMCPY_FIXED_BLOCK_SIZE
        //-----------------------------------------------------------------------------
        //! (S/D)GEMM matrix-matrix product C = alpha * A * B + beta * C using CUDA with a fixed block size and extern shared memory.
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
        void matmul_gemm_par_cuda_memcpy_fixed_block_size_1d_extern_shared(
            TIdx const m, TIdx const n, TIdx const k,
            TElem const alpha,
            TElem const * const MATMUL_RESTRICT A, TIdx const lda,
            TElem const * const MATMUL_RESTRICT B, TIdx const ldb,
            TElem const beta,
            TElem * const MATMUL_RESTRICT C, TIdx const ldc);
    #endif
    #ifdef MATMUL_BUILD_PAR_CUDA_DYN_BLOCK_SIZE
        //-----------------------------------------------------------------------------
        //! (S/D)GEMM matrix-matrix product C = alpha * A * B + beta * C using CUDA with a dynamic block size and extern shared memory.
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
        void matmul_gemm_par_cuda_dyn_block_size_1d_extern_shared(
            TIdx const m, TIdx const n, TIdx const k,
            TElem const alpha,
            TElem const * const MATMUL_RESTRICT A, TIdx const lda,
            TElem const * const MATMUL_RESTRICT B, TIdx const ldb,
            TElem const beta,
            TElem * const MATMUL_RESTRICT C, TIdx const ldc);
    #endif
    #ifdef MATMUL_BUILD_PAR_CUDA_MEMCPY_DYN_BLOCK_SIZE
        //-----------------------------------------------------------------------------
        //! (S/D)GEMM matrix-matrix product C = alpha * A * B + beta * C using CUDA with a dynamic block size and extern shared memory.
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
        void matmul_gemm_par_cuda_memcpy_dyn_block_size_1d_extern_shared(
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
