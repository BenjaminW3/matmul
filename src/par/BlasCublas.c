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

#if defined(MATMUL_BUILD_PAR_BLAS_CUBLAS_MEMCPY) || defined(MATMUL_BUILD_PAR_BLAS_CUBLAS)

    #include <matmul/par/BlasCublas.h>

    #include <matmul/common/Cuda.h>     // matmul_gemm_wrap_memcpy_host_cuda
    #include <matmul/common/Mat.h>      // matmul_mat_gemm_early_out

    #include <cuda_runtime.h>
    #include <cublas_v2.h>

    #include <stdio.h>                  // printf

    #define MATMUL_CUDA_RT_CHECK(cmd) {cudaError_t error = cmd; if(error!=cudaSuccess){printf("<%s>:%i ",__FILE__,__LINE__); printf("[CUDA] Error: %s\n", cudaGetErrorString(error));}}
    #define MATMUL_CUBLAS_CHECK(cmd) {cublasStatus_t ret = cmd; if(ret!=CUBLAS_STATUS_SUCCESS){printf("<%s>:%i [CUBLAS] Error code %d\n", __FILE__, __LINE__, ret);}}

    #ifdef MATMUL_BUILD_PAR_BLAS_CUBLAS
        //-----------------------------------------------------------------------------
        //
        //-----------------------------------------------------------------------------
        TReturn matmul_gemm_par_blas_cublas2(
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

            // Initialize cublas
            cublasHandle_t handle;
            MATMUL_CUBLAS_CHECK(cublasCreate(&handle));

            // NOTE:
            // In contrast to many other libraries, cuBLAS expects the matrices in column major storage order.
            // Because the storage order of multidimensional C arrays is row-major we have to change the order of the arguments given to `cublasDgemm`.
            // By swapping the matrix A with the matrix B the result is still correct.
            // Because cuBLAS sees the matrices in transposed order due to the inverse storage order it expects, the following computation is executed:
            //  C^T <- alpha * B^T * A^T + beta * C^T.
            // By reading the transposed matrix C^T that has been written in column major order as row major matrix we receive the expected untransposed result C.
            MATMUL_TIME_START;

        #ifdef MATMUL_ELEMENT_TYPE_DOUBLE
            MATMUL_CUBLAS_CHECK(cublasDgemm(
                handle, CUBLAS_OP_N, CUBLAS_OP_N,
                (int)m, (int)n, (int)k,
                &alpha,
                B, (int)ldb,
                A, (int)lda,
                &beta,
                C, (int)ldc));
        #else
            MATMUL_CUBLAS_CHECK(cublasSgemm(
                handle, CUBLAS_OP_N, CUBLAS_OP_N,
                (int)m, (int)n, (int)k,
                &alpha,
                B, (int)ldb,
                A, (int)lda,
                &beta,
                C, (int)ldc));
        #endif

            MATMUL_CUDA_RT_CHECK(cudaDeviceSynchronize());

            MATMUL_TIME_END;

            MATMUL_CUBLAS_CHECK(cublasDestroy(handle));

            MATMUL_TIME_RETURN;
        }
    #endif
    #ifdef MATMUL_BUILD_PAR_BLAS_CUBLAS_MEMCPY
        //-----------------------------------------------------------------------------
        //
        //-----------------------------------------------------------------------------
        TReturn matmul_gemm_par_blas_cublas2_memcpy(
            TSize const m, TSize const n, TSize const k,
            TElem const alpha,
            TElem const * const MATMUL_RESTRICT A, TSize const lda,
            TElem const * const MATMUL_RESTRICT B, TSize const ldb,
            TElem const beta,
            TElem * const MATMUL_RESTRICT C, TSize const ldc)
        {
            return
                matmul_gemm_wrap_memcpy_host_cuda(
                    m, n, k,
                    alpha,
                    A, lda,
                    B, ldb,
                    beta,
                    C, ldc,
                    matmul_gemm_par_blas_cublas2);
        }
    #endif
#endif
