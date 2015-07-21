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
        void matmul_gemm_par_blas_cublas2(
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

            MATMUL_CUBLAS_CHECK(cublasDestroy(handle));
        }
    #endif
    #ifdef MATMUL_BUILD_PAR_BLAS_CUBLAS_MEMCPY
        //-----------------------------------------------------------------------------
        //
        //-----------------------------------------------------------------------------
        void matmul_gemm_par_blas_cublas2_memcpy(
            TIdx const m, TIdx const n, TIdx const k,
            TElem const alpha,
            TElem const * const MATMUL_RESTRICT A, TIdx const lda,
            TElem const * const MATMUL_RESTRICT B, TIdx const ldb,
            TElem const beta,
            TElem * const MATMUL_RESTRICT C, TIdx const ldc)
        {
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
