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

#ifdef MATMUL_BUILD_CUDA_MEMCPY

    #include <matmul/common/Cuda.h>

    #include <matmul/common/Mat.h>      // matmul_mat_gemm_early_out

    #include <cuda_runtime.h>

    #include <stdio.h>                  // printf

    #define MATMUL_CUDA_RT_CHECK(cmd) {cudaError_t error = cmd; if(error!=cudaSuccess){printf("<%s>:%i ",__FILE__,__LINE__); printf("[CUDA] Error: %s\n", cudaGetErrorString(error));}}

    //-----------------------------------------------------------------------------
    //
    //-----------------------------------------------------------------------------
    void matmul_gemm_wrap_memcpy_host_cuda(
        TIdx const m, TIdx const n, TIdx const k,
        TElem const alpha,
        TElem const * const MATMUL_RESTRICT A, TIdx const lda,
        TElem const * const MATMUL_RESTRICT B, TIdx const ldb,
        TElem const beta,
        TElem * const MATMUL_RESTRICT C, TIdx const ldc,
        void(*pMatMul)(TIdx const, TIdx const, TIdx const, TElem const, TElem const * const, TIdx const, TElem const * const, TIdx const, TElem const, TElem * const, TIdx const))
    {
        if(matmul_mat_gemm_early_out(m, n, k, alpha, beta))
        {
            return;
        }

        MATMUL_CUDA_RT_CHECK(cudaSetDevice(0));

        TIdx const uiBytesA = lda*m*sizeof(TElem);
        TIdx const uiBytesB = ldb*k*sizeof(TElem);
        TIdx const uiBytesC = ldc*m*sizeof(TElem);

        TElem *pADev, *pBDev, *pCDev;
        MATMUL_CUDA_RT_CHECK(cudaMalloc((void **)&pADev, uiBytesA));
        MATMUL_CUDA_RT_CHECK(cudaMemcpy(pADev, A, uiBytesA, cudaMemcpyHostToDevice));
        MATMUL_CUDA_RT_CHECK(cudaMalloc((void **)&pBDev, uiBytesB));
        MATMUL_CUDA_RT_CHECK(cudaMemcpy(pBDev, B, uiBytesB, cudaMemcpyHostToDevice));
        MATMUL_CUDA_RT_CHECK(cudaMalloc((void **)&pCDev, uiBytesC));
        MATMUL_CUDA_RT_CHECK(cudaMemcpy(pCDev, C, uiBytesC, cudaMemcpyHostToDevice));

        pMatMul(
            m, n, k,
            alpha,
            pADev, lda,
            pBDev, ldb,
            beta,
            pCDev, ldc);

        MATMUL_CUDA_RT_CHECK(cudaMemcpy(C, pCDev, uiBytesC, cudaMemcpyDeviceToHost));

        cudaFree(pADev);
        cudaFree(pBDev);
        cudaFree(pCDev);
    }
    //-----------------------------------------------------------------------------
    //
    //-----------------------------------------------------------------------------
    void matmul_gemm_wrap_memcpy_host_cuda_2d(
        TIdx const m, TIdx const n, TIdx const k,
        TElem const alpha,
        TElem const * const MATMUL_RESTRICT A, TIdx const lda,
        TElem const * const MATMUL_RESTRICT B, TIdx const ldb,
        TElem const beta,
        TElem * const MATMUL_RESTRICT C, TIdx const ldc,
        void(*pMatMul)(TIdx const, TIdx const, TIdx const, TElem const, TElem const * const, TIdx const, TElem const * const, TIdx const, TElem const, TElem * const, TIdx const))
    {
        if(matmul_mat_gemm_early_out(m, n, k, alpha, beta))
        {
            return;
        }

        MATMUL_CUDA_RT_CHECK(cudaSetDevice(0));

        size_t uiPitchBytesADev = 0;
        size_t uiPitchBytesBDev = 0;
        size_t uiPitchBytesCDev = 0;
        size_t const uHeightBytesA = m;
        size_t const uiWidthBytesA = k*sizeof(TElem);
        size_t const uHeightBytesB = k;
        size_t const uiWidthBytesB = n*sizeof(TElem);
        size_t const uHeightBytesC = m;
        size_t const uiWidthBytesC = n*sizeof(TElem);
        TElem * pADev = 0;
        TElem * pBDev = 0;
        TElem * pCDev = 0;
        MATMUL_CUDA_RT_CHECK(cudaMallocPitch((void **)&pADev, &uiPitchBytesADev, uiWidthBytesA, uHeightBytesA));
        MATMUL_CUDA_RT_CHECK(cudaMemcpy2D(pADev, uiPitchBytesADev, A, lda * sizeof(TElem), uiWidthBytesA, uHeightBytesA, cudaMemcpyHostToDevice));
        MATMUL_CUDA_RT_CHECK(cudaMallocPitch((void **)&pBDev, &uiPitchBytesBDev, uiWidthBytesB, uHeightBytesB));
        MATMUL_CUDA_RT_CHECK(cudaMemcpy2D(pBDev, uiPitchBytesBDev, B, ldb * sizeof(TElem), uiWidthBytesB, uHeightBytesB, cudaMemcpyHostToDevice));
        MATMUL_CUDA_RT_CHECK(cudaMallocPitch((void **)&pCDev, &uiPitchBytesCDev, uiWidthBytesC, uHeightBytesC));
        MATMUL_CUDA_RT_CHECK(cudaMemcpy2D(pCDev, uiPitchBytesCDev, C, ldc * sizeof(TElem), uiWidthBytesC, uHeightBytesC, cudaMemcpyHostToDevice));

        pMatMul(
            m, n, k,
            alpha,
            pADev, static_cast<TIdx>(uiPitchBytesADev / sizeof(TElem)),
            pBDev, static_cast<TIdx>(uiPitchBytesBDev / sizeof(TElem)),
            beta,
            pCDev, static_cast<TIdx>(uiPitchBytesCDev / sizeof(TElem)));

        MATMUL_CUDA_RT_CHECK(cudaMemcpy2D(C, ldc * sizeof(TElem), pCDev, uiPitchBytesCDev, uiWidthBytesC, uHeightBytesC, cudaMemcpyDeviceToHost));

        cudaFree(pADev);
        cudaFree(pBDev);
        cudaFree(pCDev);
    }
#endif