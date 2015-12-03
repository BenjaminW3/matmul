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

#ifdef MATMUL_BUILD_CUDA_MEMCPY

    #include <matmul/common/Cuda.h>

    #include <matmul/common/Mat.h>      // matmul_mat_gemm_early_out

    #include <cuda_runtime.h>

    #include <stdio.h>                  // printf

    #define MATMUL_CUDA_RT_CHECK(cmd) {cudaError_t error = cmd; if(error!=cudaSuccess){printf("<%s>:%i ",__FILE__,__LINE__); printf("[CUDA] Error: %s\n", cudaGetErrorString(error));}}

    //-----------------------------------------------------------------------------
    //
    //-----------------------------------------------------------------------------
    TReturn matmul_gemm_wrap_memcpy_host_cuda(
        TSize const m, TSize const n, TSize const k,
        TElem const alpha,
        TElem const * const MATMUL_RESTRICT A, TSize const lda,
        TElem const * const MATMUL_RESTRICT B, TSize const ldb,
        TElem const beta,
        TElem * const MATMUL_RESTRICT C, TSize const ldc,
        TReturn(*pGemm)(TSize const, TSize const, TSize const, TElem const, TElem const * const, TSize const, TElem const * const, TSize const, TElem const, TElem * const, TSize const))
    {
        if(matmul_mat_gemm_early_out(m, n, k, alpha, beta))
        {
            MATMUL_TIME_RETURN_EARLY_OUT;
        }

        MATMUL_CUDA_RT_CHECK(cudaSetDevice(0));

        TSize const bytesA = lda*m*sizeof(TElem);
        TSize const bytesB = ldb*k*sizeof(TElem);
        TSize const bytesC = ldc*m*sizeof(TElem);

        TElem *pADev, *pBDev, *pCDev;
        MATMUL_CUDA_RT_CHECK(cudaMalloc((void **)&pADev, bytesA));
        MATMUL_CUDA_RT_CHECK(cudaMemcpy(pADev, A, bytesA, cudaMemcpyHostToDevice));
        MATMUL_CUDA_RT_CHECK(cudaMalloc((void **)&pBDev, bytesB));
        MATMUL_CUDA_RT_CHECK(cudaMemcpy(pBDev, B, bytesB, cudaMemcpyHostToDevice));
        MATMUL_CUDA_RT_CHECK(cudaMalloc((void **)&pCDev, bytesC));
        MATMUL_CUDA_RT_CHECK(cudaMemcpy(pCDev, C, bytesC, cudaMemcpyHostToDevice));

        MATMUL_TIME_STORE
            pGemm(
                m, n, k,
                alpha,
                pADev, lda,
                pBDev, ldb,
                beta,
                pCDev, ldc);

        MATMUL_CUDA_RT_CHECK(cudaMemcpy(C, pCDev, bytesC, cudaMemcpyDeviceToHost));

        cudaFree(pADev);
        cudaFree(pBDev);
        cudaFree(pCDev);

        MATMUL_TIME_RETURN;
    }
    //-----------------------------------------------------------------------------
    //
    //-----------------------------------------------------------------------------
    TReturn matmul_gemm_wrap_memcpy_host_cuda_2d(
        TSize const m, TSize const n, TSize const k,
        TElem const alpha,
        TElem const * const MATMUL_RESTRICT A, TSize const lda,
        TElem const * const MATMUL_RESTRICT B, TSize const ldb,
        TElem const beta,
        TElem * const MATMUL_RESTRICT C, TSize const ldc,
        TReturn(*pGemm)(TSize const, TSize const, TSize const, TElem const, TElem const * const, TSize const, TElem const * const, TSize const, TElem const, TElem * const, TSize const))
    {
        if(matmul_mat_gemm_early_out(m, n, k, alpha, beta))
        {
            MATMUL_TIME_RETURN_EARLY_OUT;
        }

        MATMUL_CUDA_RT_CHECK(cudaSetDevice(0));

        size_t pitchBytesADev = 0;
        size_t pitchBytesBDev = 0;
        size_t pitchBytesCDev = 0;
        size_t const heightBytesA = m;
        size_t const widthBytesA = k*sizeof(TElem);
        size_t const heightBytesB = k;
        size_t const widthBytesB = n*sizeof(TElem);
        size_t const heightBytesC = m;
        size_t const widthBytesC = n*sizeof(TElem);
        TElem * pADev = 0;
        TElem * pBDev = 0;
        TElem * pCDev = 0;
        MATMUL_CUDA_RT_CHECK(cudaMallocPitch((void **)&pADev, &pitchBytesADev, widthBytesA, heightBytesA));
        MATMUL_CUDA_RT_CHECK(cudaMemcpy2D(pADev, pitchBytesADev, A, lda * sizeof(TElem), widthBytesA, heightBytesA, cudaMemcpyHostToDevice));
        MATMUL_CUDA_RT_CHECK(cudaMallocPitch((void **)&pBDev, &pitchBytesBDev, widthBytesB, heightBytesB));
        MATMUL_CUDA_RT_CHECK(cudaMemcpy2D(pBDev, pitchBytesBDev, B, ldb * sizeof(TElem), widthBytesB, heightBytesB, cudaMemcpyHostToDevice));
        MATMUL_CUDA_RT_CHECK(cudaMallocPitch((void **)&pCDev, &pitchBytesCDev, widthBytesC, heightBytesC));
        MATMUL_CUDA_RT_CHECK(cudaMemcpy2D(pCDev, pitchBytesCDev, C, ldc * sizeof(TElem), widthBytesC, heightBytesC, cudaMemcpyHostToDevice));

        MATMUL_TIME_STORE
            pGemm(
                m, n, k,
                alpha,
                pADev, static_cast<TSize>(pitchBytesADev / sizeof(TElem)),
                pBDev, static_cast<TSize>(pitchBytesBDev / sizeof(TElem)),
                beta,
                pCDev, static_cast<TSize>(pitchBytesCDev / sizeof(TElem)));

        MATMUL_CUDA_RT_CHECK(cudaMemcpy2D(C, ldc * sizeof(TElem), pCDev, pitchBytesCDev, widthBytesC, heightBytesC, cudaMemcpyDeviceToHost));

        cudaFree(pADev);
        cudaFree(pBDev);
        cudaFree(pCDev);

        MATMUL_TIME_RETURN;
    }
#endif