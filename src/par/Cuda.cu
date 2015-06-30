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

#ifdef MATMUL_BUILD_PAR_CUDA

    #include <matmul/par/Cuda.h>

    #include <matmul/common/Mat.h>  // matmul_mat_gemm_early_out

    #include <cuda_runtime.h>

    #include <stdio.h>              // printf
    #include <math.h>               // ceil

    #define MATMUL_CUDA_RT_CHECK(cmd) {cudaError_t error = cmd; if(error!=cudaSuccess){printf("<%s>:%i ",__FILE__,__LINE__); printf("[CUDA] Error: %s\n", cudaGetErrorString(error));}}

    //-----------------------------------------------------------------------------
    // This function only works for square blocks.
    //-----------------------------------------------------------------------------
    __global__ void matmul_gemm_par_cuda_kernel(
        TIdx const m, TIdx const n, TIdx const k,
        TElem const alpha,
        TElem const * const MATMUL_RESTRICT A, TIdx const lda,
        TElem const * const MATMUL_RESTRICT B, TIdx const ldb,
        TElem const beta,
        TElem * const MATMUL_RESTRICT C, TIdx const ldc)
    {
        // blockIdx.x and blockIdx.y are the indices of the block to calculate inside C.
        TIdx const uiGridThreadIdxX = blockIdx.x*blockDim.x + threadIdx.x;    // Column inside C to calculate.
        TIdx const uiGridThreadIdxY = blockIdx.y*blockDim.y + threadIdx.y;    // Row inside C to calculate.

        TIdx const uiBlockThreadIdxX = threadIdx.x;    // Column inside the block of C to calculate.
        TIdx const uiBlockThreadIdxY = threadIdx.y;    // Row inside the block of C to calculate.

        TIdx const uiBlockThreadsExtentX = blockDim.x;
        TIdx const uiBlockThreadsExtentY = blockDim.y;
        //assert(uiBlockThreadsExtentX == uiBlockThreadsExtentY);
        TIdx const uiBlockThreadsExtent = uiBlockThreadsExtentX;

        // Shared memory used to store the current blocks of A and B.
        __shared__ TElem pSharedBlockA[MATMUL_CUDA_BLOCKSIZE][MATMUL_CUDA_BLOCKSIZE];
        __shared__ TElem pSharedBlockB[MATMUL_CUDA_BLOCKSIZE][MATMUL_CUDA_BLOCKSIZE];

        // If the element is outside of the matrix, write zero into the shared block.
        bool const bInsideA = (uiGridThreadIdxY < m);
        bool const bInsideB = (uiGridThreadIdxX < n);
        bool const bInsideC = (bInsideA && bInsideB);

        TElem fCSum(0);

        // Loop over all blocks of A and B that are required to compute the C block.
        auto const uiBlockMulCount(
            static_cast<TIdx>(
                ceil(
                    static_cast<float>(k)/static_cast<float>(uiBlockThreadsExtent))));
        for(TIdx k2=0; k2<uiBlockMulCount; ++k2)
        {
            // Copy data to shared memory.
            TIdx const uiAIdxX(k2*uiBlockThreadsExtentX + uiBlockThreadIdxX);
            TIdx const uiAIdx1d(uiGridThreadIdxY*lda + uiAIdxX);
            pSharedBlockA[uiBlockThreadIdxY][uiBlockThreadIdxX] =
                ((!bInsideA) || (uiAIdxX>=k))
                ? static_cast<TElem>(0)
                : A[uiAIdx1d];

            TIdx const uiBIdxY(k2*uiBlockThreadsExtentY + uiBlockThreadIdxY);
            TIdx const uiBIdx1d(uiBIdxY*ldb + uiGridThreadIdxX);
            pSharedBlockB[uiBlockThreadIdxY][uiBlockThreadIdxX] =
                ((!bInsideB) || (uiBIdxY>=k))
                ? static_cast<TElem>(0)
                : B[uiBIdx1d];

            // Synchronize to make sure the sub-matrices are loaded before starting the computation.
            __syncthreads();

            // Dyadic product within shared memory.
            for(TIdx k3 = 0; k3<uiBlockThreadsExtent; ++k3)
            {
                fCSum += alpha * pSharedBlockA[uiBlockThreadIdxY][k3]
                    * pSharedBlockB[k3][uiBlockThreadIdxX];
            }

            // Synchronize to make sure that the preceding computation is done before loading two new sub-matrices of A and B in the next iteration.
            __syncthreads();
        }

        if(bInsideC)
        {
            auto const uiIdxC1d(uiGridThreadIdxY*ldc + uiGridThreadIdxX);
            C[uiIdxC1d] = alpha * fCSum + beta * C[uiIdxC1d];
        }
    }
    //-----------------------------------------------------------------------------
    //
    //-----------------------------------------------------------------------------
    void matmul_gemm_par_cuda(
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

        cudaStream_t stream;
        MATMUL_CUDA_RT_CHECK(cudaStreamCreate(&stream));

        TIdx const uiBytesA = lda*m*sizeof(TElem);
        TIdx const uiBytesB = ldb*k*sizeof(TElem);
        TIdx const uiBytesC = ldc*m*sizeof(TElem);

        TElem *pADev, *pBDev, *pCDev;
        MATMUL_CUDA_RT_CHECK(cudaMalloc((void **)&pADev, uiBytesA));
        MATMUL_CUDA_RT_CHECK(cudaMemcpyAsync(pADev, A, uiBytesA, cudaMemcpyHostToDevice, stream));
        MATMUL_CUDA_RT_CHECK(cudaMalloc((void **)&pBDev, uiBytesB));
        MATMUL_CUDA_RT_CHECK(cudaMemcpyAsync(pBDev, B, uiBytesB, cudaMemcpyHostToDevice, stream));
        MATMUL_CUDA_RT_CHECK(cudaMalloc((void **)&pCDev, uiBytesC));
        MATMUL_CUDA_RT_CHECK(cudaMemcpyAsync(pCDev, C, uiBytesC, cudaMemcpyHostToDevice, stream));

        dim3 const dimBlock(MATMUL_CUDA_BLOCKSIZE, MATMUL_CUDA_BLOCKSIZE);
        float const fGridThreadExtentX = ceil(((float)n)/((float)MATMUL_CUDA_BLOCKSIZE));
        float const fGridThreadExtentY = ceil(((float)m)/((float)MATMUL_CUDA_BLOCKSIZE));
        unsigned int const uiGridThreadExtentX = (unsigned int)fGridThreadExtentX;
        unsigned int const uiGridThreadExtentY = (unsigned int)fGridThreadExtentY;
        dim3 const dimGrid(uiGridThreadExtentX, uiGridThreadExtentY);

        matmul_gemm_par_cuda_kernel<<<dimGrid, dimBlock, 0, stream>>>(
            m, n, k,
            alpha,
            pADev, lda,
            pBDev, ldb,
            beta,
            pCDev, ldc);

        MATMUL_CUDA_RT_CHECK(cudaMemcpyAsync(C, pCDev, uiBytesC, cudaMemcpyDeviceToHost, stream));

        MATMUL_CUDA_RT_CHECK(cudaStreamSynchronize(stream));

        cudaFree(pADev);
        cudaFree(pBDev);
        cudaFree(pCDev);

        MATMUL_CUDA_RT_CHECK(cudaStreamDestroy(stream));
    }
#endif
