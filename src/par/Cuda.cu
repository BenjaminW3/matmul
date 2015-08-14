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

#if defined(MATMUL_BUILD_PAR_CUDA_FIXED_BLOCK_SIZE) || defined(MATMUL_BUILD_PAR_CUDA_MEMCPY_FIXED_BLOCK_SIZE) || defined(MATMUL_BUILD_PAR_CUDA_DYN_BLOCK_SIZE) || defined(MATMUL_BUILD_PAR_CUDA_MEMCPY_DYN_BLOCK_SIZE)

    #include <matmul/par/Cuda.h>

    #include <matmul/common/Cuda.h> // matmul_gemm_wrap_memcpy_host_cuda_2d
    #include <matmul/common/Mat.h>  // matmul_mat_gemm_early_out

    #include <cuda_runtime.h>

    #include <stdio.h>              // printf
    #include <math.h>               // ceil
    #include <algorithm>            // std::min

    #define MATMUL_CUDA_RT_CHECK(cmd) {cudaError_t error = cmd; if(error!=cudaSuccess){printf("<%s>:%i ",__FILE__,__LINE__); printf("[CUDA] Error: %s\n", cudaGetErrorString(error));}}

    #ifdef MATMUL_BUILD_PAR_CUDA_FIXED_BLOCK_SIZE
        //-----------------------------------------------------------------------------
        // This function only works for square blocks.
        //-----------------------------------------------------------------------------
        __global__ void matmul_gemm_par_cuda_fixed_block_size_2d_static_shared_kernel(
            TIdx const m, TIdx const n, TIdx const k,
            TElem const alpha,
            TElem const * const MATMUL_RESTRICT A, TIdx const lda,
            TElem const * const MATMUL_RESTRICT B, TIdx const ldb,
            TElem const beta,
            TElem * const MATMUL_RESTRICT C, TIdx const ldc)
        {
            // Column and row of C to calculate.
            TIdx const gridThreadIdxX = blockIdx.x*blockDim.x + threadIdx.x;
            TIdx const gridThreadIdxY = blockIdx.y*blockDim.y + threadIdx.y;

            // Column and row inside the block of C to calculate.
            TIdx const blockThreadIdxX = threadIdx.x;
            TIdx const blockThreadIdxY = threadIdx.y;

            // The block threads extents.
            TIdx const blockThreadsExtentX = blockDim.x;
            TIdx const blockThreadsExtentY = blockDim.y;
            //assert(blockThreadsExtentX == blockThreadsExtentY);
            TIdx const & blockThreadsExtent = blockThreadsExtentX;

            // Shared memory used to store the current blocks of A and B.
            __shared__ TElem pBlockSharedA[MATMUL_CUDA_FIXED_BLOCK_SIZE][MATMUL_CUDA_FIXED_BLOCK_SIZE];
            __shared__ TElem pBlockSharedB[MATMUL_CUDA_FIXED_BLOCK_SIZE][MATMUL_CUDA_FIXED_BLOCK_SIZE];

            // If the element corresponding to the current thread is outside of the respective matrix.
            bool const insideA = (gridThreadIdxY < m);
            bool const insideB = (gridThreadIdxX < n);
            bool const insideC = (insideA && insideB);

            TElem dotProduct(0);

            // Loop over all blocks of A and B that are required to compute the C block.
            auto const blockMulCount(
                static_cast<TIdx>(
                    ceil(
                        static_cast<float>(k)/static_cast<float>(blockThreadsExtent))));
            for(TIdx k2=0; k2<blockMulCount; ++k2)
            {
                // Copy the current blocks of A and B into shared memory in parallel.
                // If the element of the current thread is outside of the matrix, zero is written into the shared memory.
                // This is possible because zero is a result neutral extension of the matrices regarding the dot product.
                TIdx const AIdxX(k2*blockThreadsExtentX + blockThreadIdxX);
                TIdx const AIdx1d(gridThreadIdxY*lda + AIdxX);
                pBlockSharedA[blockThreadIdxY][blockThreadIdxX] =
                    ((!insideA) || (AIdxX>=k))
                    ? static_cast<TElem>(0)
                    : A[AIdx1d];

                TIdx const BIdxY(k2*blockThreadsExtentY + blockThreadIdxY);
                TIdx const BIdx1d(BIdxY*ldb + gridThreadIdxX);
                pBlockSharedB[blockThreadIdxY][blockThreadIdxX] =
                    ((!insideB) || (BIdxY>=k))
                    ? static_cast<TElem>(0)
                    : B[BIdx1d];

                // Synchronize to make sure the complete blocks are loaded before starting the computation.
                __syncthreads();

                // Compute the dot products within shared memory.
                for(TIdx k3 = 0; k3<blockThreadsExtent; ++k3)
                {
                    dotProduct += pBlockSharedA[blockThreadIdxY][k3]
                        * pBlockSharedB[k3][blockThreadIdxX];
                }

                // Synchronize to make sure that the preceding computation is done before loading the next blocks of A and B.
                __syncthreads();
            }

            if(insideC)
            {
                auto const CIdx1d(gridThreadIdxY*ldc + gridThreadIdxX);
                C[CIdx1d] = alpha * dotProduct + beta * C[CIdx1d];
            }
        }
        //-----------------------------------------------------------------------------
        //
        //-----------------------------------------------------------------------------
        void matmul_gemm_par_cuda_fixed_block_size_2d_static_shared(
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

            dim3 const dimBlock(MATMUL_CUDA_FIXED_BLOCK_SIZE, MATMUL_CUDA_FIXED_BLOCK_SIZE);
            float const fGridThreadExtentX = ceil(((float)n) / ((float)MATMUL_CUDA_FIXED_BLOCK_SIZE));
            float const fGridThreadExtentY = ceil(((float)m) / ((float)MATMUL_CUDA_FIXED_BLOCK_SIZE));
            unsigned int const gridThreadExtentX = (unsigned int)fGridThreadExtentX;
            unsigned int const gridThreadExtentY = (unsigned int)fGridThreadExtentY;
            dim3 const dimGrid(gridThreadExtentX, gridThreadExtentY);

            matmul_gemm_par_cuda_fixed_block_size_2d_static_shared_kernel<<<
                dimGrid,
                dimBlock,
                0>>>(
                    m, n, k,
                    alpha,
                    A, lda,
                    B, ldb,
                    beta,
                    C, ldc);

            MATMUL_CUDA_RT_CHECK(cudaDeviceSynchronize());
        }
    #endif
    #ifdef MATMUL_BUILD_PAR_CUDA_MEMCPY_FIXED_BLOCK_SIZE
        //-----------------------------------------------------------------------------
        //
        //-----------------------------------------------------------------------------
        void matmul_gemm_par_cuda_memcpy_fixed_block_size_2d_static_shared(
            TIdx const m, TIdx const n, TIdx const k,
            TElem const alpha,
            TElem const * const MATMUL_RESTRICT A, TIdx const lda,
            TElem const * const MATMUL_RESTRICT B, TIdx const ldb,
            TElem const beta,
            TElem * const MATMUL_RESTRICT C, TIdx const ldc)
        {
            matmul_gemm_wrap_memcpy_host_cuda_2d(
                m, n, k,
                alpha,
                A, lda,
                B, ldb,
                beta,
                C, ldc,
                matmul_gemm_par_cuda_fixed_block_size_2d_static_shared);
        }
    #endif
    #ifdef MATMUL_BUILD_PAR_CUDA_FIXED_BLOCK_SIZE
        //-----------------------------------------------------------------------------
        // This function only works for square blocks.
        //-----------------------------------------------------------------------------
        __global__ void matmul_gemm_par_cuda_fixed_block_size_1d_static_shared_kernel(
            TIdx const m, TIdx const n, TIdx const k,
            TElem const alpha,
            TElem const * const MATMUL_RESTRICT A, TIdx const lda,
            TElem const * const MATMUL_RESTRICT B, TIdx const ldb,
            TElem const beta,
            TElem * const MATMUL_RESTRICT C, TIdx const ldc)
        {
            // Column and row of C to calculate.
            TIdx const gridThreadIdxX = blockIdx.x*blockDim.x + threadIdx.x;
            TIdx const gridThreadIdxY = blockIdx.y*blockDim.y + threadIdx.y;

            // Column and row inside the block of C to calculate.
            TIdx const blockThreadIdxX = threadIdx.x;
            TIdx const blockThreadIdxY = threadIdx.y;

            // The block threads extents.
            TIdx const blockThreadsExtentX = blockDim.x;
            TIdx const blockThreadsExtentY = blockDim.y;
            //assert(blockThreadsExtentX == blockThreadsExtentY);
            TIdx const & blockThreadsExtent = blockThreadsExtentX;

            // Shared memory used to store the current blocks of A and B.
            __shared__ TElem pBlockSharedA[MATMUL_CUDA_FIXED_BLOCK_SIZE*MATMUL_CUDA_FIXED_BLOCK_SIZE];
            __shared__ TElem pBlockSharedB[MATMUL_CUDA_FIXED_BLOCK_SIZE*MATMUL_CUDA_FIXED_BLOCK_SIZE];

            auto const sharedBlockIdx1d(blockThreadIdxY*blockThreadsExtentX + blockThreadIdxX);

            // If the element corresponding to the current thread is outside of the respective matrix.
            bool const insideA = (gridThreadIdxY < m);
            bool const insideB = (gridThreadIdxX < n);
            bool const insideC = (insideA && insideB);

            TElem dotProduct(0);

            // Loop over all blocks of A and B that are required to compute the C block.
            auto const blockMulCount(
                static_cast<TIdx>(
                    ceil(
                        static_cast<float>(k)/static_cast<float>(blockThreadsExtent))));
            for(TIdx k2=0; k2<blockMulCount; ++k2)
            {
                // Copy the current blocks of A and B into shared memory in parallel.
                // If the element of the current thread is outside of the matrix, zero is written into the shared memory.
                // This is possible because zero is a result neutral extension of the matrices regarding the dot product.
                TIdx const AIdxX(k2*blockThreadsExtentX + blockThreadIdxX);
                TIdx const AIdx1d(gridThreadIdxY*lda + AIdxX);
                pBlockSharedA[sharedBlockIdx1d] =
                    ((!insideA) || (AIdxX>=k))
                    ? static_cast<TElem>(0)
                    : A[AIdx1d];

                TIdx const BIdxY(k2*blockThreadsExtentY + blockThreadIdxY);
                TIdx const BIdx1d(BIdxY*ldb + gridThreadIdxX);
                pBlockSharedB[sharedBlockIdx1d] =
                    ((!insideB) || (BIdxY>=k))
                    ? static_cast<TElem>(0)
                    : B[BIdx1d];

                // Synchronize to make sure the complete blocks are loaded before starting the computation.
                __syncthreads();

                // Compute the dot products within shared memory.
                for(TIdx k3 = 0; k3<blockThreadsExtent; ++k3)
                {
                    dotProduct += pBlockSharedA[blockThreadIdxY*blockThreadsExtentX + k3]
                        * pBlockSharedB[k3*blockThreadsExtentY + blockThreadIdxX];
                }

                // Synchronize to make sure that the preceding computation is done before loading the next blocks of A and B.
                __syncthreads();
            }

            if(insideC)
            {
                auto const CIdx1d(gridThreadIdxY*ldc + gridThreadIdxX);
                C[CIdx1d] = alpha * dotProduct + beta * C[CIdx1d];
            }
        }
        //-----------------------------------------------------------------------------
        //
        //-----------------------------------------------------------------------------
        void matmul_gemm_par_cuda_fixed_block_size_1d_static_shared(
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

            dim3 const dimBlock(MATMUL_CUDA_FIXED_BLOCK_SIZE, MATMUL_CUDA_FIXED_BLOCK_SIZE);
            float const fGridThreadExtentX = ceil(((float)n) / ((float)MATMUL_CUDA_FIXED_BLOCK_SIZE));
            float const fGridThreadExtentY = ceil(((float)m) / ((float)MATMUL_CUDA_FIXED_BLOCK_SIZE));
            unsigned int const gridThreadExtentX = (unsigned int)fGridThreadExtentX;
            unsigned int const gridThreadExtentY = (unsigned int)fGridThreadExtentY;
            dim3 const dimGrid(gridThreadExtentX, gridThreadExtentY);

            matmul_gemm_par_cuda_fixed_block_size_1d_static_shared_kernel<<<
                dimGrid,
                dimBlock,
                0>>>(
                    m, n, k,
                    alpha,
                    A, lda,
                    B, ldb,
                    beta,
                    C, ldc);

            MATMUL_CUDA_RT_CHECK(cudaDeviceSynchronize());
        }
    #endif
    #ifdef MATMUL_BUILD_PAR_CUDA_MEMCPY_FIXED_BLOCK_SIZE
        //-----------------------------------------------------------------------------
        //
        //-----------------------------------------------------------------------------
        void matmul_gemm_par_cuda_memcpy_fixed_block_size_1d_static_shared(
            TIdx const m, TIdx const n, TIdx const k,
            TElem const alpha,
            TElem const * const MATMUL_RESTRICT A, TIdx const lda,
            TElem const * const MATMUL_RESTRICT B, TIdx const ldb,
            TElem const beta,
            TElem * const MATMUL_RESTRICT C, TIdx const ldc)
        {
            matmul_gemm_wrap_memcpy_host_cuda_2d(
                m, n, k,
                alpha,
                A, lda,
                B, ldb,
                beta,
                C, ldc,
                matmul_gemm_par_cuda_fixed_block_size_1d_static_shared);
        }
    #endif
    #ifdef MATMUL_BUILD_PAR_CUDA_FIXED_BLOCK_SIZE
        //-----------------------------------------------------------------------------
        // This function only works for square blocks.
        //-----------------------------------------------------------------------------
        __global__ void matmul_gemm_par_cuda_fixed_block_size_1d_extern_shared_kernel(
            TIdx const m, TIdx const n, TIdx const k,
            TElem const alpha,
            TElem const * const MATMUL_RESTRICT A, TIdx const lda,
            TElem const * const MATMUL_RESTRICT B, TIdx const ldb,
            TElem const beta,
            TElem * const MATMUL_RESTRICT C, TIdx const ldc)
        {
            // Column and row of C to calculate.
            TIdx const gridThreadIdxX = blockIdx.x*blockDim.x + threadIdx.x;
            TIdx const gridThreadIdxY = blockIdx.y*blockDim.y + threadIdx.y;

            // Column and row inside the block of C to calculate.
            TIdx const blockThreadIdxX = threadIdx.x;
            TIdx const blockThreadIdxY = threadIdx.y;

            // The block threads extents.
            TIdx const blockThreadsExtentX = blockDim.x;
            TIdx const blockThreadsExtentY = blockDim.y;
            //assert(blockThreadsExtentX == blockThreadsExtentY);
            TIdx const & blockThreadsExtent = blockThreadsExtentX;

            // Shared memory used to store the current blocks of A and B.
            extern __shared__ TElem pBlockSharedA[];
            auto * const pBlockSharedB(pBlockSharedA + blockThreadsExtentX*blockThreadsExtentY);

            auto const sharedBlockIdx1d(blockThreadIdxY*blockThreadsExtentX + blockThreadIdxX);

            // If the element corresponding to the current thread is outside of the respective matrix.
            bool const insideA = (gridThreadIdxY < m);
            bool const insideB = (gridThreadIdxX < n);
            bool const insideC = (insideA && insideB);

            TElem dotProduct(0);

            // Loop over all blocks of A and B that are required to compute the C block.
            auto const blockMulCount(
                static_cast<TIdx>(
                    ceil(
                        static_cast<float>(k)/static_cast<float>(blockThreadsExtent))));
            for(TIdx k2=0; k2<blockMulCount; ++k2)
            {
                // Copy the current blocks of A and B into shared memory in parallel.
                // If the element of the current thread is outside of the matrix, zero is written into the shared memory.
                // This is possible because zero is a result neutral extension of the matrices regarding the dot product.
                TIdx const AIdxX(k2*blockThreadsExtentX + blockThreadIdxX);
                TIdx const AIdx1d(gridThreadIdxY*lda + AIdxX);
                pBlockSharedA[sharedBlockIdx1d] =
                    ((!insideA) || (AIdxX>=k))
                    ? static_cast<TElem>(0)
                    : A[AIdx1d];

                TIdx const BIdxY(k2*blockThreadsExtentY + blockThreadIdxY);
                TIdx const BIdx1d(BIdxY*ldb + gridThreadIdxX);
                pBlockSharedB[sharedBlockIdx1d] =
                    ((!insideB) || (BIdxY>=k))
                    ? static_cast<TElem>(0)
                    : B[BIdx1d];

                // Synchronize to make sure the complete blocks are loaded before starting the computation.
                __syncthreads();

                // Compute the dot products within shared memory.
                for(TIdx k3 = 0; k3<blockThreadsExtent; ++k3)
                {
                    dotProduct += pBlockSharedA[blockThreadIdxY*blockThreadsExtentX + k3]
                        * pBlockSharedB[k3*blockThreadsExtentY + blockThreadIdxX];
                }

                // Synchronize to make sure that the preceding computation is done before loading the next blocks of A and B.
                __syncthreads();
            }

            if(insideC)
            {
                auto const CIdx1d(gridThreadIdxY*ldc + gridThreadIdxX);
                C[CIdx1d] = alpha * dotProduct + beta * C[CIdx1d];
            }
        }
        //-----------------------------------------------------------------------------
        //
        //-----------------------------------------------------------------------------
        void matmul_gemm_par_cuda_fixed_block_size_1d_extern_shared(
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

            dim3 const dimBlock(MATMUL_CUDA_FIXED_BLOCK_SIZE, MATMUL_CUDA_FIXED_BLOCK_SIZE);
            float const fGridThreadExtentX = ceil(((float)n) / ((float)MATMUL_CUDA_FIXED_BLOCK_SIZE));
            float const fGridThreadExtentY = ceil(((float)m) / ((float)MATMUL_CUDA_FIXED_BLOCK_SIZE));
            unsigned int const gridThreadExtentX = (unsigned int)fGridThreadExtentX;
            unsigned int const gridThreadExtentY = (unsigned int)fGridThreadExtentY;
            dim3 const dimGrid(gridThreadExtentX, gridThreadExtentY);

            matmul_gemm_par_cuda_fixed_block_size_1d_extern_shared_kernel<<<
                dimGrid,
                dimBlock,
                2u*sizeof(TElem)*MATMUL_CUDA_FIXED_BLOCK_SIZE*MATMUL_CUDA_FIXED_BLOCK_SIZE>>>(
                    m, n, k,
                    alpha,
                    A, lda,
                    B, ldb,
                    beta,
                    C, ldc);

            MATMUL_CUDA_RT_CHECK(cudaDeviceSynchronize());
        }
    #endif
    #ifdef MATMUL_BUILD_PAR_CUDA_MEMCPY_FIXED_BLOCK_SIZE
        //-----------------------------------------------------------------------------
        //
        //-----------------------------------------------------------------------------
        void matmul_gemm_par_cuda_memcpy_fixed_block_size_1d_extern_shared(
            TIdx const m, TIdx const n, TIdx const k,
            TElem const alpha,
            TElem const * const MATMUL_RESTRICT A, TIdx const lda,
            TElem const * const MATMUL_RESTRICT B, TIdx const ldb,
            TElem const beta,
            TElem * const MATMUL_RESTRICT C, TIdx const ldc)
        {
            matmul_gemm_wrap_memcpy_host_cuda_2d(
                m, n, k,
                alpha,
                A, lda,
                B, ldb,
                beta,
                C, ldc,
                matmul_gemm_par_cuda_fixed_block_size_1d_extern_shared);
        }
    #endif
    #ifdef MATMUL_BUILD_PAR_CUDA_DYN_BLOCK_SIZE
        //-----------------------------------------------------------------------------
        // This function only works for square blocks.
        //-----------------------------------------------------------------------------
        __global__ void matmul_gemm_par_cuda_dyn_block_size_1d_extern_shared_kernel(
            TIdx const m, TIdx const n, TIdx const k,
            TElem const alpha,
            TElem const * const MATMUL_RESTRICT A, TIdx const lda,
            TElem const * const MATMUL_RESTRICT B, TIdx const ldb,
            TElem const beta,
            TElem * const MATMUL_RESTRICT C, TIdx const ldc)
        {
            // Column and row of C to calculate.
            TIdx const gridThreadIdxX = blockIdx.x*blockDim.x + threadIdx.x;
            TIdx const gridThreadIdxY = blockIdx.y*blockDim.y + threadIdx.y;

            // Column and row inside the block of C to calculate.
            TIdx const blockThreadIdxX = threadIdx.x;
            TIdx const blockThreadIdxY = threadIdx.y;

            // The block threads extents.
            TIdx const blockThreadsExtentX = blockDim.x;
            TIdx const blockThreadsExtentY = blockDim.y;
            //assert(blockThreadsExtentX == blockThreadsExtentY);
            TIdx const & blockThreadsExtent = blockThreadsExtentX;

            // Shared memory used to store the current blocks of A and B.
            extern __shared__ TElem pBlockSharedA[];
            TElem * const pBlockSharedB(pBlockSharedA + blockThreadsExtentX*blockThreadsExtentY);

            TIdx const sharedBlockIdx1d(blockThreadIdxY*blockThreadsExtentX + blockThreadIdxX);

            // If the element corresponding to the current thread is outside of the respective matrix.
            bool const insideA = (gridThreadIdxY < m);
            bool const insideB = (gridThreadIdxX < n);
            bool const insideC = (insideA && insideB);

            TElem dotProduct(0);

            // Loop over all blocks of A and B that are required to compute the C block.
            TIdx const blockMulCount(
                static_cast<TIdx>(
                    ceil(
                        static_cast<float>(k) / static_cast<float>(blockThreadsExtent))));
            for (TIdx k2(0); k2<blockMulCount; ++k2)
            {
                // Copy the current blocks of A and B into shared memory in parallel.
                // If the element of the current thread is outside of the matrix, zero is written into the shared memory.
                // This is possible because zero is a result neutral extension of the matrices regarding the dot product.
                TIdx const AIdxX(k2*blockThreadsExtentX + blockThreadIdxX);
                TIdx const AIdx1d(gridThreadIdxY*lda + AIdxX);
                pBlockSharedA[sharedBlockIdx1d] =
                    ((!insideA) || (AIdxX >= k))
                    ? static_cast<TElem>(0)
                    : A[AIdx1d];

                TIdx const BIdxY(k2*blockThreadsExtentY + blockThreadIdxY);
                TIdx const BIdx1d(BIdxY*ldb + gridThreadIdxX);
                pBlockSharedB[sharedBlockIdx1d] =
                    ((!insideB) || (BIdxY >= k))
                    ? static_cast<TElem>(0)
                    : B[BIdx1d];

                // Synchronize to make sure the complete blocks are loaded before starting the computation.
                __syncthreads();

                // Compute the dot products within shared memory.
                for (TIdx k3(0); k3<blockThreadsExtent; ++k3)
                {
                    dotProduct += pBlockSharedA[blockThreadIdxY*blockThreadsExtentX + k3]
                        * pBlockSharedB[k3*blockThreadsExtentY + blockThreadIdxX];
                }

                // Synchronize to make sure that the preceding computation is done before loading the next blocks of A and B.
                __syncthreads();
            }

            if (insideC)
            {
                TIdx const CIdx1d(gridThreadIdxY*ldc + gridThreadIdxX);
                C[CIdx1d] = alpha * dotProduct + beta * C[CIdx1d];
            }
        }
        //-----------------------------------------------------------------------------
        //
        //-----------------------------------------------------------------------------
        void matmul_gemm_par_cuda_dyn_block_size_1d_extern_shared(
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

            // MATMUL_CUDA_RT_CHECK(cudaSetDevice(0));
            cudaStream_t stream;
            MATMUL_CUDA_RT_CHECK(cudaStreamCreate(&stream));

            // Get its properties.
            cudaDeviceProp cudaDevProp;
            MATMUL_CUDA_RT_CHECK(cudaGetDeviceProperties(
                &cudaDevProp,
                0));

            TIdx gridThreadExtents[] = {m, n};
            TIdx blockThreadExtents[] = {cudaDevProp.maxThreadsDim[0], cudaDevProp.maxThreadsDim[1]};

            // Restrict the max block thread extents with the grid thread extents.
            // This removes dimensions not required in the given grid thread extents.
            // This has to be done before the maxThreadsPerBlock clipping to get the maximum correctly.
            for (TIdx i(0); i<2; ++i)
            {
                blockThreadExtents[i] = std::min(blockThreadExtents[i], gridThreadExtents[i]);
            }

            // Restrict it to its minimum component.
            // For example (512, 256) will get (256, 256).
            auto minBlockThreadExtent(blockThreadExtents[0]);
            for (TIdx i(1); i<2; ++i)
            {
                minBlockThreadExtent = std::min(minBlockThreadExtent, blockThreadExtents[i]);
            }
            for (TIdx i(0); i<2; ++i)
            {
                blockThreadExtents[i] = minBlockThreadExtent;
            }

            // Adjust blockThreadExtents if its product is too large.
            if ((blockThreadExtents[0] * blockThreadExtents[1]) > cudaDevProp.maxThreadsPerBlock)
            {
                // Satisfy the following equation:
                // udaDevProp.maxThreadsPerBlock >= blockThreadExtents[0]*blockThreadExtents[1]
                // For example 1024 >= 512 * 512

                // For equal block thread extent this is easily the nth root of cudaDevProp.maxThreadsPerBlock.
                double const fNthRoot(std::pow(cudaDevProp.maxThreadsPerBlock, 1.0 / 2.0));
                auto const nthRoot(static_cast<TIdx>(fNthRoot));
                for (TIdx i(0); i<2; ++i)
                {
                    blockThreadExtents[i] = nthRoot;
                }
            }

            // Set the grid block extents (rounded to the next integer not less then the quotient.
            TIdx gridBlockExtents[] = {1, 1};
            for (TIdx i(0); i<2; ++i)
            {
                gridBlockExtents[i] =
                    static_cast<TIdx>(
                        std::ceil(static_cast<double>(gridThreadExtents[i])
                            / static_cast<double>(blockThreadExtents[i])));
            }

            dim3 const dimBlock(blockThreadExtents[0], blockThreadExtents[1]);
            dim3 const dimGrid(gridBlockExtents[0], gridBlockExtents[1]);

            matmul_gemm_par_cuda_dyn_block_size_1d_extern_shared_kernel<<<
                dimGrid,
                dimBlock,
                2u*sizeof(TElem)*blockThreadExtents[0] * blockThreadExtents[1],
                stream>>>(
                    m, n, k,
                    alpha,
                    A, lda,
                    B, ldb,
                    beta,
                    C, ldc);

            MATMUL_CUDA_RT_CHECK(cudaStreamSynchronize(stream));
            MATMUL_CUDA_RT_CHECK(cudaStreamDestroy(stream));
        }
    #endif
    #ifdef MATMUL_BUILD_PAR_CUDA_MEMCPY_DYN_BLOCK_SIZE
        //-----------------------------------------------------------------------------
        //
        //-----------------------------------------------------------------------------
        void matmul_gemm_par_cuda_memcpy_dyn_block_size_1d_extern_shared(
            TIdx const m, TIdx const n, TIdx const k,
            TElem const alpha,
            TElem const * const MATMUL_RESTRICT A, TIdx const lda,
            TElem const * const MATMUL_RESTRICT B, TIdx const ldb,
            TElem const beta,
            TElem * const MATMUL_RESTRICT C, TIdx const ldc)
        {
            matmul_gemm_wrap_memcpy_host_cuda_2d(
                m, n, k,
                alpha,
                A, lda,
                B, ldb,
                beta,
                C, ldc,
                matmul_gemm_par_cuda_dyn_block_size_1d_extern_shared);
        }
    #endif
#endif
