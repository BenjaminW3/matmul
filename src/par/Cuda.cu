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
            TIdx const uiGridThreadIdxX = blockIdx.x*blockDim.x + threadIdx.x;
            TIdx const uiGridThreadIdxY = blockIdx.y*blockDim.y + threadIdx.y;

            // Column and row inside the block of C to calculate.
            TIdx const uiBlockThreadIdxX = threadIdx.x;
            TIdx const uiBlockThreadIdxY = threadIdx.y;

            // The block threads extents.
            TIdx const uiBlockThreadsExtentX = blockDim.x;
            TIdx const uiBlockThreadsExtentY = blockDim.y;
            //assert(uiBlockThreadsExtentX == uiBlockThreadsExtentY);
            TIdx const & uiBlockThreadsExtent = uiBlockThreadsExtentX;

            // Shared memory used to store the current blocks of A and B.
            __shared__ TElem pBlockSharedA[MATMUL_CUDA_FIXED_BLOCK_SIZE][MATMUL_CUDA_FIXED_BLOCK_SIZE];
            __shared__ TElem pBlockSharedB[MATMUL_CUDA_FIXED_BLOCK_SIZE][MATMUL_CUDA_FIXED_BLOCK_SIZE];

            // If the element corresponding to the current thread is outside of the respective matrix.
            bool const bInsideA = (uiGridThreadIdxY < m);
            bool const bInsideB = (uiGridThreadIdxX < n);
            bool const bInsideC = (bInsideA && bInsideB);

            TElem dotProduct(0);

            // Loop over all blocks of A and B that are required to compute the C block.
            auto const uiBlockMulCount(
                static_cast<TIdx>(
                    ceil(
                        static_cast<float>(k)/static_cast<float>(uiBlockThreadsExtent))));
            for(TIdx k2=0; k2<uiBlockMulCount; ++k2)
            {
                // Copy the current blocks of A and B into shared memory in parallel.
                // If the element of the current thread is outside of the matrix, zero is written into the shared memory.
                // This is possible because zero is a result neutral extension of the matrices regarding the dot product.
                TIdx const uiAIdxX(k2*uiBlockThreadsExtentX + uiBlockThreadIdxX);
                TIdx const uiAIdx1d(uiGridThreadIdxY*lda + uiAIdxX);
                pBlockSharedA[uiBlockThreadIdxY][uiBlockThreadIdxX] =
                    ((!bInsideA) || (uiAIdxX>=k))
                    ? static_cast<TElem>(0)
                    : A[uiAIdx1d];

                TIdx const uiBIdxY(k2*uiBlockThreadsExtentY + uiBlockThreadIdxY);
                TIdx const uiBIdx1d(uiBIdxY*ldb + uiGridThreadIdxX);
                pBlockSharedB[uiBlockThreadIdxY][uiBlockThreadIdxX] =
                    ((!bInsideB) || (uiBIdxY>=k))
                    ? static_cast<TElem>(0)
                    : B[uiBIdx1d];

                // Synchronize to make sure the complete blocks are loaded before starting the computation.
                __syncthreads();

                // Compute the dot products within shared memory.
                for(TIdx k3 = 0; k3<uiBlockThreadsExtent; ++k3)
                {
                    dotProduct += pBlockSharedA[uiBlockThreadIdxY][k3]
                        * pBlockSharedB[k3][uiBlockThreadIdxX];
                }

                // Synchronize to make sure that the preceding computation is done before loading the next blocks of A and B.
                __syncthreads();
            }

            if(bInsideC)
            {
                auto const uiIdxC1d(uiGridThreadIdxY*ldc + uiGridThreadIdxX);
                C[uiIdxC1d] = alpha * dotProduct + beta * C[uiIdxC1d];
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
            unsigned int const uiGridThreadExtentX = (unsigned int)fGridThreadExtentX;
            unsigned int const uiGridThreadExtentY = (unsigned int)fGridThreadExtentY;
            dim3 const dimGrid(uiGridThreadExtentX, uiGridThreadExtentY);

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
            TIdx const uiGridThreadIdxX = blockIdx.x*blockDim.x + threadIdx.x;
            TIdx const uiGridThreadIdxY = blockIdx.y*blockDim.y + threadIdx.y;

            // Column and row inside the block of C to calculate.
            TIdx const uiBlockThreadIdxX = threadIdx.x;
            TIdx const uiBlockThreadIdxY = threadIdx.y;

            // The block threads extents.
            TIdx const uiBlockThreadsExtentX = blockDim.x;
            TIdx const uiBlockThreadsExtentY = blockDim.y;
            //assert(uiBlockThreadsExtentX == uiBlockThreadsExtentY);
            TIdx const & uiBlockThreadsExtent = uiBlockThreadsExtentX;

            // Shared memory used to store the current blocks of A and B.
            __shared__ TElem pBlockSharedA[MATMUL_CUDA_FIXED_BLOCK_SIZE*MATMUL_CUDA_FIXED_BLOCK_SIZE];
            __shared__ TElem pBlockSharedB[MATMUL_CUDA_FIXED_BLOCK_SIZE*MATMUL_CUDA_FIXED_BLOCK_SIZE];

            auto const uiSharedBlockIdx1d(uiBlockThreadIdxY*uiBlockThreadsExtentX + uiBlockThreadIdxX);

            // If the element corresponding to the current thread is outside of the respective matrix.
            bool const bInsideA = (uiGridThreadIdxY < m);
            bool const bInsideB = (uiGridThreadIdxX < n);
            bool const bInsideC = (bInsideA && bInsideB);

            TElem dotProduct(0);

            // Loop over all blocks of A and B that are required to compute the C block.
            auto const uiBlockMulCount(
                static_cast<TIdx>(
                    ceil(
                        static_cast<float>(k)/static_cast<float>(uiBlockThreadsExtent))));
            for(TIdx k2=0; k2<uiBlockMulCount; ++k2)
            {
                // Copy the current blocks of A and B into shared memory in parallel.
                // If the element of the current thread is outside of the matrix, zero is written into the shared memory.
                // This is possible because zero is a result neutral extension of the matrices regarding the dot product.
                TIdx const uiAIdxX(k2*uiBlockThreadsExtentX + uiBlockThreadIdxX);
                TIdx const uiAIdx1d(uiGridThreadIdxY*lda + uiAIdxX);
                pBlockSharedA[uiSharedBlockIdx1d] =
                    ((!bInsideA) || (uiAIdxX>=k))
                    ? static_cast<TElem>(0)
                    : A[uiAIdx1d];

                TIdx const uiBIdxY(k2*uiBlockThreadsExtentY + uiBlockThreadIdxY);
                TIdx const uiBIdx1d(uiBIdxY*ldb + uiGridThreadIdxX);
                pBlockSharedB[uiSharedBlockIdx1d] =
                    ((!bInsideB) || (uiBIdxY>=k))
                    ? static_cast<TElem>(0)
                    : B[uiBIdx1d];

                // Synchronize to make sure the complete blocks are loaded before starting the computation.
                __syncthreads();

                // Compute the dot products within shared memory.
                for(TIdx k3 = 0; k3<uiBlockThreadsExtent; ++k3)
                {
                    dotProduct += pBlockSharedA[uiBlockThreadIdxY*uiBlockThreadsExtentX + k3]
                        * pBlockSharedB[k3*uiBlockThreadsExtentY + uiBlockThreadIdxX];
                }

                // Synchronize to make sure that the preceding computation is done before loading the next blocks of A and B.
                __syncthreads();
            }

            if(bInsideC)
            {
                auto const uiIdxC1d(uiGridThreadIdxY*ldc + uiGridThreadIdxX);
                C[uiIdxC1d] = alpha * dotProduct + beta * C[uiIdxC1d];
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
            unsigned int const uiGridThreadExtentX = (unsigned int)fGridThreadExtentX;
            unsigned int const uiGridThreadExtentY = (unsigned int)fGridThreadExtentY;
            dim3 const dimGrid(uiGridThreadExtentX, uiGridThreadExtentY);

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
            TIdx const uiGridThreadIdxX = blockIdx.x*blockDim.x + threadIdx.x;
            TIdx const uiGridThreadIdxY = blockIdx.y*blockDim.y + threadIdx.y;

            // Column and row inside the block of C to calculate.
            TIdx const uiBlockThreadIdxX = threadIdx.x;
            TIdx const uiBlockThreadIdxY = threadIdx.y;

            // The block threads extents.
            TIdx const uiBlockThreadsExtentX = blockDim.x;
            TIdx const uiBlockThreadsExtentY = blockDim.y;
            //assert(uiBlockThreadsExtentX == uiBlockThreadsExtentY);
            TIdx const & uiBlockThreadsExtent = uiBlockThreadsExtentX;

            // Shared memory used to store the current blocks of A and B.
            extern __shared__ TElem pBlockSharedA[];
            auto * const pBlockSharedB(pBlockSharedA + uiBlockThreadsExtentX*uiBlockThreadsExtentY);

            auto const uiSharedBlockIdx1d(uiBlockThreadIdxY*uiBlockThreadsExtentX + uiBlockThreadIdxX);

            // If the element corresponding to the current thread is outside of the respective matrix.
            bool const bInsideA = (uiGridThreadIdxY < m);
            bool const bInsideB = (uiGridThreadIdxX < n);
            bool const bInsideC = (bInsideA && bInsideB);

            TElem dotProduct(0);

            // Loop over all blocks of A and B that are required to compute the C block.
            auto const uiBlockMulCount(
                static_cast<TIdx>(
                    ceil(
                        static_cast<float>(k)/static_cast<float>(uiBlockThreadsExtent))));
            for(TIdx k2=0; k2<uiBlockMulCount; ++k2)
            {
                // Copy the current blocks of A and B into shared memory in parallel.
                // If the element of the current thread is outside of the matrix, zero is written into the shared memory.
                // This is possible because zero is a result neutral extension of the matrices regarding the dot product.
                TIdx const uiAIdxX(k2*uiBlockThreadsExtentX + uiBlockThreadIdxX);
                TIdx const uiAIdx1d(uiGridThreadIdxY*lda + uiAIdxX);
                pBlockSharedA[uiSharedBlockIdx1d] =
                    ((!bInsideA) || (uiAIdxX>=k))
                    ? static_cast<TElem>(0)
                    : A[uiAIdx1d];

                TIdx const uiBIdxY(k2*uiBlockThreadsExtentY + uiBlockThreadIdxY);
                TIdx const uiBIdx1d(uiBIdxY*ldb + uiGridThreadIdxX);
                pBlockSharedB[uiSharedBlockIdx1d] =
                    ((!bInsideB) || (uiBIdxY>=k))
                    ? static_cast<TElem>(0)
                    : B[uiBIdx1d];

                // Synchronize to make sure the complete blocks are loaded before starting the computation.
                __syncthreads();

                // Compute the dot products within shared memory.
                for(TIdx k3 = 0; k3<uiBlockThreadsExtent; ++k3)
                {
                    dotProduct += pBlockSharedA[uiBlockThreadIdxY*uiBlockThreadsExtentX + k3]
                        * pBlockSharedB[k3*uiBlockThreadsExtentY + uiBlockThreadIdxX];
                }

                // Synchronize to make sure that the preceding computation is done before loading the next blocks of A and B.
                __syncthreads();
            }

            if(bInsideC)
            {
                auto const uiIdxC1d(uiGridThreadIdxY*ldc + uiGridThreadIdxX);
                C[uiIdxC1d] = alpha * dotProduct + beta * C[uiIdxC1d];
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
            unsigned int const uiGridThreadExtentX = (unsigned int)fGridThreadExtentX;
            unsigned int const uiGridThreadExtentY = (unsigned int)fGridThreadExtentY;
            dim3 const dimGrid(uiGridThreadExtentX, uiGridThreadExtentY);

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
            TIdx const uiGridThreadIdxX = blockIdx.x*blockDim.x + threadIdx.x;
            TIdx const uiGridThreadIdxY = blockIdx.y*blockDim.y + threadIdx.y;

            // Column and row inside the block of C to calculate.
            TIdx const uiBlockThreadIdxX = threadIdx.x;
            TIdx const uiBlockThreadIdxY = threadIdx.y;

            // The block threads extents.
            TIdx const uiBlockThreadsExtentX = blockDim.x;
            TIdx const uiBlockThreadsExtentY = blockDim.y;
            //assert(uiBlockThreadsExtentX == uiBlockThreadsExtentY);
            TIdx const & uiBlockThreadsExtent = uiBlockThreadsExtentX;

            // Shared memory used to store the current blocks of A and B.
            extern __shared__ TElem pBlockSharedA[];
            TElem * const pBlockSharedB(pBlockSharedA + uiBlockThreadsExtentX*uiBlockThreadsExtentY);

            TIdx const uiSharedBlockIdx1d(uiBlockThreadIdxY*uiBlockThreadsExtentX + uiBlockThreadIdxX);

            // If the element corresponding to the current thread is outside of the respective matrix.
            bool const bInsideA = (uiGridThreadIdxY < m);
            bool const bInsideB = (uiGridThreadIdxX < n);
            bool const bInsideC = (bInsideA && bInsideB);

            TElem dotProduct(0);

            // Loop over all blocks of A and B that are required to compute the C block.
            TIdx const uiBlockMulCount(
                static_cast<TIdx>(
                    ceil(
                        static_cast<float>(k) / static_cast<float>(uiBlockThreadsExtent))));
            for (TIdx k2(0); k2<uiBlockMulCount; ++k2)
            {
                // Copy the current blocks of A and B into shared memory in parallel.
                // If the element of the current thread is outside of the matrix, zero is written into the shared memory.
                // This is possible because zero is a result neutral extension of the matrices regarding the dot product.
                TIdx const uiAIdxX(k2*uiBlockThreadsExtentX + uiBlockThreadIdxX);
                TIdx const uiAIdx1d(uiGridThreadIdxY*lda + uiAIdxX);
                pBlockSharedA[uiSharedBlockIdx1d] =
                    ((!bInsideA) || (uiAIdxX >= k))
                    ? static_cast<TElem>(0)
                    : A[uiAIdx1d];

                TIdx const uiBIdxY(k2*uiBlockThreadsExtentY + uiBlockThreadIdxY);
                TIdx const uiBIdx1d(uiBIdxY*ldb + uiGridThreadIdxX);
                pBlockSharedB[uiSharedBlockIdx1d] =
                    ((!bInsideB) || (uiBIdxY >= k))
                    ? static_cast<TElem>(0)
                    : B[uiBIdx1d];

                // Synchronize to make sure the complete blocks are loaded before starting the computation.
                __syncthreads();

                // Compute the dot products within shared memory.
                for (TIdx k3(0); k3<uiBlockThreadsExtent; ++k3)
                {
                    dotProduct += pBlockSharedA[uiBlockThreadIdxY*uiBlockThreadsExtentX + k3]
                        * pBlockSharedB[k3*uiBlockThreadsExtentY + uiBlockThreadIdxX];
                }

                // Synchronize to make sure that the preceding computation is done before loading the next blocks of A and B.
                __syncthreads();
            }

            if (bInsideC)
            {
                TIdx const uiIdxC1d(uiGridThreadIdxY*ldc + uiGridThreadIdxX);
                C[uiIdxC1d] = alpha * dotProduct + beta * C[uiIdxC1d];
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

            TIdx vuiGridThreadExtents[] = { m, n };
            TIdx vuiBlockThreadExtents[] = { cudaDevProp.maxThreadsDim[0], cudaDevProp.maxThreadsDim[1] };

            // Restrict the max block thread extents with the grid thread extents.
            // This removes dimensions not required in the given grid thread extents.
            // This has to be done before the uiMaxBlockThreadsCount clipping to get the maximum correctly.
            for (TIdx i(0); i<2; ++i)
            {
                vuiBlockThreadExtents[i] = std::min(vuiBlockThreadExtents[i], vuiGridThreadExtents[i]);
            }

            // Restrict it to its minimum component.
            // For example (512, 256) will get (256, 256).
            auto uiMinBlockThreadExtent(vuiBlockThreadExtents[0]);
            for (TIdx i(1); i<2; ++i)
            {
                uiMinBlockThreadExtent = std::min(uiMinBlockThreadExtent, vuiBlockThreadExtents[i]);
            }
            for (TIdx i(0); i<2; ++i)
            {
                vuiBlockThreadExtents[i] = uiMinBlockThreadExtent;
            }

            // Adjust vuiBlockThreadExtents if its product is too large.
            if ((vuiBlockThreadExtents[0] * vuiBlockThreadExtents[1]) > cudaDevProp.maxThreadsPerBlock)
            {
                // Satisfy the following equation:
                // udaDevProp.maxThreadsPerBlock >= vuiBlockThreadExtents[0]*vuiBlockThreadExtents[1]
                // For example 1024 >= 512 * 512

                // For equal block thread extent this is easily the nth root of cudaDevProp.maxThreadsPerBlock.
                double const fNthRoot(std::pow(cudaDevProp.maxThreadsPerBlock, 1.0 / 2.0));
                auto const uiNthRoot(static_cast<TIdx>(fNthRoot));
                for (TIdx i(0); i<2; ++i)
                {
                    vuiBlockThreadExtents[i] = uiNthRoot;
                }
            }

            // Set the grid block extents (rounded to the next integer not less then the quotient.
            TIdx vuiGridBlockExtents[] = { 1, 1 };
            for (TIdx i(0); i<2; ++i)
            {
                vuiGridBlockExtents[i] =
                    static_cast<TIdx>(
                        std::ceil(static_cast<double>(vuiGridThreadExtents[i])
                            / static_cast<double>(vuiBlockThreadExtents[i])));
            }

            dim3 const dimBlock(vuiBlockThreadExtents[0], vuiBlockThreadExtents[1]);
            dim3 const dimGrid(vuiGridBlockExtents[0], vuiGridBlockExtents[1]);

            MATMUL_CUDA_RT_CHECK(cudaSetDevice(0));
            matmul_gemm_par_cuda_dyn_block_size_1d_extern_shared_kernel<<<
                dimGrid,
                dimBlock,
                2u*sizeof(TElem)*vuiBlockThreadExtents[0] * vuiBlockThreadExtents[1],
                stream>>>(
                    m, n, k,
                    alpha,
                    A, lda,
                    B, ldb,
                    beta,
                    C, ldc);

            // MATMUL_CUDA_RT_CHECK(cudaSetDevice(0));
            MATMUL_CUDA_RT_CHECK(cudaStreamSynchronize(stream));
            // MATMUL_CUDA_RT_CHECK(cudaSetDevice(0));
            MATMUL_CUDA_RT_CHECK(cudaStreamDestroy(stream));

            //MATMUL_CUDA_RT_CHECK(cudaDeviceSynchronize());
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
