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
            TSize const m, TSize const n, TSize const k,
            TElem const alpha,
            TElem const * const MATMUL_RESTRICT A, TSize const lda,
            TElem const * const MATMUL_RESTRICT B, TSize const ldb,
            TElem const beta,
            TElem * const MATMUL_RESTRICT C, TSize const ldc)
        {
            // Column and row of C to calculate.
            TSize const gridThreadIdxX = blockIdx.x*blockDim.x + threadIdx.x;
            TSize const gridThreadIdxY = blockIdx.y*blockDim.y + threadIdx.y;

            // Column and row inside the block of C to calculate.
            TSize const blockThreadIdxX = threadIdx.x;
            TSize const blockThreadIdxY = threadIdx.y;

            // The block threads extents.
            TSize const blockThreadExtentX = blockDim.x;
            TSize const blockThreadExtentY = blockDim.y;
            //assert(blockThreadExtentX == blockThreadExtentY);
            TSize const & blockThreadExtent = blockThreadExtentX;

            // Shared memory used to store the current blocks of A and B.
            __shared__ TElem pBlockSharedA[MATMUL_CUDA_FIXED_BLOCK_SIZE][MATMUL_CUDA_FIXED_BLOCK_SIZE];
            __shared__ TElem pBlockSharedB[MATMUL_CUDA_FIXED_BLOCK_SIZE][MATMUL_CUDA_FIXED_BLOCK_SIZE];

            // If the element corresponding to the current thread is outside of the respective matrix.
            bool const insideA = (gridThreadIdxY < m);
            bool const insideB = (gridThreadIdxX < n);
            bool const insideC = (insideA && insideB);

            TElem dotProduct(0);

            // Loop over all blocks of A and B that are required to compute the C block.
            TSize const blockMulCount(
                static_cast<TSize>(
                    ceil(
                        static_cast<float>(k)/static_cast<float>(blockThreadExtent))));
            for(TSize k2=0; k2<blockMulCount; ++k2)
            {
                // Copy the current blocks of A and B into shared memory in parallel.
                // If the element of the current thread is outside of the matrix, zero is written into the shared memory.
                // This is possible because zero is a result neutral extension of the matrices regarding the dot product.
                TSize const AIdxX(k2*blockThreadExtentX + blockThreadIdxX);
                TSize const AIdx1d(gridThreadIdxY*lda + AIdxX);
                pBlockSharedA[blockThreadIdxY][blockThreadIdxX] =
                    ((!insideA) || (AIdxX>=k))
                    ? static_cast<TElem>(0)
                    : A[AIdx1d];

                TSize const BIdxY(k2*blockThreadExtentY + blockThreadIdxY);
                TSize const BIdx1d(BIdxY*ldb + gridThreadIdxX);
                pBlockSharedB[blockThreadIdxY][blockThreadIdxX] =
                    ((!insideB) || (BIdxY>=k))
                    ? static_cast<TElem>(0)
                    : B[BIdx1d];

                // Synchronize to make sure the complete blocks are loaded before starting the computation.
                __syncthreads();

                // Compute the dot products within shared memory.
                for(TSize k3 = 0; k3<blockThreadExtent; ++k3)
                {
                    dotProduct += pBlockSharedA[blockThreadIdxY][k3]
                        * pBlockSharedB[k3][blockThreadIdxX];
                }

                // Synchronize to make sure that the preceding computation is done before loading the next blocks of A and B.
                __syncthreads();
            }

            if(insideC)
            {
                TSize const CIdx1d(gridThreadIdxY*ldc + gridThreadIdxX);
                C[CIdx1d] = alpha * dotProduct + beta * C[CIdx1d];
            }
        }
        //-----------------------------------------------------------------------------
        //
        //-----------------------------------------------------------------------------
        TReturn matmul_gemm_par_cuda_fixed_block_size_2d_static_shared(
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

            dim3 const dimBlock(MATMUL_CUDA_FIXED_BLOCK_SIZE, MATMUL_CUDA_FIXED_BLOCK_SIZE);
            float const fGridThreadExtentX = ceil(((float)n) / ((float)MATMUL_CUDA_FIXED_BLOCK_SIZE));
            float const fGridThreadExtentY = ceil(((float)m) / ((float)MATMUL_CUDA_FIXED_BLOCK_SIZE));
            unsigned int const gridThreadExtentX = (unsigned int)fGridThreadExtentX;
            unsigned int const gridThreadExtentY = (unsigned int)fGridThreadExtentY;
            dim3 const dimGrid(gridThreadExtentX, gridThreadExtentY);

            MATMUL_TIME_START;

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
            
            MATMUL_TIME_END;
            MATMUL_TIME_RETURN;
        }
    #endif
    #ifdef MATMUL_BUILD_PAR_CUDA_MEMCPY_FIXED_BLOCK_SIZE
        //-----------------------------------------------------------------------------
        //
        //-----------------------------------------------------------------------------
        TReturn matmul_gemm_par_cuda_memcpy_fixed_block_size_2d_static_shared(
            TSize const m, TSize const n, TSize const k,
            TElem const alpha,
            TElem const * const MATMUL_RESTRICT A, TSize const lda,
            TElem const * const MATMUL_RESTRICT B, TSize const ldb,
            TElem const beta,
            TElem * const MATMUL_RESTRICT C, TSize const ldc)
        {
            return
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
            TSize const m, TSize const n, TSize const k,
            TElem const alpha,
            TElem const * const MATMUL_RESTRICT A, TSize const lda,
            TElem const * const MATMUL_RESTRICT B, TSize const ldb,
            TElem const beta,
            TElem * const MATMUL_RESTRICT C, TSize const ldc)
        {
            // Column and row of C to calculate.
            TSize const gridThreadIdxX = blockIdx.x*blockDim.x + threadIdx.x;
            TSize const gridThreadIdxY = blockIdx.y*blockDim.y + threadIdx.y;

            // Column and row inside the block of C to calculate.
            TSize const blockThreadIdxX = threadIdx.x;
            TSize const blockThreadIdxY = threadIdx.y;

            // The block threads extents.
            TSize const blockThreadExtentX = blockDim.x;
            TSize const blockThreadExtentY = blockDim.y;
            //assert(blockThreadExtentX == blockThreadExtentY);
            TSize const & blockThreadExtent = blockThreadExtentX;

            // Shared memory used to store the current blocks of A and B.
            __shared__ TElem pBlockSharedA[MATMUL_CUDA_FIXED_BLOCK_SIZE*MATMUL_CUDA_FIXED_BLOCK_SIZE];
            __shared__ TElem pBlockSharedB[MATMUL_CUDA_FIXED_BLOCK_SIZE*MATMUL_CUDA_FIXED_BLOCK_SIZE];

            TSize const sharedBlockIdx1d(blockThreadIdxY*blockThreadExtentX + blockThreadIdxX);

            // If the element corresponding to the current thread is outside of the respective matrix.
            bool const insideA = (gridThreadIdxY < m);
            bool const insideB = (gridThreadIdxX < n);
            bool const insideC = (insideA && insideB);

            TElem dotProduct(0);

            // Loop over all blocks of A and B that are required to compute the C block.
            TSize const blockMulCount(
                static_cast<TSize>(
                    ceil(
                        static_cast<float>(k)/static_cast<float>(blockThreadExtent))));
            for(TSize k2=0; k2<blockMulCount; ++k2)
            {
                // Copy the current blocks of A and B into shared memory in parallel.
                // If the element of the current thread is outside of the matrix, zero is written into the shared memory.
                // This is possible because zero is a result neutral extension of the matrices regarding the dot product.
                TSize const AIdxX(k2*blockThreadExtentX + blockThreadIdxX);
                TSize const AIdx1d(gridThreadIdxY*lda + AIdxX);
                pBlockSharedA[sharedBlockIdx1d] =
                    ((!insideA) || (AIdxX>=k))
                    ? static_cast<TElem>(0)
                    : A[AIdx1d];

                TSize const BIdxY(k2*blockThreadExtentY + blockThreadIdxY);
                TSize const BIdx1d(BIdxY*ldb + gridThreadIdxX);
                pBlockSharedB[sharedBlockIdx1d] =
                    ((!insideB) || (BIdxY>=k))
                    ? static_cast<TElem>(0)
                    : B[BIdx1d];

                // Synchronize to make sure the complete blocks are loaded before starting the computation.
                __syncthreads();

                // Compute the dot products within shared memory.
                for(TSize k3 = 0; k3<blockThreadExtent; ++k3)
                {
                    dotProduct += pBlockSharedA[blockThreadIdxY*blockThreadExtentX + k3]
                        * pBlockSharedB[k3*blockThreadExtentY + blockThreadIdxX];
                }

                // Synchronize to make sure that the preceding computation is done before loading the next blocks of A and B.
                __syncthreads();
            }

            if(insideC)
            {
                TSize const CIdx1d(gridThreadIdxY*ldc + gridThreadIdxX);
                C[CIdx1d] = alpha * dotProduct + beta * C[CIdx1d];
            }
        }
        //-----------------------------------------------------------------------------
        //
        //-----------------------------------------------------------------------------
        TReturn matmul_gemm_par_cuda_fixed_block_size_1d_static_shared(
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

            dim3 const dimBlock(MATMUL_CUDA_FIXED_BLOCK_SIZE, MATMUL_CUDA_FIXED_BLOCK_SIZE);
            float const fGridThreadExtentX = ceil(((float)n) / ((float)MATMUL_CUDA_FIXED_BLOCK_SIZE));
            float const fGridThreadExtentY = ceil(((float)m) / ((float)MATMUL_CUDA_FIXED_BLOCK_SIZE));
            unsigned int const gridThreadExtentX = (unsigned int)fGridThreadExtentX;
            unsigned int const gridThreadExtentY = (unsigned int)fGridThreadExtentY;
            dim3 const dimGrid(gridThreadExtentX, gridThreadExtentY);

            MATMUL_TIME_START;

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
            
            MATMUL_TIME_END;
            MATMUL_TIME_RETURN;
        }
    #endif
    #ifdef MATMUL_BUILD_PAR_CUDA_MEMCPY_FIXED_BLOCK_SIZE
        //-----------------------------------------------------------------------------
        //
        //-----------------------------------------------------------------------------
        TReturn matmul_gemm_par_cuda_memcpy_fixed_block_size_1d_static_shared(
            TSize const m, TSize const n, TSize const k,
            TElem const alpha,
            TElem const * const MATMUL_RESTRICT A, TSize const lda,
            TElem const * const MATMUL_RESTRICT B, TSize const ldb,
            TElem const beta,
            TElem * const MATMUL_RESTRICT C, TSize const ldc)
        {
            return
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
            TSize const m, TSize const n, TSize const k,
            TElem const alpha,
            TElem const * const MATMUL_RESTRICT A, TSize const lda,
            TElem const * const MATMUL_RESTRICT B, TSize const ldb,
            TElem const beta,
            TElem * const MATMUL_RESTRICT C, TSize const ldc)
        {
            // Column and row of C to calculate.
            TSize const gridThreadIdxX = blockIdx.x*blockDim.x + threadIdx.x;
            TSize const gridThreadIdxY = blockIdx.y*blockDim.y + threadIdx.y;

            // Column and row inside the block of C to calculate.
            TSize const blockThreadIdxX = threadIdx.x;
            TSize const blockThreadIdxY = threadIdx.y;

            // The block threads extents.
            TSize const blockThreadExtentX = blockDim.x;
            TSize const blockThreadExtentY = blockDim.y;
            //assert(blockThreadExtentX == blockThreadExtentY);
            TSize const & blockThreadExtent = blockThreadExtentX;

            // Shared memory used to store the current blocks of A and B.
            extern __shared__ TElem pBlockSharedA[];
            TElem * const pBlockSharedB(pBlockSharedA + blockThreadExtentX*blockThreadExtentY);

            TSize const sharedBlockIdx1d(blockThreadIdxY*blockThreadExtentX + blockThreadIdxX);

            // If the element corresponding to the current thread is outside of the respective matrix.
            bool const insideA = (gridThreadIdxY < m);
            bool const insideB = (gridThreadIdxX < n);
            bool const insideC = (insideA && insideB);

            TElem dotProduct(0);

            // Loop over all blocks of A and B that are required to compute the C block.
            TSize const blockMulCount(
                static_cast<TSize>(
                    ceil(
                        static_cast<float>(k)/static_cast<float>(blockThreadExtent))));
            for(TSize k2=0; k2<blockMulCount; ++k2)
            {
                // Copy the current blocks of A and B into shared memory in parallel.
                // If the element of the current thread is outside of the matrix, zero is written into the shared memory.
                // This is possible because zero is a result neutral extension of the matrices regarding the dot product.
                TSize const AIdxX(k2*blockThreadExtentX + blockThreadIdxX);
                TSize const AIdx1d(gridThreadIdxY*lda + AIdxX);
                pBlockSharedA[sharedBlockIdx1d] =
                    ((!insideA) || (AIdxX>=k))
                    ? static_cast<TElem>(0)
                    : A[AIdx1d];

                TSize const BIdxY(k2*blockThreadExtentY + blockThreadIdxY);
                TSize const BIdx1d(BIdxY*ldb + gridThreadIdxX);
                pBlockSharedB[sharedBlockIdx1d] =
                    ((!insideB) || (BIdxY>=k))
                    ? static_cast<TElem>(0)
                    : B[BIdx1d];

                // Synchronize to make sure the complete blocks are loaded before starting the computation.
                __syncthreads();

                // Compute the dot products within shared memory.
                for(TSize k3 = 0; k3<blockThreadExtent; ++k3)
                {
                    dotProduct += pBlockSharedA[blockThreadIdxY*blockThreadExtentX + k3]
                        * pBlockSharedB[k3*blockThreadExtentY + blockThreadIdxX];
                }

                // Synchronize to make sure that the preceding computation is done before loading the next blocks of A and B.
                __syncthreads();
            }

            if(insideC)
            {
                TSize const CIdx1d(gridThreadIdxY*ldc + gridThreadIdxX);
                C[CIdx1d] = alpha * dotProduct + beta * C[CIdx1d];
            }
        }
        //-----------------------------------------------------------------------------
        //
        //-----------------------------------------------------------------------------
        TReturn matmul_gemm_par_cuda_fixed_block_size_1d_extern_shared(
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

            dim3 const dimBlock(MATMUL_CUDA_FIXED_BLOCK_SIZE, MATMUL_CUDA_FIXED_BLOCK_SIZE);
            float const fGridThreadExtentX = ceil(((float)n) / ((float)MATMUL_CUDA_FIXED_BLOCK_SIZE));
            float const fGridThreadExtentY = ceil(((float)m) / ((float)MATMUL_CUDA_FIXED_BLOCK_SIZE));
            unsigned int const gridThreadExtentX = (unsigned int)fGridThreadExtentX;
            unsigned int const gridThreadExtentY = (unsigned int)fGridThreadExtentY;
            dim3 const dimGrid(gridThreadExtentX, gridThreadExtentY);

            MATMUL_TIME_START;

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
            
            MATMUL_TIME_END;
            MATMUL_TIME_RETURN;
        }
    #endif
    #ifdef MATMUL_BUILD_PAR_CUDA_MEMCPY_FIXED_BLOCK_SIZE
        //-----------------------------------------------------------------------------
        //
        //-----------------------------------------------------------------------------
        TReturn matmul_gemm_par_cuda_memcpy_fixed_block_size_1d_extern_shared(
            TSize const m, TSize const n, TSize const k,
            TElem const alpha,
            TElem const * const MATMUL_RESTRICT A, TSize const lda,
            TElem const * const MATMUL_RESTRICT B, TSize const ldb,
            TElem const beta,
            TElem * const MATMUL_RESTRICT C, TSize const ldc)
        {
            return
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
            TSize const m, TSize const n, TSize const k,
            TElem const alpha,
            TElem const * const MATMUL_RESTRICT A, TSize const lda,
            TElem const * const MATMUL_RESTRICT B, TSize const ldb,
            TElem const beta,
            TElem * const MATMUL_RESTRICT C, TSize const ldc)
        {
            // Column and row of C to calculate.
            TSize const gridThreadIdxX = blockIdx.x*blockDim.x + threadIdx.x;
            TSize const gridThreadIdxY = blockIdx.y*blockDim.y + threadIdx.y;

            // Column and row inside the block of C to calculate.
            TSize const blockThreadIdxX = threadIdx.x;
            TSize const blockThreadIdxY = threadIdx.y;

            // The block threads extents.
            TSize const blockThreadExtentX = blockDim.x;
            TSize const blockThreadExtentY = blockDim.y;
            //assert(blockThreadExtentX == blockThreadExtentY);
            TSize const & blockThreadExtent1d = blockThreadExtentX;

            // Shared memory used to store the current blocks of A and B.
            extern __shared__ TElem pBlockSharedA[];
            TElem * const pBlockSharedB(pBlockSharedA + blockThreadExtentX*blockThreadExtentY);

            TSize const sharedBlockIdx1d(blockThreadIdxY*blockThreadExtentX + blockThreadIdxX);

            // If the element corresponding to the current thread is outside of the respective matrix.
            bool const insideA = (gridThreadIdxY < m);
            bool const insideB = (gridThreadIdxX < n);
            bool const insideC = (insideA && insideB);

            TElem dotProduct(0);

            // Loop over all blocks of A and B that are required to compute the C block.
            TSize const blockMulCount(
                static_cast<TSize>(
                    ceil(
                        static_cast<float>(k) / static_cast<float>(blockThreadExtent1d))));
            for(TSize k2(0); k2<blockMulCount; ++k2)
            {
                // Copy the current blocks of A and B into shared memory in parallel.
                // If the element of the current thread is outside of the matrix, zero is written into the shared memory.
                // This is possible because zero is a result neutral extension of the matrices regarding the dot product.
                TSize const AIdxX(k2*blockThreadExtentX + blockThreadIdxX);
                TSize const AIdx1d(gridThreadIdxY*lda + AIdxX);
                pBlockSharedA[sharedBlockIdx1d] =
                    ((!insideA) || (AIdxX >= k))
                    ? static_cast<TElem>(0)
                    : A[AIdx1d];

                TSize const BIdxY(k2*blockThreadExtentY + blockThreadIdxY);
                TSize const BIdx1d(BIdxY*ldb + gridThreadIdxX);
                pBlockSharedB[sharedBlockIdx1d] =
                    ((!insideB) || (BIdxY >= k))
                    ? static_cast<TElem>(0)
                    : B[BIdx1d];

                // Synchronize to make sure the complete blocks are loaded before starting the computation.
                __syncthreads();

                // Compute the dot products within shared memory.
                for(TSize k3(0); k3<blockThreadExtent1d; ++k3)
                {
                    dotProduct += pBlockSharedA[blockThreadIdxY*blockThreadExtentX + k3]
                        * pBlockSharedB[k3*blockThreadExtentY + blockThreadIdxX];
                }

                // Synchronize to make sure that the preceding computation is done before loading the next blocks of A and B.
                __syncthreads();
            }

            if (insideC)
            {
                TSize const CIdx1d(gridThreadIdxY*ldc + gridThreadIdxX);
                C[CIdx1d] = alpha * dotProduct + beta * C[CIdx1d];
            }
        }
        //-----------------------------------------------------------------------------
        //
        //-----------------------------------------------------------------------------
        TReturn matmul_gemm_par_cuda_dyn_block_size_1d_extern_shared(
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

            // MATMUL_CUDA_RT_CHECK(cudaSetDevice(0));
            cudaStream_t stream;
            MATMUL_CUDA_RT_CHECK(cudaStreamCreate(&stream));

            // Get its properties.
            cudaDeviceProp cudaDevProp;
            MATMUL_CUDA_RT_CHECK(cudaGetDeviceProperties(
                &cudaDevProp,
                0));

            TSize gridThreadExtent[] = {m, n};
            TSize blockThreadExtent[] = {cudaDevProp.maxThreadsDim[0], cudaDevProp.maxThreadsDim[1]};

            // Restrict the max block thread extent with the grid thread extent.
            // This removes dimensions not required in the given grid thread extent.
            // This has to be done before the maxThreadsPerBlock clipping to get the maximum correctly.
            for(TSize i(0); i<2; ++i)
            {
                blockThreadExtent[i] = std::min(blockThreadExtent[i], gridThreadExtent[i]);
            }

            // Restrict it to its minimum component.
            // For example (512, 256) will get (256, 256).
            TSize minBlockThreadExtent(blockThreadExtent[0]);
            for(TSize i(1); i<2; ++i)
            {
                minBlockThreadExtent = std::min(minBlockThreadExtent, blockThreadExtent[i]);
            }
            for(TSize i(0); i<2; ++i)
            {
                blockThreadExtent[i] = minBlockThreadExtent;
            }

            // Adjust blockThreadExtent if its product is too large.
            if ((blockThreadExtent[0] * blockThreadExtent[1]) > cudaDevProp.maxThreadsPerBlock)
            {
                // Satisfy the following equation:
                // udaDevProp.maxThreadsPerBlock >= blockThreadExtent[0]*blockThreadExtent[1]
                // For example 1024 >= 512 * 512

                // For equal block thread extent this is easily the nth root of cudaDevProp.maxThreadsPerBlock.
                double const fNthRoot(std::pow(cudaDevProp.maxThreadsPerBlock, 1.0 / 2.0));
                TSize const nthRoot(static_cast<TSize>(fNthRoot));
                for(TSize i(0); i<2; ++i)
                {
                    blockThreadExtent[i] = nthRoot;
                }
            }

            // Set the grid block extent (rounded to the next integer not less then the quotient.
            TSize gridBlockExtent[] = {1, 1};
            for(TSize i(0); i<2; ++i)
            {
                gridBlockExtent[i] =
                    static_cast<TSize>(
                        std::ceil(static_cast<double>(gridThreadExtent[i])
                            / static_cast<double>(blockThreadExtent[i])));
            }

            dim3 const dimBlock(blockThreadExtent[0], blockThreadExtent[1]);
            dim3 const dimGrid(gridBlockExtent[0], gridBlockExtent[1]);

            MATMUL_TIME_START;

            matmul_gemm_par_cuda_dyn_block_size_1d_extern_shared_kernel<<<
                dimGrid,
                dimBlock,
                2u*sizeof(TElem)*blockThreadExtent[0] * blockThreadExtent[1],
                stream>>>(
                    m, n, k,
                    alpha,
                    A, lda,
                    B, ldb,
                    beta,
                    C, ldc);

            MATMUL_CUDA_RT_CHECK(cudaStreamSynchronize(stream));

            MATMUL_TIME_END;

            MATMUL_CUDA_RT_CHECK(cudaStreamDestroy(stream));

            MATMUL_TIME_RETURN;
        }
    #endif
    #ifdef MATMUL_BUILD_PAR_CUDA_MEMCPY_DYN_BLOCK_SIZE
        //-----------------------------------------------------------------------------
        //
        //-----------------------------------------------------------------------------
        TReturn matmul_gemm_par_cuda_memcpy_dyn_block_size_1d_extern_shared(
            TSize const m, TSize const n, TSize const k,
            TElem const alpha,
            TElem const * const MATMUL_RESTRICT A, TSize const lda,
            TElem const * const MATMUL_RESTRICT B, TSize const ldb,
            TElem const beta,
            TElem * const MATMUL_RESTRICT C, TSize const ldc)
        {
            return
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
