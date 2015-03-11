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

#include "matmul_par_cuda.cuh"

#ifdef MATMUL_BUILD_PAR_CUDA

	#include <stdio.h>		// printf

	#include <cuda.h>

	#define MATMUL_CUDA_CHECK(cmd) {cudaError_t error = cmd; if(error!=cudaSuccess){printf("<%s>:%i ",__FILE__,__LINE__); printf("[CUDA] Error: %s\n", cudaGetErrorString(error));}}

	//-----------------------------------------------------------------------------
	// This function only works for squared blocks and grids.
	//-----------------------------------------------------------------------------
	__global__ void matmul_par_cuda_kernel(
		size_t const n,
		TElement const * const A,
		TElement const * const B,
		TElement * const C)
	{
		// blockIdx.x and blockIdx.y are the indices of the block to calculate inside C.
		const int cx = blockIdx.x*blockDim.x + threadIdx.x;	// Column inside C to calculate.
		const int cy = blockIdx.y*blockDim.y + threadIdx.y;	// Row inside C to calculate.

		const int col = threadIdx.x;	// Column inside the block of C to calculate.
		const int row = threadIdx.y;	// Row inside the block of C to calculate.

		// Shared memory used to store the current blocks of A and B.
		__shared__ TElement pSharedBlockA[MATMUL_CUDA_BLOCKSIZE][MATMUL_CUDA_BLOCKSIZE];
		__shared__ TElement pSharedBlockB[MATMUL_CUDA_BLOCKSIZE][MATMUL_CUDA_BLOCKSIZE];

		TElement fCSum = 0.0f;

		// If the element is outside of the matrix, write zero into the shared block.
		bool const bOutsideMatrix = (cx>=n) || (cy>=n);
		
		// Loop over all blocks of A and B that are required to compute the C block. 
		for (int l=0; l<gridDim.x; ++l)
		{
			// Copy data to shared memory.
			const int uiIndexA = cy*n+l*MATMUL_CUDA_BLOCKSIZE + col;
			pSharedBlockA[row][col] = bOutsideMatrix ? 0 : A[uiIndexA];
			const int uiIndexB = (l*MATMUL_CUDA_BLOCKSIZE+row)*n + cx;
			pSharedBlockB[row][col] = bOutsideMatrix ? 0 : B[uiIndexB];

			// Synchronize to make sure the sub-matrices are loaded before starting the computation.
			__syncthreads();

			// Dyadic product within shared memory.
			for (int k=0; k<MATMUL_CUDA_BLOCKSIZE; ++k)
			{
				fCSum += pSharedBlockA[row][k] * pSharedBlockB[k][col];
			}

			// Synchronize to make sure that the preceding computation is done before loading two new sub-matrices of A and B in the next iteration.
			__syncthreads();
		}

		if(!bOutsideMatrix)
		{
			C[cy*n + cx] += fCSum;
		}
	}

	//-----------------------------------------------------------------------------
	//
	//-----------------------------------------------------------------------------
	void matmul_par_cuda(
		size_t const n,
		TElement const * const restrict A,
		TElement const * const restrict B,
		TElement * const restrict C)
	{
		TElement *pADev, *pBDev, *pCDev;

		dim3 dimBlock(MATMUL_CUDA_BLOCKSIZE, MATMUL_CUDA_BLOCKSIZE);
		float const fGridNumElements = ceil(((float)n)/((float)MATMUL_CUDA_BLOCKSIZE));
		dim3 dimGrid(fGridNumElements, fGridNumElements);
		
		size_t const uiBytes = n*n*sizeof(TElement);
		
		MATMUL_CUDA_CHECK(cudaMalloc((void **) &pADev, uiBytes));
		MATMUL_CUDA_CHECK(cudaMemcpy(pADev, A, uiBytes, cudaMemcpyHostToDevice));
		MATMUL_CUDA_CHECK(cudaMalloc((void **) &pBDev, uiBytes));
		MATMUL_CUDA_CHECK(cudaMemcpy(pBDev, B, uiBytes, cudaMemcpyHostToDevice));
		MATMUL_CUDA_CHECK(cudaMalloc((void **) &pCDev, uiBytes));
		MATMUL_CUDA_CHECK(cudaMemcpy(pCDev, C, uiBytes, cudaMemcpyHostToDevice));

		matmul_par_cuda_kernel<<<dimGrid,dimBlock>>>(
			n,
			pADev,
			pBDev,
			pCDev);

		MATMUL_CUDA_CHECK(cudaDeviceSynchronize());
		MATMUL_CUDA_CHECK(cudaMemcpy(C, pCDev, uiBytes, cudaMemcpyDeviceToHost));

		cudaFree(pADev);
		cudaFree(pBDev);
		cudaFree(pCDev);
	}
#endif
