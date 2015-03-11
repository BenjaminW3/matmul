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

#include "matmul_par_blas_cublas.cuh"

#ifdef MATMUL_BUILD_PAR_BLAS_CUBLAS

	#include <stdio.h>		// printf

	#include <cuda.h>

	#define MATMUL_CUDA_CHECK(cmd) {cudaError_t error = cmd; if(error!=cudaSuccess){printf("<%s>:%i ",__FILE__,__LINE__); printf("[CUDA] Error: %s\n", cudaGetErrorString(error));}}

	#include <cuda_runtime.h>
	#include <cublas_v2.h>
	
	#define MATMUL_CUBLAS_CHECK(cmd) {cublasStatus_t ret = cmd; if(ret!=CUBLAS_STATUS_SUCCESS){printf("cublasCreate returned error code %d, line(%d)\n", ret, __LINE__); exit(EXIT_FAILURE);}}

	//-----------------------------------------------------------------------------
	//
	//-----------------------------------------------------------------------------
	void matmul_par_blas_cublas2(
		size_t const n,
		TElement const * const restrict A,
		TElement const * const restrict B,
		TElement * const restrict C)
	{
		TElement *pADev, *pBDev, *pCDev;

		size_t const uiBytes = n*n*sizeof(TElement);
		
		MATMUL_CUDA_CHECK(cudaMalloc((void **) &pADev, uiBytes));
		MATMUL_CUDA_CHECK(cudaMemcpy(pADev, A, uiBytes, cudaMemcpyHostToDevice));
		MATMUL_CUDA_CHECK(cudaMalloc((void **) &pBDev, uiBytes));
		MATMUL_CUDA_CHECK(cudaMemcpy(pBDev, B, uiBytes, cudaMemcpyHostToDevice));
		MATMUL_CUDA_CHECK(cudaMalloc((void **) &pCDev, uiBytes));

		// Initialise cublas
		cublasHandle_t handle;
		MATMUL_CUBLAS_CHECK(cublasCreate(&handle));
		
		// Do the calculation.
		TElement const alpha = 1;
		TElement const beta  = 1;
		//Note: cublas is column primary! So we need to transpose the order.
		#ifdef MATMUL_ELEMENT_TYPE_DOUBLE
			MATMUL_CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, pBDev, n, pADev, n, &beta, pCDev, n));
		#else
			MATMUL_CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, pBDev, n, pADev, n, &beta, pCDev, n));
		#endif

		MATMUL_CUDA_CHECK(cudaDeviceSynchronize());
		MATMUL_CUDA_CHECK(cudaMemcpy(C, pCDev, uiBytes, cudaMemcpyDeviceToHost));

		cudaFree(pADev);
		cudaFree(pBDev);
		cudaFree(pCDev);

		MATMUL_CUBLAS_CHECK(cublasDestroy(handle));
	}
#endif
