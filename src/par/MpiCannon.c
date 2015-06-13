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

#if defined(MATMUL_BUILD_PAR_MPI_CANNON_STD) || defined(MATMUL_BUILD_PAR_MPI_CANNON_MKL) || defined(MATMUL_BUILD_PAR_MPI_CANNON_CUBLAS)

    #include <matmul/par/MpiCannon.h>

    #include <matmul/seq/MultipleOpts.h>
    #include <matmul/common/Alloc.h>
    #include <matmul/common/Mat.h>      // matmul_mat_get_block, matmul_mat_set_block, matmul_mat_gemm_early_out

    #include <stdbool.h>                // bool, true, false
    #include <math.h>                   // sqrt
    #include <stdio.h>                  // printf
    #include <assert.h>                 // assert

    #include <mpi.h>

    //#define MATMUL_MPI_ADDITIONAL_DEBUG_OUTPUT

    //-----------------------------------------------------------------------------
    //
    //-----------------------------------------------------------------------------
    void matmul_gemm_par_mpi_cannon_block_local(
        size_t const b,
        TElem const alpha,
        TElem * const MATMUL_RESTRICT pALocal,
        TElem * const MATMUL_RESTRICT pBLocal,
        TElem const beta,
        TElem * const MATMUL_RESTRICT pCLocal,
        MPI_Comm * const comm2D,
        int aiGridCoords[2],
        size_t q,
        void(*pMatMul)(size_t const, size_t const, size_t const, TElem const, TElem const * const, size_t const, TElem const * const, size_t const, TElem const, TElem * const, size_t const))
    {
        assert(comm2D);
        assert(q>0);

        size_t const uiNumElementsBlock = b * b;

        MPI_Status status;
        int const iInitialShiftSendRecTag = 5;
        int iRankShiftSource, iRankShiftDest;
        // Perform the initial matrix alignment for A. (1 == dir) = horizontal dimesion to transport in.
        MPI_Cart_shift(*comm2D, 1, -aiGridCoords[0], &iRankShiftSource, &iRankShiftDest);
        MPI_Sendrecv_replace(pALocal, (int)uiNumElementsBlock, MATMUL_MPI_ELEMENT_TYPE, iRankShiftDest, iInitialShiftSendRecTag, iRankShiftSource, iInitialShiftSendRecTag, *comm2D, &status);

        // Perform the initial matrix alignment for B. (0 == dir) = vertical dimesion to transport in.
        MPI_Cart_shift(*comm2D, 0, -aiGridCoords[1], &iRankShiftSource, &iRankShiftDest);
        MPI_Sendrecv_replace(pBLocal, (int)uiNumElementsBlock, MATMUL_MPI_ELEMENT_TYPE, iRankShiftDest, iInitialShiftSendRecTag, iRankShiftSource, iInitialShiftSendRecTag, *comm2D, &status);

        // Compute ranks of the up and left shifts. (-1 == disp) < 0 -> shift down.
        int iRankUp, iRankDown, iRankLeft, iRankRight;
        MPI_Cart_shift(*comm2D, 1, -1, &iRankRight, &iRankLeft);
        MPI_Cart_shift(*comm2D, 0, -1, &iRankDown, &iRankUp);

        // Compute the current block.
        int const iComputeShiftSendRecTag = 7;
        for(size_t i = 0; i<q; ++i)
        {
            // Perform the local calculation.
            pMatMul(b, b, b, alpha, pALocal, b, pBLocal, b, beta, pCLocal, b);

            // Shift matrix A left by one.
            MPI_Sendrecv_replace(pALocal, (int)uiNumElementsBlock, MATMUL_MPI_ELEMENT_TYPE, iRankLeft, iComputeShiftSendRecTag, iRankRight, iComputeShiftSendRecTag, *comm2D, &status);

            // Shift matrix B up by one.
            MPI_Sendrecv_replace(pBLocal, (int)uiNumElementsBlock, MATMUL_MPI_ELEMENT_TYPE, iRankUp, iComputeShiftSendRecTag, iRankDown, iComputeShiftSendRecTag, *comm2D, &status);
        }

        // Restore the original distribution of A and B.
        /*
        int const iRestoreShiftSendRecTag = 7;

        MPI_Cart_shift(*comm2D, 1, aiGridCoords[0], &iRankShiftSource, &iRankShiftDest);
        MPI_Sendrecv_replace(pALocal, uiNumElementsBlock, MATMUL_MPI_ELEMENT_TYPE, iRankShiftDest, iRestoreShiftSendRecTag, iRankShiftSource, iRestoreShiftSendRecTag, *comm2D, &status);

        printf("6:%d \n", (int)(aiGridCoords[1]+aiGridCoords[0]*q));

        MPI_Cart_shift(*comm2D, 0, aiGridCoords[1], &iRankShiftSource, &iRankShiftDest);
        MPI_Sendrecv_replace(pBLocal, uiNumElementsBlock, MATMUL_MPI_ELEMENT_TYPE, iRankShiftDest, iRestoreShiftSendRecTag, iRankShiftSource, iRestoreShiftSendRecTag, *comm2D, &status);

        printf("7:%d \n", (int)(aiGridCoords[1]+aiGridCoords[0]*q));*/
    }

    //-----------------------------------------------------------------------------
    //
    //-----------------------------------------------------------------------------
    void matmul_gemm_par_mpi_cannon_nonblock_local(
        size_t const b,
        TElem const alpha,
        TElem * const MATMUL_RESTRICT pALocal,
        TElem * const MATMUL_RESTRICT pBLocal,
        TElem const beta,
        TElem * const MATMUL_RESTRICT pCLocal,
        MPI_Comm * const comm2D,
        int aiGridCoords[2],
        size_t q,
        void(*pMatMul)(size_t const, size_t const, size_t const, TElem const, TElem const * const, size_t const, TElem const * const, size_t const, TElem const, TElem * const, size_t const))
    {
        assert(comm2D);
        assert(q>0);

        size_t const uiNumElementsBlock = b * b;

        MPI_Status status;
        int const iShiftSendRecTag = 1;
        int iRankShiftSource, iRankShiftDest;
        // Perform the initial matrix alignment for A.
        MPI_Cart_shift(*comm2D, 1, -aiGridCoords[0], &iRankShiftSource, &iRankShiftDest);
        //printf("iRankShiftSource %d iRankShiftDest %d", iRankShiftSource, iRankShiftDest);
        MPI_Sendrecv_replace(pALocal, (int)uiNumElementsBlock, MATMUL_MPI_ELEMENT_TYPE, iRankShiftDest, iShiftSendRecTag, iRankShiftSource, iShiftSendRecTag, *comm2D, &status);

        // Perform the initial matrix alignment for B
        MPI_Cart_shift(*comm2D, 0, -aiGridCoords[1], &iRankShiftSource, &iRankShiftDest);
        MPI_Sendrecv_replace(pBLocal, (int)uiNumElementsBlock, MATMUL_MPI_ELEMENT_TYPE, iRankShiftDest, iShiftSendRecTag, iRankShiftSource, iShiftSendRecTag, *comm2D, &status);

        // Setup the A and B buffers that are swapped between the current calculation and the current receiver buffer.
        TElem * apALocal[2];
        apALocal[0] = pALocal;
        apALocal[1] = matmul_arr_alloc(uiNumElementsBlock);
        TElem * apBLocal[2];
        apBLocal[0] = pBLocal;
        apBLocal[1] = matmul_arr_alloc(uiNumElementsBlock);

        // Compute ranks of the up and left shifts. (-1 == disp) < 0 -> shift down.
        int iRankUp, iRankDown, iRankLeft, iRankRight;
        MPI_Cart_shift(*comm2D, 1, -1, &iRankRight, &iRankLeft);
        MPI_Cart_shift(*comm2D, 0, -1, &iRankDown, &iRankUp);

        // Compute the current block.
        for(size_t i = 0; i<q; ++i)
        {
            MPI_Request reqs[4];

            // Shift matrix A left and B up by one.
            MPI_Isend(apALocal[i%2], (int)uiNumElementsBlock, MATMUL_MPI_ELEMENT_TYPE, iRankLeft, 1, *comm2D, &reqs[0]);
            MPI_Isend(apBLocal[i%2], (int)uiNumElementsBlock, MATMUL_MPI_ELEMENT_TYPE, iRankUp, 1, *comm2D, &reqs[1]);
            MPI_Irecv(apALocal[(i+1)%2], (int)uiNumElementsBlock, MATMUL_MPI_ELEMENT_TYPE, iRankRight, 1, *comm2D, &reqs[2]);
            MPI_Irecv(apBLocal[(i+1)%2], (int)uiNumElementsBlock, MATMUL_MPI_ELEMENT_TYPE, iRankDown, 1, *comm2D, &reqs[3]);

            // Perform the local calculation.
            pMatMul(b, b, b, alpha, apALocal[i%2], b, apBLocal[i%2], b, beta, pCLocal, b);

            for(size_t j = 0; j<4; ++j)
            {
                MPI_Wait(&reqs[j], &status);
            }
        }

        // Restore the original distribution of A and B.
        /*MPI_Cart_shift(*comm2D, 1, aiGridCoords[0], &iRankShiftSource, &iRankShiftDest);
        MPI_Sendrecv_replace(pALocal, uiNumElementsBlock, MATMUL_MPI_ELEMENT_TYPE, iRankShiftDest, iShiftSendRecTag, iRankShiftSource, iShiftSendRecTag, *comm2D, &status);

        MPI_Cart_shift(*comm2D, 0, aiGridCoords[1], &iRankShiftSource, &iRankShiftDest);
        MPI_Sendrecv_replace(pBLocal, uiNumElementsBlock, MATMUL_MPI_ELEMENT_TYPE, iRankShiftDest, iShiftSendRecTag, iRankShiftSource, iShiftSendRecTag, *comm2D, &status);
        */

        // Free up the resources.
        matmul_arr_free(apALocal[1]);
        matmul_arr_free(apBLocal[1]);
    }

    //-----------------------------------------------------------------------------
    //
    //-----------------------------------------------------------------------------
    void matmul_gemm_par_mpi_cannon(
        size_t const m, size_t const n, size_t const k,
        TElem const alpha,
        TElem const * const MATMUL_RESTRICT A, size_t const lda,
        TElem const * const MATMUL_RESTRICT B, size_t const ldb,
        TElem const beta,
        TElem * const MATMUL_RESTRICT C, size_t const ldc,
        bool const bBlockingComm,
        void(*pMatMul)(size_t const, size_t const, size_t const, TElem const, TElem const * const, size_t const, TElem const * const, size_t const, TElem const, TElem * const, size_t const))
    {
        if(matmul_mat_gemm_early_out(m, n, k, alpha, beta))
        {
            return;
        }

        // \TODO: Implement for non square matrices?
        if(m!=n || m!=k)
        {
            printf("Invalid matrix size! The matrices have to be square for the MPI Cannon GEMM.\n");
            return;
        }

        // Apply beta multiplication to C.
        if(beta != (TElem)1)
        {
            for(size_t i = 0; i < m; ++i)
            {
                for(size_t j = 0; j < n; ++j)
                {
                    C[i*ldc + j] *= beta;
                }
            }
        }

        // Get the number of processes.
        int iNumProcesses;
        MPI_Comm_size(MATMUL_MPI_COMM, &iNumProcesses);

        // Get the local Rank
        int iRank1D;
        MPI_Comm_rank(MATMUL_MPI_COMM, &iRank1D);

#ifdef MATMUL_MPI_ADDITIONAL_DEBUG_OUTPUT
        if(iRank1D==MATMUL_MPI_ROOT)
        {
            printf(" p=%d", iNumProcesses);
        }
#endif

        // Set up the sizes for a cartesian 2d grid topology.
        size_t const q = (int)sqrt((double)iNumProcesses);

        // Test if it is a square.
        if(q * q != iNumProcesses)
        {
            //MPI_Finalize();
            if(iRank1D==MATMUL_MPI_ROOT)
            {
                printf("\nInvalid environment! The number of processors (%d given) should be perfect square.\n", iNumProcesses);
            }
            return;
        }
#ifdef MATMUL_MPI_ADDITIONAL_DEBUG_OUTPUT
        if(iRank1D==MATMUL_MPI_ROOT)
        {
            printf(" -> %"MATMUL_PRINTF_SIZE_T" x %"MATMUL_PRINTF_SIZE_T" grid", q, q);
        }
#endif

        // Determine block size of the local block.
        size_t const b = n/q;

        // Test if the matrix can be divided equally. This can fail if e.g. the matrix is 3x3 and the preocesses are 2x2.
        if(n % q != 0)
        {
            //MPI_Finalize();
            if(iRank1D==MATMUL_MPI_ROOT)
            {
                printf("\nThe matrices can't be divided among processors equally!\n");
            }
            return;
        }

        // Set that the structure is periodical around the given dimension for wraparound connections.
        int aiPeriods[2];
        aiPeriods[0] = aiPeriods[1] = 1;

        int aiProcesses[2];
        aiProcesses[0] = aiProcesses[1] = (int)q;
        // Create the cartesian 2d grid topology. Ranks can be reordered.
        MPI_Comm comm2D;
        MPI_Cart_create(MATMUL_MPI_COMM, 2, aiProcesses, aiPeriods, 1, &comm2D);

        // Get the rank and coordinates with respect to the new 2D grid topology.
        int iRank2D;
        MPI_Comm_rank(comm2D, &iRank2D);
        int aiGridCoords[2];
        MPI_Cart_coords(comm2D, iRank2D, 2, aiGridCoords);
#ifdef MATMUL_MPI_ADDITIONAL_DEBUG_OUTPUT
        printf(" iRank2D=%d, x=%d y=%d\n", iRank2D, aiGridCoords[1], aiGridCoords[0]);
#endif

        // Compute ranks of the up and left shifts. (-1 == disp) < 0 -> shift down.
        /*int iRankUp, iRankDown, iRankLeft, iRankRight;
        MPI_Cart_shift(comm2D, 1, -1, &iRankRight, &iRankLeft);
        printf("%d iRankRight:%d iRankLeft:%d\n", iRank2D, iRankRight, iRankLeft);
        MPI_Cart_shift(comm2D, 0, -1, &iRankDown, &iRankUp);
        printf("%d iRankDown:%d iRankUp:%d\n", iRank2D, iRankDown, iRankUp);*/

        // Initialize the local buffers
        size_t const uiNumElementsBlock = b * b;

        TElem * const pALocal = matmul_arr_alloc(uiNumElementsBlock);
        TElem * const pBLocal = matmul_arr_alloc(uiNumElementsBlock);
        TElem * const pCLocal = matmul_arr_alloc(uiNumElementsBlock);

        TElem * pBufferCopyLocal = 0;

        if(iRank1D==MATMUL_MPI_ROOT)
        {
            pBufferCopyLocal = matmul_arr_alloc(uiNumElementsBlock);
        }

        // Send the blocks.
        TElem * apBuffersLocal[3] = {pALocal, pBLocal, pCLocal};
        TElem const * apBuffersGlobal[3] = {A, B, C};
        size_t const ald[3] = {lda, ldb, ldc};

#ifdef MATMUL_MPI_ADDITIONAL_DEBUG_OUTPUT
        if(iRank1D==MATMUL_MPI_ROOT)
        {
            printf(" Begin sending Blocks.\n");
        }
#endif
        for(size_t uiBuffer = 0; uiBuffer<3; ++uiBuffer)
        {
            int const iInitSendRecTag = 2;

            if(iRank1D==MATMUL_MPI_ROOT)
            {
                for(int iRankDestination = 1; iRankDestination<iNumProcesses; ++iRankDestination)
                {
                    int aiGridCoordsDest[2];
                    MPI_Cart_coords(comm2D, iRankDestination, 2, aiGridCoordsDest);

                    // Copy the blocks so that they lay linearly in memory.
                    matmul_mat_get_block(apBuffersGlobal[uiBuffer], ald[uiBuffer], aiGridCoordsDest[1], aiGridCoordsDest[0], pBufferCopyLocal, b);

                    MPI_Send(pBufferCopyLocal, (int)uiNumElementsBlock, MATMUL_MPI_ELEMENT_TYPE, iRankDestination, iInitSendRecTag, MATMUL_MPI_COMM);
                }

                // Copy the root block.
                matmul_mat_get_block(apBuffersGlobal[uiBuffer], ald[uiBuffer], aiGridCoords[1], aiGridCoords[0], apBuffersLocal[uiBuffer], b);
            }
            else
            {
                MPI_Status status;
                MPI_Recv(apBuffersLocal[uiBuffer], (int)uiNumElementsBlock, MATMUL_MPI_ELEMENT_TYPE, MATMUL_MPI_ROOT, iInitSendRecTag, MATMUL_MPI_COMM, &status);
            }
        }
#ifdef MATMUL_MPI_ADDITIONAL_DEBUG_OUTPUT
        if(iRank1D==MATMUL_MPI_ROOT)
        {
            printf(" Finished sending Blocks.\n");
        }
#endif

        // Do the node local calculation.
        if(bBlockingComm)
        {
            matmul_gemm_par_mpi_cannon_block_local(b, alpha, pALocal, pBLocal, (TElem)1, pCLocal, &comm2D, aiGridCoords, q, pMatMul);
        }
        else
        {
            matmul_gemm_par_mpi_cannon_nonblock_local(b, alpha, pALocal, pBLocal, (TElem)1, pCLocal, &comm2D, aiGridCoords, q, pMatMul);
        }

        // Collect the results and integrate into C
        int const iCollectSendRecTag = 3;

#ifdef MATMUL_MPI_ADDITIONAL_DEBUG_OUTPUT
        if(iRank1D==MATMUL_MPI_ROOT)
        {
            printf(" Begin collecting Blocks.\n");
        }
#endif
        if(iRank1D==MATMUL_MPI_ROOT)
        {
            for(int iRankOrigin = 1; iRankOrigin<iNumProcesses; ++iRankOrigin)
            {
                int aiGridCoordsDest[2];
                MPI_Cart_coords(comm2D, iRankOrigin, 2, aiGridCoordsDest);

                MPI_Status status;
                MPI_Recv(pBufferCopyLocal, (int)uiNumElementsBlock, MATMUL_MPI_ELEMENT_TYPE, iRankOrigin, iCollectSendRecTag, MATMUL_MPI_COMM, &status);

                // Copy the blocks so that they lay linearly in memory.
                matmul_mat_set_block(pBufferCopyLocal, b, C, ldc, aiGridCoordsDest[1], aiGridCoordsDest[0]);
            }

            // Copy the root block.
            matmul_mat_set_block(pCLocal, b, C, ldc, aiGridCoords[1], aiGridCoords[0]);
        }
        else
        {
            MPI_Send(pCLocal, (int)uiNumElementsBlock, MATMUL_MPI_ELEMENT_TYPE, MATMUL_MPI_ROOT, iCollectSendRecTag, MATMUL_MPI_COMM);
        }
#ifdef MATMUL_MPI_ADDITIONAL_DEBUG_OUTPUT
        if(iRank1D==MATMUL_MPI_ROOT)
        {
            matmul_arr_free(pBufferCopyLocal);
            printf(" Finished collecting Blocks.\n");
        }
#endif

        // Free up the resources.
        matmul_arr_free(pALocal);
        matmul_arr_free(pBLocal);
        matmul_arr_free(pCLocal);
        MPI_Comm_free(&comm2D);
    }

#ifdef MATMUL_BUILD_PAR_MPI_CANNON_STD
    //-----------------------------------------------------------------------------
    //
    //-----------------------------------------------------------------------------
    void matmul_gemm_par_mpi_cannon_block(
        size_t const m, size_t const n, size_t const k,
        TElem const alpha,
        TElem const * const MATMUL_RESTRICT A, size_t const lda,
        TElem const * const MATMUL_RESTRICT B, size_t const ldb,
        TElem const beta,
        TElem * const MATMUL_RESTRICT C, size_t const ldc)
    {
        if(matmul_mat_gemm_early_out(m, n, k, alpha, beta))
        {
            return;
        }

        matmul_gemm_par_mpi_cannon(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, true, matmul_gemm_seq_multiple_opts);
    }

    //-----------------------------------------------------------------------------
    //
    //-----------------------------------------------------------------------------
    void matmul_gemm_par_mpi_cannon_nonblock(
        size_t const m, size_t const n, size_t const k,
        TElem const alpha,
        TElem const * const MATMUL_RESTRICT A, size_t const lda,
        TElem const * const MATMUL_RESTRICT B, size_t const ldb,
        TElem const beta,
        TElem * const MATMUL_RESTRICT C, size_t const ldc)
    {
        if(matmul_mat_gemm_early_out(m, n, k, alpha, beta))
        {
            return;
        }

        matmul_gemm_par_mpi_cannon(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, false, matmul_gemm_seq_multiple_opts);
    }
#endif
#ifdef MATMUL_BUILD_PAR_MPI_CANNON_MKL

    #include <matmul/par/BlasMkl.h>

    //-----------------------------------------------------------------------------
    //
    //-----------------------------------------------------------------------------
    void matmul_gemm_par_mpi_cannon_nonblock_blas_mkl(
        size_t const m, size_t const n, size_t const k,
        TElem const alpha,
        TElem const * const MATMUL_RESTRICT A, size_t const lda,
        TElem const * const MATMUL_RESTRICT B, size_t const ldb,
        TElem const beta,
        TElem * const MATMUL_RESTRICT C, size_t const ldc)
    {
        if(matmul_mat_gemm_early_out(m, n, k, alpha, beta))
        {
            return;
        }

        matmul_gemm_par_mpi_cannon(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, false, matmul_gemm_par_blas_mkl);
    }
#endif
#ifdef MATMUL_BUILD_PAR_MPI_CANNON_CUBLAS

    #include <matmul/par/BlasCublas.h>

    //-----------------------------------------------------------------------------
    //
    //-----------------------------------------------------------------------------
    void matmul_gemm_par_mpi_cannon_nonblock_blas_cublas(
        size_t const m, size_t const n, size_t const k,
        TElem const alpha,
        TElem const * const MATMUL_RESTRICT A, size_t const lda,
        TElem const * const MATMUL_RESTRICT B, size_t const ldb,
        TElem const beta,
        TElem * const MATMUL_RESTRICT C, size_t const ldc)
    {
        if(matmul_mat_gemm_early_out(m, n, k, alpha, beta))
        {
            return;
        }

        matmul_gemm_par_mpi_cannon(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, false, matmul_gemm_par_blas_cublas2);
    }
#endif
#endif
