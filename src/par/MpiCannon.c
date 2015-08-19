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
    void matmul_gemm_par_mpi_cannon_local_block(
        TIdx const b,
        TElem const alpha,
        TElem * const MATMUL_RESTRICT pALocal,
        TElem * const MATMUL_RESTRICT pBLocal,
        TElem * const MATMUL_RESTRICT pCLocal,
        MPI_Comm * const pComm2D,
        int const iRankLeft,
        int const iRankRight,
        int const iRankUp,
        int const iRankDown,
        TIdx const q,
        TReturn(*pGemm)(TIdx const, TIdx const, TIdx const, TElem const, TElem const * const, TIdx const, TElem const * const, TIdx const, TElem const, TElem * const, TIdx const))
    {
        assert(pComm2D);
        assert(q>0);

        TIdx const numElementsBlock = b * b;
        int const iNumElementsBlock = (int)numElementsBlock;

        MPI_Status status;

        // Compute the current block.
        int const iComputeShiftSendRecTagA = 6;
        int const iComputeShiftSendRecTagB = 7;
        for(TIdx i = 0; i<q; ++i)
        {
            // Perform the local calculation.
            pGemm(b, b, b, alpha, pALocal, b, pBLocal, b, (TElem)1, pCLocal, b);

            // Shift matrix A left by one.
            MPI_Sendrecv_replace(pALocal, iNumElementsBlock, MATMUL_MPI_ELEMENT_TYPE, iRankLeft, iComputeShiftSendRecTagA, iRankRight, iComputeShiftSendRecTagA, *pComm2D, &status);

            // Shift matrix B up by one.
            MPI_Sendrecv_replace(pBLocal, iNumElementsBlock, MATMUL_MPI_ELEMENT_TYPE, iRankUp, iComputeShiftSendRecTagB, iRankDown, iComputeShiftSendRecTagB, *pComm2D, &status);
        }
    }
    //-----------------------------------------------------------------------------
    //
    //-----------------------------------------------------------------------------
    void matmul_gemm_par_mpi_cannon_local_nonblock(
        TIdx const b,
        TElem const alpha,
        TElem * const MATMUL_RESTRICT pALocal,
        TElem * const MATMUL_RESTRICT pBLocal,
        TElem * const MATMUL_RESTRICT pCLocal,
        MPI_Comm * const pComm2D,
        int const iRankLeft,
        int const iRankRight,
        int const iRankUp,
        int const iRankDown,
        TIdx const q,
        TReturn(*pGemm)(TIdx const, TIdx const, TIdx const, TElem const, TElem const * const, TIdx const, TElem const * const, TIdx const, TElem const, TElem * const, TIdx const))
    {
        assert(pComm2D);
        assert(q>0);

        TIdx const numElementsBlock = b * b;
        int const iNumElementsBlock = (int)numElementsBlock;

        MPI_Status status;

        // Setup the A and B buffers that are swapped between the current calculation and the current receiver buffer.
        TElem * apALocal[2];
        apALocal[0] = pALocal;
        apALocal[1] = matmul_arr_alloc(numElementsBlock);
        TElem * apBLocal[2];
        apBLocal[0] = pBLocal;
        apBLocal[1] = matmul_arr_alloc(numElementsBlock);

        // Compute the current block.
        int const iComputeShiftSendRecTagA = 6;
        int const iComputeShiftSendRecTagB = 7;
        for(TIdx i = 0; i<q; ++i)
        {
            MPI_Request reqs[4];

            bool const bLastIteration = ((i+1) == q);
            if(!bLastIteration)
            {
                // Shift matrix A left and B up by one.
                MPI_Isend(apALocal[i%2], iNumElementsBlock, MATMUL_MPI_ELEMENT_TYPE, iRankLeft, iComputeShiftSendRecTagA, *pComm2D, &reqs[0]);
                MPI_Isend(apBLocal[i%2], iNumElementsBlock, MATMUL_MPI_ELEMENT_TYPE, iRankUp, iComputeShiftSendRecTagB, *pComm2D, &reqs[1]);
                MPI_Irecv(apALocal[(i+1)%2], iNumElementsBlock, MATMUL_MPI_ELEMENT_TYPE, iRankRight, iComputeShiftSendRecTagA, *pComm2D, &reqs[2]);
                MPI_Irecv(apBLocal[(i+1)%2], iNumElementsBlock, MATMUL_MPI_ELEMENT_TYPE, iRankDown, iComputeShiftSendRecTagB, *pComm2D, &reqs[3]);
            }

            // Perform the local calculation.
            pGemm(b, b, b, alpha, apALocal[i%2], b, apBLocal[i%2], b, (TElem)1, pCLocal, b);

            if(!bLastIteration)
            {
                for(TIdx j = 0; j<4; ++j)
                {
                    MPI_Wait(&reqs[j], &status);
                }
            }
        }

        // Free up the resources.
        matmul_arr_free(apALocal[1]);
        matmul_arr_free(apBLocal[1]);
    }
    //-----------------------------------------------------------------------------
    //
    //-----------------------------------------------------------------------------
    void matmul_gemm_par_mpi_cannon_local(
        TIdx const b,
        TElem const alpha,
        TElem * const MATMUL_RESTRICT pALocal,
        TElem * const MATMUL_RESTRICT pBLocal,
        TElem * const MATMUL_RESTRICT pCLocal,
        MPI_Comm * const pComm2D,
        bool const bBlockingComm,
        int aiGridCoords[2],
        TIdx const q,
        TReturn(*pGemm)(TIdx const, TIdx const, TIdx const, TElem const, TElem const * const, TIdx const, TElem const * const, TIdx const, TElem const, TElem * const, TIdx const))
    {
        assert(pComm2D);
        assert(q>0);

        TIdx const numElementsBlock = b * b;
        int const iNumElementsBlock = (int)numElementsBlock;

        MPI_Status status;
        int const iInitialShiftSendRecTagA = 4;
        int const iInitialShiftSendRecTagB = 5;
        int iRankShiftSource, iRankShiftDest;
        // Perform the initial matrix alignment for A.
        MPI_Cart_shift(*pComm2D, 1, -aiGridCoords[0], &iRankShiftSource, &iRankShiftDest);
        MPI_Sendrecv_replace(pALocal, iNumElementsBlock, MATMUL_MPI_ELEMENT_TYPE, iRankShiftDest, iInitialShiftSendRecTagA, iRankShiftSource, iInitialShiftSendRecTagA, *pComm2D, &status);

        // Perform the initial matrix alignment for B
        MPI_Cart_shift(*pComm2D, 0, -aiGridCoords[1], &iRankShiftSource, &iRankShiftDest);
        MPI_Sendrecv_replace(pBLocal, iNumElementsBlock, MATMUL_MPI_ELEMENT_TYPE, iRankShiftDest, iInitialShiftSendRecTagB, iRankShiftSource, iInitialShiftSendRecTagB, *pComm2D, &status);

        // Compute ranks of the up and left shifts. (-1 == disp) < 0 -> shift down.
        int iRankUp, iRankDown, iRankLeft, iRankRight;
        MPI_Cart_shift(*pComm2D, 1, -1, &iRankRight, &iRankLeft);
        MPI_Cart_shift(*pComm2D, 0, -1, &iRankDown, &iRankUp);

        if(bBlockingComm)
        {
            matmul_gemm_par_mpi_cannon_local_block(b, alpha, pALocal, pBLocal, pCLocal, pComm2D, iRankLeft, iRankRight, iRankUp, iRankDown, q, pGemm);
        }
        else
        {
            matmul_gemm_par_mpi_cannon_local_nonblock(b, alpha, pALocal, pBLocal, pCLocal, pComm2D, iRankLeft, iRankRight, iRankUp, iRankDown, q, pGemm);
        }

        // Restore the original distribution of A and B.
        // This is not necessary for our implementation because the A and B sub-matrices are initially scattered and not used any more.
        /*
        int const iRestoreShiftSendRecTagA = 8;
        int const iRestoreShiftSendRecTagB = 9;
        MPI_Cart_shift(*pComm2D, 1, aiGridCoords[0], &iRankShiftSource, &iRankShiftDest);
        MPI_Sendrecv_replace(pALocal, iNumElementsBlock, MATMUL_MPI_ELEMENT_TYPE, iRankShiftDest, iRestoreShiftSendRecTagA, iRankShiftSource, iRestoreShiftSendRecTagA, *pComm2D, &status);

        MPI_Cart_shift(*pComm2D, 0, aiGridCoords[1], &iRankShiftSource, &iRankShiftDest);
        MPI_Sendrecv_replace(pBLocal, iNumElementsBlock, MATMUL_MPI_ELEMENT_TYPE, iRankShiftDest, iRestoreShiftSendRecTagB, iRankShiftSource, iRestoreShiftSendRecTagB, *pComm2D, &status);
        */
    }
    //-----------------------------------------------------------------------------
    //
    //-----------------------------------------------------------------------------
    TReturn matmul_gemm_par_mpi_cannon_algo(
        TIdx const m, TIdx const n, TIdx const k,
        TElem const alpha,
        TElem const * const MATMUL_RESTRICT A, TIdx const lda,
        TElem const * const MATMUL_RESTRICT B, TIdx const ldb,
        TElem const beta,
        TElem * const MATMUL_RESTRICT C, TIdx const ldc,
        bool const bBlockingComm,
        TReturn(*pGemm)(TIdx const, TIdx const, TIdx const, TElem const, TElem const * const, TIdx const, TElem const * const, TIdx const, TElem const, TElem * const, TIdx const))
    {
        if(matmul_mat_gemm_early_out(m, n, k, alpha, beta))
        {
            MATMUL_TIME_RETURN_EARLY_OUT;
        }

        // \TODO: Implement for non square matrices?
        if((m!=n) || (m!=k))
        {
            printf("[GEMM MPI Cannon] Invalid matrix size! The matrices have to be square for the MPI Cannon GEMM.\n");
            MATMUL_TIME_RETURN_EARLY_OUT;
        }

        // \FIXME: Fix alpha != 1!
        if(alpha!=(TElem)1)
        {
            printf("[GEMM MPI Cannon] alpha != 1 currently not implemented.\n");
            MATMUL_TIME_RETURN_EARLY_OUT;
        }

        // Get the number of processes.
        int iNumProcesses;
        MPI_Comm_size(MATMUL_MPI_COMM, &iNumProcesses);

        // Get the local Rank.
        int rank1D;
        MPI_Comm_rank(MATMUL_MPI_COMM, &rank1D);

#ifdef MATMUL_MPI_ADDITIONAL_DEBUG_OUTPUT
        if(rank1D == MATMUL_MPI_ROOT)
        {
            printf(" p=%d", iNumProcesses);
        }
#endif

        // Set up the sizes for a cartesian 2d grid topology.
        TIdx const q = (TIdx)sqrt((double)iNumProcesses);

        // Test if it is a square.
        if(q * q != iNumProcesses)
        {
            if(rank1D == MATMUL_MPI_ROOT)
            {
                printf("\n[GEMM MPI Cannon] Invalid environment! The number of processors (%d given) should be perfect square.\n", iNumProcesses);
            }
            MATMUL_TIME_RETURN_EARLY_OUT;
        }
#ifdef MATMUL_MPI_ADDITIONAL_DEBUG_OUTPUT
        if(rank1D == MATMUL_MPI_ROOT)
        {
            printf(" -> %"MATMUL_PRINTF_SIZE_T" x %"MATMUL_PRINTF_SIZE_T" grid", q, q);
        }
#endif

        // Test if the matrix can be divided equally. This can fail if e.g. the matrix is 3x3 and the processes are 2x2.
        if(n % q != 0)
        {
            if(rank1D == MATMUL_MPI_ROOT)
            {
                printf("\n[GEMM MPI Cannon] The matrices can't be divided among processors equally!\n");
            }
            MATMUL_TIME_RETURN_EARLY_OUT;
        }

        // Determine block size of the local block.
        TIdx const b = n/q;

        // Set that the structure is periodical around the given dimension for wraparound connections.
        int aiPeriods[2];
        aiPeriods[0] = aiPeriods[1] = (int)true;

        int aiProcesses[2];
        aiProcesses[0] = aiProcesses[1] = (int)q;
        // Create the cartesian 2d grid topology. Ranks can be reordered.
        MPI_Comm comm2D;
        MPI_Cart_create(MATMUL_MPI_COMM, 2, aiProcesses, aiPeriods, (int)true, &comm2D);

        // Get the rank and coordinates with respect to the new 2D grid topology.
        int iRank2D;
        MPI_Comm_rank(comm2D, &iRank2D);
        int aiGridCoords[2];
        MPI_Cart_coords(comm2D, iRank2D, 2, aiGridCoords);
#ifdef MATMUL_MPI_ADDITIONAL_DEBUG_OUTPUT
        printf(" iRank2D=%d, x=%d y=%d\n", iRank2D, aiGridCoords[1], aiGridCoords[0]);
#endif

        MATMUL_TIME_START;

        // Initialize the local buffers
        TIdx const numElementsBlock = b * b;

        TElem * const pALocal = matmul_arr_alloc(numElementsBlock);
        TElem * const pBLocal = matmul_arr_alloc(numElementsBlock);
        TElem * const pCLocal = matmul_arr_alloc(numElementsBlock);

        TElem * pBufferCopyLocal = 0;

        if(rank1D == MATMUL_MPI_ROOT)
        {
            pBufferCopyLocal = matmul_arr_alloc(numElementsBlock);
        }

        // Send the blocks.
        TElem * apBuffersLocal[3] = {pALocal, pBLocal, pCLocal};
        TElem const * apBuffersGlobal[3] = {A, B, C};
        TIdx const ald[3] = {lda, ldb, ldc};

#ifdef MATMUL_MPI_ADDITIONAL_DEBUG_OUTPUT
        if(rank1D == MATMUL_MPI_ROOT)
        {
            printf(" Begin sending Blocks.\n");
        }
#endif
        for(TIdx bufferIdx = 0; bufferIdx<3; ++bufferIdx)
        {
            int const iInitSendRecTag = 2;

            if(rank1D == MATMUL_MPI_ROOT)
            {
                for(int iRankDestination = 1; iRankDestination<iNumProcesses; ++iRankDestination)
                {
                    int aiGridCoordsDest[2];
                    MPI_Cart_coords(comm2D, iRankDestination, 2, aiGridCoordsDest);

                    // Copy the blocks so that they lay linearly in memory.
                    matmul_mat_get_block(apBuffersGlobal[bufferIdx], ald[bufferIdx], aiGridCoordsDest[1], aiGridCoordsDest[0], pBufferCopyLocal, b);

                    MPI_Send(pBufferCopyLocal, (int)numElementsBlock, MATMUL_MPI_ELEMENT_TYPE, iRankDestination, iInitSendRecTag, MATMUL_MPI_COMM);
                }

                // Copy the root block.
                matmul_mat_get_block(apBuffersGlobal[bufferIdx], ald[bufferIdx], aiGridCoords[1], aiGridCoords[0], apBuffersLocal[bufferIdx], b);
            }
            else
            {
                MPI_Status status;
                MPI_Recv(apBuffersLocal[bufferIdx], (int)numElementsBlock, MATMUL_MPI_ELEMENT_TYPE, MATMUL_MPI_ROOT, iInitSendRecTag, MATMUL_MPI_COMM, &status);
            }
        }
#ifdef MATMUL_MPI_ADDITIONAL_DEBUG_OUTPUT
        if(rank1D == MATMUL_MPI_ROOT)
        {
            printf(" Finished sending Blocks.\n");
        }
#endif

        // Apply beta multiplication to local C.
        if(beta != (TElem)1)
        {
            for(TIdx i = 0; i < b; ++i)
            {
                for(TIdx j = 0; j < b; ++j)
                {
                    pCLocal[i*b + j] *= beta;
                }
            }
        }

        // Do the node local calculation.
        matmul_gemm_par_mpi_cannon_local(b, alpha, pALocal, pBLocal, pCLocal, &comm2D, bBlockingComm, aiGridCoords, q, pGemm);

        // Collect the results and integrate into C
        int const iCollectSendRecTag = 3;

#ifdef MATMUL_MPI_ADDITIONAL_DEBUG_OUTPUT
        if(rank1D == MATMUL_MPI_ROOT)
        {
            printf(" Begin collecting Blocks.\n");
        }
#endif
        if(rank1D == MATMUL_MPI_ROOT)
        {
            for(int iRankOrigin = 1; iRankOrigin<iNumProcesses; ++iRankOrigin)
            {
                int aiGridCoordsDest[2];
                MPI_Cart_coords(comm2D, iRankOrigin, 2, aiGridCoordsDest);

                MPI_Status status;
                MPI_Recv(pBufferCopyLocal, (int)numElementsBlock, MATMUL_MPI_ELEMENT_TYPE, iRankOrigin, iCollectSendRecTag, MATMUL_MPI_COMM, &status);

                // Copy the blocks so that they lay linearly in memory.
                matmul_mat_set_block(pBufferCopyLocal, b, C, ldc, aiGridCoordsDest[1], aiGridCoordsDest[0]);
            }

            // Copy the root block.
            matmul_mat_set_block(pCLocal, b, C, ldc, aiGridCoords[1], aiGridCoords[0]);
        }
        else
        {
            MPI_Send(pCLocal, (int)numElementsBlock, MATMUL_MPI_ELEMENT_TYPE, MATMUL_MPI_ROOT, iCollectSendRecTag, MATMUL_MPI_COMM);
        }
#ifdef MATMUL_MPI_ADDITIONAL_DEBUG_OUTPUT
        if(rank1D == MATMUL_MPI_ROOT)
        {
            matmul_arr_free(pBufferCopyLocal);
            printf(" Finished collecting Blocks.\n");
        }
#endif

        // Free up the resources.
        matmul_arr_free(pALocal);
        matmul_arr_free(pBLocal);
        matmul_arr_free(pCLocal);

        MATMUL_TIME_END;

        MPI_Comm_free(&comm2D);

        MATMUL_TIME_RETURN;
    }

#ifdef MATMUL_BUILD_PAR_MPI_CANNON_STD
    //-----------------------------------------------------------------------------
    //
    //-----------------------------------------------------------------------------
    TReturn matmul_gemm_par_mpi_cannon_block(
        TIdx const m, TIdx const n, TIdx const k,
        TElem const alpha,
        TElem const * const MATMUL_RESTRICT A, TIdx const lda,
        TElem const * const MATMUL_RESTRICT B, TIdx const ldb,
        TElem const beta,
        TElem * const MATMUL_RESTRICT C, TIdx const ldc)
    {
        if(matmul_mat_gemm_early_out(m, n, k, alpha, beta))
        {
            MATMUL_TIME_RETURN_EARLY_OUT;
        }

        return
            matmul_gemm_par_mpi_cannon_algo(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, true, matmul_gemm_seq_multiple_opts);
    }

    //-----------------------------------------------------------------------------
    //
    //-----------------------------------------------------------------------------
    TReturn matmul_gemm_par_mpi_cannon_nonblock(
        TIdx const m, TIdx const n, TIdx const k,
        TElem const alpha,
        TElem const * const MATMUL_RESTRICT A, TIdx const lda,
        TElem const * const MATMUL_RESTRICT B, TIdx const ldb,
        TElem const beta,
        TElem * const MATMUL_RESTRICT C, TIdx const ldc)
    {
        if(matmul_mat_gemm_early_out(m, n, k, alpha, beta))
        {
            MATMUL_TIME_RETURN_EARLY_OUT;
        }

        return
            matmul_gemm_par_mpi_cannon_algo(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, false, matmul_gemm_seq_multiple_opts);
    }
#endif
#ifdef MATMUL_BUILD_PAR_MPI_CANNON_MKL

    #include <matmul/par/BlasMkl.h>

    //-----------------------------------------------------------------------------
    //
    //-----------------------------------------------------------------------------
    TReturn matmul_gemm_par_mpi_cannon_nonblock_blas_mkl(
        TIdx const m, TIdx const n, TIdx const k,
        TElem const alpha,
        TElem const * const MATMUL_RESTRICT A, TIdx const lda,
        TElem const * const MATMUL_RESTRICT B, TIdx const ldb,
        TElem const beta,
        TElem * const MATMUL_RESTRICT C, TIdx const ldc)
    {
        if(matmul_mat_gemm_early_out(m, n, k, alpha, beta))
        {
            MATMUL_TIME_RETURN_EARLY_OUT;
        }

        return
            matmul_gemm_par_mpi_cannon_algo(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, false, matmul_gemm_par_blas_mkl);
    }
#endif
#ifdef MATMUL_BUILD_PAR_MPI_CANNON_CUBLAS

    #include <matmul/par/BlasCublas.h>

    //-----------------------------------------------------------------------------
    //
    //-----------------------------------------------------------------------------
    TReturn matmul_gemm_par_mpi_cannon_nonblock_blas_cublas(
        TIdx const m, TIdx const n, TIdx const k,
        TElem const alpha,
        TElem const * const MATMUL_RESTRICT A, TIdx const lda,
        TElem const * const MATMUL_RESTRICT B, TIdx const ldb,
        TElem const beta,
        TElem * const MATMUL_RESTRICT C, TIdx const ldc)
    {
        if(matmul_mat_gemm_early_out(m, n, k, alpha, beta))
        {
            MATMUL_TIME_RETURN_EARLY_OUT;
        }

        return
            matmul_gemm_par_mpi_cannon_algo(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, false, matmul_gemm_par_blas_cublas2_memcpy);
    }
#endif
#endif
