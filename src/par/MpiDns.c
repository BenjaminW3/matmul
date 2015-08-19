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

#ifdef MATMUL_BUILD_PAR_MPI_DNS

    #include <matmul/par/MpiDns.h>

    #include <matmul/seq/MultipleOpts.h>
    #include <matmul/common/Alloc.h>
    #include <matmul/common/Mat.h>          // matmul_mat_row_major_to_mat_x_block_major, matmul_mat_gemm_early_out
    #include <matmul/common/Array.h>        // matmul_arr_alloc_fill_zero

    #include <stdbool.h>                    // bool
    #include <math.h>                       // cbrt
    #include <stdio.h>                      // printf
    #include <assert.h>                     // assert

    #include <mpi.h>

    //#define MATMUL_MPI_ADDITIONAL_DEBUG_OUTPUT

    //-----------------------------------------------------------------------------
    // Adapted from Camillo Lugaresi and Cosmin Stroe https://github.com/cstroe/PP-MM-A03/blob/master/code/dns.c
    //-----------------------------------------------------------------------------

    #define I_DIM 0
    #define J_DIM 1
    #define K_DIM 2

    //-----------------------------------------------------------------------------
    // Holds the information about the current topology and the matrix sizes.
    //-----------------------------------------------------------------------------
    typedef struct STopologyInfo
    {
        MPI_Comm commMesh3D;

        MPI_Comm commMeshIK, commRingJ;
        MPI_Comm commMeshJK, commRingI;
        MPI_Comm commMeshIJ, commRingK;

        int iLocalRank1D;               // Local rank in 1D communicator.

        int aiGridCoords[3];            // Local coordinates.

        TIdx n;                         // The size of the full matrices is n x n
        TIdx b;                         // b = (n/q)
    } STopologyInfo;

    //-----------------------------------------------------------------------------
    //
    //-----------------------------------------------------------------------------
    void matmul_gemm_par_mpi_dns_scatter_mat_blocks_2d(
        STopologyInfo const * const MATMUL_RESTRICT info,
        TElem const * const MATMUL_RESTRICT pX,
        TIdx const ldx,
        TElem * const MATMUL_RESTRICT pXSub,
        bool const columnFirst,
        MPI_Comm const mesh)
    {
        TElem * pXBlocks = 0;
        if(info->iLocalRank1D == MATMUL_MPI_ROOT)
        {
            TIdx const elemCount =  info->n * info->n;
            pXBlocks = matmul_arr_alloc(elemCount);

            matmul_mat_row_major_to_mat_x_block_major(pX, info->n, info->n, ldx, pXBlocks, info->b, columnFirst);
        }

        TIdx const numElementsBlock = info->b * info->b;

        MPI_Scatter(pXBlocks, (int)numElementsBlock, MATMUL_MPI_ELEMENT_TYPE, pXSub, (int)numElementsBlock, MATMUL_MPI_ELEMENT_TYPE, MATMUL_MPI_ROOT, mesh);

        if(info->iLocalRank1D == MATMUL_MPI_ROOT)
        {
            matmul_arr_free(pXBlocks);
        }
    }

    //-----------------------------------------------------------------------------
    // Distribute the A or B matrix.
    //-----------------------------------------------------------------------------
    void matmul_gemm_par_mpi_dns_distribute_mat(
        STopologyInfo const * const MATMUL_RESTRICT info,
        TElem const * const MATMUL_RESTRICT pX,
        TIdx const ldx,
        TElem * const MATMUL_RESTRICT pXSub,
        TIdx const ringdim,
        bool const columnFirst)
    {
        MPI_Comm mesh, ring;
        if(ringdim == J_DIM)
        {
            mesh = info->commMeshIK;
            ring = info->commRingJ;
        }
        else
        {
            mesh = info->commMeshJK;
            ring = info->commRingI;
        }

        // Scatter the matrix in its plane.
        if(info->aiGridCoords[ringdim] == MATMUL_MPI_ROOT)
        {
            matmul_gemm_par_mpi_dns_scatter_mat_blocks_2d(
                info,
                pX,
                ldx,
                pXSub,
                columnFirst,
                mesh);
        }

        TIdx const numElementsBlock = info->b * info->b;

        // Broadcast the matrix into the cube.
        MPI_Bcast(pXSub, (int)numElementsBlock, MATMUL_MPI_ELEMENT_TYPE, MATMUL_MPI_ROOT, ring);
    }

    //-----------------------------------------------------------------------------
    //
    //-----------------------------------------------------------------------------
    void matmul_gemm_par_mpi_dns_reduce_c(
        STopologyInfo const * const MATMUL_RESTRICT info,
        TElem * const MATMUL_RESTRICT pCSub)
    {
        TIdx const numElementsBlock = info->b * info->b;

        // Reduce along k dimension to the i-j plane
        if(info->aiGridCoords[K_DIM] == MATMUL_MPI_ROOT)
        {
            MPI_Reduce(MPI_IN_PLACE, pCSub, (int)numElementsBlock, MATMUL_MPI_ELEMENT_TYPE, MPI_SUM, MATMUL_MPI_ROOT, info->commRingK);
        }
        else
        {
            MPI_Reduce(pCSub, 0, (int)numElementsBlock, MATMUL_MPI_ELEMENT_TYPE, MPI_SUM, MATMUL_MPI_ROOT, info->commRingK);
        }
    }

    //-----------------------------------------------------------------------------
    //
    //-----------------------------------------------------------------------------
    void matmul_gemm_par_mpi_dns_scatter_c_blocks_2d(
        STopologyInfo const * const MATMUL_RESTRICT info,
        TElem * const MATMUL_RESTRICT C,
        TIdx const ldc,
        TElem * const MATMUL_RESTRICT pCSub)
    {
        if(info->aiGridCoords[K_DIM] == MATMUL_MPI_ROOT)
        {
            matmul_gemm_par_mpi_dns_scatter_mat_blocks_2d(
                info,
                C,
                ldc,
                pCSub,
                false,
                info->commMeshIJ);
        }
    }

    //-----------------------------------------------------------------------------
    //
    //-----------------------------------------------------------------------------
    void matmul_gemm_par_mpi_dns_gather_c_blocks_2d(
        STopologyInfo const * const MATMUL_RESTRICT info,
        TElem * const MATMUL_RESTRICT pCSub,
        TElem * const MATMUL_RESTRICT C,
        TIdx const ldc)
    {
        if(info->aiGridCoords[K_DIM] == MATMUL_MPI_ROOT)
        {
            TElem * pCBlocks = 0;
            if(info->iLocalRank1D == MATMUL_MPI_ROOT)
            {
                TIdx const elemCount = info->n * info->n;
                pCBlocks = matmul_arr_alloc(elemCount);
            }

            TIdx const numElementsBlock = info->b * info->b;

            MPI_Gather(pCSub, (int)numElementsBlock, MATMUL_MPI_ELEMENT_TYPE, pCBlocks, (int)numElementsBlock, MATMUL_MPI_ELEMENT_TYPE, MATMUL_MPI_ROOT, info->commMeshIJ);

            if(info->iLocalRank1D == MATMUL_MPI_ROOT)
            {
                matmul_mat_x_block_major_to_mat_row_major(pCBlocks, info->b, C, info->n, info->n, ldc, false);
                matmul_arr_free(pCBlocks);
            }
        }
    }

    //-----------------------------------------------------------------------------
    //
    //-----------------------------------------------------------------------------
    void matmul_gemm_par_mpi_dns_local(
        STopologyInfo const * const MATMUL_RESTRICT info,
        TElem const alpha,
        TElem const * const MATMUL_RESTRICT A, TIdx const lda,
        TElem const * const MATMUL_RESTRICT B, TIdx const ldb,
        TElem const beta,
        TElem * const MATMUL_RESTRICT C, TIdx const ldc,
        TReturn(*pGemm)(TIdx const, TIdx const, TIdx const, TElem const, TElem const * const, TIdx const, TElem const * const, TIdx const, TElem const, TElem * const, TIdx const))
    {
        assert(info->commMesh3D);
        assert(info->n>0);
        assert(info->b>0);

        // Allocate the local matrices
        TIdx const numElementsBlock = info->b * info->b;
        TElem * const ASub = matmul_arr_alloc(numElementsBlock);
        TElem * const BSub = matmul_arr_alloc(numElementsBlock);
        // The elements in the root IJ plane get sub-matrices of the input C.
        // All others are zero. Even though we multiply with beta=0, an uninitialized array is not enough because values could be NaN and 0*NaN=NaN.
        bool const bIJPlane = (info->aiGridCoords[K_DIM] == MATMUL_MPI_ROOT);
        TElem * const CSub = bIJPlane
                                ? matmul_arr_alloc(numElementsBlock)
                                : matmul_arr_alloc_fill_zero(numElementsBlock);

        // Scatter C on the i-j plane.
        matmul_gemm_par_mpi_dns_scatter_c_blocks_2d(info, C, ldc, CSub);
        // Distribute A along the i-k plane and then in the j direction.
        matmul_gemm_par_mpi_dns_distribute_mat(info, A, lda, ASub, J_DIM, false);
        // Distribute B along the k-j plane and then in the i direction.
        matmul_gemm_par_mpi_dns_distribute_mat(info, B, ldb, BSub, I_DIM, true);

        // Apply beta multiplication to local C.
        if(bIJPlane)
        {
            if(alpha != (TElem)1)
            {
                for(TIdx i = 0; i < info->b; ++i)
                {
                    for(TIdx j = 0; j < info->b; ++j)
                    {
                        CSub[i*info->b + j] *= beta;
                    }
                }
            }
        }

        // Do the local matrix multiplication.
        pGemm(info->b, info->b, info->b, alpha, ASub, info->b, BSub, info->b, (TElem)1, CSub, info->b);

        // Reduce along k dimension to the i-j plane
        matmul_gemm_par_mpi_dns_reduce_c(info, CSub);

        // Gather C on the i-j plane to the root node.
        matmul_gemm_par_mpi_dns_gather_c_blocks_2d(info, CSub, C, ldc);

        matmul_arr_free(CSub);
        matmul_arr_free(BSub);
        matmul_arr_free(ASub);
    }

    //-----------------------------------------------------------------------------
    // \param info The topology information structure to be filled.
    // \param n The matrix size.
    // \return The success status.
    //-----------------------------------------------------------------------------
    bool matmul_gemm_par_mpi_dns_create_topology_info(
        STopologyInfo * const info,
        TIdx const n)
    {
        info->n = n;

        // Get the number of processes.
        int iNumProcesses;
        MPI_Comm_size(MATMUL_MPI_COMM, &iNumProcesses);

        // Get the local Rank
        MPI_Comm_rank(MATMUL_MPI_COMM, &info->iLocalRank1D);

#ifdef MATMUL_MPI_ADDITIONAL_DEBUG_OUTPUT
        if(info->iLocalRank1D == MATMUL_MPI_ROOT)
        {
            printf(" p=%d", iNumProcesses);
        }
#endif

        // Set up the sizes for a cartesian 3d mesh topology.
        // The number of processors in each dimension of the 3D-Mesh. Each processor will receive a block of (n/q)*(n/q) elements of A and B.
        TIdx const q = (TIdx)cbrt((double)iNumProcesses);

        // Test if it is a cube.
        if(q * q * q != iNumProcesses)
        {
            if(info->iLocalRank1D==MATMUL_MPI_ROOT)
            {
                printf("\n[GEMM MPI DNS] Invalid environment! The number of processors (%d given) should be perfect cube.\n", iNumProcesses);
            }
            return false;
        }
#ifdef MATMUL_MPI_ADDITIONAL_DEBUG_OUTPUT
        if(info->iLocalRank1D == MATMUL_MPI_ROOT)
        {
            printf(" -> %"MATMUL_PRINTF_SIZE_T" x %"MATMUL_PRINTF_SIZE_T" x %"MATMUL_PRINTF_SIZE_T" mesh", (size_t)info->q, (size_t)info->q, (size_t)info->q);
        }
#endif

        // Test if the matrix can be divided equally. This can fail if e.g. the matrix is 3x3 and the processes are 2x2.
        if(n % q != 0)
        {
            if(info->iLocalRank1D == MATMUL_MPI_ROOT)
            {
                printf("\n[GEMM MPI DNS] The matrices can't be divided among processors equally!\n");
            }
            return false;
        }

        // Determine block size of the local block.
        info->b = n/q;

        // Set that the structure is periodical around the given dimension for wraparound connections.
        int aiPeriods[3] = {(int)true, (int)true, (int)true};
        int aiProcesses[3] = {(int)q, (int)q, (int)q};

        // Create the cartesian 2d grid topology. Ranks can be reordered.
        MPI_Cart_create(MATMUL_MPI_COMM, 3, aiProcesses, aiPeriods, (int)true, &info->commMesh3D);

        // Get the rank and coordinates with respect to the new 3D grid topology.
        int iLocalRank3D = 0;
        MPI_Comm_rank(info->commMesh3D, &iLocalRank3D);
        MPI_Cart_coords(info->commMesh3D, iLocalRank3D, 3, info->aiGridCoords);

#ifdef MATMUL_MPI_ADDITIONAL_DEBUG_OUTPUT
        printf(" iLocalRank3D=%d, i=%d j=%d k=%d\n", iLocalRank3D, info->aiGridCoords[I_DIM], info->aiGridCoords[J_DIM], info->aiGridCoords[K_DIM]);
#endif

        int dims[3];

        // Create the i-k plane.
        dims[I_DIM] = dims[K_DIM] = (int)true;
        dims[J_DIM] = (int)false;
        MPI_Cart_sub(info->commMesh3D, dims, &info->commMeshIK);

        // Create the j-k plane.
        dims[J_DIM] = dims[K_DIM] = (int)true;
        dims[I_DIM] = (int)false;
        MPI_Cart_sub(info->commMesh3D, dims, &info->commMeshJK);

        // Create the i-j plane.
        dims[I_DIM] = dims[J_DIM] = (int)true;
        dims[K_DIM] = (int)false;
        MPI_Cart_sub(info->commMesh3D, dims, &info->commMeshIJ);

        // Create the i ring.
        dims[J_DIM] = dims[K_DIM] = (int)false;
        dims[I_DIM] = (int)true;
        MPI_Cart_sub(info->commMesh3D, dims, &info->commRingI);

        // Create the j ring.
        dims[I_DIM] = dims[K_DIM] = (int)false;
        dims[J_DIM] = (int)true;
        MPI_Cart_sub(info->commMesh3D, dims, &info->commRingJ);

        // Create the k rings.
        dims[I_DIM] = dims[J_DIM] = (int)false;
        dims[K_DIM] = (int)true;
        MPI_Cart_sub(info->commMesh3D, dims, &info->commRingK);

        return true;
    }
    //-----------------------------------------------------------------------------
    //
    //-----------------------------------------------------------------------------
    void matmul_gemm_par_mpi_dns_destroy_topology_info(STopologyInfo * const info)
    {
        MPI_Comm_free(&info->commMeshIK);
        MPI_Comm_free(&info->commMeshJK);
        MPI_Comm_free(&info->commMeshIJ);
        MPI_Comm_free(&info->commRingI);
        MPI_Comm_free(&info->commRingJ);
        MPI_Comm_free(&info->commRingK);
        MPI_Comm_free(&info->commMesh3D);
    }
    //-----------------------------------------------------------------------------
    //
    //-----------------------------------------------------------------------------
    TReturn matmul_gemm_par_mpi_dns_algo(
        TIdx const m, TIdx const n, TIdx const k,
        TElem const alpha,
        TElem const * const MATMUL_RESTRICT A, TIdx const lda,
        TElem const * const MATMUL_RESTRICT B, TIdx const ldb,
        TElem const beta,
        TElem * const MATMUL_RESTRICT C, TIdx const ldc,
        TReturn(*pGemm)(TIdx const, TIdx const, TIdx const, TElem const, TElem const * const, TIdx const, TElem const * const, TIdx const, TElem const, TElem * const, TIdx const))
    {
        if(matmul_mat_gemm_early_out(m, n, k, alpha, beta))
        {
            MATMUL_TIME_RETURN_EARLY_OUT;
        }

        // \TODO: Implement for non square matrices?
        if((m!=n) || (m!=k))
        {
            printf("[GEMM MPI DNS] Invalid matrix size! The matrices have to be square for the MPI DNS GEMM.\n");
            MATMUL_TIME_RETURN_EARLY_OUT;
        }

        // \FIXME: Fix alpha != 1!
        if(alpha!=(TElem)1)
        {
            printf("[GEMM MPI DNS] alpha != 1 currently not implemented.\n");
            MATMUL_TIME_RETURN_EARLY_OUT;
        }

        struct STopologyInfo info;
        if(!matmul_gemm_par_mpi_dns_create_topology_info(&info, n))
        {
            MATMUL_TIME_RETURN_EARLY_OUT;
        }
        else
        {
            MATMUL_TIME_START;

            matmul_gemm_par_mpi_dns_local(&info, alpha, A, lda, B, ldb, beta, C, ldc, pGemm);

            MATMUL_TIME_END;

            matmul_gemm_par_mpi_dns_destroy_topology_info(&info);

            MATMUL_TIME_RETURN;
        }
    }

    //-----------------------------------------------------------------------------
    //
    //-----------------------------------------------------------------------------
    TReturn matmul_gemm_par_mpi_dns(
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
            matmul_gemm_par_mpi_dns_algo(
                m, n, k,
                alpha,
                A, lda,
                B, ldb,
                beta,
                C, ldc,
                matmul_gemm_seq_multiple_opts);
    }
#endif
