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

#ifdef MATMUL_BUILD_PAR_STRASSEN_OMP2

    #include <matmul/par/StrassenOmp2.h>

    #include <matmul/par/Omp.h>         // matmul_gemm_par_omp2_guided_schedule

    #include <matmul/common/Alloc.h>    // matmul_arr_alloc
    #include <matmul/common/Array.h>    // matmul_arr_alloc_fill_zero
    #include <matmul/common/Mat.h>      // matmul_mat_gemm_early_out

    #include <assert.h>                 // assert
    #include <stdio.h>                  // printf

    #include <omp.h>

    #if _OPENMP >= 200203   // OpenMP 2.0
        //-----------------------------------------------------------------------------
        //
        //-----------------------------------------------------------------------------
        void matmul_mat_add_pitch_par_omp2(
            TSize const m, TSize const n,
            TElem const * const MATMUL_RESTRICT A, TSize const lda,
            TElem const * const MATMUL_RESTRICT B, TSize const ldb,
            TElem * const MATMUL_RESTRICT C, TSize const ldc)
        {
        #if _OPENMP < 200805    // For OpenMP < 3.0 you have to declare the loop index outside of the loop header.
            int iM = (int)m;
            int i;
            #pragma omp parallel for
            for(i = 0; i < iM; ++i)
        #else
            #pragma omp parallel for
            for(TSize i = 0; i < m; ++i)
        #endif
            {
                for(TSize j = 0; j < n; ++j)
                {
                    C[i*ldc + j] = A[i*lda + j] + B[i*ldb + j];
                }
            }
        }
        //-----------------------------------------------------------------------------
        //
        //-----------------------------------------------------------------------------
        void matmul_mat_add2_pitch_par_omp2(
            TSize const m, TSize const n,
            TElem const * const MATMUL_RESTRICT A, TSize const lda,
            TElem * const MATMUL_RESTRICT C, TSize const ldc)
        {
        #if _OPENMP < 200805    // For OpenMP < 3.0 you have to declare the loop index outside of the loop header.
            int iM = (int)m;
            int i;
            #pragma omp parallel for
            for(i = 0; i < iM; ++i)
        #else
            #pragma omp parallel for
            for(TSize i = 0; i < m; ++i)
        #endif
            {
                for(TSize j = 0; j < n; ++j)
                {
                    C[i*ldc + j] += A[i*lda + j];
                }
            }
        }
        //-----------------------------------------------------------------------------
        //
        //-----------------------------------------------------------------------------
        void matmul_mat_sub_pitch_par_omp2(
            TSize const m, TSize const n,
            TElem const * const MATMUL_RESTRICT A, TSize const lda,
            TElem const * const MATMUL_RESTRICT B, TSize const ldb,
            TElem * const MATMUL_RESTRICT C, TSize const ldc)
        {
        #if _OPENMP < 200805    // For OpenMP < 3.0 you have to declare the loop index outside of the loop header.
            int iM = (int)m;
            int i;
            #pragma omp parallel for
            for(i = 0; i < iM; ++i)
        #else
            #pragma omp parallel for
            for(TSize i = 0; i < m; ++i)
        #endif
            {
                for(TSize j = 0; j < n; ++j)
                {
                    C[i*ldc + j] = A[i*lda + j] - B[i*ldb + j];
                }
            }
        }

        //-----------------------------------------------------------------------------
        //
        //-----------------------------------------------------------------------------
        TReturn matmul_gemm_par_strassen_omp2(
            TSize const m, TSize const n, TSize const k,
            TElem const alpha,
            TElem const * const MATMUL_RESTRICT X, TSize const lda,
            TElem const * const MATMUL_RESTRICT Y, TSize const ldb,
            TElem const beta,
            TElem * const MATMUL_RESTRICT Z, TSize const ldc)
        {
            if(matmul_mat_gemm_early_out(m, n, k, alpha, beta))
            {
                MATMUL_TIME_RETURN_EARLY_OUT;
            }

            MATMUL_TIME_START;

            // Apply beta multiplication to C.
            if(beta != (TElem)1)
            {
        #if _OPENMP < 200805    // For OpenMP < 3.0 you have to declare the loop index outside of the loop header.
                int iM = (int)m;
                int i;
                #pragma omp parallel for
                for(i = 0; i < iM; ++i)
        #else
                #pragma omp parallel for
                for(TSize i = 0; i < m; ++i)
        #endif
                {
                    for(TSize j = 0; j < n; ++j)
                    {
                        Z[i*ldc + j] *= beta;
                    }
                }
            }

            // Recursive base case.
            // If the matrices are smaller then the cutoff size we just use the conventional algorithm.
            if((n % 2 == 1) || (n <= MATMUL_STRASSEN_OMP_CUT_OFF))
            {
                matmul_gemm_par_omp2_static_schedule(m, n, k, alpha, X, lda, Y, ldb, (TElem)1, Z, ldc);
            }
            else
            {
                // \TODO: Implement for non square matrices?
                if(m!=n || m!=k)
                {
                    printf("[GEMM Strassen OpenMP] Invalid matrix size! The matrices have to be square for the MPI Cannon GEMM.\n");
                    MATMUL_TIME_RETURN_EARLY_OUT;
                }

                TSize const h = n/2;             // size of sub-matrices

                TElem const * const A = X;      // A-D matrices embedded in X
                TElem const * const B = X + h;
                TElem const * const C = X + h*lda;
                TElem const * const D = C + h;

                TElem const * const E = Y;      // E-H matrices embeded in Y
                TElem const * const F = Y + h;
                TElem const * const G = Y + h*ldb;
                TElem const * const H = G + h;

                // Allocate temporary matrices.
                TSize const elemCount = h * h;
                TElem * P[7];
                for(TSize i = 0; i < 7; ++i)
                {
                    P[i] = matmul_arr_alloc_fill_zero(elemCount);
                }
                TElem * const T = matmul_arr_alloc(elemCount);
                TElem * const U = matmul_arr_alloc(elemCount);

                //#pragma omp parallel sections    // Parallel sections decrease the performance!
                {
                    //#pragma omp section
                    {
                        // P0 = A*(F - H);
                        matmul_mat_sub_pitch_par_omp2(h, h, F, ldb, H, ldb, T, h);
                        matmul_gemm_par_strassen_omp2(h, h, h, alpha, A, lda, T, h, (TElem)1, P[0], h);
                    }
                    //#pragma omp section
                    {
                        // P1 = (A + B)*H
                        matmul_mat_add_pitch_par_omp2(h, h, A, lda, B, lda, T, h);
                        matmul_gemm_par_strassen_omp2(h, h, h, alpha, T, h, H, ldb, (TElem)1, P[1], h);
                    }
                    //#pragma omp section
                    {
                        // P2 = (C + D)*E
                        matmul_mat_add_pitch_par_omp2(h, h, C, lda, D, lda, T, h);
                        matmul_gemm_par_strassen_omp2(h, h, h, alpha, T, h, E, ldb, (TElem)1, P[2], h);
                    }
                    //#pragma omp section
                    {
                        // P3 = D*(G - E);
                        matmul_mat_sub_pitch_par_omp2(h, h, G, ldb, E, ldb, T, h);
                        matmul_gemm_par_strassen_omp2(h, h, h, alpha, D, lda, T, h, (TElem)1, P[3], h);
                    }
                    //#pragma omp section
                    {
                        // P4 = (A + D)*(E + H)
                        matmul_mat_add_pitch_par_omp2(h, h, A, lda, D, lda, T, h);
                        matmul_mat_add_pitch_par_omp2(h, h, E, ldb, H, ldb, U, h);
                        matmul_gemm_par_strassen_omp2(h, h, h, alpha, T, h, U, h, (TElem)1, P[4], h);
                    }
                    //#pragma omp section
                    {
                        // P5 = (B - D)*(G + H)
                        matmul_mat_sub_pitch_par_omp2(h, h, B, lda, D, lda, T, h);
                        matmul_mat_add_pitch_par_omp2(h, h, G, ldb, H, ldb, U, h);
                        matmul_gemm_par_strassen_omp2(h, h, h, alpha, T, h, U, h, (TElem)1, P[5], h);
                    }
                    //#pragma omp section
                    {
                        // P6 = (A - C)*(E + F)
                        matmul_mat_sub_pitch_par_omp2(h, h, A, lda, C, lda, T, h);
                        matmul_mat_add_pitch_par_omp2(h, h, E, ldb, F, ldb, U, h);
                        matmul_gemm_par_strassen_omp2(h, h, h, alpha, T, h, U, h, (TElem)1, P[6], h);
                    }
                }

                //#pragma omp parallel sections
                {
                    //#pragma omp section
                    // Z upper left = (P3 + P4) + (P5 - P1)
                    matmul_mat_add_pitch_par_omp2(h, h, P[4], h, P[3], h, T, h);
                    //#pragma omp section
                    matmul_mat_sub_pitch_par_omp2(h, h, P[5], h, P[1], h, U, h);
                }
                TElem * const V = P[5];    // P[5] is only used once, so we reuse it as temporary buffer.
                matmul_mat_add_pitch_par_omp2(h, h, T, h, U, h, V, h);
                matmul_mat_add2_pitch_par_omp2(h, h, V, h, Z, ldc);

                // Z lower left = P2 + P3
                matmul_mat_add_pitch_par_omp2(h, h, P[2], h, P[3], h, V, h);
                matmul_mat_add2_pitch_par_omp2(h, h, V, h, Z + h*ldc, ldc);

                // Z upper right = P0 + P1
                matmul_mat_add_pitch_par_omp2(h, h, P[0], h, P[1], h, V, h);
                matmul_mat_add2_pitch_par_omp2(h, h, V, h, Z + h, ldc);

                //#pragma omp parallel sections
                {
                    //#pragma omp section
                    // Z lower right = (P0 + P4) - (P2 + P6)
                    matmul_mat_add_pitch_par_omp2(h, h, P[0], h, P[4], h, T, h);
                    //#pragma omp section
                    matmul_mat_add_pitch_par_omp2(h, h, P[2], h, P[6], h, U, h);
                }
                matmul_mat_add_pitch_par_omp2(h, h, T, h, U, h, V, h);
                matmul_mat_add2_pitch_par_omp2(h, h, V, h, Z + h*(ldc + 1), ldc);

                // Deallocate temporary matrices.
                matmul_arr_free(U);
                matmul_arr_free(T);
                for(TSize i = 0; i < 7; ++i)
                {
                    matmul_arr_free(P[i]);
                }
            }

            MATMUL_TIME_END;
            MATMUL_TIME_RETURN;
        }
    #endif
#endif
