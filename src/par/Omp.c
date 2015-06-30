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

#if defined(MATMUL_BUILD_PAR_OMP2) || defined(MATMUL_BUILD_PAR_OMP3) || defined(MATMUL_BUILD_PAR_OMP4)

    #include <matmul/par/Omp.h>

    #include <matmul/common/Mat.h>  // matmul_mat_gemm_early_out

    #include <omp.h>

    #include <stdio.h>              // printf

    #ifdef MATMUL_BUILD_PAR_OMP2
        #if _OPENMP >= 200203   // OpenMP 2.0
            //-----------------------------------------------------------------------------
            //
            //-----------------------------------------------------------------------------
            void matmul_gemm_par_omp2_guided_schedule(
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

                #pragma omp parallel// shared(n,lda,A,ldb,B,ldc,C)
                {
            #ifdef MATMUL_OMP_PRINT_NUM_CORES
                    #pragma omp single
                    {
                        printf(" p=%d ", omp_get_num_threads());
                    }
            #endif

            #if _OPENMP < 200805    // For OpenMP < 3.0 you have to declare the loop index outside of the loop header.
                    int iM = (int)m;
                    int i;
                    #pragma omp for schedule(guided)
                    for(i = 0; i < iM; ++i)
            #else
                    #pragma omp for schedule(guided)
                    for(TIdx i = 0; i < m; ++i)
            #endif
                    {
                        for(TIdx j = 0; j < n; ++j)
                        {
                            C[i*ldc + j] *= beta;
                        }
                        for(TIdx k2 = 0; k2 < k; ++k2)
                        {
                            TElem const a = alpha * A[i*lda + k2];

                            for(TIdx j = 0; j < n; ++j)
                            {
                                C[i*ldc + j] += a * B[k2*ldb + j];
                            }
                        }
                    }
                }
            }
            //-----------------------------------------------------------------------------
            //
            //-----------------------------------------------------------------------------
            void matmul_gemm_par_omp2_static_schedule(
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

                #pragma omp parallel //shared(A,B,C)
                {
                    #ifdef MATMUL_OMP_PRINT_NUM_CORES
            #pragma omp single
                    {
                        printf(" p=%d ", omp_get_num_threads());
                    }
            #endif

            #if _OPENMP < 200805    // For OpenMP < 3.0 you have to declare the loop index outside of the loop header.
                    int iM = (int)m;
                    int i;
                    #pragma omp for schedule(static)
                    for(i = 0; i < iM; ++i)
            #else
                    #pragma omp for schedule(static)
                    for(TIdx i = 0; i < m; ++i)
            #endif
                    {
                        for(TIdx j = 0; j < n; ++j)
                        {
                            C[i*ldc + j] *= beta;
                        }
                        for(TIdx k2 = 0; k2 < k; ++k2)
                        {
                            TElem const a = alpha * A[i*lda + k2];

                            for(TIdx j = 0; j < n; ++j)
                            {
                                C[i*ldc + j] += a * B[k2*ldb + j];
                            }
                        }
                    }
                }
            }
        #endif
    #endif
    #ifdef MATMUL_BUILD_PAR_OMP3
        #if _OPENMP >= 200805   // OpenMP 3.0
            //-----------------------------------------------------------------------------
            //
            //-----------------------------------------------------------------------------
            void matmul_gemm_par_omp3_static_schedule_collapse(
                TIdx const m, TIdx const n, TIdx const k,
                TElem const alpha,
                TElem const * const MATMUL_RESTRICT A,  TIdx const lda,
                TElem const * const MATMUL_RESTRICT B,  TIdx const ldb,
                TElem const beta,
                TElem * const MATMUL_RESTRICT C,  TIdx const ldc)
            {
                if(matmul_mat_gemm_early_out(m, n, k, alpha, beta))
                {
                    return;
                }

                #pragma omp parallel //shared(A,B,C)
                {
            #ifdef MATMUL_OMP_PRINT_NUM_CORES
                    #pragma omp single
                    {
                        printf(" p=%d ", omp_get_num_threads());
                    }
            #endif

                    #pragma omp for collapse(2) schedule(static)
                    for(TIdx i = 0; i < m; ++i)
                    {
                        for(TIdx j = 0; j < n; ++j)
                        {
                            C[i*ldc + j] *= beta;
                        }
                    }

                    // NOTE:
                    // - ikj-order not possible.
                    // - In ijk order we can only collapse the outer two loops.
                    // Both restrictions are due to the non-atomic write to C (multiple threads could write to the same indices i and j of C)
                    #pragma omp for collapse(2) schedule(static)    // http://software.intel.com/en-us/articles/openmp-loop-collapse-directive
                    for(TIdx i = 0; i < m; ++i)
                    {
                        for(TIdx j = 0; j < n; ++j)
                        {
                            for(TIdx k2 = 0; k2 < k; ++k2)
                            {
                                C[i*ldc + j] += alpha * A[i*lda + k2] * B[k2*ldb + j];
                            }
                        }
                    }
                }
            }
        #endif
    #endif
    #ifdef MATMUL_BUILD_PAR_OMP2
        #if _OPENMP >= 201307   // OpenMP 4.0
            //-----------------------------------------------------------------------------
            //
            //-----------------------------------------------------------------------------
            void matmul_gemm_par_omp4(
                TIdx const m, TIdx const n, TIdx const k,
                TElem const alpha,
                TElem const * const MATMUL_RESTRICT A,  TIdx const lda,
                TElem const * const MATMUL_RESTRICT B,  TIdx const ldb,
                TElem const beta,
                TElem * const MATMUL_RESTRICT C,  TIdx const ldc)
            {
                if(matmul_mat_gemm_early_out(m, n, k, alpha, beta))
                {
                    return;
                }
                #pragma omp target if(0) map(to: m, n, k, alpha, A[0:lda*m], lda, B[0:ldb*k], ldb, beta, ldc) map(tofrom: C[0:ldc*m])

                #pragma omp teams /*num_teams(...) thread_limit(...)*/
                {
                    #pragma omp distribute
                    for(TIdx i = 0; i < m; ++i)
                    {
                        #pragma omp parallel for  /*num_threads(...)*/ schedule(static)
                        for(TIdx j = 0; j < n; ++j)
                        {
                            C[i*ldc + j] *= beta;
                        }
                        // NOTE: ikj-order not possible due to the non-atomic write to C (multiple threads could write to the same indices i and j of C)
                        #pragma omp parallel for  /*num_threads(...)*/ schedule(static)
                        for(TIdx j = 0; j < n; ++j)
                        {
                            for(TIdx k2 = 0; k2 < k; ++k2)
                            {
                                C[i*ldc + j] += alpha * A[i*lda + k2] * B[k2*ldb + j];
                            }
                        }
                    }
                }
            }
        #endif
    #endif
#endif
