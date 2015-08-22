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

#ifdef MATMUL_BUILD_PAR_OPENACC

    #include <matmul/par/OpenAcc.h>

    #include <matmul/common/Mat.h>  // matmul_mat_gemm_early_out

    #include <openacc.h>

    //-----------------------------------------------------------------------------
    //
    //-----------------------------------------------------------------------------
    TReturn matmul_gemm_par_openacc_kernels(
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

        MATMUL_TIME_START;

#pragma acc kernels copyin(A[0:(lda*m)], B[0:(ldb*k)]) copy(C[0:(ldc*m)])
        {
#pragma acc loop independent gang(MATMUL_OPENACC_GANG_SIZE)
            for(TSize i = 0; i < m; ++i)
            {
#pragma acc loop independent vector(MATMUL_OPENACC_VECTOR_SIZE)
                for(TSize j = 0; j < n; ++j)
                {
                    TElem ctmp = 0;
#pragma acc loop seq//reduction(+:ctmp) // Reduction here is much slower then sequential execution!
                    for(TSize k2 = 0; k2 < k; ++k2)
                    {
                        ctmp += alpha * A[i*lda + k2] * B[k2*ldb +j];
                    }
                    C[i*ldc + j] += C[i*ldc + j] * beta + ctmp;
                }
            }
        }

        MATMUL_TIME_END;
        MATMUL_TIME_RETURN;
    }

    //-----------------------------------------------------------------------------
    //
    //-----------------------------------------------------------------------------
    TReturn matmul_gemm_par_openacc_parallel(
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

        MATMUL_TIME_START;

#pragma acc parallel copyin(A[0:(lda*m)], B[0:(ldb*k)]) copy(C[0:(ldc*m)])
        {
#pragma acc loop
            for(TSize i = 0; i < m; ++i)
            {
#pragma acc loop
                for(TSize j = 0; j < n; ++j)
                {
                    C[i*ldc + j] *= beta;
#pragma acc loop seq // Reduction here is much slower then sequential execution!
                    for(TSize k2 = 0; k2 < k; ++k2)
                    {
                        C[i*ldc + j] += alpha * A[i*lda + k2] * B[k2*ldb + j];
                    }
                }
            }
        }

        MATMUL_TIME_END;
        MATMUL_TIME_RETURN;
    }

#endif
