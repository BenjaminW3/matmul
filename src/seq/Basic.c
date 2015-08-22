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

#ifdef MATMUL_BUILD_SEQ_BASIC

    #include <matmul/seq/Basic.h>

    #include <matmul/common/Mat.h>  // matmul_mat_gemm_early_out

    //-----------------------------------------------------------------------------
    //
    //-----------------------------------------------------------------------------
    TReturn matmul_gemm_seq_basic(
        TSize const m, TSize const n, TSize const k,
        TElem const alpha,
        TElem const * const A, TSize const lda,
        TElem const * const B, TSize const ldb,
        TElem const beta,
        TElem * const C, TSize const ldc)
    {
        if(matmul_mat_gemm_early_out(m, n, k, alpha, beta))
        {
            MATMUL_TIME_RETURN_EARLY_OUT;
        }

        MATMUL_TIME_START;

        for(TSize i = 0; i < m; ++i)
        {
            for(TSize j = 0; j < n; ++j)
            {
                C[i*ldc + j] *= beta;
                for(TSize k2 = 0; k2 < k; ++k2)
                {
                    C[i*ldc + j] += alpha * A[i*lda + k2] * B[k2*ldb + j];
                }
            }
        }

        MATMUL_TIME_END;
        MATMUL_TIME_RETURN;
    }
#endif
