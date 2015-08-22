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

#pragma once

#if defined(MATMUL_BUILD_PAR_PHI_OFF_OMP2_GUIDED) ||  defined(MATMUL_BUILD_PAR_PHI_OFF_OMP2_STATIC) ||  defined(MATMUL_BUILD_PAR_PHI_OFF_OMP3) ||  defined(MATMUL_BUILD_PAR_PHI_OFF_OMP4)

    #include <matmul/common/Config.h>   // TElem, TSize, TReturn

    #ifdef __cplusplus
        extern "C"
        {
    #endif
    #if _OPENMP >= 200203   // OpenMP 2.0
        #ifdef MATMUL_BUILD_PAR_PHI_OFF_OMP2_GUIDED
            //-----------------------------------------------------------------------------
            //! (S/D)GEMM matrix-matrix product C = alpha * A * B + beta * C using OpenMP 2.0 parallel for guided schedule offloading to Xeon Phi.
            //!
            //! \param m Specifies the number of rows of the matrix A and of the matrix C.
            //! \param n Specifies the number of columns of the matrix B and the number of columns of the matrix C.
            //! \param k Specifies the number of columns of the matrix A and the number of rows of the matrix B.
            //! \param alpha Scalar value used to scale the product of matrices A and B.
            //! \param A Array, size lda-by-k. The leading m-by-k part of the array must contain the matrix A.
            //! \param lda Specifies the leading dimension of A.
            //! \param B Array, size ldb-by-n. The leading k-by-n part of the array must contain the matrix B.
            //! \param ldb Specifies the leading dimension of B.
            //! \param beta Scalar value used to scale matrix C.
            //! \param C Array, size ldc-by-n. The leading m-by-n part of the array must contain the matrix C.
            //! \param ldc Specifies the leading dimension of C.
            //-----------------------------------------------------------------------------
            TReturn matmul_gemm_par_phi_off_omp2_guided_schedule(
                TSize const m, TSize const n, TSize const k,
                TElem const alpha,
                TElem const * const MATMUL_RESTRICT A, TSize const lda,
                TElem const * const MATMUL_RESTRICT B, TSize const ldb,
                TElem const beta,
                TElem * const MATMUL_RESTRICT C, TSize const ldc);
        #endif
        #ifdef MATMUL_BUILD_PAR_PHI_OFF_OMP2_STATIC
            //-----------------------------------------------------------------------------
            //! (S/D)GEMM matrix-matrix product C = alpha * A * B + beta * C using OpenMP 2.0 parallel for static schedule offloading to Xeon Phi.
            //!
            //! \param m Specifies the number of rows of the matrix A and of the matrix C.
            //! \param n Specifies the number of columns of the matrix B and the number of columns of the matrix C.
            //! \param k Specifies the number of columns of the matrix A and the number of rows of the matrix B.
            //! \param alpha Scalar value used to scale the product of matrices A and B.
            //! \param A Array, size lda-by-k. The leading m-by-k part of the array must contain the matrix A.
            //! \param lda Specifies the leading dimension of A.
            //! \param B Array, size ldb-by-n. The leading k-by-n part of the array must contain the matrix B.
            //! \param ldb Specifies the leading dimension of B.
            //! \param beta Scalar value used to scale matrix C.
            //! \param C Array, size ldc-by-n. The leading m-by-n part of the array must contain the matrix C.
            //! \param ldc Specifies the leading dimension of C.
            //-----------------------------------------------------------------------------
            TReturn matmul_gemm_par_phi_off_omp2_static_schedule(
                TSize const m, TSize const n, TSize const k,
                TElem const alpha,
                TElem const * const MATMUL_RESTRICT A, TSize const lda,
                TElem const * const MATMUL_RESTRICT B, TSize const ldb,
                TElem const beta,
                TElem * const MATMUL_RESTRICT C, TSize const ldc);
        #endif
    #endif
    #if _OPENMP >= 200805   // OpenMP 3.0
        #ifdef MATMUL_BUILD_PAR_PHI_OFF_OMP3
            //-----------------------------------------------------------------------------
            //! (S/D)GEMM matrix-matrix product C = alpha * A * B + beta * C using OpenMP 3.0 parallel for collapse static schedule offloading to Xeon Phi.
            //!
            //! \param m Specifies the number of rows of the matrix A and of the matrix C.
            //! \param n Specifies the number of columns of the matrix B and the number of columns of the matrix C.
            //! \param k Specifies the number of columns of the matrix A and the number of rows of the matrix B.
            //! \param alpha Scalar value used to scale the product of matrices A and B.
            //! \param A Array, size lda-by-k. The leading m-by-k part of the array must contain the matrix A.
            //! \param lda Specifies the leading dimension of A.
            //! \param B Array, size ldb-by-n. The leading k-by-n part of the array must contain the matrix B.
            //! \param ldb Specifies the leading dimension of B.
            //! \param beta Scalar value used to scale matrix C.
            //! \param C Array, size ldc-by-n. The leading m-by-n part of the array must contain the matrix C.
            //! \param ldc Specifies the leading dimension of C.
            //-----------------------------------------------------------------------------
            TReturn matmul_gemm_par_phi_off_omp3_static_schedule_collapse(
                TSize const m, TSize const n, TSize const k,
                TElem const alpha,
                TElem const * const MATMUL_RESTRICT A, TSize const lda,
                TElem const * const MATMUL_RESTRICT B, TSize const ldb,
                TElem const beta,
                TElem * const MATMUL_RESTRICT C, TSize const ldc);
        #endif
    #endif
    #if _OPENMP >= 201307   // OpenMP 4.0
        #ifdef MATMUL_BUILD_PAR_PHI_OFF_OMP4
            //-----------------------------------------------------------------------------
            //! (S/D)GEMM matrix-matrix product C = alpha * A * B + beta * C using OpenMP 4.0 teams distribute parallel for static schedule offloading to Xeon Phi.
            //!
            //! \param m Specifies the number of rows of the matrix A and of the matrix C.
            //! \param n Specifies the number of columns of the matrix B and the number of columns of the matrix C.
            //! \param k Specifies the number of columns of the matrix A and the number of rows of the matrix B.
            //! \param alpha Scalar value used to scale the product of matrices A and B.
            //! \param A Array, size lda-by-k. The leading m-by-k part of the array must contain the matrix A.
            //! \param lda Specifies the leading dimension of A.
            //! \param B Array, size ldb-by-n. The leading k-by-n part of the array must contain the matrix B.
            //! \param ldb Specifies the leading dimension of B.
            //! \param beta Scalar value used to scale matrix C.
            //! \param C Array, size ldc-by-n. The leading m-by-n part of the array must contain the matrix C.
            //! \param ldc Specifies the leading dimension of C.
            //-----------------------------------------------------------------------------
            TReturn matmul_gemm_par_phi_off_omp4(
                TSize const m, TSize const n, TSize const k,
                TElem const alpha,
                TElem const * const MATMUL_RESTRICT A, TSize const lda,
                TElem const * const MATMUL_RESTRICT B, TSize const ldb,
                TElem const beta,
                TElem * const MATMUL_RESTRICT C, TSize const ldc);
        #endif
    #endif
    #ifdef __cplusplus
        }
    #endif
#endif
