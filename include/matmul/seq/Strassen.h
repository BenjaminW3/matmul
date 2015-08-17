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

#ifdef MATMUL_BUILD_SEQ_STRASSEN

    #include <matmul/common/Config.h>   // TElem, TIdx, TReturn

    #ifdef __cplusplus
        extern "C"
        {
    #endif
    //-----------------------------------------------------------------------------
    //! Matrix-matrix addition C := A + B using OpenMP parallel for.
    //!
    //! \param m Specifies the number of rows of the matrix A and of the matrix C.
    //! \param n Specifies the number of columns of the matrix B and the number of columns of the matrix C.
    //! \param k Specifies the number of columns of the matrix A and the number of rows of the matrix B.
    //! \param A Array, size lda-by-k. The leading m-by-k part of the array must contain the matrix A.
    //! \param lda Specifies the leading dimension of A.
    //! \param B Array, size ldb-by-n. The leading k-by-n part of the array must contain the matrix B.
    //! \param ldb Specifies the leading dimension of B.
    //! \param C Array, size ldc-by-n. The leading m-by-n part of the array must contain the matrix C.
    //! \param ldc Specifies the leading dimension of C.
    //! \param C The input and result matrix.
    //-----------------------------------------------------------------------------
    void matmul_mat_add_pitch_seq(
        TIdx const m, TIdx const n,
        TElem const * const MATMUL_RESTRICT A, TIdx const lda,
        TElem const * const MATMUL_RESTRICT B, TIdx const ldb,
        TElem * const MATMUL_RESTRICT C, TIdx const ldc);

    //-----------------------------------------------------------------------------
    //! Matrix-matrix subtraction C := A - B using OpenMP parallel for.
    //!
    //! \param m Specifies the number of rows of the matrix A and of the matrix C.
    //! \param n Specifies the number of columns of the matrix B and the number of columns of the matrix C.
    //! \param k Specifies the number of columns of the matrix A and the number of rows of the matrix B.
    //! \param A Array, size lda-by-k. The leading m-by-k part of the array must contain the matrix A.
    //! \param lda Specifies the leading dimension of A.
    //! \param B Array, size ldb-by-n. The leading k-by-n part of the array must contain the matrix B.
    //! \param ldb Specifies the leading dimension of B.
    //! \param C Array, size ldc-by-n. The leading m-by-n part of the array must contain the matrix C.
    //! \param ldc Specifies the leading dimension of C.
    //! \param C The input and result matrix.
    //-----------------------------------------------------------------------------
    void matmul_mat_sub_pitch_seq(
        TIdx const m, TIdx const n,
        TElem const * const MATMUL_RESTRICT A, TIdx const lda,
        TElem const * const MATMUL_RESTRICT B, TIdx const ldb,
        TElem * const MATMUL_RESTRICT C, TIdx const ldc);

    //-----------------------------------------------------------------------------
    //! (S/D)GEMM matrix-matrix product C = alpha * A * B + beta * C using the (Volker) Strassen algorithm.
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
    //!
    //! Theoretical Runtime is O(n^log2(7)) = O(n^2.807).
    //! Assume NxN matrices where n is a power of two.
    //! Algorithm:
    //!   Matrices X and Y are split into four smaller
    //!   (n/2)x(n/2) matrices as follows:
    //!          _    _          _   _
    //!     X = | A  B |    Y = | E F |
    //!         | C  D |        | G H |
    //!          -    -          -   -
    //!   Then we build the following 7 matrices (requiring seven (n/2)x(n/2) matrix multiplications -- this is where the 2.807 = log2(7) improvement comes from):
    //!     P0 = A*(F - H);
    //!     P1 = (A + B)*H
    //!     P2 = (C + D)*E
    //!     P3 = D*(G - E);
    //!     P4 = (A + D)*(E + H)
    //!     P5 = (B - D)*(G + H)
    //!     P6 = (A - C)*(E + F)
    //!   The final result is
    //!        _                                            _
    //!   Z = | (P3 + P4) + (P5 - P1)   P0 + P1              |
    //!       | P2 + P3                 (P0 + P4) - (P2 + P6)|
    //!        -                                            -
    //! 7*mul, 18*add
    //! http://mathworld.wolfram.com/StrassenFormulas.html
    //! http://en.wikipedia.org/wiki/Strassen_algorithm
    //-----------------------------------------------------------------------------
    TReturn matmul_gemm_seq_strassen(
        TIdx const m, TIdx const n, TIdx const k,
        TElem const alpha,
        TElem const * const MATMUL_RESTRICT A, TIdx const lda,
        TElem const * const MATMUL_RESTRICT B, TIdx const ldb,
        TElem const beta,
        TElem * const MATMUL_RESTRICT C, TIdx const ldc);
    #ifdef __cplusplus
        }
    #endif
#endif
