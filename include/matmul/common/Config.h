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

//-----------------------------------------------------------------------------
// If MPI tests are to be build, set some additional definitions.
//-----------------------------------------------------------------------------
#if defined(MATMUL_BUILD_PAR_MPI_CANNON_STD) || defined(MATMUL_BUILD_PAR_MPI_CANNON_MKL) || defined(MATMUL_BUILD_PAR_MPI_CANNON_CUBLAS) || defined(MATMUL_BUILD_PAR_MPI_DNS)
    #define MATMUL_MPI
#endif

#ifdef MATMUL_MPI
    #include <mpi.h>
    #define MATMUL_MPI_COMM MPI_COMM_WORLD
    #define MATMUL_MPI_ROOT 0
#endif

//-----------------------------------------------------------------------------
// Return type.
//-----------------------------------------------------------------------------
#ifdef MATMUL_RETURN_COMPUTATION_TIME
    #include <matmul/common/Time.h>
    typedef double TReturn;
    #define MATMUL_TIME_START double const matmulTimeStart = getTimeSec()
    #define MATMUL_TIME_END double const matmulTimeEnd = getTimeSec(); double const matmulTimeDiff = matmulTimeEnd - matmulTimeStart
    #define MATMUL_TIME_STORE double const matmulTimeDiff =
    #define MATMUL_TIME_RETURN return matmulTimeDiff
    #define MATMUL_TIME_RETURN_EARLY_OUT return 0.0
#else
    typedef void TReturn;
    #define MATMUL_TIME_START
    #define MATMUL_TIME_END
    #define MATMUL_TIME_STORE
    #define MATMUL_TIME_RETURN
    #define MATMUL_TIME_RETURN_EARLY_OUT return;
#endif

//-----------------------------------------------------------------------------
// Data type depending definitions.
//-----------------------------------------------------------------------------
#ifdef MATMUL_ELEMENT_TYPE_DOUBLE
    typedef double TElem;
    #define MATMUL_EPSILON DBL_EPSILON    //!< This is used to calculate whether a result value is within a matrix size dependent error range.
    #ifdef MATMUL_MPI
        #define MATMUL_MPI_ELEMENT_TYPE MPI_DOUBLE
    #endif
#else
    typedef float TElem;
    #define MATMUL_EPSILON FLT_EPSILON    //!< This is used to calculate whether a result value is within a matrix size dependent error range.
    #ifdef MATMUL_MPI
        #define MATMUL_MPI_ELEMENT_TYPE MPI_FLOAT
    #endif
#endif

#include <stddef.h>                 // size_t
#include <stdint.h>                 // int32_t

typedef MATMUL_INDEX_TYPE TIdx;

//-----------------------------------------------------------------------------
// Compiler Settings.
//-----------------------------------------------------------------------------
#if defined __INTEL_COMPILER                    // ICC additionally defines _MSC_VER if used in VS so this has to come first
    #ifdef __cplusplus
        #define MATMUL_RESTRICT __restrict
    #else
        #define MATMUL_RESTRICT restrict
    #endif
    #if defined(_MSC_VER) && _MSC_VER<=1800
        #define MATMUL_PRINTF_SIZE_T "Iu"
    #else
        #define MATMUL_PRINTF_SIZE_T "zu"
    #endif

#elif defined __clang__
    #ifdef __cplusplus
        #define MATMUL_RESTRICT __restrict__
    #else
        #define MATMUL_RESTRICT restrict
    #endif
    #if defined(_MSC_VER) && _MSC_VER<=1800
        #define MATMUL_PRINTF_SIZE_T "Iu"
    #else
        #define MATMUL_PRINTF_SIZE_T "zu"
    #endif

#elif defined __GNUC__
    #ifdef __cplusplus
        #define MATMUL_RESTRICT __restrict__
    #else
        #define MATMUL_RESTRICT restrict
    #endif
    #define MATMUL_PRINTF_SIZE_T "zu"

#elif defined _MSC_VER
    #define MATMUL_RESTRICT __restrict          // Visual C++ 2013 and below do not define C99 restrict keyword under its supposed name. (And its not fully standard conformant)
    #if _MSC_VER<=1800
        #define MATMUL_PRINTF_SIZE_T "Iu"       // Visual C++ 2013 and below do not support C99 printf specifiers.
    #else
        #define MATMUL_PRINTF_SIZE_T "zu"
    #endif

#elif defined __CUDACC__
    #define MATMUL_RESTRICT __restrict__

#elif defined __PGI
    #define MATMUL_RESTRICT restrict
    #define MATMUL_PRINTF_SIZE_T "zu"

#else
    #define MATMUL_RESTRICT restrict
    #define MATMUL_PRINTF_SIZE_T "zu"
#endif

//-----------------------------------------------------------------------------
//! The no debug level.
//-----------------------------------------------------------------------------
#define MATMUL_DEBUG_DISABLED 0
//-----------------------------------------------------------------------------
//! The minimal debug level.
//-----------------------------------------------------------------------------
#define MATMUL_DEBUG_MINIMAL 1
//-----------------------------------------------------------------------------
//! The full debug level.
//-----------------------------------------------------------------------------
#define MATMUL_DEBUG_FULL 2