#pragma once

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

//-----------------------------------------------------------------------------
// Compiler Settings.
//-----------------------------------------------------------------------------
#if defined __INTEL_COMPILER                    // ICC additionally defines _MSC_VER if used in VS so this has to come first
    #define MATMUL_ICC
    #define MATMUL_RESTRICT restrict
    #if defined(_MSC_VER) && _MSC_VER<=1800
        #define MATMUL_PRINTF_SIZE_T "Iu"
    #else
        #define MATMUL_PRINTF_SIZE_T "zu"
    #endif

#elif defined __clang__
    #define MATMUL_CLANG
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
    #define MATMUL_GCC
    #ifdef __cplusplus
        #define MATMUL_RESTRICT __restrict__
    #else
        #define MATMUL_RESTRICT restrict
    #endif
    #define MATMUL_PRINTF_SIZE_T "zu"

#elif defined _MSC_VER
    #define MATMUL_MSVC
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