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

#include <matmul/common/Alloc.h>

#include <stdlib.h>             // malloc, free

#if defined MATMUL_ALIGNED_MALLOC
    #ifndef _MSC_VER
        #include <malloc.h>     // memalign or valloc are not always declared in stdlib.h.
    #endif
#endif

//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
void * matmul_arr_aligned_alloc_internal(
    TIdx const numBytes)
{
#if defined MATMUL_ALIGNED_MALLOC
    // If c11 were supported we could use void *aligned_alloc(size_t alignment, size_t size);
#if defined _MSC_VER
    return _aligned_malloc(numBytes, 64);
#elif defined __linux__
    return memalign(64, numBytes);
#elif defined __MACH__      // Mac OS X
    return malloc(numBytes);    // malloc is always 16 byte aligned on Mac.
#else
    return valloc(numBytes);    // other (use valloc for page-aligned memory)
#endif
#else
    return malloc(numBytes);
#endif
}

//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
TElem * matmul_arr_alloc(
    TIdx const elemCount)
{
    TIdx const numBytes = elemCount * sizeof(TElem);

    return (TElem*) matmul_arr_aligned_alloc_internal(numBytes);
}

//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
void matmul_arr_aligned_free_internal(
    void * const MATMUL_RESTRICT ptr)
{
#if defined MATMUL_ALIGNED_MALLOC
#if defined _MSC_VER
    _aligned_free(ptr);
#elif defined __linux__
    free(ptr);
#elif defined __MACH__
    free(ptr);
#else
    free(ptr);
#endif
#else
    free(ptr);
#endif
}

//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
void matmul_arr_free(
    TElem * const MATMUL_RESTRICT ptr)
{
    matmul_arr_aligned_free_internal((void*)ptr);
}
