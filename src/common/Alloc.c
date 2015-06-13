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

#include <matmul/common/Alloc.h>

#include <stdlib.h>        // malloc, free

#if defined MATMUL_ALIGNED_MALLOC
    #ifndef _MSC_VER
        #include <malloc.h>    // memalign or valloc are not always declared in stdlib.h.
    #endif
#endif

//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
void * matmul_arr_aligned_alloc_internal(
    size_t const uiNumBytes)
{
#if defined MATMUL_ALIGNED_MALLOC
    // If c11 were supported we could use void *aligned_alloc(size_t alignment, size_t size);
#if defined _MSC_VER
    return _aligned_malloc(uiNumBytes, 64);
#elif defined __linux__
    return memalign(64, uiNumBytes);
#elif defined __MACH__      // Mac OS X
    return malloc(uiNumBytes);    // malloc is always 16 byte aligned on Mac.
#else
    return valloc(uiNumBytes);    // other (use valloc for page-aligned memory)
#endif
#else
    return malloc(uiNumBytes);
#endif
}

//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
TElem * matmul_arr_alloc(
    size_t const uiNumElements)
{
    size_t const uiNumBytes = uiNumElements * sizeof(TElem);

    return (TElem*) matmul_arr_aligned_alloc_internal(uiNumBytes);
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
