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

#include <matmul/common/Array.h>

#include <matmul/common/Alloc.h>

#include <assert.h>
#include <stdlib.h>        // RAND_MAX, srand

//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
TElem matmul_gen_rand_val(
    TElem const min,
    TElem const max)
{
    assert(min <= max); // bad input

    return ((TElem)rand()/(TElem)(RAND_MAX)) * (max-min) + min;
}

//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
void matmul_arr_fill_val(
    TElem * const pArray,
    TIdx const uiNumElements,
    TElem const val)
{
    assert(pArray);

    for(TIdx i = 0; i<uiNumElements; ++i)
    {
        pArray[i] = val;
    }
}
//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
void matmul_arr_fill_zero(
    TElem * const pArray,
    TIdx const uiNumElements)
{
    matmul_arr_fill_val(
        pArray,
        uiNumElements,
        (TElem)0);
}
//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
void matmul_arr_fill_idx(
    TElem * const pArray,
    TIdx const uiNumElements)
{
    assert(pArray);

    for(TIdx i = 0; i<uiNumElements; ++i)
    {
        pArray[i] = (TElem)i;
    }
}
//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
void matmul_arr_fill_rand(
    TElem * const pArray,
    TIdx const uiNumElements,
    TElem const min,
    TElem const max)
{
    assert(pArray);

    for(TIdx i = 0; i<uiNumElements; ++i)
    {
        pArray[i] = matmul_gen_rand_val(min, max);
    }
}

//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
TElem * matmul_arr_alloc_fill_val(
    TIdx const uiNumElements,
    TElem const val)
{
    TElem * arr = matmul_arr_alloc(uiNumElements);

    matmul_arr_fill_val(arr, uiNumElements, val);

    return arr;
}
//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
TElem * matmul_arr_alloc_fill_zero(
    TIdx const uiNumElements)
{
    TElem * arr = matmul_arr_alloc(uiNumElements);

    matmul_arr_fill_zero(arr, uiNumElements);

    return arr;
}
//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
TElem * matmul_arr_alloc_fill_idx(
    TIdx const uiNumElements)
{
    TElem * arr = matmul_arr_alloc(uiNumElements);

    matmul_arr_fill_idx(arr, uiNumElements);

    return arr;
}
//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
TElem * matmul_arr_alloc_fill_rand(
    TIdx const uiNumElements,
    TElem const min,
    TElem const max)
{
    TElem * arr = matmul_arr_alloc(uiNumElements);

    matmul_arr_fill_rand(arr, uiNumElements, min, max);

    return arr;
}
