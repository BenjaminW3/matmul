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
    TElem const fMin,
    TElem const fMax)
{
    assert(fMin < fMax); // bad input

    return ((TElem)rand()/(TElem)(RAND_MAX)) * (fMax-fMin) - fMin;
}

//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
void matmul_arr_zero_fill(
    TElem * pArray,
    size_t const uiNumElements)
{
    assert(pArray);

    for(size_t i = 0; i<uiNumElements; ++i)
    {
        pArray[i] = (TElem)0;
    }
}

//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
void matmul_arr_rand_fill(
    TElem * pArray,
    size_t const uiNumElements)
{
    assert(pArray);

    for(size_t i = 0; i<uiNumElements; ++i)
    {
        pArray[i] = matmul_gen_rand_val((TElem)0, (TElem)1);
    }
}

//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
TElem * matmul_arr_alloc_zero_fill(
    size_t const uiNumElements)
{
    TElem * arr = matmul_arr_alloc(uiNumElements);

    matmul_arr_zero_fill(arr, uiNumElements);

    return arr;
}
//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
TElem * matmul_arr_alloc_rand_fill(
    size_t const uiNumElements)
{
    TElem * arr = matmul_arr_alloc(uiNumElements);

    matmul_arr_rand_fill(arr, uiNumElements);

    return arr;
}
