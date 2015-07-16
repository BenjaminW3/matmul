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

#include <matmul/common/Config.h>   // TElem, TIdx

#ifdef __cplusplus
    extern "C"
    {
#endif
//-----------------------------------------------------------------------------
//! \return A random value in the range [fMin, fMax].
//-----------------------------------------------------------------------------
TElem matmul_gen_rand_val(
    TElem const min,
    TElem const max);

//-----------------------------------------------------------------------------
//! Fills the array with the given value.
//! \param pArray The array.
//! \param uiNumElements The number of elements in the array.
//-----------------------------------------------------------------------------
void matmul_arr_fill_val(
    TElem * const pArray,
    TIdx const uiNumElements,
    TElem const val);
//-----------------------------------------------------------------------------
//! Fills the array with zeros.
//! \param pArray The array.
//! \param uiNumElements The number of elements in the array.
//-----------------------------------------------------------------------------
void matmul_arr_fill_zero(
    TElem * const pArray,
    TIdx const uiNumElements);
//-----------------------------------------------------------------------------
//! Fills the array with the indices as values.
//! \param pArray The array.
//! \param uiNumElements The number of elements in the array.
//-----------------------------------------------------------------------------
void matmul_arr_fill_idx(
    TElem * const pArray,
    TIdx const uiNumElements);
//-----------------------------------------------------------------------------
//! Fills the array with random numbers.
//! \param pArray The array.
//! \param uiNumElements The number of elements in the array.
//-----------------------------------------------------------------------------
void matmul_arr_fill_rand(
    TElem * const pArray,
    TIdx const uiNumElements,
    TElem const min,
    TElem const max);

//-----------------------------------------------------------------------------
//! \return A array of the given type initialized with the given value.
//! \param uiNumElements The number of elements in the array.
//-----------------------------------------------------------------------------
TElem * matmul_arr_alloc_fill_val(
    TIdx const uiNumElements,
    TElem const val);
//-----------------------------------------------------------------------------
//! \return A array of the given type initialized with zero.
//! \param uiNumElements The number of elements in the array.
//-----------------------------------------------------------------------------
TElem * matmul_arr_alloc_fill_zero(
    TIdx const uiNumElements);
//-----------------------------------------------------------------------------
//! \return A array of the given type initialized with the indices as values.
//! \param uiNumElements The number of elements in the array.
//-----------------------------------------------------------------------------
TElem * matmul_arr_alloc_fill_idx(
    TIdx const uiNumElements);
//-----------------------------------------------------------------------------
//! \return A array of random values of the given type.
//! \param uiNumElements The number of elements in the array.
//-----------------------------------------------------------------------------
TElem * matmul_arr_alloc_fill_rand(
    TIdx const uiNumElements,
    TElem const min,
    TElem const max);
#ifdef __cplusplus
    }
#endif
