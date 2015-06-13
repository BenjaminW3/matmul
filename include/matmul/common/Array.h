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

#include <matmul/common/Config.h>

#include <stddef.h>    // size_t

#ifdef __cplusplus
    extern "C"
    {
#endif
//-----------------------------------------------------------------------------
//! \return A random value in the range [fMin, fMax].
//-----------------------------------------------------------------------------
TElem matmul_gen_rand_val(
    TElem const fMin,
    TElem const fMax);

//-----------------------------------------------------------------------------
//! Fills the array with zeros.
//! \param pArray The array.
//! \param uiNumElements The number of elements in the matrix.
//-----------------------------------------------------------------------------
void matmul_arr_zero_fill(
    TElem * pArray,
    size_t const uiNumElements);

//-----------------------------------------------------------------------------
//! Fills the array with random numbers.
//! \param pArray The array.
//! \param uiNumElements The number of elements in the matrix.
//-----------------------------------------------------------------------------
void matmul_arr_rand_fill(
    TElem * pArray,
    size_t const uiNumElements);

//-----------------------------------------------------------------------------
//! \return A array of the given type initialized with zero.
//! \param uiNumElements The number of elements in the array.
//-----------------------------------------------------------------------------
TElem * matmul_arr_alloc_zero_fill(
    size_t const uiNumElements);

//-----------------------------------------------------------------------------
//! \return A array of random values of the given type.
//! \param uiNumElements The number of elements in the array.
//-----------------------------------------------------------------------------
TElem * matmul_arr_alloc_rand_fill(
    size_t const uiNumElements);
#ifdef __cplusplus
    }
#endif
