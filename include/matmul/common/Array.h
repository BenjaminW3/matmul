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
