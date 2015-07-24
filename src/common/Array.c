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
