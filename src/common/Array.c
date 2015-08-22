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
    TSize const elemCount,
    TElem const val)
{
    assert(pArray);

    for(TSize i = 0; i<elemCount; ++i)
    {
        pArray[i] = val;
    }
}
//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
void matmul_arr_fill_zero(
    TElem * const pArray,
    TSize const elemCount)
{
    matmul_arr_fill_val(
        pArray,
        elemCount,
        (TElem)0);
}
//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
void matmul_arr_fill_idx(
    TElem * const pArray,
    TSize const elemCount)
{
    assert(pArray);

    for(TSize i = 0; i<elemCount; ++i)
    {
        pArray[i] = (TElem)i;
    }
}
//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
void matmul_arr_fill_rand(
    TElem * const pArray,
    TSize const elemCount,
    TElem const min,
    TElem const max)
{
    assert(pArray);

    for(TSize i = 0; i<elemCount; ++i)
    {
        pArray[i] = matmul_gen_rand_val(min, max);
    }
}

//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
TElem * matmul_arr_alloc_fill_val(
    TSize const elemCount,
    TElem const val)
{
    TElem * arr = matmul_arr_alloc(elemCount);

    matmul_arr_fill_val(arr, elemCount, val);

    return arr;
}
//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
TElem * matmul_arr_alloc_fill_zero(
    TSize const elemCount)
{
    TElem * arr = matmul_arr_alloc(elemCount);

    matmul_arr_fill_zero(arr, elemCount);

    return arr;
}
//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
TElem * matmul_arr_alloc_fill_idx(
    TSize const elemCount)
{
    TElem * arr = matmul_arr_alloc(elemCount);

    matmul_arr_fill_idx(arr, elemCount);

    return arr;
}
//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
TElem * matmul_arr_alloc_fill_rand(
    TSize const elemCount,
    TElem const min,
    TElem const max)
{
    TElem * arr = matmul_arr_alloc(elemCount);

    matmul_arr_fill_rand(arr, elemCount, min, max);

    return arr;
}
