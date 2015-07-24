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

#include <matmul/common/Config.h>   // TElem

#ifdef __cplusplus
    extern "C"
    {
#endif
//-----------------------------------------------------------------------------
//! Tries to allocate the memory on 64 Byte boundary if the operating system allows this.
//! \return A array of random values of the given type.
//! \param uiNumElements The number of elements in the matrix.
//-----------------------------------------------------------------------------
TElem * matmul_arr_alloc(
    TIdx const uiNumBytes);

//-----------------------------------------------------------------------------
//! \return A array of random values of the given type.
//! \param uiNumElements The number of elements in the matrix.
//-----------------------------------------------------------------------------
void matmul_arr_free(
    TElem * const MATMUL_RESTRICT ptr);
#ifdef __cplusplus
    }
#endif
