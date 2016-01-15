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

#include <matmul/seq/Basic.h>
#include <matmul/seq/SingleOpts.h>
#include <matmul/seq/MultipleOpts.h>
#include <matmul/seq/Strassen.h>
#include <matmul/par/Alpaka.h>
#include <matmul/par/AlpakaOmpNative.h>
#include <matmul/par/AlpakaTiling.h>
#include <matmul/par/BlasCublas.h>
#include <matmul/par/BlasMkl.h>
#include <matmul/par/Cuda.h>
#include <matmul/par/MpiCannon.h>
#include <matmul/par/MpiDns.h>
#include <matmul/par/OpenAcc.h>
#include <matmul/par/Omp.h>
#include <matmul/par/StrassenOmp2.h>
#include <matmul/par/PhiOffOmp.h>
#include <matmul/par/PhiOffBlasMkl.h>

#include <matmul/common/Alloc.h>
#include <matmul/common/Array.h>
#include <matmul/common/Config.h>
#include <matmul/common/Mat.h>
#include <matmul/common/Time.h>
