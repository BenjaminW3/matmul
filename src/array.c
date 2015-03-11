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

#include "array.h"

#include "malloc.h"

#include <assert.h>
#include <stdlib.h>		// RAND_MAX, srand

//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
TElement mat_gen_rand_val(
	TElement const fMin,
	TElement const fMax)
{
	assert(fMin < fMax); // bad input

	return ((TElement)rand()/(TElement)(RAND_MAX)) * (fMax-fMin) - fMin;
}

//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
void mat_zero_fill(TElement * pArray,
	size_t const uiNumElements)
{
	assert(pArray);

	for(size_t i = 0; i<uiNumElements; ++i)
	{
		pArray[i] = 0;
	}
}

//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
void mat_rand_fill(TElement * pArray,
	size_t const uiNumElements)
{
	assert(pArray);

	for(size_t i = 0; i<uiNumElements; ++i)
	{
		pArray[i] = mat_gen_rand_val(0, 1);
	}
}

//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
TElement * mat_alloc_zero_fill(
	size_t const uiNumElements)
{
	TElement * arr = mat_alloc(uiNumElements);

	mat_zero_fill(arr, uiNumElements);

	return arr;
}
//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
TElement * mat_alloc_rand_fill(
	size_t const uiNumElements)
{
	TElement * arr = mat_alloc(uiNumElements);

	mat_rand_fill(arr, uiNumElements);

	return arr;
}
