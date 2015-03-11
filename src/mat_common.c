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

#include "mat_common.h"

#include <assert.h>		// assert
#include <math.h>		// fabs
#include <string.h>		// memcpy
#include <stdio.h>		// printf

#include <float.h>

//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
bool mat_cmp(
	size_t const n,
	TElement const * const restrict X,
	TElement const * const restrict Y)
{
	// The maximum number of error values being printed.
	static size_t const uiMaxErrorsPrint = 5;

	// The threshold difference from where the value is considered to be a real error.
	TElement const fErrorThreshold = (TElement)(MATMUL_EPSILON * pow((TElement)n * (TElement)2.0, (TElement)1.6));
	//printf("fErrorThreshold %32.28lf: ", fErrorThreshold);

	size_t uiNumPrintedErrors = 0;

	// Loop through all values, print out errors and get the maximum error.
	TElement fMaxError = (TElement)0.0;
	for(size_t i = 0; i < n; ++i)
	{
		for(size_t j = 0; j < n; ++j)
		{
			size_t const uiIndex = i*n+j;
			TElement const fError = (TElement)fabs(X[uiIndex] - Y[uiIndex]);
			if(fError>fErrorThreshold && uiNumPrintedErrors<uiMaxErrorsPrint)
			{
				if(uiNumPrintedErrors==0){printf("\n");}
				printf("Error in Cell [%"PRINTF_SIZE_T"][%"PRINTF_SIZE_T"] of %16.16lf X: %f Y: %f\n", i, j, fError, X[uiIndex], Y[uiIndex]);
				++uiNumPrintedErrors;
			}

			fMaxError = (fMaxError<fError) ? fError : fMaxError;
		}
	}
	if(uiNumPrintedErrors>=uiMaxErrorsPrint) 
	{ 
		printf("... %"PRINTF_SIZE_T" more errors in the matrix.\n", uiNumPrintedErrors);
	}

	// Print the maximum error.
	printf("fMaxError %32.28lf: ", fMaxError);

	return true;
}

//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
void mat_print(
	size_t const n,
	TElement const * const restrict A)
{
	printf("{");
	for(size_t i = 0; i < n; ++i)
	{
		if(i>0)
		{
			printf(",\n");
		}
		printf("{");
		for(size_t j = 0; j < n; ++j)
		{
			if(j>0)
			{
				printf(",");
			}
			printf("%f", A[i*n+j]);
		}
		printf("}");
	}
	printf("}");
}

//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
void mat_copy_block(
	size_t const b,
	TElement const * const restrict pSrcMat,
	size_t const sn,
	size_t const sr,
	size_t const sc,
	TElement * const restrict pDstMat,
	size_t const dn,
	size_t const dr,
	size_t const dc)
{
	// The start indices for the copy.
	TElement const * pSrcBlock = pSrcMat + sn * sr + sc;
	TElement * pDstBlock = pDstMat + dn * dr + dc;

	// Copy line by line.
	for(size_t i = 0; i < b; ++i)
	{
		memcpy(pDstBlock, pSrcBlock, sizeof(TElement)*b);
		// Add the pitch -> next line start index.
		pSrcBlock += sn;
		pDstBlock += dn;
	}
}

//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
void mat_copy(
	TElement const * const restrict pSrcMat,
	TElement * const restrict pDstMat,
	size_t const n)
{
	mat_copy_block(
		n,
		pSrcMat,
		n,
		0,
		0,
		pDstMat,
		n,
		0,
		0);
}

//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
void mat_get_block(
	TElement const * const restrict pSrcMat,
	size_t const n,
	size_t const uiBlockIndexHorizontal,
	size_t const uiBlockIndexVertical,
	TElement * const restrict pDstBlock,
	size_t const b)
{
	size_t const uiBlockOffsetHorizontal = uiBlockIndexHorizontal * b;
	size_t const uiBlockOffsetVertical = uiBlockIndexVertical * b;

	// Reorder the block of the input so that it is laying linearly in memory.
	for(size_t i = 0; i < b; ++i)
	{
		size_t const uiOffsetVerticalLocal = i*b;
		size_t const uiOffsetVerticalGlobal = (uiBlockOffsetVertical + i) * n;
		size_t const uiOffsetBlockRowGlobal = uiOffsetVerticalGlobal + uiBlockOffsetHorizontal;
		for(size_t j = 0; j < b; ++j)
		{
			size_t const uiOffsetLocal = uiOffsetVerticalLocal + j;
			size_t const uiOffsetGlobal = uiOffsetBlockRowGlobal + j;

			pDstBlock[uiOffsetLocal] = pSrcMat[uiOffsetGlobal];
		}
	}
}

//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
void mat_set_block(
	TElement const * const restrict pSrcBlock,
	size_t const b,
	TElement * const restrict pDstMat,
	size_t const n,
	size_t const uiBlockIndexHorizontal,
	size_t const uiBlockIndexVertical)
{
	size_t const uiBlockOffsetHorizontal = uiBlockIndexHorizontal * b;
	size_t const uiBlockOffsetVertical = uiBlockIndexVertical * b;

	// Reorder the block of the input so that it is laying linearly in memory.
	for(size_t i = 0; i < b; ++i)
	{
		size_t const uiOffsetVerticalLocal = i*b;
		size_t const uiOffsetVerticalGlobal = (uiBlockOffsetVertical + i) * n;
		size_t const uiOffsetBlockRowGlobal = uiOffsetVerticalGlobal + uiBlockOffsetHorizontal;
		for(size_t j = 0; j < b; ++j)
		{
			size_t const uiOffsetLocal = uiOffsetVerticalLocal + j;
			size_t const uiOffsetGlobal = uiOffsetBlockRowGlobal + j;

			pDstMat[uiOffsetGlobal] = pSrcBlock[uiOffsetLocal];
		}
	}
}

//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
/*void mat_get_block(
	TElement const * const restrict pSrcMat,
	size_t const n,
	size_t const uiBlockIndexHorizontal,
	size_t const uiBlockIndexVertical,
	TElement * const restrict pDstBlock,
	size_t const b)
{
	size_t const uiBlockOffsetHorizontal = uiBlockIndexHorizontal * b;
	size_t const uiBlockOffsetVertical = uiBlockIndexVertical * b;

	mat_copy_block(
		b, 
		pSrcMat, n, uiBlockOffsetVertical, uiBlockIndexHorizontal,
		pDstBlock, b, 0, 0
		);
}

//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
void mat_set_block(
	TElement const * const restrict pSrcBlock,
	size_t const b,
	TElement * const restrict pDstMat,
	size_t const n,
	size_t const uiBlockIndexHorizontal,
	size_t const uiBlockIndexVertical)
{
	size_t const uiBlockOffsetHorizontal = uiBlockIndexHorizontal * b;
	size_t const uiBlockOffsetVertical = uiBlockIndexVertical * b;

	mat_copy_block(
		b,
		pSrcBlock, b, 0, 0,
		pDstMat, n, uiBlockOffsetVertical, uiBlockIndexHorizontal
		);
}*/

//-----------------------------------------------------------------------------
// 
//-----------------------------------------------------------------------------
void mat_row_major_to_mat_x_block_major(
	TElement const * const restrict pSrcMat,
	size_t const n,
	TElement * restrict pBlockMajorMat,
	size_t const b,
	bool const bColumnFirst)
{
	assert(n % b == 0);

	size_t const q = n / b;
	if(bColumnFirst)
	{
		for(size_t j = 0; j < q; ++j)
		{
			for(size_t i = 0; i < q; ++i)
			{
				mat_copy_block(b, pSrcMat, n, i * b, j * b, pBlockMajorMat, b, 0, 0);
				pBlockMajorMat += b*b;
			}
		}
	}
	else
	{
		for(size_t i = 0; i < q; ++i)
		{
			for(size_t j = 0; j < q; ++j)
			{
				mat_copy_block(b, pSrcMat, n, i * b, j * b, pBlockMajorMat, b, 0, 0);
				pBlockMajorMat += b*b;
			}
		}
	}
}
//-----------------------------------------------------------------------------
// 
//-----------------------------------------------------------------------------
void mat_x_block_major_to_mat_row_major(
	TElement const * restrict pBlockMajorMat,
	size_t const b,
	TElement * const restrict pDstMat,
	size_t const n,
	bool const bColumnFirst)
{
	assert(n % b == 0);

	size_t const q = n / b;
	if(bColumnFirst)
	{
		for(size_t j = 0; j < q; j++)
		{
			for(size_t i = 0; i < q; i++)
			{
				mat_copy_block(b, pBlockMajorMat, b, 0, 0, pDstMat, n, i * b, j * b);
				pBlockMajorMat += b*b;
			}
		}
	}
	else
	{
		for(size_t i = 0; i < q; i++)
		{
			for(size_t j = 0; j < q; j++)
			{
				mat_copy_block(b, pBlockMajorMat, b, 0, 0, pDstMat, n, i * b, j * b);
				pBlockMajorMat += b*b;
			}
		}
	}
}
