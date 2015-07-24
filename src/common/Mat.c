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

#include <matmul/common/Mat.h>

#include <assert.h>             // assert
#include <math.h>               // fabs
#include <string.h>             // memcpy
#include <stdio.h>              // printf

#include <float.h>

//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
bool matmul_mat_cmp(
    TIdx const m, TIdx const n,
    TElem const * const MATMUL_RESTRICT A, TIdx const lda,
    TElem const * const MATMUL_RESTRICT B, TIdx const ldb,
    TElem const fErrorThreshold)
{
    // The maximum number of error values being printed.
    static TIdx const uiMaxErrorsPrint = 100;

    TIdx uiNumErrors = 0;

    // Loop through all values, print out errors and get the maximum error.
    TElem fMaxError = (TElem)0.0;
    for(TIdx i = 0; i < m; ++i)
    {
        for(TIdx j = 0; j < n; ++j)
        {
            TIdx const uiIdxA = i*lda+j;
            TIdx const uiIdxB = i*ldb+j;
            TElem const fError = (TElem)fabs(A[uiIdxA] - B[uiIdxB]);
            if(fError > fErrorThreshold)
            {
                if(uiNumErrors < uiMaxErrorsPrint)
                {
                    printf("\nError in Cell [%"MATMUL_PRINTF_SIZE_T"][%"MATMUL_PRINTF_SIZE_T"] of %16.16lf A: %f B: %f", (size_t)i, (size_t)j, fError, A[uiIdxA], B[uiIdxB]);
                }
                ++uiNumErrors;
            }

            fMaxError = (fMaxError<fError) ? fError : fMaxError;
        }
    }
    // Print the number of errors not printed.
    if(uiNumErrors > uiMaxErrorsPrint)
    {
        printf("\n... %"MATMUL_PRINTF_SIZE_T" more errors in the matrix.", (size_t)uiNumErrors-uiMaxErrorsPrint);
    }
    // Print the maximum error.
    if(fMaxError > fErrorThreshold)
    {
        printf("\nfMaxDiff=%32.28lf", fMaxError);
    }
    // If something has been printed, add a newline.
    if(uiNumErrors > 0)
    {
        printf("\n");
    }

    return (uiNumErrors==0);
}

//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
void matmul_mat_print(
    TIdx const m, TIdx const n,
    TElem const * const MATMUL_RESTRICT A, TIdx const lda,
    char * const pElemSeperator, char * const pRowSeperator,
    char * const pDimBegin, char * const pDimEnd)
{
    printf("%s", pDimBegin);
    for(TIdx i = 0; i < m; ++i)
    {
        if(i>0)
        {
            printf("%s", pRowSeperator);
        }
        printf("%s", pDimBegin);
        for(TIdx j = 0; j < n; ++j)
        {
            if(j>0)
            {
                printf("%s", pElemSeperator);
            }
            printf("%f", A[i*lda+j]);
        }
        printf("%s", pDimEnd);
    }
    printf("%s", pDimEnd);
}
//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
void matmul_mat_print_simple(
    TIdx const m, TIdx const n,
    TElem const * const MATMUL_RESTRICT A, TIdx const lda)
{
    matmul_mat_print(
        m, n,
        A, lda,
        ",", "\n",
        "", "");
}
//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
void matmul_mat_print_mathematica(
    TIdx const m, TIdx const n,
    TElem const * const MATMUL_RESTRICT A, TIdx const lda)
{
    matmul_mat_print(
        m, n,
        A, lda,
        ",", ",\n",
        "{", "}");
}

//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
bool matmul_mat_gemm_early_out(
    TIdx const m, TIdx const n, TIdx const k,
    TElem const alpha,
    TElem const beta)
{
    // Early out if nothing has to be computed.
    if((m == 0) || (n == 0)
        || (((alpha == (TElem)0) || (k == 0)) && (beta == (TElem)1)))
    {
        return true;
    }
    else
    {
        return false;
    }
}

//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
void matmul_mat_copy_block(
    TIdx const m,
    TIdx const n,
    TElem const * const MATMUL_RESTRICT pSrcMat, TIdx const lds,
    TIdx const sr, TIdx const sc,
    TElem * const MATMUL_RESTRICT pDstMat, TIdx const ldd,
    TIdx const dr, TIdx const dc)
{
    // The start indices for the copy.
    TElem const * pSrcBlock = pSrcMat + lds * sr + sc;
    TElem * pDstBlock = pDstMat + ldd * dr + dc;

    // Copy line by line.
    for(TIdx i = 0; i < m; ++i)
    {
        memcpy(pDstBlock, pSrcBlock, sizeof(TElem)*n);
        // Add the pitch -> next line start index.
        pSrcBlock += lds;
        pDstBlock += ldd;
    }
}

//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
void matmul_mat_copy(
    TIdx const m,
    TIdx const n,
    TElem const * const MATMUL_RESTRICT pSrcMat, TIdx const lds,
    TElem * const MATMUL_RESTRICT pDstMat, TIdx const ldd)
{
    matmul_mat_copy_block(
        m,
        n,
        pSrcMat,
        lds,
        0,
        0,
        pDstMat,
        ldd,
        0,
        0);
}

//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
void matmul_mat_row_major_to_mat_x_block_major(
    TElem const * const MATMUL_RESTRICT pSrcMat, TIdx const m, TIdx const n, TIdx const lds,
    TElem * MATMUL_RESTRICT pBlockMajorMat, TIdx const b,
    bool const bColumnFirst)
{
    assert(n == m);
    assert(n % b == 0);

    TIdx const q = n / b;
    if(bColumnFirst)
    {
        for(TIdx j = 0; j < q; ++j)
        {
            for(TIdx i = 0; i < q; ++i)
            {
                matmul_mat_copy_block(b, b, pSrcMat, lds, i * b, j * b, pBlockMajorMat, b, 0, 0);
                pBlockMajorMat += b*b;
            }
        }
    }
    else
    {
        for(TIdx i = 0; i < q; ++i)
        {
            for(TIdx j = 0; j < q; ++j)
            {
                matmul_mat_copy_block(b, b, pSrcMat, lds, i * b, j * b, pBlockMajorMat, b, 0, 0);
                pBlockMajorMat += b*b;
            }
        }
    }
}

//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
void matmul_mat_x_block_major_to_mat_row_major(
    TElem const * MATMUL_RESTRICT pBlockMajorMat, TIdx const b,
    TElem * const MATMUL_RESTRICT pDstMat, TIdx const m, TIdx const n, TIdx const ldd,
    bool const bColumnFirst)
{
    assert(n == m);
    assert(n % b == 0);

    TIdx const q = n / b;
    if(bColumnFirst)
    {
        for(TIdx j = 0; j < q; j++)
        {
            for(TIdx i = 0; i < q; i++)
            {
                matmul_mat_copy_block(b, b, pBlockMajorMat, b, 0, 0, pDstMat, ldd, i * b, j * b);
                pBlockMajorMat += b*b;
            }
        }
    }
    else
    {
        for(TIdx i = 0; i < q; i++)
        {
            for(TIdx j = 0; j < q; j++)
            {
                matmul_mat_copy_block(b, b, pBlockMajorMat, b, 0, 0, pDstMat, ldd, i * b, j * b);
                pBlockMajorMat += b*b;
            }
        }
    }
}

//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
void matmul_mat_get_block(
    TElem const * const MATMUL_RESTRICT pSrcMat, TIdx const lds,
    TIdx const uiBlockIdxHorizontal,
    TIdx const uiBlockIdxVertical,
    TElem * const MATMUL_RESTRICT pDstBlock, TIdx const b)
{
    TIdx const uiBlockOffsetHorizontal = uiBlockIdxHorizontal * b;
    TIdx const uiBlockOffsetVertical = uiBlockIdxVertical * b;

    // Reorder the block of the input so that it is laying linearly in memory.
    for(TIdx i = 0; i < b; ++i)
    {
        TIdx const uiOffsetVerticalLocal = i*b;
        TIdx const uiOffsetVerticalGlobal = (uiBlockOffsetVertical + i) * lds;
        TIdx const uiOffsetBlockRowGlobal = uiOffsetVerticalGlobal + uiBlockOffsetHorizontal;
        for(TIdx j = 0; j < b; ++j)
        {
            TIdx const uiOffsetLocal = uiOffsetVerticalLocal + j;
            TIdx const uiOffsetGlobal = uiOffsetBlockRowGlobal + j;

            pDstBlock[uiOffsetLocal] = pSrcMat[uiOffsetGlobal];
        }
    }
}

//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
void matmul_mat_set_block(
    TElem const * const MATMUL_RESTRICT pSrcBlock, TIdx const b,
    TElem * const MATMUL_RESTRICT pDstMat, TIdx const ldd,
    TIdx const uiBlockIdxHorizontal,
    TIdx const uiBlockIdxVertical)
{
    TIdx const uiBlockOffsetHorizontal = uiBlockIdxHorizontal * b;
    TIdx const uiBlockOffsetVertical = uiBlockIdxVertical * b;

    // Reorder the block of the input so that it is laying linearly in memory.
    for(TIdx i = 0; i < b; ++i)
    {
        TIdx const uiOffsetVerticalLocal = i*b;
        TIdx const uiOffsetVerticalGlobal = (uiBlockOffsetVertical + i) * ldd;
        TIdx const uiOffsetBlockRowGlobal = uiOffsetVerticalGlobal + uiBlockOffsetHorizontal;
        for(TIdx j = 0; j < b; ++j)
        {
            TIdx const uiOffsetLocal = uiOffsetVerticalLocal + j;
            TIdx const uiOffsetGlobal = uiOffsetBlockRowGlobal + j;

            pDstMat[uiOffsetGlobal] = pSrcBlock[uiOffsetLocal];
        }
    }
}
