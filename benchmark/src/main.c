//-----------------------------------------------------------------------------
//! \file
//! Copyright 2013-2015 Benjamin Worpitz, Rene Widera
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

#include <matmul/matmul.h>

#include <stdio.h>                  // printf
#include <assert.h>                 // assert
#include <stdlib.h>                 // malloc, free
#include <stdbool.h>                // bool, true, false
#include <time.h>                   // time()
#include <float.h>                  // DBL_MAX
#include <math.h>                   // pow

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

//-----------------------------------------------------------------------------
//! A struct holding the algorithms to benchmark.
//-----------------------------------------------------------------------------
typedef struct GemmAlgo
{
    TReturn(*pGemm)(TSize const, TSize const, TSize const, TElem const, TElem const * const, TSize const, TElem const * const, TSize const, TElem const, TElem * const, TSize const);
    char const * pszName;
    double const exponentOmega;
} GemmAlgo;

#ifdef MATMUL_BENCHMARK_CUDA_NO_COPY
    #include <cuda_runtime.h>

    #define MATMUL_CUDA_RT_CHECK(cmd) {cudaError_t error = cmd; if(error!=cudaSuccess){printf("<%s>:%i ",__FILE__,__LINE__); printf("[CUDA] Error: %s\n", cudaGetErrorString(error));}}
#endif

//-----------------------------------------------------------------------------
//! \return The time in milliseconds required to multiply 2 random matrices of the given type and size
//! \param n The matrix dimension.
//-----------------------------------------------------------------------------
double measureRandomMatMul(
    GemmAlgo const * const algo,
    TSize const m, TSize const n, TSize const k,
    TSize const repeatCount,
    bool const bRepeatTakeMinimum
 #ifdef MATMUL_BENCHMARK_VERIFY_RESULT
    ,bool * pResultsCorrect
#endif
    )
{
    TElem const minVal = MATMUL_EPSILON;
    TElem const maxVal = (TElem)10;

#ifdef MATMUL_BENCHMARK_VERIFY_RESULT
    // The threshold difference from where the value is considered to be a real error.
    TElem const errorThreshold = (TElem)(((TElem)2) * MATMUL_EPSILON * ((TElem)m) * ((TElem)n) * ((TElem)k) * maxVal);
#endif

    // Generate random alpha and beta.
#ifdef MATMUL_MPI
    TElem const alpha = (TElem)1;
    TElem const beta = (TElem)0;
#else
    TElem const alpha = matmul_gen_rand_val(minVal, maxVal);
    TElem const beta = matmul_gen_rand_val(minVal, maxVal);
#endif

    // Allocate and initialize the matrices of the given size.
    TSize const elemCount = n * n;
#ifdef MATMUL_MPI
    TElem const * /*const*/ A = 0;
    TElem const * /*const*/ B = 0;
    TElem * /*const*/ C = 0;
    #ifdef MATMUL_BENCHMARK_VERIFY_RESULT
        TElem * /*const*/ D = 0;
    #endif

    int rank1D;
    MPI_Comm_rank(MATMUL_MPI_COMM, &rank1D);
    if(rank1D == MATMUL_MPI_ROOT)
    {
        A = matmul_arr_alloc_fill_rand(elemCount, minVal, maxVal);
        B = matmul_arr_alloc_fill_rand(elemCount, minVal, maxVal);
        C = matmul_arr_alloc(elemCount);
    #ifdef MATMUL_BENCHMARK_VERIFY_RESULT
        D = matmul_arr_alloc(elemCount);
    #endif
    }
#else
    TElem const * const A = matmul_arr_alloc_fill_rand(elemCount, minVal, maxVal);
    TElem const * const B = matmul_arr_alloc_fill_rand(elemCount, minVal, maxVal);
    TElem * const C = matmul_arr_alloc(elemCount);
    #ifdef MATMUL_BENCHMARK_VERIFY_RESULT
        TElem * const D = matmul_arr_alloc(elemCount);
    #endif
#endif

#ifdef MATMUL_BENCHMARK_CUDA_NO_COPY
    MATMUL_CUDA_RT_CHECK(cudaSetDevice(0));

    size_t pitchBytesADev = 0;
    size_t pitchBytesBDev = 0;
    size_t pitchBytesCDev = 0;
    size_t const heightBytesA = m;
    size_t const widthBytesA = k*sizeof(TElem);
    size_t const heightBytesB = k;
    size_t const widthBytesB = n*sizeof(TElem);
    size_t const heightBytesC = m;
    size_t const widthBytesC = n*sizeof(TElem);
    TElem * pADev = 0;
    TElem * pBDev = 0;
    TElem * pCDev = 0;
    MATMUL_CUDA_RT_CHECK(cudaMallocPitch((void **)&pADev, &pitchBytesADev, widthBytesA, heightBytesA));
    MATMUL_CUDA_RT_CHECK(cudaMemcpy2D(pADev, pitchBytesADev, A, n * sizeof(TElem), widthBytesA, heightBytesA, cudaMemcpyHostToDevice));
    MATMUL_CUDA_RT_CHECK(cudaMallocPitch((void **)&pBDev, &pitchBytesBDev, widthBytesB, heightBytesB));
    MATMUL_CUDA_RT_CHECK(cudaMemcpy2D(pBDev, pitchBytesBDev, B, n * sizeof(TElem), widthBytesB, heightBytesB, cudaMemcpyHostToDevice));
    MATMUL_CUDA_RT_CHECK(cudaMallocPitch((void **)&pCDev, &pitchBytesCDev, widthBytesC, heightBytesC));
    TSize const lda = (TSize)(pitchBytesADev / sizeof(TElem));
    TSize const ldb = (TSize)(pitchBytesBDev / sizeof(TElem));
    TSize const ldc = (TSize)(pitchBytesCDev / sizeof(TElem));
#endif

    // Initialize the measurement result.
    double timeMeasuredSec = 0.0;
    if(bRepeatTakeMinimum)
    {
        timeMeasuredSec = DBL_MAX;
    }

    // Iterate.
    for(TSize i = 0; i < repeatCount; ++i)
    {
#ifdef MATMUL_MPI
        if(rank1D == MATMUL_MPI_ROOT)
        {
#endif
            // Because we calculate C += A*B we need to initialize C.
            // Even if we would not need this, we would have to initialize the C array with data before using it because else we would measure page table time on first write.
            // We have to fill C with new data in subsequent iterations because else the values in C would get bigger and bigger in each iteration.
            matmul_arr_fill_rand(C, elemCount, minVal, maxVal);
    #ifdef MATMUL_BENCHMARK_VERIFY_RESULT
            matmul_mat_copy(D, n, C, n, n, n);
    #endif

#ifdef MATMUL_MPI
        }
#endif

#ifdef MATMUL_BENCHMARK_CUDA_NO_COPY
        MATMUL_CUDA_RT_CHECK(cudaMemcpy2D(pCDev, pitchBytesCDev, C, n * sizeof(TElem), widthBytesC, heightBytesC, cudaMemcpyHostToDevice));
#endif

#ifdef MATMUL_MPI
        double timeStart = 0;
        // Only the root process does the printing.
        if(rank1D == MATMUL_MPI_ROOT)
        {
#endif

#ifdef MATMUL_BENCHMARK_PRINT_ITERATIONS
            // If there are multiple repetitions, print the iteration we are at now.
            if(repeatCount!=1)
            {
                if(i>0)
                {
                    printf("; ");
                }
                printf("\ti=%"MATMUL_PRINTF_SIZE_T, (size_t)i);
            }
#endif

#ifdef MATMUL_BENCHMARK_PRINT_MATRICES
            printf("\n");
            printf("%f\n*\n", alpha);
            matmul_mat_print_simple(A, n, n, n);
            printf("\n*\n");
            matmul_mat_print_simple(B, n, n, n);
            printf("\n+\n");
            printf("%f\n*\n", beta);
            matmul_mat_print_simple(C, n, n, n);
#endif

#ifdef MATMUL_MPI
    #ifndef MATMUL_BENCHMARK_COMPUTATION_TIME
            timeStart = getTimeSec();
    #endif
        }
#else
    #ifndef MATMUL_BENCHMARK_COMPUTATION_TIME
        double const timeStart = getTimeSec();
    #endif
#endif

        // Matrix multiplication.
#ifdef MATMUL_BENCHMARK_COMPUTATION_TIME
        double const timeElapsed = 
#endif
#ifdef MATMUL_BENCHMARK_CUDA_NO_COPY
        (algo->pGemm)(n, n, n, alpha, pADev, lda, pBDev, ldb, beta, pCDev, ldc);
#else
        (algo->pGemm)(n, n, n, alpha, A, n, B, n, beta, C, n);
#endif

#ifdef MATMUL_MPI
        // Only the root process does the printing.
        if(rank1D == MATMUL_MPI_ROOT)
        {
#endif
#ifndef MATMUL_BENCHMARK_COMPUTATION_TIME
            double const timeEnd = getTimeSec();
            double const timeElapsed = timeEnd - timeStart;
#endif

#ifdef MATMUL_BENCHMARK_PRINT_MATRICES
            printf("\n=\n");
            matmul_mat_print_simple(C, n, n, n);
#endif

#ifdef MATMUL_BENCHMARK_VERIFY_RESULT

    #ifdef MATMUL_BENCHMARK_CUDA_NO_COPY
            MATMUL_CUDA_RT_CHECK(cudaMemcpy2D(C, n * sizeof(TElem), pCDev, pitchBytesCDev, widthBytesC, heightBytesC, cudaMemcpyDeviceToHost));
    #endif
            matmul_gemm_seq_basic(n, n, n, alpha, A, n, B, n, beta, D, n);

    #ifdef MATMUL_BENCHMARK_PRINT_MATRICES
            printf("\n=\n");
            matmul_mat_print_simple(D, n, n, n);
    #endif

            bool const resultCorrect = matmul_mat_cmp(C, n, D, n, n, n, errorThreshold);
            if(!resultCorrect)
            {
                printf("%s iteration %"MATMUL_PRINTF_SIZE_T" result incorrect!", algo->pszName, (size_t)i);
            }
            *pResultsCorrect = (*pResultsCorrect) && resultCorrect;
#endif

#ifdef MATMUL_BENCHMARK_PRINT_MATRICES
            printf("\n");
#endif

            if(bRepeatTakeMinimum)
            {
                timeMeasuredSec = (timeElapsed<timeMeasuredSec) ? timeElapsed : timeMeasuredSec;
            }
            else
            {
                timeMeasuredSec += timeElapsed * (1.0/(double)repeatCount);
            }

#ifdef MATMUL_MPI
        }
#endif
    }

#ifdef MATMUL_BENCHMARK_CUDA_NO_COPY
    cudaFree(pADev);
    cudaFree(pBDev);
    cudaFree(pCDev);
#endif

#ifdef MATMUL_MPI
    // Only the root process does the printing.
    if(rank1D == MATMUL_MPI_ROOT)
    {
#endif

#ifndef MATMUL_BENCHMARK_PRINT_GFLOPS
        // Print the time needed for the calculation.
        printf("\t%12.8lf", timeMeasuredSec);
#else
        // Print the GFLOPS.
        double const operationCount = 2.0*pow((double)n, algo->exponentOmega);
        double const flops = (timeMeasuredSec!=0) ? (operationCount/timeMeasuredSec) : 0.0;
        printf("\t%12.8lf", flops*1.0e-9);
#endif

        matmul_arr_free((TElem * const)A);
        matmul_arr_free((TElem * const)B);
        matmul_arr_free(C);
#ifdef MATMUL_BENCHMARK_VERIFY_RESULT
        matmul_arr_free(D);
#endif

#ifdef MATMUL_MPI
    }
#endif

    return timeMeasuredSec;
}

//-----------------------------------------------------------------------------
//! A struct containing an array of all matrix sizes to test.
//-----------------------------------------------------------------------------
typedef struct GemmSizes
{
    TSize sizeCount;
    TSize * pSizes;
} GemmSizes;

//-----------------------------------------------------------------------------
//! Fills the matrix sizes struct.
//! \param minN The start matrix dimension.
//! \param stepN The step width for each iteration. If set to 0 the size is doubled on each iteration.
//! \param maxN The maximum matrix dimension.
//-----------------------------------------------------------------------------
GemmSizes buildSizes(
    TSize const minN,
    TSize const maxN,
    TSize const stepN)
{
    GemmSizes sizes;
    sizes.sizeCount = 0;
    sizes.pSizes = 0;

    TSize n;
    for(n = minN; n <= maxN; n += (stepN == 0) ? n : stepN)
    {
        ++sizes.sizeCount;
    }

    sizes.pSizes = (TSize *)malloc(sizes.sizeCount * sizeof(TSize));

    TSize idx = 0;
    for(n = minN; n <= maxN; n += (stepN == 0) ? n : stepN)
    {
        sizes.pSizes[idx] = n;
        ++idx;
    }

    return sizes;
}

//-----------------------------------------------------------------------------
//! Class template with static member templates because function templates do not allow partial specialization.
//! \param minN The start matrix dimension.
//! \param stepN The step width for each iteration. If set to 0 the size is doubled on each iteration.
//! \param maxN The maximum matrix dimension.
//! \return True, if all results are correct.
//-----------------------------------------------------------------------------
#ifdef MATMUL_BENCHMARK_VERIFY_RESULT
    bool
#else
    void
#endif
measureRandomMatMuls(
    GemmAlgo const * const pMatMulAlgos,
    TSize const algoCount,
    GemmSizes const * const pSizes,
    TSize const repeatCount)
{
#ifdef MATMUL_MPI
    int rank1D;
    MPI_Comm_rank(MATMUL_MPI_COMM, &rank1D);
    if(rank1D==MATMUL_MPI_ROOT)
    {
#endif
#ifndef MATMUL_BENCHMARK_PRINT_GFLOPS
        printf("\n#time in s");
#else
        printf("\n#GFLOPS");
#endif
        printf("\nm=n=k");
        // Table heading
        for(TSize algoIdx = 0; algoIdx < algoCount; ++algoIdx)
        {
                printf(" \t%s", pMatMulAlgos[algoIdx].pszName);
        }
#ifdef MATMUL_MPI
    }
#endif

#ifdef MATMUL_BENCHMARK_VERIFY_RESULT
    bool allResultsCorrect = true;
#endif
    if(pSizes)
    {
        for(TSize sizeIdx = 0; sizeIdx < pSizes->sizeCount; ++sizeIdx)
        {
            TSize const n = pSizes->pSizes[sizeIdx];
#ifdef MATMUL_MPI
            if(rank1D==MATMUL_MPI_ROOT)
            {
#endif
                // Print the operation
                printf("\n%"MATMUL_PRINTF_SIZE_T, (size_t)n);
#ifdef MATMUL_MPI
            }
#endif

            for(TSize algoIdx = 0; algoIdx < algoCount; ++algoIdx)
            {
#ifdef MATMUL_BENCHMARK_VERIFY_RESULT
                bool resultsCorrectAlgo = true;
#endif
                // Execute the operation and measure the time taken.
                measureRandomMatMul(
                    &pMatMulAlgos[algoIdx],
                    n,
                    n,
                    n,
                    repeatCount,
#ifdef MATMUL_BENCHMARK_REPEAT_TAKE_MINIMUM
                    true
#else
                    false
#endif
#ifdef MATMUL_BENCHMARK_VERIFY_RESULT
                    , &resultsCorrectAlgo
#endif
                    );

#ifdef MATMUL_BENCHMARK_VERIFY_RESULT
                    allResultsCorrect &= resultsCorrectAlgo;
#endif
            }
        }
    }
    else
    {
        printf("Pointer to structure of test sizes 'pSizes' is not allowed to be nullptr!\n");
    }

#ifdef MATMUL_BENCHMARK_VERIFY_RESULT
    return allResultsCorrect;
#endif
}

#define MATMUL_STRINGIFY(s) MATMUL_STRINGIFY_INTERNAL(s)
#define MATMUL_STRINGIFY_INTERNAL(s) #s

//-----------------------------------------------------------------------------
//! Prints some startup informations.
//-----------------------------------------------------------------------------
void main_print_startup(
    TSize minN,
    TSize maxN,
    TSize stepN,
    TSize repeatCount)
{
    printf("# matmul benchmark copyright (c) 2013-2015, Benjamin Worpitz");
    printf(" | config:");
#ifdef NDEBUG
    printf("Release");
#else
    printf("Debug");
#endif
    printf("; element type:");
#ifdef MATMUL_ELEMENT_TYPE_DOUBLE
    printf("double");
#else
    printf("float");
#endif
    printf("; index type:%s", MATMUL_STRINGIFY(MATMUL_INDEX_TYPE));
    printf("; min n:%"MATMUL_PRINTF_SIZE_T, (size_t)minN);
    printf("; max n:%"MATMUL_PRINTF_SIZE_T, (size_t)maxN);
    printf("; step n:%"MATMUL_PRINTF_SIZE_T, (size_t)stepN);
    printf("; repeat count:%"MATMUL_PRINTF_SIZE_T, (size_t)repeatCount);
#ifdef MATMUL_BENCHMARK_COMPUTATION_TIME
    printf("; MATMUL_BENCHMARK_COMPUTATION_TIME=ON");
#else
    printf("; MATMUL_BENCHMARK_COMPUTATION_TIME=OFF");
#endif
#ifdef MATMUL_BENCHMARK_PRINT_GFLOPS
    printf("; MATMUL_BENCHMARK_PRINT_GFLOPS=ON");
#else
    printf("; MATMUL_BENCHMARK_PRINT_GFLOPS=OFF");
#endif
#ifdef MATMUL_BENCHMARK_REPEAT_TAKE_MINIMUM
    printf("; MATMUL_BENCHMARK_REPEAT_TAKE_MINIMUM=ON");
#else
    printf("; MATMUL_BENCHMARK_REPEAT_TAKE_MINIMUM=OFF");
#endif
#ifdef MATMUL_BENCHMARK_VERIFY_RESULT
    printf("; MATMUL_BENCHMARK_VERIFY_RESULT");
#endif
#ifdef MATMUL_BENCHMARK_PRINT_MATRICES
    printf("; MATMUL_BENCHMARK_PRINT_MATRICES");
#endif
#ifdef MATMUL_BENCHMARK_PRINT_ITERATIONS
    printf("; MATMUL_BENCHMARK_PRINT_ITERATIONS");
#endif
#ifdef MATMUL_MPI
    printf("; MATMUL_MPI");
#endif
}

//-----------------------------------------------------------------------------
//! Main method initiating the measurements of all algorithms selected in config.h.
//-----------------------------------------------------------------------------
int main(
    int argc,
    char ** argv
    )
{
    // Disable buffering of printf. Always print it immediately.
    setvbuf(stdout, 0, _IONBF, 0);

    // Set the initial seed to make the measurements repeatable.
    srand(42u);

    TSize minN = 1;
    TSize maxN = 1;
    TSize stepN = 1;
    TSize repeatCount = 1;

    // Read all arguments.
    if(argc != (4+1))
    {
        printf("\nExactly four arguments are required!");
        printf("\nOptions: min max step repeat!\n");
        return EXIT_FAILURE;
    }
    else
    {
        minN = atoi(argv[1]);
        maxN = atoi(argv[2]);
        stepN = atoi(argv[3]);
        repeatCount = MAX(1, atoi(argv[4]));
    }

#ifdef MATMUL_MPI
    // Initialize MPI before calling any MPI methods.
    int mpiStatus = MPI_Init(&argc, &argv);
    int rank1D;
    if(mpiStatus != MPI_SUCCESS)
    {
        printf("\nUnable to initialize MPI. MPI_Init failed.\n");
    }
    else
    {
        MPI_Comm_rank(MATMUL_MPI_COMM, &rank1D);

        if(rank1D==MATMUL_MPI_ROOT)
        {
            main_print_startup(minN, maxN, stepN, repeatCount);
        }
    }
#else
    main_print_startup(minN, maxN, stepN, repeatCount);
#endif

    double const timeStart = getTimeSec();

    srand((unsigned int)time(0));

    static GemmAlgo const algos[] = {
    #ifdef MATMUL_BENCHMARK_SEQ_BASIC
        {matmul_gemm_seq_basic, "gemm_seq_basic", 3.0},
    #endif
    #ifdef MATMUL_BENCHMARK_SEQ_SINGLE_OPTS
        {matmul_gemm_seq_index_pointer, "gemm_seq_index_pointer", 3.0},
        {matmul_gemm_seq_restrict, "gemm_seq_restrict", 3.0},
        {matmul_gemm_seq_loop_reorder, "gemm_seq_loop_reorder", 3.0},
        {matmul_gemm_seq_index_precalculate, "gemm_seq_index_precalculate", 3.0},
        {matmul_gemm_seq_loop_unroll_4, "gemm_seq_loop_unroll_4", 3.0},
        {matmul_gemm_seq_loop_unroll_8, "gemm_seq_loop_unroll_8", 3.0},
        {matmul_gemm_seq_loop_unroll_16, "gemm_seq_loop_unroll_16", 3.0},
        {matmul_gemm_seq_block, "gemm_seq_block", 3.0},
    #endif
    #ifdef MATMUL_BENCHMARK_SEQ_MULTIPLE_OPTS_BLOCK
        {matmul_gemm_seq_multiple_opts_block, "gemm_seq_complete_opt_block", 3.0},
    #endif
    #ifdef MATMUL_BENCHMARK_SEQ_MULTIPLE_OPTS
        {matmul_gemm_seq_multiple_opts, "gemm_seq_complete_opt", 3.0},
    #endif
    #ifdef MATMUL_BENCHMARK_SEQ_STRASSEN
        {matmul_gemm_seq_strassen, "gemm_seq_strassen", 2.80735},   // 2.80735 = log(7.0) / log(2.0)
    #endif
    #if _OPENMP >= 200203   // OpenMP 2.0
        #ifdef MATMUL_BENCHMARK_PAR_OMP2_GUIDED
            {matmul_gemm_par_omp2_guided_schedule, "gemm_par_omp2_guided_schedule", 3.0},
        #endif
        #ifdef MATMUL_BENCHMARK_PAR_OMP2_STATIC
            {matmul_gemm_par_omp2_static_schedule, "gemm_par_omp2_static_schedule", 3.0},
        #endif
    #endif
    #if _OPENMP >= 200805   // OpenMP 3.0 (3.1=201107)
        #ifdef MATMUL_BENCHMARK_PAR_OMP3
            {matmul_gemm_par_omp3_static_schedule_collapse, "gemm_par_omp3_static_schedule_collapse", 3.0},
        #endif
    #endif
    #if _OPENMP >= 201307   // OpenMP 4.0
        #ifdef MATMUL_BENCHMARK_PAR_OMP4
            {matmul_gemm_par_omp4, "gemm_par_omp4", 3.0},
        #endif
    #endif
    #if _OPENMP >= 200203   // OpenMP 2.0
        #ifdef MATMUL_BENCHMARK_PAR_PHI_OFF_OMP2_GUIDED
            {matmul_gemm_par_phi_off_omp2_guided_schedule, "gemm_par_phi_off_omp2_guided_schedule", 3.0},
        #endif
        #ifdef MATMUL_BENCHMARK_PAR_PHI_OFF_OMP2_STATIC
            {matmul_gemm_par_phi_off_omp2_static_schedule, "gemm_par_phi_off_omp2_static_schedule", 3.0},
        #endif
    #endif
    #if _OPENMP >= 200805   // OpenMP 3.0
        #ifdef MATMUL_BENCHMARK_PAR_PHI_OFF_OMP3
            {matmul_gemm_par_phi_off_omp3_static_schedule_collapse, "gemm_par_phi_off_omp3_static_schedule_collapse", 3.0},
        #endif
    #endif
    #if _OPENMP >= 201307   // OpenMP 4.0
        #ifdef MATMUL_BENCHMARK_PAR_PHI_OFF_OMP4
            {matmul_gemm_par_phi_off_omp4, "gemm_par_phi_off_omp4", 3.0},
        #endif
    #endif
    #if _OPENMP >= 200203   // OpenMP 2.0
        #ifdef MATMUL_BENCHMARK_PAR_STRASSEN_OMP2
            {matmul_gemm_par_strassen_omp2, "gemm_par_strassen_omp", 2.80735},   // 2.80735 = log(7.0) / log(2.0)
        #endif
    #endif
    #ifdef MATMUL_BENCHMARK_PAR_OPENACC
        {matmul_gemm_par_openacc_kernels, "gemm_par_openacc_kernels", 3.0},
        {matmul_gemm_par_openacc_parallel, "gemm_par_openacc_parallel", 3.0},
    #endif
    #ifdef MATMUL_BENCHMARK_PAR_ALPAKA_ACC_CPU_B_OMP2_T_SEQ
        #ifdef MATMUL_BENCHMARK_ALPAKA_CUDASDK_KERNEL
            {matmul_gemm_par_alpaka_cpu_b_omp2_t_seq, "gemm_par_alpaka_cpu_b_omp2_t_seq", 3.0},
        #endif
        #ifdef MATMUL_BENCHMARK_ALPAKA_OMPNATIVE_KERNEL
            {matmul_gemm_par_alpaka_cpu_b_omp2_t_seq_ompNative, "gemm_par_alpaka_cpu_b_omp2_t_seq_ompNative", 3.0},
        #endif
        #ifdef MATMUL_BENCHMARK_ALPAKA_TILING_KERNEL
            {matmul_gemm_par_alpaka_cpu_b_omp2_t_seq_tiling, "gemm_par_alpaka_cpu_b_omp2_t_seq_tiling", 3.0},
        #endif
    #endif
    #ifdef MATMUL_BENCHMARK_PAR_ALPAKA_ACC_CPU_B_SEQ_T_OMP2
        #ifdef MATMUL_BENCHMARK_ALPAKA_CUDASDK_KERNEL
            {matmul_gemm_par_alpaka_cpu_b_seq_t_omp2, "gemm_par_alpaka_cpu_b_seq_t_omp2", 3.0},
        #endif
        #ifdef MATMUL_BENCHMARK_ALPAKA_OMPNATIVE_KERNEL
            {matmul_gemm_par_alpaka_cpu_b_seq_t_omp2_ompNative, "gemm_par_alpaka_cpu_b_seq_t_omp2_ompNative", 3.0},
        #endif
        #ifdef MATMUL_BENCHMARK_ALPAKA_TILING_KERNEL
            {matmul_gemm_par_alpaka_cpu_b_seq_t_omp2_tiling, "gemm_par_alpaka_cpu_b_seq_t_omp2_tiling", 3.0},
        #endif
    #endif
    #ifdef MATMUL_BENCHMARK_PAR_ALPAKA_ACC_CPU_BT_OMP4
        #ifdef MATMUL_BENCHMARK_ALPAKA_CUDASDK_KERNEL
            {matmul_gemm_par_alpaka_cpu_bt_omp4, "gemm_par_alpaka_cpu_bt_omp4", 3.0},
        #endif
        #ifdef MATMUL_BENCHMARK_ALPAKA_OMPNATIVE_KERNEL
            {matmul_gemm_par_alpaka_cpu_bt_omp4_ompNative, "gemm_par_alpaka_cpu_bt_omp4_ompNative", 3.0},
        #endif
        #ifdef MATMUL_BENCHMARK_ALPAKA_TILING_KERNEL
            {matmul_gemm_par_alpaka_cpu_bt_omp4_tiling, "gemm_par_alpaka_cpu_bt_omp4_tiling", 3.0},
        #endif
    #endif
    #ifdef MATMUL_BENCHMARK_PAR_ALPAKA_ACC_CPU_B_SEQ_T_THREADS
        #ifdef MATMUL_BENCHMARK_ALPAKA_CUDASDK_KERNEL
            {matmul_gemm_par_alpaka_cpu_b_seq_t_threads, "gemm_par_alpaka_cpu_b_seq_t_threads", 3.0},
        #endif
        #ifdef MATMUL_BENCHMARK_ALPAKA_OMPNATIVE_KERNEL
            {matmul_gemm_par_alpaka_cpu_b_seq_t_threads_ompNative, "gemm_par_alpaka_cpu_b_seq_t_threads_ompNative", 3.0},
        #endif
        #ifdef MATMUL_BENCHMARK_ALPAKA_TILING_KERNEL
            {matmul_gemm_par_alpaka_cpu_b_seq_t_threads_tiling, "gemm_par_alpaka_cpu_b_seq_t_threads_tiling", 3.0},
        #endif
    #endif
    #ifdef MATMUL_BENCHMARK_PAR_ALPAKA_ACC_CPU_B_SEQ_T_FIBERS
        #ifdef MATMUL_BENCHMARK_ALPAKA_CUDASDK_KERNEL
            {matmul_gemm_seq_alpaka_cpu_b_seq_t_fibers, "gemm_seq_alpaka_cpu_b_seq_t_fibers", 3.0},
        #endif
        #ifdef MATMUL_BENCHMARK_ALPAKA_OMPNATIVE_KERNEL
            {matmul_gemm_seq_alpaka_cpu_b_seq_t_fibers_ompNative, "gemm_seq_alpaka_cpu_b_seq_t_fibers_ompNative", 3.0},
        #endif
        #ifdef MATMUL_BENCHMARK_ALPAKA_TILING_KERNEL
            {matmul_gemm_seq_alpaka_cpu_b_seq_t_fibers_tiling, "gemm_seq_alpaka_cpu_b_seq_t_fibers_tiling", 3.0},
        #endif
    #endif
    #ifdef MATMUL_BENCHMARK_PAR_ALPAKA_ACC_CPU_B_SEQ_T_SEQ
        #ifdef MATMUL_BENCHMARK_ALPAKA_CUDASDK_KERNEL
            {matmul_gemm_seq_alpaka_cpu_b_seq_t_seq, "gemm_seq_alpaka_cpu_b_seq_t_seq", 3.0},
        #endif
        #ifdef MATMUL_BENCHMARK_ALPAKA_OMPNATIVE_KERNEL
            {matmul_gemm_seq_alpaka_cpu_b_seq_t_seq_ompNative, "gemm_seq_alpaka_cpu_b_seq_t_seq_ompNative", 3.0},
        #endif
        #ifdef MATMUL_BENCHMARK_ALPAKA_TILING_KERNEL
            {matmul_gemm_seq_alpaka_cpu_b_seq_t_seq_tiling, "gemm_seq_alpaka_cpu_b_seq_t_seq_tiling", 3.0},
        #endif
    #endif
    #ifdef MATMUL_BENCHMARK_PAR_ALPAKA_ACC_GPU_CUDA_MEMCPY
        #ifdef MATMUL_BENCHMARK_ALPAKA_CUDASDK_KERNEL
            {matmul_gemm_par_alpaka_gpu_cuda_memcpy, "gemm_par_alpaka_gpu_cuda_memcpy", 3.0},
        #endif
        #ifdef MATMUL_BENCHMARK_ALPAKA_OMPNATIVE_KERNEL
            {matmul_gemm_par_alpaka_gpu_cuda_memcpy_ompNative, "gemm_par_alpaka_gpu_cuda_memcpy_ompNative", 3.0},
        #endif
        #ifdef MATMUL_BENCHMARK_ALPAKA_TILING_KERNEL
            {matmul_gemm_par_alpaka_gpu_cuda_memcpy_tiling, "gemm_par_alpaka_gpu_cuda_memcpy_tiling", 3.0},
        #endif
    #endif
    #ifdef MATMUL_BENCHMARK_PAR_CUDA_MEMCPY_FIXED_BLOCK_SIZE
        {matmul_gemm_par_cuda_memcpy_fixed_block_size_2d_static_shared, "gemm_par_cuda_memcpy_fixed_block_size_2d_static_shared", 3.0},
        {matmul_gemm_par_cuda_memcpy_fixed_block_size_1d_static_shared, "gemm_par_cuda_memcpy_fixed_block_size_1d_static_shared", 3.0},
        {matmul_gemm_par_cuda_memcpy_fixed_block_size_1d_extern_shared, "gemm_par_cuda_memcpy_fixed_block_size_1d_extern_shared", 3.0},
    #endif
    #ifdef MATMUL_BENCHMARK_PAR_CUDA_MEMCPY_DYN_BLOCK_SIZE
        {matmul_gemm_par_cuda_memcpy_dyn_block_size_1d_extern_shared, "gemm_par_cuda_memcpy_dyn_block_size_1d_extern_shared", 3.0},
    #endif
    #ifdef MATMUL_BENCHMARK_PAR_BLAS_CUBLAS_MEMCPY
        {matmul_gemm_par_blas_cublas2_memcpy, "gemm_par_blas_cublas2_memcpy", 3.0},
    #endif
    #ifdef MATMUL_BENCHMARK_PAR_BLAS_MKL
        {matmul_gemm_par_blas_mkl, "gemm_par_blas_mkl", 3.0},
    #endif
    #ifdef MATMUL_BENCHMARK_PAR_PHI_OFF_BLAS_MKL
        {matmul_gemm_par_phi_off_blas_mkl, "gemm_par_phi_off_blas_mkl", 3.0},
    #endif

    #ifdef MATMUL_BENCHMARK_PAR_ALPAKA_ACC_GPU_CUDA
        #ifdef MATMUL_BENCHMARK_ALPAKA_CUDASDK_KERNEL
            {matmul_gemm_par_alpaka_gpu_cuda, "gemm_par_alpaka_gpu_cuda", 3.0},
        #endif
        #ifdef MATMUL_BENCHMARK_ALPAKA_OMPNATIVE_KERNEL
            {matmul_gemm_par_alpaka_gpu_cuda_ompNative, "gemm_par_alpaka_gpu_cuda_ompNative", 3.0},
        #endif
        #ifdef MATMUL_BENCHMARK_ALPAKA_TILING_KERNEL
            {matmul_gemm_par_alpaka_gpu_cuda_tiling, "gemm_par_alpaka_gpu_cuda_tiling", 3.0},
        #endif
    #endif
    #ifdef MATMUL_BENCHMARK_PAR_CUDA_FIXED_BLOCK_SIZE
        {matmul_gemm_par_cuda_fixed_block_size_2d_static_shared, "gemm_par_cuda_fixed_block_size_2d_static_shared", 3.0},
        {matmul_gemm_par_cuda_fixed_block_size_1d_static_shared, "gemm_par_cuda_fixed_block_size_1d_static_shared", 3.0},
        {matmul_gemm_par_cuda_fixed_block_size_1d_extern_shared, "gemm_par_cuda_fixed_block_size_1d_extern_shared", 3.0},
    #endif
    #ifdef MATMUL_BENCHMARK_PAR_CUDA_DYN_BLOCK_SIZE
        {matmul_gemm_par_cuda_dyn_block_size_1d_extern_shared, "gemm_par_cuda_dyn_block_size_1d_extern_shared", 3.0 },
    #endif
    #ifdef MATMUL_BENCHMARK_PAR_BLAS_CUBLAS
        {matmul_gemm_par_blas_cublas2, "gemm_par_blas_cublas2", 3.0},
    #endif

    #ifdef MATMUL_BENCHMARK_PAR_MPI_CANNON_STD
        {matmul_gemm_par_mpi_cannon_block, "gemm_par_mpi_cannon_block", 3.0},
        {matmul_gemm_par_mpi_cannon_nonblock, "gemm_par_mpi_cannon_nonblock", 3.0},
    #endif
    #ifdef MATMUL_BENCHMARK_PAR_MPI_CANNON_MKL
        {matmul_gemm_par_mpi_cannon_nonblock_blas_mkl, "gemm_par_mpi_cannon_nonblock_blas_mkl", 3.0},
    #endif
    #ifdef MATMUL_BENCHMARK_BUILD_PAR_MPI_CANNON_CUBLAS
        {matmul_gemm_par_mpi_cannon_nonblock_blas_cublas, "gemm_par_mpi_cannon_nonblock_blas_cublas", 3.0},
    #endif
    #ifdef MATMUL_BENCHMARK_PAR_MPI_DNS
        {matmul_gemm_par_mpi_dns, "gemm_par_mpi_dns", 3.0},
    #endif
    };

    GemmSizes const sizes = buildSizes(
        minN,
        maxN,
        stepN);

#ifdef MATMUL_BENCHMARK_VERIFY_RESULT
    bool const allResultsCorrect =
#endif
    measureRandomMatMuls(
        algos,
        sizeof(algos)/sizeof(algos[0]),
        &sizes,
        repeatCount);

    free(sizes.pSizes);

#ifdef MATMUL_MPI
    if(mpiStatus == MPI_SUCCESS)
    {
        if(rank1D==MATMUL_MPI_ROOT)
        {
            double const timeEnd = getTimeSec();
            double const timeElapsed = timeEnd - timeStart;
            printf("\nTotal runtime: %12.6lf s\n", timeElapsed);
        }
        MPI_Finalize();
    }
#else
    double const timeEnd = getTimeSec();
    double const timeElapsed = timeEnd - timeStart;
    printf("\nTotal runtime: %12.6lf s\n", timeElapsed);
#endif

#ifdef MATMUL_BENCHMARK_VERIFY_RESULT
    return !allResultsCorrect;
#else
    return EXIT_SUCCESS;
#endif
}
