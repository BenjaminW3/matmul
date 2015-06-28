################################################################################
# Copyright 2015 Benjamin Worpitz
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
# SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
# RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
# NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE
# USE OR PERFORMANCE OF THIS SOFTWARE.
################################################################################

################################################################################
# Required CMake version.
################################################################################

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12)

################################################################################
# matmul.
################################################################################

# Return values.
UNSET(matmul_FOUND)
UNSET(matmul_VERSION)
UNSET(matmul_DEFINITIONS)
UNSET(matmul_INCLUDE_DIR)
UNSET(matmul_INCLUDE_DIRS)
UNSET(matmul_LIBRARY)
UNSET(matmul_LIBRARIES)

# Internal usage.
UNSET(_MATMUL_FOUND)
UNSET(_MATMUL_COMPILE_OPTIONS_C_PRIVATE)
UNSET(_MATMUL_COMPILE_OPTIONS_CXX_PRIVATE)
UNSET(_MATMUL_COMPILE_DEFINITIONS_PRIVATE)
UNSET(_MATMUL_COMPILE_DEFINITIONS_PUBLIC)
UNSET(_MATMUL_INCLUDE_DIRECTORY)
UNSET(_MATMUL_INCLUDE_DIRECTORIES_PRIVATE)
UNSET(_MATMUL_INCLUDE_DIRECTORIES_PUBLIC)
UNSET(_MATMUL_LINK_LIBRARY)
UNSET(_MATMUL_LINK_LIBRARIES_PRIVATE)
UNSET(_MATMUL_LINK_LIBRARIES_INTERFACE)
UNSET(_MATMUL_FILES_HEADER)
UNSET(_MATMUL_FILES_SOURCE_CXX)
UNSET(_MATMUL_FILES_SOURCE_CU)
UNSET(_MATMUL_FILES_OTHER)

#-------------------------------------------------------------------------------
# Directory of this file.
#-------------------------------------------------------------------------------
SET(_MATMUL_ROOT_DIR ${CMAKE_CURRENT_LIST_DIR})

# Normalize the path (e.g. remove ../)
GET_FILENAME_COMPONENT(_MATMUL_ROOT_DIR "${_MATMUL_ROOT_DIR}" ABSOLUTE)

#-------------------------------------------------------------------------------
# Set found to true initially and set it on false if a required dependency is missing.
#-------------------------------------------------------------------------------
SET(_MATMUL_FOUND TRUE)

#-------------------------------------------------------------------------------
# Common.
#-------------------------------------------------------------------------------
# Add common functions.
SET(_MATMUL_COMMON_FILE "${_MATMUL_ROOT_DIR}/cmake/common.cmake")
INCLUDE("${_MATMUL_COMMON_FILE}")

#-------------------------------------------------------------------------------
# Options.
#-------------------------------------------------------------------------------
# Drop-down combo box in cmake-gui.
SET(MATMUL_DEBUG "0" CACHE STRING "Debug level")
SET_PROPERTY(CACHE MATMUL_DEBUG PROPERTY STRINGS "0;1;2")

OPTION(MATMUL_ELEMENT_TYPE_DOUBLE "If this is defined, double precision data elements are used, else single precision." ON)
IF(MATMUL_ELEMENT_TYPE_DOUBLE)
    LIST(APPEND _MATMUL_COMPILE_DEFINITIONS_PUBLIC "MATMUL_ELEMENT_TYPE_DOUBLE")
ENDIF()
OPTION(MATMUL_ALIGNED_MALLOC "The matrices will be allocated in aligned storage if this is defined." ON)
IF(MATMUL_ALIGNED_MALLOC)
    LIST(APPEND _MATMUL_COMPILE_DEFINITIONS_PUBLIC "MATMUL_ALIGNED_MALLOC")
ENDIF()

#-------------------------------------------------------------------------------
# Add definitions and dependencies.
#-------------------------------------------------------------------------------

OPTION(MATMUL_BUILD_SEQ_BASIC "Enable the basic sequential GEMM" OFF)
IF(MATMUL_BUILD_SEQ_BASIC)
    LIST(APPEND _MATMUL_COMPILE_DEFINITIONS_PUBLIC "MATMUL_BUILD_SEQ_BASIC")
ENDIF()
OPTION(MATMUL_BUILD_SEQ_SINGLE_OPTS "Enable the optimized versions of the sequential algorithm each with only one optimization" OFF)
IF(MATMUL_BUILD_SEQ_SINGLE_OPTS)
    LIST(APPEND _MATMUL_COMPILE_DEFINITIONS_PUBLIC "MATMUL_BUILD_SEQ_SINGLE_OPTS")
ENDIF()
OPTION(MATMUL_BUILD_SEQ_MULTIPLE_OPTS_BLOCK "Enable the optimized versions of the sequential algorithm with multiple optimizations and blocking at once" OFF)
IF(MATMUL_BUILD_SEQ_MULTIPLE_OPTS_BLOCK)
    LIST(APPEND _MATMUL_COMPILE_DEFINITIONS_PUBLIC "MATMUL_BUILD_SEQ_MULTIPLE_OPTS_BLOCK")
ENDIF()
OPTION(MATMUL_BUILD_SEQ_MULTIPLE_OPTS "Enable the optimized versions of the sequential algorithm with multiple optimizations at once" OFF)
IF(MATMUL_BUILD_SEQ_MULTIPLE_OPTS)
    LIST(APPEND _MATMUL_COMPILE_DEFINITIONS_PUBLIC "MATMUL_BUILD_SEQ_MULTIPLE_OPTS")
ENDIF()
OPTION(MATMUL_BUILD_SEQ_STRASSEN "Enable the basic sequential Strassen algorithm" OFF)
IF(MATMUL_BUILD_SEQ_STRASSEN)
    LIST(APPEND _MATMUL_COMPILE_DEFINITIONS_PUBLIC "MATMUL_BUILD_SEQ_STRASSEN")
    SET(MATMUL_BUILD_SEQ_MULTIPLE_OPTS ON CACHE BOOL "" FORCE)
    LIST(APPEND _MATMUL_COMPILE_DEFINITIONS_PUBLIC "MATMUL_BUILD_SEQ_MULTIPLE_OPTS")
ENDIF()
OPTION(MATMUL_BUILD_PAR_OMP2 "The optimized but not blocked algorithm with OpenMP 2 annotations" OFF)
IF(MATMUL_BUILD_PAR_OMP2)
    SET(_MATMUL_BUILD_OMP ON)
    LIST(APPEND _MATMUL_COMPILE_DEFINITIONS_PUBLIC "MATMUL_BUILD_PAR_OMP2")
ENDIF()
OPTION(MATMUL_BUILD_PAR_OMP3 "The optimized but not blocked algorithm with OpenMP 3 annotations" OFF)
IF(MATMUL_BUILD_PAR_OMP3)
    SET(_MATMUL_BUILD_OMP ON)
    LIST(APPEND _MATMUL_COMPILE_DEFINITIONS_PUBLIC "MATMUL_BUILD_PAR_OMP3")
ENDIF()
OPTION(MATMUL_BUILD_PAR_OMP4 "The optimized but not blocked algorithm with OpenMP 4 annotations" OFF)
IF(MATMUL_BUILD_PAR_OMP4)
    SET(_MATMUL_BUILD_OMP ON)
    LIST(APPEND _MATMUL_COMPILE_DEFINITIONS_PUBLIC "MATMUL_BUILD_PAR_OMP4")
ENDIF()
OPTION(MATMUL_BUILD_PAR_STRASSEN_OMP2 "The Strassen algorithm using OpenMP 2 methods for the GEMM base case and matrix addition and subtraction" OFF)
IF(MATMUL_BUILD_PAR_STRASSEN_OMP2)
    SET(_MATMUL_BUILD_OMP ON)
    LIST(APPEND _MATMUL_COMPILE_DEFINITIONS_PUBLIC "MATMUL_BUILD_PAR_STRASSEN_OMP2")
    SET(MATMUL_BUILD_PAR_OMP2 ON CACHE BOOL "" FORCE)
    LIST(APPEND _MATMUL_COMPILE_DEFINITIONS_PUBLIC "MATMUL_BUILD_PAR_OMP2")
ENDIF()
OPTION(MATMUL_BUILD_PAR_OPENACC "Enable the optimized but not blocked algorithm with OpenACC annotations" OFF)
IF(MATMUL_BUILD_PAR_OPENACC)
    SET(MATMUL_BUILD_PAR_OPENACC ON)
    LIST(APPEND _MATMUL_COMPILE_DEFINITIONS_PUBLIC "MATMUL_BUILD_PAR_OPENACC")
ENDIF()
OPTION(MATMUL_BUILD_PAR_CUDA "Enable the GEMM algorithm from the CUDA developers guide" OFF)
IF(MATMUL_BUILD_PAR_CUDA)
    SET(_MATMUL_BUILD_CUDA ON)
    LIST(APPEND _MATMUL_COMPILE_DEFINITIONS_PUBLIC "MATMUL_BUILD_PAR_CUDA")
ENDIF()
OPTION(MATMUL_BUILD_PAR_BLAS_MKL "Enable the GEMM using the Intel MKL BLAS implementation" OFF)
IF(MATMUL_BUILD_PAR_BLAS_MKL)
    SET(_MATMUL_BUILD_MKL ON)
    LIST(APPEND _MATMUL_COMPILE_DEFINITIONS_PUBLIC "MATMUL_BUILD_PAR_BLAS_MKL")
ENDIF()
OPTION(MATMUL_BUILD_PAR_BLAS_CUBLAS "Enable the GEMM using the NVIDIA cublas2 implementation." OFF)
IF(MATMUL_BUILD_PAR_BLAS_CUBLAS)
    SET(_MATMUL_BUILD_CUBLAS ON)
    LIST(APPEND _MATMUL_COMPILE_DEFINITIONS_PUBLIC "MATMUL_BUILD_PAR_BLAS_CUBLAS")
ENDIF()
OPTION(MATMUL_BUILD_PAR_PHI_OFF_OMP2 "Enable the offloading of the GEMM onto the xeon phi using OpenMP 2." OFF)
IF(MATMUL_BUILD_PAR_PHI_OFF_OMP2)
    SET(_MATMUL_BUILD_OMP ON)
    LIST(APPEND _MATMUL_COMPILE_DEFINITIONS_PUBLIC "MATMUL_BUILD_PAR_PHI_OFF_OMP2")
ENDIF()
OPTION(MATMUL_BUILD_PAR_PHI_OFF_OMP3 "Enable the offloading of the GEMM onto the xeon phi using OpenMP 3." OFF)
IF(MATMUL_BUILD_PAR_PHI_OFF_OMP3)
    SET(_MATMUL_BUILD_OMP ON)
    LIST(APPEND _MATMUL_COMPILE_DEFINITIONS_PUBLIC "MATMUL_BUILD_PAR_PHI_OFF_OMP3")
ENDIF()
OPTION(MATMUL_BUILD_PAR_PHI_OFF_OMP4 "Enable the offloading of the GEMM onto the xeon phi using OpenMP 4." OFF)
IF(MATMUL_BUILD_PAR_PHI_OFF_OMP4)
    SET(_MATMUL_BUILD_OMP ON)
    LIST(APPEND _MATMUL_COMPILE_DEFINITIONS_PUBLIC "MATMUL_BUILD_PAR_PHI_OFF_OMP4")
ENDIF()
OPTION(MATMUL_BUILD_PAR_PHI_OFF_BLAS_MKL "Enable the offloading of the GEMM onto the xeon phi using the Intel MKL BLAS implementation." OFF)
IF(MATMUL_BUILD_PAR_PHI_OFF_BLAS_MKL)
    SET(_MATMUL_BUILD_MKL ON)
    LIST(APPEND _MATMUL_COMPILE_DEFINITIONS_PUBLIC "MATMUL_BUILD_PAR_PHI_OFF_BLAS_MKL")
ENDIF()

OPTION(MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_B_SEQ_T_SEQ "Enable the GEMM using the alpaka serial accelerator back-end on the CPU" OFF)
IF(MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_B_SEQ_T_SEQ)
    SET(_MATMUL_BUILD_ALPAKA ON)
    SET(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE ON)
    LIST(APPEND _MATMUL_COMPILE_DEFINITIONS_PUBLIC "MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_B_SEQ_T_SEQ")
ELSE()
    SET(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE OFF)
ENDIF()
OPTION(MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_B_OMP2_T_SEQ "Enable the GEMM using the alpaka OpenMP 2.0 grid block accelerator back-end on the CPU" OFF)
IF(MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_B_OMP2_T_SEQ)
    SET(_MATMUL_BUILD_ALPAKA ON)
    SET(ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE ON CACHE BOOL "" FORCE)
    SET(_MATMUL_BUILD_OMP ON)
    LIST(APPEND _MATMUL_COMPILE_DEFINITIONS_PUBLIC "MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_B_OMP2_T_SEQ")
ELSE()
    SET(ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE OFF CACHE BOOL "" FORCE)
ENDIF()
OPTION(MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_B_SEQ_T_OMP2 "Enable the GEMM using the alpaka OpenMP 2.0 block thread accelerator back-end on the CPU" OFF)
IF(MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_B_SEQ_T_OMP2)
    SET(_MATMUL_BUILD_ALPAKA ON)
    SET(ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE ON CACHE BOOL "" FORCE)
    SET(_MATMUL_BUILD_OMP ON)
    LIST(APPEND _MATMUL_COMPILE_DEFINITIONS_PUBLIC "MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_B_SEQ_T_OMP2")
ELSE()
    SET(ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE OFF CACHE BOOL "" FORCE)
ENDIF()
OPTION(MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_BT_OMP4 "Enable the GEMM using the alpaka OpenMP 4.0 accelerator back-end on the CPU" OFF)
IF(MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_BT_OMP4)
    SET(_MATMUL_BUILD_ALPAKA ON)
    SET(ALPAKA_ACC_CPU_BT_OMP4_ENABLE ON CACHE BOOL "" FORCE)
    SET(_MATMUL_BUILD_OMP ON)
    LIST(APPEND _MATMUL_COMPILE_DEFINITIONS_PUBLIC "MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_BT_OMP4")
ELSE()
    SET(ALPAKA_ACC_CPU_BT_OMP4_ENABLE OFF CACHE BOOL "" FORCE)
ENDIF()
OPTION(MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_B_SEQ_T_FIBERS "Enable the GEMM using the alpaka Boost.Fiber accelerator back-end on the CPU" OFF)
IF(MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_B_SEQ_T_FIBERS)
    SET(_MATMUL_BUILD_ALPAKA ON)
    SET(ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLE ON CACHE BOOL "" FORCE)
    LIST(APPEND _MATMUL_COMPILE_DEFINITIONS_PUBLIC "MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_B_SEQ_T_FIBERS")
ELSE()
    SET(ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLE OFF CACHE BOOL "" FORCE)
ENDIF()
OPTION(MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_B_SEQ_T_THREADS "Enable the GEMM using the alpaka std::thread accelerator back-end on the CPU" OFF)
IF(MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_B_SEQ_T_THREADS)
    SET(_MATMUL_BUILD_ALPAKA ON)
    SET(ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE ON CACHE BOOL "" FORCE)
    LIST(APPEND _MATMUL_COMPILE_DEFINITIONS_PUBLIC "MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_B_SEQ_T_THREADS")
ELSE()
    SET(ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE OFF CACHE BOOL "" FORCE)
ENDIF()
OPTION(MATMUL_BUILD_PAR_ALPAKA_ACC_GPU_CUDA "Enable the GEMM using the alpaka CUDA accelerator" OFF)
IF(MATMUL_BUILD_PAR_ALPAKA_ACC_GPU_CUDA)
    SET(_MATMUL_BUILD_ALPAKA ON)
    SET(ALPAKA_ACC_GPU_CUDA_ENABLE ON CACHE BOOL "" FORCE)
    SET(_MATMUL_BUILD_CUDA ON)
    LIST(APPEND _MATMUL_COMPILE_DEFINITIONS_PUBLIC "MATMUL_BUILD_PAR_ALPAKA_ACC_GPU_CUDA")
ELSE()
    SET(ALPAKA_ACC_GPU_CUDA_ENABLE OFF CACHE BOOL "" FORCE)
ENDIF()

OPTION(MATMUL_BUILD_PAR_MPI_CANNON_STD "Enable the distributed GEMM using the cannon algorithm, MPI and the optimized sequential implementation on each node." OFF)
IF(MATMUL_BUILD_PAR_MPI_CANNON_STD)
    SET(_MATMUL_BUILD_MPI ON)
    LIST(APPEND _MATMUL_COMPILE_DEFINITIONS_PUBLIC "MATMUL_BUILD_PAR_MPI_CANNON_STD")
    SET(MATMUL_BUILD_SEQ_MULTIPLE_OPTS ON CACHE BOOL "" FORCE)
    LIST(APPEND _MATMUL_COMPILE_DEFINITIONS_PUBLIC "MATMUL_BUILD_SEQ_MULTIPLE_OPTS")
ENDIF()
OPTION(MATMUL_BUILD_PAR_MPI_CANNON_MKL "Enable the distributed GEMM using the cannon algorithm, MPI and the Intel MKL on each node." OFF)
IF(MATMUL_BUILD_PAR_MPI_CANNON_MKL)
    SET(_MATMUL_BUILD_MPI ON)
    SET(_MATMUL_BUILD_MKL ON)
    LIST(APPEND _MATMUL_COMPILE_DEFINITIONS_PUBLIC "MATMUL_BUILD_PAR_MPI_CANNON_MKL")
    SET(MATMUL_BUILD_PAR_BLAS_MKL ON CACHE BOOL "" FORCE)
    LIST(APPEND _MATMUL_COMPILE_DEFINITIONS_PUBLIC "MATMUL_BUILD_PAR_BLAS_MKL")
ENDIF()
OPTION(MATMUL_BUILD_PAR_MPI_CANNON_CUBLAS "Enable the distributed GEMM using the cannon algorithm, MPI and the cuBLAS on each node." OFF)
IF(MATMUL_BUILD_PAR_MPI_CANNON_CUBLAS)
    SET(_MATMUL_BUILD_MPI ON)
    SET(_MATMUL_BUILD_CUBLAS ON)
    LIST(APPEND _MATMUL_COMPILE_DEFINITIONS_PUBLIC "MATMUL_BUILD_PAR_MPI_CANNON_CUBLAS")
    SET(MATMUL_BUILD_PAR_BLAS_CUBLAS ON CACHE BOOL "" FORCE)
    LIST(APPEND _MATMUL_COMPILE_DEFINITIONS_PUBLIC "MATMUL_BUILD_PAR_BLAS_CUBLAS")
ENDIF()
OPTION(MATMUL_BUILD_PAR_MPI_DNS "Enable the distributed GEMM using the DNS algorithm, MPI and the optimized sequential implementation on each node" OFF)
IF(MATMUL_BUILD_PAR_MPI_DNS)
    SET(_MATMUL_BUILD_MPI ON)
    LIST(APPEND _MATMUL_COMPILE_DEFINITIONS_PUBLIC "MATMUL_BUILD_PAR_MPI_DNS")
    SET(MATMUL_BUILD_SEQ_MULTIPLE_OPTS ON CACHE BOOL "" FORCE)
    LIST(APPEND _MATMUL_COMPILE_DEFINITIONS_PUBLIC "MATMUL_BUILD_SEQ_MULTIPLE_OPTS")
ENDIF()

#-------------------------------------------------------------------------------
# Sequential settings.
#-------------------------------------------------------------------------------
IF(MATMUL_BUILD_SEQ_SINGLE_OPTS OR MATMUL_BUILD_SEQ_MULTIPLE_OPTS_BLOCK)
    SET(MATMUL_SEQ_BLOCK_FACTOR 128 CACHE INTEGER "The block factor for local GEMM.")
    IF(MATMUL_SEQ_BLOCK_FACTOR)
        LIST(APPEND _MATMUL_COMPILE_DEFINITIONS_PUBLIC "MATMUL_SEQ_BLOCK_FACTOR=${MATMUL_SEQ_BLOCK_FACTOR}")
    ENDIF()
ENDIF()

#-------------------------------------------------------------------------------
# Strassen settings.
#-------------------------------------------------------------------------------
IF(MATMUL_BUILD_SEQ_STRASSEN)
    SET(MATMUL_STRASSEN_CUT_OFF 128 CACHE INTEGER "The cut-off at which the standard algorithm is used instead of further recursive calculation.")
    IF(MATMUL_STRASSEN_CUT_OFF)
        LIST(APPEND _MATMUL_COMPILE_DEFINITIONS_PUBLIC "MATMUL_STRASSEN_CUT_OFF=${MATMUL_STRASSEN_CUT_OFF}")
    ENDIF()
ENDIF()
IF(MATMUL_BUILD_PAR_STRASSEN_OMP2)
    SET(MATMUL_STRASSEN_OMP_CUT_OFF 512 CACHE INTEGER "The cut-off at which the standard algorithm is used instead of further recursive calculation.")
    IF(MATMUL_STRASSEN_OMP_CUT_OFF)
        LIST(APPEND _MATMUL_COMPILE_DEFINITIONS_PUBLIC "MATMUL_STRASSEN_OMP_CUT_OFF=${MATMUL_STRASSEN_OMP_CUT_OFF}")
    ENDIF()
ENDIF()

#-------------------------------------------------------------------------------
# OpenMP Settings.
#-------------------------------------------------------------------------------
IF(_MATMUL_BUILD_OMP)
    OPTION(MATMUL_OMP_PRINT_NUM_CORES "If this is defined, each call to a matmul function will print out the number of cores used currently. This can have a huge performance impact especially for the recursive Strassen Method." OFF)
    IF(MATMUL_OMP_PRINT_NUM_CORES)
        LIST(APPEND _MATMUL_COMPILE_DEFINITIONS_PUBLIC "MATMUL_OMP_PRINT_NUM_CORES")
    ENDIF()
ENDIF()

#-------------------------------------------------------------------------------
# OpenACC settings.
#-------------------------------------------------------------------------------
IF(MATMUL_BUILD_PAR_OPENACC)
    SET(MATMUL_OPENACC_GANG_SIZE 32 CACHE INTEGER "The block factor used.")
    IF(MATMUL_OPENACC_GANG_SIZE)
        LIST(APPEND _MATMUL_COMPILE_DEFINITIONS_PUBLIC "MATMUL_OPENACC_GANG_SIZE=${MATMUL_OPENACC_GANG_SIZE}")
    ENDIF()
    SET(MATMUL_OPENACC_VECTOR_SIZE 64 CACHE INTEGER "The block factor used.")
    IF(MATMUL_OPENACC_VECTOR_SIZE)
        LIST(APPEND _MATMUL_COMPILE_DEFINITIONS_PUBLIC "MATMUL_OPENACC_VECTOR_SIZE=${MATMUL_OPENACC_VECTOR_SIZE}")
    ENDIF()
ENDIF()

#-------------------------------------------------------------------------------
# CUDA Settings.
#-------------------------------------------------------------------------------
IF(MATMUL_BUILD_PAR_CUDA)
    SET(MATMUL_CUDA_BLOCKSIZE 32 CACHE INTEGER "The block size used on the gpu.")
    IF(MATMUL_CUDA_BLOCKSIZE)
        LIST(APPEND _MATMUL_COMPILE_DEFINITIONS_PUBLIC "MATMUL_CUDA_BLOCKSIZE=${MATMUL_CUDA_BLOCKSIZE}")
    ENDIF()
ENDIF()

#-------------------------------------------------------------------------------
# MKL BLAS Settings.
#-------------------------------------------------------------------------------
IF(MATMUL_BUILD_PAR_PHI_OFF_BLAS_MKL)
    OPTION(MATMUL_PHI_OFF_BLAS_MKL_AUTO_WORKDIVISION "If this is not set, the GEMM will be fully computed on the phi." OFF)
    IF(MATMUL_PHI_OFF_BLAS_MKL_AUTO_WORKDIVISION)
        LIST(APPEND _MATMUL_COMPILE_DEFINITIONS_PUBLIC "MATMUL_PHI_OFF_BLAS_MKL_AUTO_WORKDIVISION")
    ENDIF()
ENDIF()

#-------------------------------------------------------------------------------
# Find OpenMP.
#-------------------------------------------------------------------------------
IF(_MATMUL_BUILD_OMP)
    FIND_PACKAGE(OpenMP)
    IF(NOT OPENMP_FOUND)
        MESSAGE(WARNING "Required matmul dependency OpenMP could not be found!")
        SET(_MATMUL_FOUND FALSE)

    ELSE()
        LIST(APPEND _MATMUL_COMPILE_OPTIONS_C_PRIVATE ${OpenMP_C_FLAGS})
        SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_C_FLAGS}")
        IF((NOT ALPAKA_ACC_GPU_CUDA_ENABLE) AND _MATMUL_BUILD_CUDA)
            SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        ENDIF()
    ENDIF()
ENDIF()

#-------------------------------------------------------------------------------
# Find OpenACC.
#-------------------------------------------------------------------------------
IF(MATMUL_BUILD_PAR_OPENACC)
    FIND_PACKAGE(OpenACC)
    IF(NOT OpenACC_FOUND)
        MESSAGE(WARNING "Required matmul dependency OpenACC could not be found!")
        SET(_MATMUL_FOUND FALSE)

    ELSE()
        MESSAGE(FATAL_ERROR "Setting matmul OpenACC dependencies not implemented!")
        # FIXME: -acc -ta=nvidia -Minfo=accel -mp=allcores # -march=native PGI compiler may need --enable-cuda=basic
    ENDIF()
ENDIF()

#-------------------------------------------------------------------------------
# Find MKL.
#-------------------------------------------------------------------------------
IF(_MATMUL_BUILD_MKL)
    FIND_PACKAGE(MKL)
    IF(NOT MKL_FOUND)
        MESSAGE(WARNING "Required matmul dependency MKL could not be found!")
        SET(_MATMUL_FOUND FALSE)

    ELSE()
        MESSAGE(FATAL_ERROR "Setting matmul MKL dependencies not implemented!")
        # FIXME: -mkl=parallel
    ENDIF()
ENDIF()

#-------------------------------------------------------------------------------
# Find MPI
#-------------------------------------------------------------------------------
IF(_MATMUL_BUILD_MPI)
    FIND_PACKAGE(MPI)
    IF((NOT MPI_C_FOUND) OR (NOT MPI_CXX_FOUND))
        MESSAGE(WARNING "Required matmul dependency MPI could not be found!")
        SET(_MATMUL_FOUND FALSE)

    ELSE()
        LIST(APPEND _MATMUL_COMPILE_OPTIONS_C_PRIVATE ${MPI_C_COMPILE_FLAGS})
        LIST(APPEND _MATMUL_COMPILE_OPTIONS_CXX_PRIVATE ${MPI_CXX_COMPILE_FLAGS})

        LIST(APPEND _MATMUL_INCLUDE_DIRECTORIES_PUBLIC ${MPI_C_INCLUDE_PATH})
        LIST(APPEND _MATMUL_INCLUDE_DIRECTORIES_PUBLIC ${MPI_CXX_INCLUDE_PATH})

        SET(_MATMUL_MPI_LIBRARIES ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})
        list_add_prefix("general;" _MATMUL_MPI_LIBRARIES)
        LIST(APPEND _MATMUL_LINK_LIBRARIES_PRIVATE ${_MATMUL_MPI_LIBRARIES})
        UNSET(_MATMUL_MPI_LIBRARIES)
    ENDIF()
ENDIF()

#-------------------------------------------------------------------------------
# Find CUDA.
#-------------------------------------------------------------------------------
IF(_MATMUL_BUILD_CUDA OR _MATMUL_BUILD_CUBLAS)
    IF(_MATMUL_BUILD_CUBLAS)
        # cuBLAS 2 is supported in CUDA 4.0+.
        FIND_PACKAGE(CUDA "4.0")
    ELSE()
        FIND_PACKAGE(CUDA)
    ENDIF()

    IF(NOT CUDA_FOUND)
        MESSAGE(WARNING "Required matmul dependency CUDA could not be found!")
        SET(_MATMUL_FOUND FALSE)

    ELSE()
        # If the flags have not already been set.
        IF(NOT ALPAKA_ACC_GPU_CUDA_ENABLE)
            IF(${MATMUL_DEBUG} GREATER 1)
                SET(CUDA_VERBOSE_BUILD ON)
            ENDIF()
            SET(CUDA_PROPAGATE_HOST_FLAGS ON)

            SET(MATMUL_CUDA_ARCH sm_20 CACHE STRING "Set GPU architecture")
            LIST(APPEND CUDA_NVCC_FLAGS "-arch=${MATMUL_CUDA_ARCH}")

            IF(NOT MSVC)
                SET(CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
            ENDIF()

            IF(CMAKE_BUILD_TYPE MATCHES "Debug")
                LIST(APPEND CUDA_NVCC_FLAGS "-g" "-G")
            ENDIF()

            OPTION(MATMUL_CUDA_FAST_MATH "Enable fast-math" ON)
            IF(MATMUL_CUDA_FAST_MATH)
                LIST(APPEND CUDA_NVCC_FLAGS "--use_fast_math")
            ENDIF()

            OPTION(MATMUL_CUDA_FTZ "Set flush to zero for GPU" OFF)
            IF(MATMUL_CUDA_FTZ)
                LIST(APPEND CUDA_NVCC_FLAGS "--ftz=true")
            ELSE()
                LIST(APPEND CUDA_NVCC_FLAGS "--ftz=false")
            ENDIF()

            OPTION(MATMUL_CUDA_SHOW_REGISTER "Show kernel registers and create PTX" OFF)
            IF(MATMUL_CUDA_SHOW_REGISTER)
                LIST(APPEND CUDA_NVCC_FLAGS "-Xptxas=-v")
            ENDIF()

            OPTION(MATMUL_CUDA_KEEP_FILES "Keep all intermediate files that are generated during internal compilation steps (folder: nvcc_tmp)" OFF)
            IF(MATMUL_CUDA_KEEP_FILES)
                MAKE_DIRECTORY("${PROJECT_BINARY_DIR}/nvcc_tmp")
                LIST(APPEND CUDA_NVCC_FLAGS "--keep" "--keep-dir" "${PROJECT_BINARY_DIR}/nvcc_tmp")
            ENDIF()

            OPTION(MATMUL_CUDA_SHOW_CODELINES "Show kernel lines in cuda-gdb and cuda-memcheck" OFF)
            IF(MATMUL_CUDA_SHOW_CODELINES)
                LIST(APPEND CUDA_NVCC_FLAGS "--source-in-ptx" "-lineinfo")
                IF(NOT MSVC)
                    LIST(APPEND CUDA_NVCC_FLAGS "-Xcompiler" "-rdynamic")
                ENDIF()
                SET(MATMUL_CUDA_KEEP_FILES ON CACHE BOOL "activate keep files" FORCE)
            ENDIF()

            LIST(APPEND _MATMUL_LINK_LIBRARIES_PRIVATE "general;${CUDA_CUDART_LIBRARY}")
            LIST(APPEND _MATMUL_INCLUDE_DIRECTORIES_PRIVATE ${CUDA_INCLUDE_DIRS})
        ENDIF()

        IF(_MATMUL_BUILD_CUBLAS)
            LIST(APPEND _MATMUL_LINK_LIBRARIES_PRIVATE "general;${CUDA_CUBLAS_LIBRARIES}")
        ENDIF()
    ENDIF()
ENDIF()

#-------------------------------------------------------------------------------
# Find alpaka.
#-------------------------------------------------------------------------------
IF(_MATMUL_BUILD_ALPAKA)
    LIST(APPEND CMAKE_MODULE_PATH "${ALPAKA_ROOT}")
    FIND_PACKAGE(alpaka)
    IF(NOT alpaka_FOUND)
        MESSAGE(WARNING "Required matmul dependency alpaka could not be found!")
        SET(_MATMUL_FOUND FALSE)

    ELSE()
        LIST(APPEND _MATMUL_COMPILE_OPTIONS_CXX_PRIVATE ${alpaka_COMPILE_OPTIONS})
        LIST(APPEND _MATMUL_COMPILE_DEFINITIONS_PRIVATE ${alpaka_COMPILE_DEFINITIONS})
        LIST(APPEND _MATMUL_INCLUDE_DIRECTORIES_PRIVATE ${alpaka_INCLUDE_DIRS})
        LIST(APPEND _MATMUL_LINK_LIBRARIES_PRIVATE ${alpaka_LIBRARIES})
    ENDIF()
ENDIF()

#-------------------------------------------------------------------------------
# matmul.
#-------------------------------------------------------------------------------
SET(_MATMUL_INCLUDE_DIRECTORY "${_MATMUL_ROOT_DIR}/include")
LIST(APPEND _MATMUL_INCLUDE_DIRECTORIES_PUBLIC ${_MATMUL_INCLUDE_DIRECTORY})
SET(_MATMUL_LINK_LIBRARY "$<TARGET_FILE:>")
LIST(APPEND _MATMUL_LINK_LIBRARIES_INTERFACE ${_MATMUL_LINK_LIBRARY})

IF(NOT MSVC)
    LIST(APPEND _MATMUL_COMPILE_OPTIONS_C_PRIVATE "-std=c99")
ENDIF()

LIST(APPEND _MATMUL_COMPILE_DEFINITIONS_PUBLIC "MATMUL_DEBUG=${MATMUL_DEBUG}")

SET(_MATMUL_SUFFIXED_INCLUDE_DIR "${_MATMUL_INCLUDE_DIRECTORY}/matmul")
SET(_MATMUL_SOURCE_DIR "${_MATMUL_ROOT_DIR}/src")

SET(_MATMUL_FILES_OTHER "${_MATMUL_ROOT_DIR}/matmulConfig.cmake" "${_MATMUL_COMMON_FILE}" "${_MATMUL_ROOT_DIR}/.travis.yml" "${_MATMUL_ROOT_DIR}/README.md")

# Add all the source and include files in all recursive subdirectories and group them accordingly.
append_recursive_files_add_to_src_group("${_MATMUL_SUFFIXED_INCLUDE_DIR}" "${_MATMUL_SUFFIXED_INCLUDE_DIR}" "h" _MATMUL_FILES_HEADER)
append_recursive_files_add_to_src_group("${_MATMUL_SUFFIXED_INCLUDE_DIR}" "${_MATMUL_SUFFIXED_INCLUDE_DIR}" "hpp" _MATMUL_FILES_HEADER)
append_recursive_files_add_to_src_group("${_MATMUL_SOURCE_DIR}" "${_MATMUL_SOURCE_DIR}" "cpp" _MATMUL_FILES_SOURCE_CXX)
append_recursive_files_add_to_src_group("${_MATMUL_SOURCE_DIR}" "${_MATMUL_SOURCE_DIR}" "c" _MATMUL_FILES_SOURCE_C)
append_recursive_files_add_to_src_group("${_MATMUL_SOURCE_DIR}" "${_MATMUL_SOURCE_DIR}" "cu" _MATMUL_FILES_SOURCE_CU)

# Compile options (PRIVATE).
# C
IF(${MATMUL_DEBUG} GREATER 1)
    MESSAGE(STATUS "_MATMUL_COMPILE_OPTIONS_C_PRIVATE: ${_MATMUL_COMPILE_OPTIONS_C_PRIVATE}")
ENDIF()
# COMPILE_FLAGS is NOT a list so we have to append the options (prefixed by a space) one by one to not insert a semicolon in between.
FOREACH(_MATMUL_COMPILE_OPTION_C_PRIVATE ${_MATMUL_COMPILE_OPTIONS_C_PRIVATE})
    SET_PROPERTY(
        SOURCE ${_MATMUL_FILES_SOURCE_C}
        APPEND_STRING
        PROPERTY COMPILE_FLAGS " ${_MATMUL_COMPILE_OPTION_C_PRIVATE}")
ENDFOREACH()
# CXX
IF(${MATMUL_DEBUG} GREATER 1)
    MESSAGE(STATUS "_MATMUL_COMPILE_OPTIONS_CXX_PRIVATE: ${_MATMUL_COMPILE_OPTIONS_CXX_PRIVATE}")
ENDIF()
FOREACH(_MATMUL_COMPILE_OPTION_CXX_PRIVATE ${_MATMUL_COMPILE_OPTIONS_CXX_PRIVATE})
    SET_PROPERTY(
        SOURCE ${_MATMUL_FILES_SOURCE_CXX} ${_MATMUL_FILES_SOURCE_CU}
        APPEND_STRING
        PROPERTY COMPILE_FLAGS " ${_MATMUL_COMPILE_OPTION_CXX_PRIVATE}")
ENDFOREACH()

#-------------------------------------------------------------------------------
# Target.
#-------------------------------------------------------------------------------
IF(NOT TARGET "matmul")
    IF(NOT _MATMUL_BUILD_CUDA)
        # Always add all files to the target executable build call to add them to the build project.
        ADD_LIBRARY(
            "matmul"
            ${_MATMUL_FILES_HEADER} ${_MATMUL_FILES_SOURCE_C} ${_MATMUL_FILES_SOURCE_CXX} ${_MATMUL_FILES_SOURCE_CU} ${_MATMUL_FILES_OTHER})

        # Compile definitions.
        IF(${MATMUL_DEBUG} GREATER 1)
            MESSAGE(STATUS "_MATMUL_COMPILE_DEFINITIONS_PRIVATE: ${_MATMUL_COMPILE_DEFINITIONS_PRIVATE}")
            MESSAGE(STATUS "_MATMUL_COMPILE_DEFINITIONS_PUBLIC: ${_MATMUL_COMPILE_DEFINITIONS_PUBLIC}")
        ENDIF()
        LIST(
            LENGTH
            _MATMUL_COMPILE_DEFINITIONS_PRIVATE
            _MATMUL_COMPILE_DEFINITIONS_PRIVATE_LENGTH)
        IF(${_MATMUL_COMPILE_DEFINITIONS_PRIVATE_LENGTH} GREATER 0)
            TARGET_COMPILE_DEFINITIONS(
                "matmul"
                PRIVATE ${_MATMUL_COMPILE_DEFINITIONS_PRIVATE})
        ENDIF()
        LIST(
            LENGTH
            _MATMUL_COMPILE_DEFINITIONS_PUBLIC
            _MATMUL_COMPILE_DEFINITIONS_PUBLIC_LENGTH)
        IF(${_MATMUL_COMPILE_DEFINITIONS_PUBLIC_LENGTH} GREATER 0)
            TARGET_COMPILE_DEFINITIONS(
                "matmul"
                PUBLIC ${_MATMUL_COMPILE_DEFINITIONS_PUBLIC})
        ENDIF()

        # Include directories.
        IF(${MATMUL_DEBUG} GREATER 1)
            MESSAGE(STATUS "_MATMUL_INCLUDE_DIRECTORIES_PRIVATE: ${_MATMUL_INCLUDE_DIRECTORIES_PRIVATE}")
        ENDIF()
        LIST(
            LENGTH
            _MATMUL_INCLUDE_DIRECTORIES_PRIVATE
            _MATMUL_INCLUDE_DIRECTORIES_PRIVATE_LENGTH)
        IF(${_MATMUL_INCLUDE_DIRECTORIES_PRIVATE_LENGTH} GREATER 0)
            TARGET_INCLUDE_DIRECTORIES(
                "matmul"
                PRIVATE ${_MATMUL_INCLUDE_DIRECTORIES_PRIVATE})
        ENDIF()
        LIST(
            LENGTH
            _MATMUL_INCLUDE_DIRECTORIES_PUBLIC
            _MATMUL_INCLUDE_DIRECTORIES_PUBLIC_LENGTH)
        IF(${_MATMUL_INCLUDE_DIRECTORIES_PUBLIC_LENGTH} GREATER 0)
            TARGET_INCLUDE_DIRECTORIES(
                "matmul"
                PUBLIC ${_MATMUL_INCLUDE_DIRECTORIES_PUBLIC})
        ENDIF()
    ENDIF()
ENDIF()

SET(matmul_DEFINITIONS ${_MATMUL_COMPILE_DEFINITIONS_PUBLIC})
# Add '-D' to the definitions
list_add_prefix("-D" matmul_DEFINITIONS)
# Add the compile options to the definitions.
SET(matmul_INCLUDE_DIR ${_MATMUL_INCLUDE_DIRECTORY})
SET(matmul_INCLUDE_DIRS ${_MATMUL_INCLUDE_DIRECTORIES_PUBLIC})
SET(matmul_LIBRARY ${_MATMUL_LINK_LIBRARY})
SET(matmul_LIBRARIES ${_MATMUL_LINK_LIBRARIES_INTERFACE})
SET(matmul_VERSION "1.0.0")

IF(NOT TARGET "matmul")
    IF(_MATMUL_BUILD_CUDA)
        # CUDA does not work well with the much better target dependent TARGET_XXX commands but requires the settings to be available globally: https://www.cmake.org/Bug/view.php?id=14201&nbn=1
        INCLUDE_DIRECTORIES(
            ${_MATMUL_INCLUDE_DIRECTORY}
            ${_MATMUL_INCLUDE_DIRECTORIES_PRIVATE})
        SET(_MATMUL_COMPILE_DEFINITIONS_COPY ${_MATMUL_COMPILE_DEFINITIONS_PRIVATE} ${_MATMUL_COMPILE_DEFINITIONS_PUBLIC})
        list_add_prefix("-D" _MATMUL_COMPILE_DEFINITIONS_COPY)
        ADD_DEFINITIONS(
            ${_MATMUL_COMPILE_DEFINITIONS_COPY})
        UNSET(_MATMUL_COMPILE_DEFINITIONS_COPY)
        CMAKE_POLICY(SET CMP0023 OLD)   # CUDA_ADD_EXECUTABLE calls TARGET_LINK_LIBRARIES without keywords.
        CUDA_ADD_LIBRARY(
            "matmul"
            ${_MATMUL_FILES_HEADER} ${_MATMUL_FILES_SOURCE_C} ${_MATMUL_FILES_SOURCE_CXX} ${_MATMUL_FILES_SOURCE_CU} ${_MATMUL_FILES_OTHER})
    ENDIF()
ENDIF()

# Link libraries.
IF(${MATMUL_DEBUG} GREATER 0)
    MESSAGE(STATUS "_MATMUL_LINK_LIBRARIES_PRIVATE: ${_MATMUL_LINK_LIBRARIES_PRIVATE}")
ENDIF()
LIST(
    LENGTH
    _MATMUL_LINK_LIBRARIES_PRIVATE
    _MATMUL_LINK_LIBRARIES_PRIVATE_LENGTH)
IF(${_MATMUL_LINK_LIBRARIES_PRIVATE_LENGTH} GREATER 0)
    TARGET_LINK_LIBRARIES(
        "matmul"
        PRIVATE ${_MATMUL_LINK_LIBRARIES_PRIVATE})
ENDIF()

#-------------------------------------------------------------------------------
# Print the return values.
#-------------------------------------------------------------------------------
IF(${MATMUL_DEBUG} GREATER 0)
    MESSAGE(STATUS "matmul_FOUND: ${matmul_FOUND}")
    MESSAGE(STATUS "matmul_VERSION: ${matmul_VERSION}")
    MESSAGE(STATUS "matmul_DEFINITIONS: ${matmul_DEFINITIONS}")
    MESSAGE(STATUS "matmul_INCLUDE_DIR: ${matmul_INCLUDE_DIR}")
    MESSAGE(STATUS "matmul_INCLUDE_DIRS: ${matmul_INCLUDE_DIRS}")
    MESSAGE(STATUS "matmul_LIBRARY: ${matmul_LIBRARY}")
    MESSAGE(STATUS "matmul_LIBRARIES: ${matmul_LIBRARIES}")
ENDIF()

# Unset already set variables if not found.
IF(NOT _MATMUL_FOUND)
    UNSET(matmul_FOUND)
    UNSET(matmul_VERSION)
    UNSET(matmul_DEFINITIONS)
    UNSET(matmul_INCLUDE_DIR)
    UNSET(matmul_INCLUDE_DIRS)
    UNSET(matmul_LIBRARY)
    UNSET(matmul_LIBRARIES)

    UNSET(_MATMUL_FOUND)
    UNSET(_MATMUL_COMPILE_OPTIONS_C_PRIVATE)
    UNSET(_MATMUL_COMPILE_OPTIONS_CXX_PRIVATE)
    UNSET(_MATMUL_COMPILE_DEFINITIONS_PRIVATE)
    UNSET(_MATMUL_COMPILE_DEFINITIONS_PUBLIC)
    UNSET(_MATMUL_INCLUDE_DIRECTORY)
    UNSET(_MATMUL_INCLUDE_DIRECTORIES_PRIVATE)
    UNSET(_MATMUL_INCLUDE_DIRECTORIES_PUBLIC)
    UNSET(_MATMUL_LINK_LIBRARY)
    UNSET(_MATMUL_LINK_LIBRARIES_PRIVATE)
    UNSET(_MATMUL_LINK_LIBRARIES_INTERFACE)
    UNSET(_MATMUL_FILES_HEADER)
    UNSET(_MATMUL_FILES_SOURCE_CXX)
    UNSET(_MATMUL_FILES_SOURCE_CU)
    UNSET(_MATMUL_FILES_OTHER)
ELSE()
    # Make internal variables advanced options in the GUI.
    MARK_AS_ADVANCED(
        matmul_INCLUDE_DIR
        matmul_LIBRARY
        _MATMUL_COMPILE_OPTIONS_C_PRIVATE
        _MATMUL_COMPILE_OPTIONS_CXX_PRIVATE
        _MATMUL_COMPILE_DEFINITIONS_PRIVATE
        _MATMUL_COMPILE_DEFINITIONS_PUBLIC
        _MATMUL_INCLUDE_DIRECTORY
        _MATMUL_INCLUDE_DIRECTORIES_PRIVATE
        _MATMUL_INCLUDE_DIRECTORIES_PUBLIC
        _MATMUL_LINK_LIBRARY
        _MATMUL_LINK_LIBRARIES_PRIVATE
        _MATMUL_LINK_LIBRARIES_INTERFACE
        _MATMUL_FILES_HEADER
        _MATMUL_FILES_SOURCE_CXX
        _MATMUL_FILES_SOURCE_CU
        _MATMUL_FILES_OTHER)
ENDIF()

###############################################################################
# FindPackage options
###############################################################################

# Handles the REQUIRED, QUIET and version-related arguments for FIND_PACKAGE.
# NOTE: We do not check for matmul_LIBRARIES and matmul_DEFINITIONS because they can be empty.
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(
    "matmul"
    FOUND_VAR matmul_FOUND
    REQUIRED_VARS matmul_INCLUDE_DIR matmul_LIBRARY
    VERSION_VAR matmul_VERSION)
