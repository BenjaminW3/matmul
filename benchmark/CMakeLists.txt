#
# Copyright 2014-2015 Benjamin Worpitz, Rene Widera
#
# This file is part of matmul.
#
# matmul is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# matmul is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with matmul.
# If not, see <http://www.gnu.org/licenses/>.
#
# Set the following CMake variables to change the behavior:
# - ``MATMUL_BENCHMARK_VERIFY_RESULT`` {ON, OFF}
# - ``MATMUL_BENCHMARK_REPEAT_TAKE_MINIMUM`` {ON, OFF}
#
# Set the following CMake variables to select the versions being compiled:
# NOTE: Either MPI or CUDA device only or host timings can be activated.
# So only elements of one of the following 3 blocks can be active:
# - ``MATMUL_BENCHMARK_BUILD_SEQ_BASIC`` {ON, OFF}
# - ``MATMUL_BENCHMARK_BUILD_SEQ_SINGLE_OPTS`` {ON, OFF}
# - ``MATMUL_BENCHMARK_BUILD_SEQ_MULTIPLE_OPTS`` {ON, OFF}
# - ``MATMUL_BENCHMARK_BUILD_SEQ_STRASSEN`` {ON, OFF}
# - ``MATMUL_BENCHMARK_BUILD_PAR_OMP2_GUIDED`` {ON, OFF}
# - ``MATMUL_BENCHMARK_BUILD_PAR_OMP2_STATIC`` {ON, OFF}
# - ``MATMUL_BENCHMARK_BUILD_PAR_OMP3`` {ON, OFF}
# - ``MATMUL_BENCHMARK_BUILD_PAR_OMP4`` {ON, OFF}
# - ``MATMUL_BENCHMARK_BUILD_PAR_STRASSEN_OMP2`` {ON, OFF}
# - ``MATMUL_BENCHMARK_BUILD_PAR_OPENACC`` {ON, OFF}
# - ``MATMUL_BENCHMARK_BUILD_PAR_CUDA_MEMCPY_FIXED_BLOCK_SIZE`` {ON, OFF}
# - ``MATMUL_BENCHMARK_BUILD_PAR_CUDA_MEMCPY_DYN_BLOCK_SIZE`` {ON, OFF}
# - ``MATMUL_BENCHMARK_BUILD_PAR_BLAS_CUBLAS_MEMCPY`` {ON, OFF}
# - ``MATMUL_BENCHMARK_BUILD_PAR_BLAS_MKL`` {ON, OFF}
# - ``MATMUL_BENCHMARK_BUILD_PAR_PHI_OFF_OMP2_GUIDED`` {ON, OFF}
# - ``MATMUL_BENCHMARK_BUILD_PAR_PHI_OFF_OMP2_STATIC`` {ON, OFF}
# - ``MATMUL_BENCHMARK_BUILD_PAR_PHI_OFF_OMP3`` {ON, OFF}
# - ``MATMUL_BENCHMARK_BUILD_PAR_PHI_OFF_OMP4`` {ON, OFF}
# - ``MATMUL_BENCHMARK_BUILD_PAR_PHI_OFF_BLAS_MKL`` {ON, OFF}
# - ``MATMUL_BENCHMARK_BUILD_PAR_ALPAKA_ACC_CPU_B_SEQ_T_SEQ`` {ON, OFF}
# - ``MATMUL_BENCHMARK_BUILD_PAR_ALPAKA_ACC_CPU_B_OMP2_T_SEQ`` {ON, OFF}
# - ``MATMUL_BENCHMARK_BUILD_PAR_ALPAKA_ACC_CPU_B_SEQ_T_OMP2`` {ON, OFF}
# - ``MATMUL_BENCHMARK_BUILD_PAR_ALPAKA_ACC_CPU_BT_OMP4`` {ON, OFF}
# - ``MATMUL_BENCHMARK_BUILD_PAR_ALPAKA_ACC_CPU_B_SEQ_T_FIBERS`` {ON, OFF}
# - ``MATMUL_BENCHMARK_BUILD_PAR_ALPAKA_ACC_CPU_B_SEQ_T_THREADS`` {ON, OFF}
# - ``MATMUL_BENCHMARK_BUILD_PAR_ALPAKA_ACC_GPU_CUDA_MEMCPY`` {ON, OFF}
#
# - ``MATMUL_BENCHMARK_BUILD_PAR_ALPAKA_ACC_GPU_CUDA`` {ON, OFF}
# - ``MATMUL_BENCHMARK_BUILD_PAR_CUDA_FIXED_BLOCK_SIZE`` {ON, OFF}
# - ``MATMUL_BENCHMARK_BUILD_PAR_CUDA_DYN_BLOCK_SIZE`` {ON, OFF}
# - ``MATMUL_BENCHMARK_BUILD_PAR_BLAS_CUBLAS`` {ON, OFF}
#
# - ``MATMUL_BENCHMARK_BUILD_PAR_MPI_CANNON_STD`` {ON, OFF}
# - ``MATMUL_BENCHMARK_BUILD_PAR_MPI_CANNON_MKL`` {ON, OFF}
# - ``MATMUL_BENCHMARK_BUILD_PAR_MPI_CANNON_CUBLAS`` {ON, OFF}
# - ``MATMUL_BENCHMARK_BUILD_PAR_MPI_DNS`` {ON, OFF}

################################################################################
# Required CMake version.
################################################################################

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12)

SET_PROPERTY(GLOBAL PROPERTY USE_FOLDERS ON)

################################################################################
# Project.
################################################################################

PROJECT("matmul_benchmark")

UNSET(_MATMUL_BENCHMARK_COMPILE_OPTIONS)
UNSET(_MATMUL_BENCHMARK_COMPILE_DEFINITIONS)
UNSET(_MATMUL_BENCHMARK_INCLUDE_DIR)
UNSET(_MATMUL_BENCHMARK_LINK_FLAGS)

#-------------------------------------------------------------------------------
# Options.
#-------------------------------------------------------------------------------

SET(MATMUL_BENCHMARK_ALPAKA_CUDASDK_KERNEL ON CACHE BOOL "Use the CUDA SDK kernel if alapka is used")
IF(MATMUL_BENCHMARK_ALPAKA_CUDASDK_KERNEL)
    LIST(APPEND _MATMUL_BENCHMARK_COMPILE_DEFINITIONS "MATMUL_BENCHMARK_ALPAKA_CUDASDK_KERNEL")
ENDIF()
SET(MATMUL_BENCHMARK_ALPAKA_OMPNATIVE_KERNEL ON CACHE BOOL "Use the OpenMP2 kernel if alapka is used")
IF(MATMUL_BENCHMARK_ALPAKA_OMPNATIVE_KERNEL)
    LIST(APPEND _MATMUL_BENCHMARK_COMPILE_DEFINITIONS "MATMUL_BENCHMARK_ALPAKA_OMPNATIVE_KERNEL")
ENDIF()
SET(MATMUL_BENCHMARK_ALPAKA_TILING_KERNEL ON CACHE BOOL "Use the tiling kernel if alapka is used")
IF(MATMUL_BENCHMARK_ALPAKA_TILING_KERNEL)
    LIST(APPEND _MATMUL_BENCHMARK_COMPILE_DEFINITIONS "MATMUL_BENCHMARK_ALPAKA_TILING_KERNEL")
ENDIF()

SET(MATMUL_BENCHMARK_SEQ_BASIC OFF CACHE BOOL "Enable the basic sequential GEMM")
IF(MATMUL_BENCHMARK_SEQ_BASIC)
    LIST(APPEND _MATMUL_BENCHMARK_COMPILE_DEFINITIONS "MATMUL_BENCHMARK_SEQ_BASIC")
    SET(MATMUL_BUILD_SEQ_BASIC ON CACHE BOOL "" FORCE)
ENDIF()
SET(MATMUL_BENCHMARK_SEQ_SINGLE_OPTS OFF CACHE BOOL "Enable the optimized versions of the sequential algorithm each with only one optimization")
IF(MATMUL_BENCHMARK_SEQ_SINGLE_OPTS)
    LIST(APPEND _MATMUL_BENCHMARK_COMPILE_DEFINITIONS "MATMUL_BENCHMARK_SEQ_SINGLE_OPTS")
    SET(MATMUL_BUILD_SEQ_SINGLE_OPTS ON CACHE BOOL "" FORCE)
ENDIF()
SET(MATMUL_BENCHMARK_SEQ_MULTIPLE_OPTS OFF CACHE BOOL "Enable the optimized versions of the sequential algorithm with multiple optimizations and blocking at once")
IF(MATMUL_BENCHMARK_SEQ_MULTIPLE_OPTS_BLOCK)
    LIST(APPEND _MATMUL_BENCHMARK_COMPILE_DEFINITIONS "MATMUL_BENCHMARK_SEQ_MULTIPLE_OPTS_BLOCK")
    SET(MATMUL_BUILD_SEQ_MULTIPLE_OPTS_BLOCK ON CACHE BOOL "" FORCE)
ENDIF()
SET(MATMUL_BENCHMARK_SEQ_MULTIPLE_OPTS OFF CACHE BOOL "Enable the optimized versions of the sequential algorithm with multiple optimizations at once")
IF(MATMUL_BENCHMARK_SEQ_MULTIPLE_OPTS)
    LIST(APPEND _MATMUL_BENCHMARK_COMPILE_DEFINITIONS "MATMUL_BENCHMARK_SEQ_MULTIPLE_OPTS")
    SET(MATMUL_BUILD_SEQ_MULTIPLE_OPTS ON CACHE BOOL "" FORCE)
ENDIF()
SET(MATMUL_BENCHMARK_SEQ_STRASSEN OFF CACHE BOOL "Enable the basic sequential Strassen algorithm")
IF(MATMUL_BENCHMARK_SEQ_STRASSEN)
    LIST(APPEND _MATMUL_BENCHMARK_COMPILE_DEFINITIONS "MATMUL_BENCHMARK_SEQ_STRASSEN")
    SET(MATMUL_BUILD_SEQ_STRASSEN ON CACHE BOOL "" FORCE)
ENDIF()
SET(MATMUL_BENCHMARK_PAR_OMP2_STATIC OFF CACHE BOOL "The optimized but not blocked algorithm with OpenMP 2 annotations")
IF(MATMUL_BENCHMARK_PAR_OMP2_STATIC)
    LIST(APPEND _MATMUL_BENCHMARK_COMPILE_DEFINITIONS "MATMUL_BENCHMARK_PAR_OMP2_STATIC")
    SET(MATMUL_BUILD_PAR_OMP2_STATIC ON CACHE BOOL "" FORCE)
ENDIF()
SET(MATMUL_BENCHMARK_PAR_OMP2_GUIDED OFF CACHE BOOL "The optimized but not blocked algorithm with OpenMP 2 annotations")
IF(MATMUL_BENCHMARK_PAR_OMP2_GUIDED)
    LIST(APPEND _MATMUL_BENCHMARK_COMPILE_DEFINITIONS "MATMUL_BENCHMARK_PAR_OMP2_GUIDED")
    SET(MATMUL_BUILD_PAR_OMP2_GUIDED ON CACHE BOOL "" FORCE)
ENDIF()
SET(MATMUL_BENCHMARK_PAR_OMP3 OFF CACHE BOOL "The optimized but not blocked algorithm with OpenMP 3 annotations")
IF(MATMUL_BENCHMARK_PAR_OMP3)
    LIST(APPEND _MATMUL_BENCHMARK_COMPILE_DEFINITIONS "MATMUL_BENCHMARK_PAR_OMP3")
    SET(MATMUL_BUILD_PAR_OMP3 ON CACHE BOOL "" FORCE)
ENDIF()
SET(MATMUL_BENCHMARK_PAR_OMP42 OFF CACHE BOOL "The optimized but not blocked algorithm with OpenMP 4 annotations")
IF(MATMUL_BENCHMARK_PAR_OMP4)
    LIST(APPEND _MATMUL_BENCHMARK_COMPILE_DEFINITIONS "MATMUL_BENCHMARK_PAR_OMP4")
    SET(MATMUL_BUILD_PAR_OMP4 ON CACHE BOOL "" FORCE)
ENDIF()
SET(MATMUL_BENCHMARK_PAR_STRASSEN_OMP2 OFF CACHE BOOL "The Strassen algorithm using OpenMP 2 methods for the GEMM base case and matrix addition and subtraction")
IF(MATMUL_BENCHMARK_PAR_STRASSEN_OMP2)
    LIST(APPEND _MATMUL_BENCHMARK_COMPILE_DEFINITIONS "MATMUL_BENCHMARK_PAR_STRASSEN_OMP2")
    SET(MATMUL_BUILD_PAR_STRASSEN_OMP2 ON CACHE BOOL "" FORCE)
ENDIF()
SET(MATMUL_BENCHMARK_PAR_OPENACC OFF CACHE BOOL "Enable the optimized but not blocked algorithm with OpenACC annotations")
IF(MATMUL_BENCHMARK_PAR_OPENACC)
    LIST(APPEND _MATMUL_BENCHMARK_COMPILE_DEFINITIONS "MATMUL_BENCHMARK_PAR_OPENACC")
    SET(MATMUL_BUILD_PAR_OPENACC ON CACHE BOOL "" FORCE)
ENDIF()
SET(MATMUL_BENCHMARK_PAR_CUDA_MEMCPY_FIXED_BLOCK_SIZE OFF CACHE BOOL "Enable the GEMM algorithm from the CUDA developers guide with fixed block size")
IF(MATMUL_BENCHMARK_PAR_CUDA_MEMCPY_FIXED_BLOCK_SIZE)
    LIST(APPEND _MATMUL_BENCHMARK_COMPILE_DEFINITIONS "MATMUL_BENCHMARK_PAR_CUDA_MEMCPY_FIXED_BLOCK_SIZE")
    SET(MATMUL_BUILD_PAR_CUDA_MEMCPY_FIXED_BLOCK_SIZE ON CACHE BOOL "" FORCE)
ENDIF()
SET(MATMUL_BENCHMARK_PAR_CUDA_FIXED_BLOCK_SIZE OFF CACHE BOOL "Enable the GEMM algorithm from the CUDA developers guide with fixed block size")
IF(MATMUL_BENCHMARK_PAR_CUDA_FIXED_BLOCK_SIZE)
    LIST(APPEND _MATMUL_BENCHMARK_COMPILE_DEFINITIONS "MATMUL_BENCHMARK_PAR_CUDA_FIXED_BLOCK_SIZE")
    SET(MATMUL_BUILD_PAR_CUDA_FIXED_BLOCK_SIZE ON CACHE BOOL "" FORCE)
ENDIF()
SET(MATMUL_BENCHMARK_PAR_CUDA_MEMCPY_DYN_BLOCK_SIZE OFF CACHE BOOL "Enable the GEMM algorithm from the CUDA developers guide with dynamic block size")
IF(MATMUL_BENCHMARK_PAR_CUDA_MEMCPY_DYN_BLOCK_SIZE)
    LIST(APPEND _MATMUL_BENCHMARK_COMPILE_DEFINITIONS "MATMUL_BENCHMARK_PAR_CUDA_MEMCPY_DYN_BLOCK_SIZE")
    SET(MATMUL_BUILD_PAR_CUDA_MEMCPY_DYN_BLOCK_SIZE ON CACHE BOOL "" FORCE)
ENDIF()
SET(MATMUL_BENCHMARK_PAR_CUDA_DYN_BLOCK_SIZE OFF CACHE BOOL "Enable the GEMM algorithm from the CUDA developers guide with dynamic block size")
IF(MATMUL_BENCHMARK_PAR_CUDA_DYN_BLOCK_SIZE)
    LIST(APPEND _MATMUL_BENCHMARK_COMPILE_DEFINITIONS "MATMUL_BENCHMARK_PAR_CUDA_DYN_BLOCK_SIZE")
    SET(MATMUL_BUILD_PAR_CUDA_DYN_BLOCK_SIZE ON CACHE BOOL "" FORCE)
ENDIF()
SET(MATMUL_BENCHMARK_PAR_BLAS_MKL OFF CACHE BOOL "Enable the GEMM using the Intel MKL BLAS implementation")
IF(MATMUL_BENCHMARK_PAR_BLAS_MKL)
    LIST(APPEND _MATMUL_BENCHMARK_COMPILE_DEFINITIONS "MATMUL_BENCHMARK_PAR_BLAS_MKL")
    SET(MATMUL_BUILD_PAR_BLAS_MKL ON CACHE BOOL "" FORCE)
ENDIF()
SET(MATMUL_BENCHMARK_PAR_BLAS_CUBLAS_MEMCPY OFF CACHE BOOL "Enable the GEMM using the NVIDIA cublas2 implementation.")
IF(MATMUL_BENCHMARK_PAR_BLAS_CUBLAS_MEMCPY)
    LIST(APPEND _MATMUL_BENCHMARK_COMPILE_DEFINITIONS "MATMUL_BENCHMARK_PAR_BLAS_CUBLAS_MEMCPY")
    SET(MATMUL_BUILD_PAR_BLAS_CUBLAS_MEMCPY ON CACHE BOOL "" FORCE)
ENDIF()
SET(MATMUL_BENCHMARK_PAR_BLAS_CUBLAS OFF CACHE BOOL "Enable the GEMM using the NVIDIA cublas2 implementation.")
IF(MATMUL_BENCHMARK_PAR_BLAS_CUBLAS)
    LIST(APPEND _MATMUL_BENCHMARK_COMPILE_DEFINITIONS "MATMUL_BENCHMARK_PAR_BLAS_CUBLAS")
    SET(MATMUL_BUILD_PAR_BLAS_CUBLAS ON CACHE BOOL "" FORCE)
ENDIF()
SET(MATMUL_BENCHMARK_PAR_PHI_OFF_OMP2_GUIDED OFF CACHE BOOL "Enable the offloading of the GEMM onto the xeon phi using OpenMP 2.")
IF(MATMUL_BENCHMARK_PAR_PHI_OFF_OMP2_GUIDED)
    LIST(APPEND _MATMUL_BENCHMARK_COMPILE_DEFINITIONS "MATMUL_BENCHMARK_PAR_PHI_OFF_OMP2_GUIDED")
    SET(MATMUL_BUILD_PAR_PHI_OFF_OMP2_GUIDED ON CACHE BOOL "" FORCE)
ENDIF()
SET(MATMUL_BENCHMARK_PAR_PHI_OFF_OMP2_STATIC OFF CACHE BOOL "Enable the offloading of the GEMM onto the xeon phi using OpenMP 2.")
IF(MATMUL_BENCHMARK_PAR_PHI_OFF_OMP2_STATIC)
    LIST(APPEND _MATMUL_BENCHMARK_COMPILE_DEFINITIONS "MATMUL_BENCHMARK_PAR_PHI_OFF_OMP2_STATIC")
    SET(MATMUL_BUILD_PAR_PHI_OFF_OMP2_STATIC ON CACHE BOOL "" FORCE)
ENDIF()
SET(MATMUL_BENCHMARK_PAR_PHI_OFF_OMP3 OFF CACHE BOOL "Enable the offloading of the GEMM onto the xeon phi using OpenMP 3.")
IF(MATMUL_BENCHMARK_PAR_PHI_OFF_OMP3)
    LIST(APPEND _MATMUL_BENCHMARK_COMPILE_DEFINITIONS "MATMUL_BENCHMARK_PAR_PHI_OFF_OMP3")
    SET(MATMUL_BUILD_PAR_PHI_OFF_OMP3 ON CACHE BOOL "" FORCE)
ENDIF()
SET(MATMUL_BENCHMARK_PAR_PHI_OFF_OMP4 OFF CACHE BOOL "Enable the offloading of the GEMM onto the xeon phi using OpenMP 4.")
IF(MATMUL_BENCHMARK_PAR_PHI_OFF_OMP4)
    LIST(APPEND _MATMUL_BENCHMARK_COMPILE_DEFINITIONS "MATMUL_BENCHMARK_PAR_PHI_OFF_OMP4")
    SET(MATMUL_BUILD_PAR_PHI_OFF_OMP4 ON CACHE BOOL "" FORCE)
ENDIF()
SET(MATMUL_BENCHMARK_PAR_PHI_OFF_BLAS_MKL OFF CACHE BOOL "Enable the offloading of the GEMM onto the xeon phi using the Intel MKL BLAS implementation.")
IF(MATMUL_BENCHMARK_PAR_PHI_OFF_BLAS_MKL)
    LIST(APPEND _MATMUL_BENCHMARK_COMPILE_DEFINITIONS "MATMUL_BENCHMARK_PAR_PHI_OFF_BLAS_MKL")
    SET(MATMUL_BUILD_PAR_PHI_OFF_BLAS_MKL ON CACHE BOOL "" FORCE)
ENDIF()

SET(MATMUL_BENCHMARK_PAR_ALPAKA_ACC_CPU_B_SEQ_T_SEQ OFF CACHE BOOL "Enable the GEMM using the alpaka serial accelerator back-end on the CPU")
IF(MATMUL_BENCHMARK_PAR_ALPAKA_ACC_CPU_B_SEQ_T_SEQ)
    LIST(APPEND _MATMUL_BENCHMARK_COMPILE_DEFINITIONS "MATMUL_BENCHMARK_PAR_ALPAKA_ACC_CPU_B_SEQ_T_SEQ")
    SET(MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_B_SEQ_T_SEQ ON CACHE BOOL "" FORCE)
ENDIF()
SET(MATMUL_BENCHMARK_PAR_ALPAKA_ACC_CPU_B_OMP2_T_SEQ OFF CACHE BOOL "Enable the GEMM using the alpaka OpenMP 2.0 grid block accelerator back-end on the CPU")
IF(MATMUL_BENCHMARK_PAR_ALPAKA_ACC_CPU_B_OMP2_T_SEQ)
    LIST(APPEND _MATMUL_BENCHMARK_COMPILE_DEFINITIONS "MATMUL_BENCHMARK_PAR_ALPAKA_ACC_CPU_B_OMP2_T_SEQ")
    SET(MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_B_OMP2_T_SEQ ON CACHE BOOL "" FORCE)
ENDIF()
SET(MATMUL_BENCHMARK_PAR_ALPAKA_ACC_CPU_B_SEQ_T_OMP2 OFF CACHE BOOL "Enable the GEMM using the alpaka OpenMP 2.0 block thread accelerator back-end on the CPU")
IF(MATMUL_BENCHMARK_PAR_ALPAKA_ACC_CPU_B_SEQ_T_OMP2)
    LIST(APPEND _MATMUL_BENCHMARK_COMPILE_DEFINITIONS "MATMUL_BENCHMARK_PAR_ALPAKA_ACC_CPU_B_SEQ_T_OMP2")
    SET(MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_B_SEQ_T_OMP2 ON CACHE BOOL "" FORCE)
ENDIF()
SET(MATMUL_BENCHMARK_PAR_ALPAKA_ACC_CPU_B_SEQ_T_THREADS OFF CACHE BOOL "Enable the GEMM using the alpaka std::thread accelerator back-end on the CPU")
IF(MATMUL_BENCHMARK_PAR_ALPAKA_ACC_CPU_B_SEQ_T_THREADS)
    LIST(APPEND _MATMUL_BENCHMARK_COMPILE_DEFINITIONS "MATMUL_BENCHMARK_PAR_ALPAKA_ACC_CPU_B_SEQ_T_THREADS")
    SET(MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_B_SEQ_T_THREADS ON CACHE BOOL "" FORCE)
ENDIF()
SET(MATMUL_BENCHMARK_PAR_ALPAKA_ACC_CPU_B_SEQ_T_FIBERS OFF CACHE BOOL "Enable the GEMM using the alpaka Boost.Fiber accelerator back-end on the CPU")
IF(MATMUL_BENCHMARK_PAR_ALPAKA_ACC_CPU_B_SEQ_T_FIBERS)
    LIST(APPEND _MATMUL_BENCHMARK_COMPILE_DEFINITIONS "MATMUL_BENCHMARK_PAR_ALPAKA_ACC_CPU_B_SEQ_T_FIBERS")
    SET(MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_B_SEQ_T_FIBERS ON CACHE BOOL "" FORCE)
ENDIF()
SET(MATMUL_BENCHMARK_PAR_ALPAKA_ACC_CPU_BT_OMP4 OFF CACHE BOOL "Enable the GEMM using the alpaka OpenMP 4.0 accelerator back-end on the CPU")
IF(MATMUL_BENCHMARK_PAR_ALPAKA_ACC_CPU_BT_OMP4)
    LIST(APPEND _MATMUL_BENCHMARK_COMPILE_DEFINITIONS "MATMUL_BENCHMARK_PAR_ALPAKA_ACC_CPU_BT_OMP4")
    SET(MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_BT_OMP4 ON CACHE BOOL "" FORCE)
ENDIF()
SET(MATMUL_BENCHMARK_PAR_ALPAKA_ACC_GPU_CUDA_MEMCPY OFF CACHE BOOL "Enable the GEMM using the alpaka CUDA accelerator")
IF(MATMUL_BENCHMARK_PAR_ALPAKA_ACC_GPU_CUDA_MEMCPY)
    LIST(APPEND _MATMUL_BENCHMARK_COMPILE_DEFINITIONS "MATMUL_BENCHMARK_PAR_ALPAKA_ACC_GPU_CUDA_MEMCPY")
    SET(MATMUL_BUILD_PAR_ALPAKA_ACC_GPU_CUDA_MEMCPY ON CACHE BOOL "" FORCE)
ENDIF()
SET(MATMUL_BENCHMARK_PAR_ALPAKA_ACC_GPU_CUDA OFF CACHE BOOL "Enable the GEMM using the alpaka CUDA accelerator")
IF(MATMUL_BENCHMARK_PAR_ALPAKA_ACC_GPU_CUDA)
    LIST(APPEND _MATMUL_BENCHMARK_COMPILE_DEFINITIONS "MATMUL_BENCHMARK_PAR_ALPAKA_ACC_GPU_CUDA")
    SET(MATMUL_BUILD_PAR_ALPAKA_ACC_GPU_CUDA ON CACHE BOOL "" FORCE)
ENDIF()

SET(MATMUL_BENCHMARK_PAR_MPI_CANNON_STD OFF CACHE BOOL "Enable the distributed GEMM using the cannon algorithm, MPI and the optimized sequential implementation on each node.")
IF(MATMUL_BENCHMARK_PAR_MPI_CANNON_STD)
    LIST(APPEND _MATMUL_BENCHMARK_COMPILE_DEFINITIONS "MATMUL_BENCHMARK_PAR_MPI_CANNON_STD")
    SET(MATMUL_BUILD_PAR_MPI_CANNON_STD ON CACHE BOOL "" FORCE)
ENDIF()
SET(MATMUL_BENCHMARK_PAR_MPI_CANNON_MKL OFF CACHE BOOL "Enable the distributed GEMM using the cannon algorithm, MPI and the Intel MKL on each node.")
IF(MATMUL_BENCHMARK_PAR_MPI_CANNON_MKL)
    LIST(APPEND _MATMUL_BENCHMARK_COMPILE_DEFINITIONS "MATMUL_BENCHMARK_PAR_MPI_CANNON_MKL")
    SET(MATMUL_BUILD_PAR_MPI_CANNON_MKL ON CACHE BOOL "" FORCE)
ENDIF()
SET(MATMUL_BENCHMARK_PAR_MPI_CANNON_CUBLAS OFF CACHE BOOL "Enable the distributed GEMM using the cannon algorithm, MPI and the cuBLAS on each node.")
IF(MATMUL_BENCHMARK_PAR_MPI_CANNON_CUBLAS)
    LIST(APPEND _MATMUL_BENCHMARK_COMPILE_DEFINITIONS "MATMUL_BENCHMARK_PAR_MPI_CANNON_CUBLAS")
    SET(MATMUL_BUILD_PAR_MPI_CANNON_CUBLAS ON CACHE BOOL "" FORCE)
ENDIF()
SET(MATMUL_BENCHMARK_PAR_MPI_DNS OFF CACHE BOOL "Enable the distributed GEMM using the DNS algorithm, MPI and the optimized sequential implementation on each node")
IF(MATMUL_BENCHMARK_PAR_MPI_DNS)
    LIST(APPEND _MATMUL_BENCHMARK_COMPILE_DEFINITIONS "MATMUL_BENCHMARK_PAR_MPI_DNS")
    SET(MATMUL_BUILD_PAR_MPI_DNS ON CACHE BOOL "" FORCE)
ENDIF()

#-------------------------------------------------------------------------------
# MPI vs Host vs CUDA.
#-------------------------------------------------------------------------------
IF(MATMUL_BENCHMARK_PAR_MPI_CANNON_STD
    OR MATMUL_BENCHMARK_PAR_MPI_CANNON_MKL
    OR MATMUL_BENCHMARK_PAR_MPI_CANNON_CUBLAS
    OR MATMUL_BENCHMARK_PAR_MPI_DNS)

    SET(_MATMUL_BENCHMARK_MPI TRUE)
ENDIF()
IF(MATMUL_BENCHMARK_SEQ_BASIC
    OR MATMUL_BENCHMARK_SEQ_SINGLE_OPTS
    OR MATMUL_BENCHMARK_SEQ_MULTIPLE_OPTS_BLOCK
    OR MATMUL_BENCHMARK_SEQ_MULTIPLE_OPTS
    OR MATMUL_BENCHMARK_SEQ_STRASSEN
    OR MATMUL_BENCHMARK_PAR_OMP2_GUIDED
    OR MATMUL_BENCHMARK_PAR_OMP2_STATIC
    OR MATMUL_BENCHMARK_PAR_OMP3
    OR MATMUL_BENCHMARK_PAR_OMP4
    OR MATMUL_BENCHMARK_PAR_STRASSEN_OMP2
    OR MATMUL_BENCHMARK_PAR_OPENACC
    OR MATMUL_BENCHMARK_PAR_CUDA_MEMCPY_FIXED_BLOCK_SIZE
    OR MATMUL_BENCHMARK_PAR_CUDA_MEMCPY_DYN_BLOCK_SIZE
    OR MATMUL_BENCHMARK_PAR_BLAS_MKL
    OR MATMUL_BENCHMARK_PAR_BLAS_CUBLAS_MEMCPY
    OR MATMUL_BENCHMARK_PAR_PHI_OFF_OMP2_GUIDED
    OR MATMUL_BENCHMARK_PAR_PHI_OFF_OMP2_STATIC
    OR MATMUL_BENCHMARK_PAR_PHI_OFF_OMP3
    OR MATMUL_BENCHMARK_PAR_PHI_OFF_OMP4
    OR MATMUL_BENCHMARK_PAR_PHI_OFF_BLAS_MKL
    OR MATMUL_BENCHMARK_PAR_ALPAKA_ACC_CPU_B_OMP2_T_SEQ
    OR MATMUL_BENCHMARK_PAR_ALPAKA_ACC_CPU_B_SEQ_T_OMP2
    OR MATMUL_BENCHMARK_PAR_ALPAKA_ACC_CPU_BT_OMP4
    OR MATMUL_BENCHMARK_PAR_ALPAKA_ACC_CPU_B_SEQ_T_FIBERS
    OR MATMUL_BENCHMARK_PAR_ALPAKA_ACC_CPU_B_SEQ_T_THREADS
    OR MATMUL_BENCHMARK_PAR_ALPAKA_ACC_GPU_CUDA_MEMCPY)

    SET(_MATMUL_BENCHMARK_SINGLE_NODE_HOST TRUE)
ENDIF()
IF(MATMUL_BENCHMARK_PAR_CUDA_FIXED_BLOCK_SIZE
    OR MATMUL_BENCHMARK_PAR_CUDA_DYN_BLOCK_SIZE
    OR MATMUL_BENCHMARK_PAR_BLAS_CUBLAS
    OR MATMUL_BENCHMARK_PAR_ALPAKA_ACC_GPU_CUDA)

    SET(_MATMUL_BENCHMARK_SINGLE_NODE_CUDA TRUE)
    LIST(APPEND _MATMUL_BENCHMARK_COMPILE_DEFINITIONS "MATMUL_BENCHMARK_CUDA_NO_COPY")
ENDIF()

IF((_MATMUL_BENCHMARK_SINGLE_NODE_HOST AND _MATMUL_BENCHMARK_MPI) OR (_MATMUL_BENCHMARK_SINGLE_NODE_CUDA AND _MATMUL_BENCHMARK_MPI))
    MESSAGE(FATAL_ERROR "If MPI tests are executed, no other benchmarks can be executed by the same executable!")
ENDIF()
IF(_MATMUL_BENCHMARK_SINGLE_NODE_CUDA AND _MATMUL_BENCHMARK_SINGLE_NODE_HOST)
    MESSAGE(FATAL_ERROR "If GPU tests without memcopy are executed, no other benchmarks can be executed by the same executable!")
ENDIF()

#-------------------------------------------------------------------------------
# Measurement settings.
#-------------------------------------------------------------------------------
SET(MATMUL_BENCHMARK_VERIFY_RESULT OFF CACHE BOOL "The result of a computation will be compared with the result of the standard sequential algorithm.")
IF(MATMUL_BENCHMARK_VERIFY_RESULT)
    LIST(APPEND _MATMUL_BENCHMARK_COMPILE_DEFINITIONS "MATMUL_BENCHMARK_VERIFY_RESULT")
    SET(MATMUL_BUILD_SEQ_BASIC ON CACHE BOOL "" FORCE)
ENDIF()
SET(MATMUL_BENCHMARK_PRINT_GFLOPS OFF CACHE BOOL "If the GFLOPS should be printed instead if the time.")
IF(MATMUL_BENCHMARK_PRINT_GFLOPS)
    LIST(APPEND _MATMUL_BENCHMARK_COMPILE_DEFINITIONS "MATMUL_BENCHMARK_PRINT_GFLOPS")
ENDIF()
SET(MATMUL_BENCHMARK_REPEAT_TAKE_MINIMUM ON CACHE BOOL "If this is defined the minimum of all repetitions is returned instead of the average.")
IF(MATMUL_BENCHMARK_REPEAT_TAKE_MINIMUM)
    LIST(APPEND _MATMUL_BENCHMARK_COMPILE_DEFINITIONS "MATMUL_BENCHMARK_REPEAT_TAKE_MINIMUM")
ENDIF()
SET(MATMUL_BENCHMARK_PRINT_ITERATIONS OFF CACHE BOOL "If the current iteration number should be printed.")
IF(MATMUL_BENCHMARK_PRINT_ITERATIONS)
    LIST(APPEND _MATMUL_BENCHMARK_COMPILE_DEFINITIONS "MATMUL_BENCHMARK_PRINT_ITERATIONS")
ENDIF()
SET(MATMUL_BENCHMARK_PRINT_MATRICES OFF CACHE BOOL "If the matrices (in and out) should be printed.")
IF(MATMUL_BENCHMARK_PRINT_MATRICES)
    LIST(APPEND _MATMUL_BENCHMARK_COMPILE_DEFINITIONS "MATMUL_BENCHMARK_PRINT_MATRICES")
ENDIF()
SET(MATMUL_BENCHMARK_COMPUTATION_TIME OFF CACHE BOOL "If this is defined, the time needed for the computation itself omitting initialization and shutdown (if possible) is measured.")
IF(MATMUL_BENCHMARK_COMPUTATION_TIME)
    LIST(APPEND _MATMUL_BENCHMARK_COMPILE_DEFINITIONS "MATMUL_BENCHMARK_COMPUTATION_TIME")
    SET(MATMUL_RETURN_COMPUTATION_TIME ON CACHE BOOL "" FORCE)
ENDIF()

IF(NOT MSVC)
    LIST(APPEND _MATMUL_BENCHMARK_COMPILE_OPTIONS "-std=c99")
ENDIF()

#-------------------------------------------------------------------------------
# Find OpenMP.
#-------------------------------------------------------------------------------
IF(MATMUL_BENCHMARK_PAR_OMP2_GUIDED OR MATMUL_BENCHMARK_PAR_OMP2_STATIC OR MATMUL_BENCHMARK_PAR_OMP3 OR MATMUL_BENCHMARK_PAR_OMP4 OR MATMUL_BENCHMARK_PAR_STRASSEN_OMP2 OR MATMUL_BENCHMARK_PAR_PHI_OFF_OMP2_GUIDED OR MATMUL_BENCHMARK_PAR_PHI_OFF_OMP2_STATIC OR MATMUL_BENCHMARK_PAR_PHI_OFF_OMP3 OR MATMUL_BENCHMARK_PAR_PHI_OFF_OMP4 OR MATMUL_BENCHMARK_PAR_ALPAKA_ACC_CPU_B_OMP2_T_SEQ OR MATMUL_BENCHMARK_PAR_ALPAKA_ACC_CPU_B_SEQ_T_OMP2 OR MATMUL_BENCHMARK_PAR_ALPAKA_ACC_CPU_BT_OMP4)
    FIND_PACKAGE(OpenMP)
    IF(NOT OPENMP_FOUND)
        MESSAGE(ERROR "benchmark dependency OpenMP could not be found!")

    ELSE()
        LIST(APPEND _MATMUL_BENCHMARK_COMPILE_OPTIONS ${OpenMP_C_FLAGS})
    ENDIF()
ENDIF()

#-------------------------------------------------------------------------------
# Find matmul.
#-------------------------------------------------------------------------------

SET(MATMUL_ROOT "${CMAKE_CURRENT_LIST_DIR}/../" CACHE STRING  "The location of the matmul library")

LIST(APPEND CMAKE_MODULE_PATH "${MATMUL_ROOT}")
FIND_PACKAGE("matmul" REQUIRED)

#-------------------------------------------------------------------------------
# Common.
#-------------------------------------------------------------------------------

INCLUDE("${MATMUL_ROOT}/cmake/common.cmake")

#-------------------------------------------------------------------------------
# Add executable.
#-------------------------------------------------------------------------------

SET(_MATMUL_BENCHMARK_INCLUDE_DIR "include/")
SET(_MATMUL_BENCHMARK_SUFFIXED_INCLUDE_DIR "${_MATMUL_BENCHMARK_INCLUDE_DIR}benchmark/")
SET(_MATMUL_BENCHMARK_SOURCE_DIR "src/")

# Add all the include files in all recursive subdirectories and group them accordingly.
append_recursive_files_add_to_src_group("${_MATMUL_BENCHMARK_SUFFIXED_INCLUDE_DIR}" "" "h" _MATMUL_BENCHMARK_FILES_HEADER)

# Add all the source files in all recursive subdirectories and group them accordingly.
append_recursive_files_add_to_src_group("${_MATMUL_BENCHMARK_SOURCE_DIR}" "" "c" _MATMUL_BENCHMARK_FILES_SOURCE)

ADD_EXECUTABLE(
    "matmul_benchmark"
    ${_MATMUL_BENCHMARK_FILES_HEADER} ${_MATMUL_BENCHMARK_FILES_SOURCE})
TARGET_COMPILE_OPTIONS(
    "matmul_benchmark"
    PRIVATE ${_MATMUL_BENCHMARK_COMPILE_OPTIONS})
TARGET_COMPILE_DEFINITIONS(
    "matmul_benchmark"
    PRIVATE ${_MATMUL_BENCHMARK_COMPILE_DEFINITIONS})
TARGET_INCLUDE_DIRECTORIES(
    "matmul_benchmark"
    PUBLIC ${_MATMUL_BENCHMARK_INCLUDE_DIR})
TARGET_LINK_LIBRARIES(
    "matmul_benchmark"
    PRIVATE "matmul"
    ${_MATMUL_BENCHMARK_LINK_FLAGS})
