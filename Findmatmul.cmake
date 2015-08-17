#.rst:
# Findmatmul
# ----------
#
# Matrix multiplication library.
# https://github.com/BenjaminW3/matmul
#
# Finding and Using matmul
# ^^^^^^^^^^^^^^^^^^^^^
#
# .. code-block:: cmake
#
#   FIND_PACKAGE(matmul
#     [version] [EXACT]     # Minimum or EXACT version, e.g. 1.0.0
#     [REQUIRED]            # Fail with an error if matmul or a required
#                           # component is not found
#     [QUIET]               # Do not warn if this module was not found
#     [COMPONENTS <...>]    # Compiled in components: ignored
#   )
#   TARGET_LINK_LIBRARIES(<target> PUBLIC matmul)
#
# To provide a hint to this module where to find the matmul installation,
# set the MATMUL_ROOT variable.
#
# Set the following CMake variables BEFORE calling FIND_PACKAGE to
# change the behavior of this module:
# - ``MATMUL_ELEMENT_TYPE_DOUBLE`` {ON, OFF}
# - ``MATMUL_INDEX_TYPE`` {int, size_t, ...}
# - ``MATMUL_ALIGNED_MALLOC`` {ON, OFF}
# - ``MATMUL_RETURN_COMPUTATION_TIME`` {ON, OFF}
#
# - ``MATMUL_SEQ_BLOCK_FACTOR`` {0<MATMUL_SEQ_BLOCK_FACTOR}
# - ``MATMUL_STRASSEN_CUT_OFF`` {0<MATMUL_STRASSEN_CUT_OFF}
# - ``MATMUL_STRASSEN_OMP_CUT_OFF`` {0<MATMUL_STRASSEN_OMP_CUT_OFF}
# - ``MATMUL_OMP_PRINT_NUM_CORES`` {ON, OFF}
# - ``MATMUL_OPENACC_GANG_SIZE`` {0<MATMUL_OPENACC_GANG_SIZE}
# - ``MATMUL_OPENACC_VECTOR_SIZE`` {0<MATMUL_OPENACC_VECTOR_SIZE}
# - ``MATMUL_CUDA_BLOCKSIZE`` {0<MATMUL_CUDA_BLOCKSIZE}
# - ``MATMUL_PHI_OFF_BLAS_MKL_AUTO_WORKDIVISION`` {ON, OFF}
#
# Set the following CMake variables BEFORE calling FIND_PACKAGE to
# select the versions being compiled:
# NOTE: Either MPI or CUDA device only or host timings can be activated.
# Only elements of one of the following 3 blocks can be active:
# - ``MATMUL_BUILD_SEQ_BASIC`` {ON, OFF}
# - ``MATMUL_BUILD_SEQ_SINGLE_OPTS`` {ON, OFF}
# - ``MATMUL_BUILD_SEQ_MULTIPLE_OPTS`` {ON, OFF}
# - ``MATMUL_BUILD_SEQ_STRASSEN`` {ON, OFF}
# - ``MATMUL_BUILD_PAR_OMP2`` {ON, OFF}
# - ``MATMUL_BUILD_PAR_OMP3`` {ON, OFF}
# - ``MATMUL_BUILD_PAR_OMP4`` {ON, OFF}
# - ``MATMUL_BUILD_PAR_STRASSEN_OMP2`` {ON, OFF}
# - ``MATMUL_BUILD_PAR_OPENACC`` {ON, OFF}
# - ``MATMUL_BUILD_PAR_CUDA_MEMCPY_FIXED_BLOCK_SIZE`` {ON, OFF}
# - ``MATMUL_BUILD_PAR_CUDA_MEMCPY_DYN_BLOCK_SIZE`` {ON, OFF}
# - ``MATMUL_BUILD_PAR_BLAS_CUBLAS_MEMCPY`` {ON, OFF}
# - ``MATMUL_BUILD_PAR_BLAS_MKL`` {ON, OFF}
# - ``MATMUL_BUILD_PAR_PHI_OFF_OMP2`` {ON, OFF}
# - ``MATMUL_BUILD_PAR_PHI_OFF_OMP3`` {ON, OFF}
# - ``MATMUL_BUILD_PAR_PHI_OFF_OMP4`` {ON, OFF}
# - ``MATMUL_BUILD_PAR_PHI_OFF_BLAS_MKL`` {ON, OFF}
# - ``MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_B_SEQ_T_SEQ`` {ON, OFF}
# - ``MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_B_OMP2_T_SEQ`` {ON, OFF}
# - ``MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_B_SEQ_T_OMP2`` {ON, OFF}
# - ``MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_BT_OMP4`` {ON, OFF}
# - ``MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_B_SEQ_T_FIBERS`` {ON, OFF}
# - ``MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_B_SEQ_T_THREADS`` {ON, OFF}
# - ``MATMUL_BUILD_PAR_ALPAKA_ACC_GPU_CUDA_MEMCPY`` {ON, OFF}
#
# - ``MATMUL_BUILD_PAR_ALPAKA_ACC_GPU_CUDA`` {ON, OFF}
# - ``MATMUL_BUILD_PAR_CUDA_FIXED_BLOCK_SIZE`` {ON, OFF}
# - ``MATMUL_BUILD_PAR_CUDA_DYN_BLOCK_SIZE`` {ON, OFF}
# - ``MATMUL_BUILD_PAR_BLAS_CUBLAS`` {ON, OFF}
#
# - ``MATMUL_BUILD_PAR_MPI_CANNON_STD`` {ON, OFF}
# - ``MATMUL_BUILD_PAR_MPI_CANNON_MKL`` {ON, OFF}
# - ``MATMUL_BUILD_PAR_MPI_CANNON_CUBLAS`` {ON, OFF}
# - ``MATMUL_BUILD_PAR_MPI_DNS`` {ON, OFF}
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# - ``matmul_DEFINITIONS``
#   Compiler definitions.
# - ``matmul_FOUND``
#   TRUE if matmul found a working install.
# - ``matmul_INCLUDE_DIRS``
#   Include directories for the matmul headers.
# - ``matmul_LIBRARIES``
#   matmul libraries.
# - ``matmul_VERSION``
#   Version in format Major.Minor.Patch
#
#
# IMPORTED Targets
# ^^^^^^^^^^^^^^^^
#
# This module defines the :prop_tgt:`IMPORTED` target ``matmul``, if matmul has
# been found.
#


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

FIND_PATH(_MATMUL_ROOT_DIR
  NAMES "include/matmul/matmul.h"
  HINTS "${MATMUL_ROOT}" ENV MATMUL_ROOT
  DOC "matmul ROOT location")

IF(_MATMUL_ROOT_DIR)
    INCLUDE("${_MATMUL_ROOT_DIR}/matmulConfig.cmake")
ELSE()
    MESSAGE(FATAL_ERROR "matmul could not be found!")
ENDIF()
