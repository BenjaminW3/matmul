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

#if defined(MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_B_SEQ_T_SEQ) || defined(MATMUL_BUILD_PAR_ALPAKA_ACC_GPU_CUDA) || defined(MATMUL_BUILD_PAR_ALPAKA_ACC_GPU_CUDA_MEMCPY) || defined(MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_B_OMP2_T_SEQ) || defined(MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_B_SEQ_T_OMP2) || defined(MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_BT_OMP4) || defined(MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_B_SEQ_T_THREADS) || defined(MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_B_SEQ_T_FIBERS)

    #include <matmul/common/Mat.h>  // matmul_mat_gemm_early_out

    #include <alpaka/alpaka.hpp>

    #include <stdio.h>              // printf
    #include <math.h>               // ceil
    #include <type_traits>          // std::is_same

    //#############################################################################
    // This function only works for square blocks.
    //#############################################################################
    class GemmAlpakaKernel
    {
    public:
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TAcc,
            typename TElem>
        ALPAKA_FN_ACC auto operator()(
            TAcc const & acc,
            TIdx const & m, TIdx const & n, TIdx const & k,
            TElem const & alpha,
            TElem const * const MATMUL_RESTRICT A, TIdx const & lda,
            TElem const * const MATMUL_RESTRICT B, TIdx const & ldb,
            TElem const & beta,
            TElem * const MATMUL_RESTRICT C, TIdx const & ldc) const
        -> void
        {
            static_assert(alpaka::dim::Dim<TAcc>::value == 2u,
                "The accelerator used for with MatMulKernel has to be 2 dimensional!");

            // Column and row of C to calculate.
            auto const v2uiGridThreadIdx(alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc));
            TIdx const & uiGridThreadIdxX(v2uiGridThreadIdx[1u]);
            TIdx const & uiGridThreadIdxY(v2uiGridThreadIdx[0u]);

            // Column and row inside the block of C to calculate.
            auto const v2uiBlockThreadIdx(alpaka::idx::getIdx<alpaka::Block, alpaka::Threads>(acc));
            TIdx const & uiBlockThreadIdxX(v2uiBlockThreadIdx[1u]);
            TIdx const & uiBlockThreadIdxY(v2uiBlockThreadIdx[0u]);

            // The block threads extents.
            auto const v2uiBlockThreadsExtents(alpaka::workdiv::getWorkDiv<alpaka::Block, alpaka::Threads>(acc));
            TIdx const & uiBlockThreadsExtentX(v2uiBlockThreadsExtents[1u]);
            TIdx const & uiBlockThreadsExtentY(v2uiBlockThreadsExtents[0u]);
            //assert(uiBlockThreadsExtentX == uiBlockThreadsExtentY);
            TIdx const & uiBlockThreadsExtent(uiBlockThreadsExtentX);

            // Shared memory used to store the current blocks of A and B.
            TElem * const pBlockSharedA(acc.template getBlockSharedExternMem<TElem>());
            TElem * const pBlockSharedB(pBlockSharedA + uiBlockThreadsExtentX*uiBlockThreadsExtentY);

            TIdx const uiSharedBlockIdx1d(uiBlockThreadIdxY*uiBlockThreadsExtentX + uiBlockThreadIdxX);

            // If the element is outside of the matrix, write zero into the shared block.
            bool const bInsideA = (uiGridThreadIdxY < m);
            bool const bInsideB = (uiGridThreadIdxX < n);
            bool const bInsideC = (bInsideA && bInsideB);

            TElem dotProduct(0);

            // Loop over all blocks of A and B that are required to compute the C block.
            TIdx const uiBlockMulCount(
                static_cast<TIdx>(
                    alpaka::math::ceil(
                        acc,
                        static_cast<float>(k)/static_cast<float>(uiBlockThreadsExtent))));
            for(TIdx k2(0); k2<uiBlockMulCount; ++k2)
            {
                // Copy data to shared memory.
                TIdx const uiAIdxX(k2*uiBlockThreadsExtentX + uiBlockThreadIdxX);
                TIdx const uiAIdx1d(uiGridThreadIdxY*lda + uiAIdxX);
                pBlockSharedA[uiSharedBlockIdx1d] =
                    ((!bInsideA) || (uiAIdxX>=k))
                    ? static_cast<TElem>(0)
                    : A[uiAIdx1d];

                TIdx const uiBIdxY(k2*uiBlockThreadsExtentY + uiBlockThreadIdxY);
                TIdx const uiBIdx1d(uiBIdxY*ldb + uiGridThreadIdxX);
                pBlockSharedB[uiSharedBlockIdx1d] =
                    ((!bInsideB) || (uiBIdxY>=k))
                    ? static_cast<TElem>(0)
                    : B[uiBIdx1d];

                // Synchronize to make sure the sub-matrices are loaded before starting the computation.
                acc.syncBlockThreads();

                // Dyadic product within shared memory.
                for(TIdx k3(0); k3<uiBlockThreadsExtent; ++k3)
                {
                    dotProduct += pBlockSharedA[uiBlockThreadIdxY*uiBlockThreadsExtentX + k3]
                        * pBlockSharedB[k3*uiBlockThreadsExtentY + uiBlockThreadIdxX];
                }

                // Synchronize to make sure that the preceding computation is done before loading two new sub-matrices of A and B in the next iteration.
                acc.syncBlockThreads();
            }

            if(bInsideC)
            {
                TIdx const uiIdxC1d(uiGridThreadIdxY*ldc + uiGridThreadIdxX);
                C[uiIdxC1d] = alpha * dotProduct + beta * C[uiIdxC1d];
            }
        }
    };

    namespace alpaka
    {
        namespace kernel
        {
            namespace traits
            {
                //#############################################################################
                //! The trait for getting the size of the block shared extern memory for a kernel.
                //#############################################################################
                template<
                    typename TAcc>
                struct BlockSharedExternMemSizeBytes<
                    GemmAlpakaKernel,
                    TAcc>
                {
                    //-----------------------------------------------------------------------------
                    //! \return The size of the shared memory allocated for a block.
                    //-----------------------------------------------------------------------------
                    template<
                        typename TElem>
                    ALPAKA_FN_HOST static auto getBlockSharedExternMemSizeBytes(
                        alpaka::Vec<alpaka::dim::Dim<TAcc>, size::Size<TAcc>> const & vuiBlockThreadsExtents,
                        TIdx const & m,
                        TIdx const & n,
                        TIdx const & k,
                        TElem const & alpha,
                        TElem const * const A,
                        TIdx const & lda,
                        TElem const * const B,
                        TIdx const & ldb,
                        TElem const & beta,
                        TElem * const C,
                        TIdx const & ldc)
                    -> size::Size<TAcc>
                    {
                        static_assert(
                            std::is_same<TIdx, size::Size<TAcc>>::value,
                            "TIdx and size::Size<TAcc> have to be identical!");

                        boost::ignore_unused(m);
                        boost::ignore_unused(n);
                        boost::ignore_unused(k);
                        boost::ignore_unused(alpha);
                        boost::ignore_unused(A);
                        boost::ignore_unused(lda);
                        boost::ignore_unused(B);
                        boost::ignore_unused(ldb);
                        boost::ignore_unused(beta);
                        boost::ignore_unused(C);
                        boost::ignore_unused(ldc);

                        // Reserve the buffer for the two blocks of A and B.
                        return 2u * vuiBlockThreadsExtents.prod() * sizeof(TElem);
                    }
                };
            }
        }
    }

    //-----------------------------------------------------------------------------
    //
    //-----------------------------------------------------------------------------
    template<
        typename TAcc>
    void matmul_gemm_par_alpaka(
        TIdx const m, TIdx const n, TIdx const k,
        TElem const alpha,
        TElem const * const MATMUL_RESTRICT A, TIdx const lda,
        TElem const * const MATMUL_RESTRICT B, TIdx const ldb,
        TElem const beta,
        TElem * const MATMUL_RESTRICT C, TIdx const ldc)
    {
        if(matmul_mat_gemm_early_out(m, n, k, alpha, beta))
        {
            return;
        }

        // Select a device to execute on.
        alpaka::dev::Dev<TAcc> devAcc(
            alpaka::dev::DevMan<TAcc>::getDevByIdx(0));

        // Get a stream on this device.
        alpaka::stream::Stream<alpaka::dev::Dev<TAcc>> stream(
            alpaka::stream::create(devAcc));

        // Result matrix is MxN. We create one worker per result matrix cell.
        alpaka::Vec2<TIdx> const v2uiExtentsC(
            m,
            n);

        // Let alpaka calculate good block and grid sizes given our full problem extents.
        alpaka::workdiv::WorkDivMembers<alpaka::dim::DimInt<2u>, TIdx> const workDiv(
            alpaka::workdiv::getValidWorkDiv<TAcc>(
                devAcc,
                v2uiExtentsC,
                false,
                alpaka::workdiv::GridBlockExtentsSubDivRestrictions::EqualExtents));

        // Create the executor.
        auto exec(alpaka::exec::create<TAcc>(workDiv, stream));

        // Create an instance of the kernel functor.
        GemmAlpakaKernel kernel;

        // Execute the kernel.
        // NOTE: We remove the __restrict__ because alpaka calls std::ref on the arguments and std::ref errors.
        // This is most probably undefined. MSVC compiles it without any warning.
        exec(
            kernel,
            m,
            n,
            k,
            alpha,
            reinterpret_cast<TElem const *>(A),
            lda,
            reinterpret_cast<TElem const *>(B),
            ldb,
            beta,
            reinterpret_cast<TElem *>(C),
            ldc);

        // Wait for the stream to finish the memory operation.
        alpaka::wait::wait(stream);
    }

    //-----------------------------------------------------------------------------
    //
    //-----------------------------------------------------------------------------
    template<
        typename TAcc>
    void matmul_gemm_par_alpaka_memcpy(
        TIdx const m, TIdx const n, TIdx const k,
        TElem const alpha,
        TElem const * const MATMUL_RESTRICT A, TIdx const lda,
        TElem const * const MATMUL_RESTRICT B, TIdx const ldb,
        TElem const beta,
        TElem * const MATMUL_RESTRICT C, TIdx const ldc)
    {
        if(matmul_mat_gemm_early_out(m, n, k, alpha, beta))
        {
            return;
        }

        // Get the host device.
        auto devHost(alpaka::dev::cpu::getDev());

        // Select a device to execute on.
        alpaka::dev::Dev<TAcc> devAcc(
            alpaka::dev::DevMan<TAcc>::getDevByIdx(0));

        // Get a stream on this device.
        alpaka::stream::Stream<alpaka::dev::Dev<TAcc>> stream(
            alpaka::stream::create(devAcc));

        alpaka::Vec2<TIdx> const v2uiExtentsA(
            m,
            k);

        alpaka::Vec2<TIdx> const v2uiExtentsB(
            k,
            n);

        // Result matrix is MxN. We create one worker per result matrix cell.
        alpaka::Vec2<TIdx> const v2uiExtentsC(
            m,
            n);

        // Wrap the Pointers into memory buffer objects.
        using BufWrapperIn = alpaka::mem::buf::BufPlainPtrWrapper<
            std::decay<decltype(devHost)>::type,
            TElem const,
            alpaka::dim::DimInt<2u>,
            TIdx>;
        BufWrapperIn bufAHost(A, devHost, v2uiExtentsA, lda);
        BufWrapperIn bufBHost(B, devHost, v2uiExtentsB, ldb);
        using BufWrapperOut = alpaka::mem::buf::BufPlainPtrWrapper<
            std::decay<decltype(devHost)>::type,
            TElem,
            alpaka::dim::DimInt<2u>,
            TIdx>;
        BufWrapperOut bufCHost(C, devHost, v2uiExtentsC, ldc);

        // Allocate the buffers on the accelerator and copy Host -> Acc (Interleaved for better performance)
        auto bufAAcc(alpaka::mem::buf::alloc<TElem, TIdx>(devAcc, v2uiExtentsA));
        alpaka::mem::view::copy(bufAAcc, bufAHost, v2uiExtentsA, stream);
        auto bufBAcc(alpaka::mem::buf::alloc<TElem, TIdx>(devAcc, v2uiExtentsB));
        alpaka::mem::view::copy(bufBAcc, bufBHost, v2uiExtentsB, stream);
        auto bufCAcc(alpaka::mem::buf::alloc<TElem, TIdx>(devAcc, v2uiExtentsC));
        alpaka::mem::view::copy(bufCAcc, bufCHost, v2uiExtentsC, stream);

        // Let alpaka calculate good block and grid sizes given our full problem extents.
        alpaka::workdiv::WorkDivMembers<alpaka::dim::DimInt<2u>, TIdx> const workDiv(
            alpaka::workdiv::getValidWorkDiv<TAcc>(
                devAcc,
                v2uiExtentsC,
                false,
                alpaka::workdiv::GridBlockExtentsSubDivRestrictions::EqualExtents));

        // Create the executor.
        auto exec(alpaka::exec::create<TAcc>(workDiv, stream));

        // Create an instance of the kernel functor.
        GemmAlpakaKernel kernel;

        // Execute the kernel.
        exec(
            kernel,
            m,
            n,
            k,
            alpha,
            alpaka::mem::view::getPtrNative(bufAAcc),
            static_cast<TIdx>(alpaka::mem::view::getPitchBytes<1u>(bufAAcc) / sizeof(TElem)),
            alpaka::mem::view::getPtrNative(bufBAcc),
            static_cast<TIdx>(alpaka::mem::view::getPitchBytes<1u>(bufBAcc) / sizeof(TElem)),
            beta,
            alpaka::mem::view::getPtrNative(bufCAcc),
            static_cast<TIdx>(alpaka::mem::view::getPitchBytes<1u>(bufCAcc) / sizeof(TElem)));

        // Copy back the result.
        alpaka::mem::view::copy(bufCHost, bufCAcc, v2uiExtentsC, stream);

        // Wait for the stream to finish the memory operation.
        alpaka::wait::wait(stream);
    }
#endif
