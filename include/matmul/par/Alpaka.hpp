//-----------------------------------------------------------------------------
//! \file
//! Copyright 2013-2016 Benjamin Worpitz, Erik Zenker
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
    class GemmAlpakaSharedKernel
    {
    public:
        template<
            typename TAcc,
            typename TElem>
        ALPAKA_FN_ACC auto operator()(
            TAcc const & acc,
            TSize const & m, TSize const & n, TSize const & k,
            TElem const & alpha,
            TElem const * const MATMUL_RESTRICT A, TSize const & lda,
            TElem const * const MATMUL_RESTRICT B, TSize const & ldb,
            TElem const & beta,
            TElem * const MATMUL_RESTRICT C, TSize const & ldc) const
        -> void
        {
            static_assert(alpaka::dim::Dim<TAcc>::value == 2u,
                "The accelerator used for the GemmAlpakaKernel has to be 2 dimensional!");

            // Column and row of C to calculate.
            auto const gridThreadIdx(alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc));
            TSize const & gridThreadIdxX(gridThreadIdx[1u]);
            TSize const & gridThreadIdxY(gridThreadIdx[0u]);

            // Column and row inside the block of C to calculate.
            auto const blockThreadIdx(alpaka::idx::getIdx<alpaka::Block, alpaka::Threads>(acc));
            TSize const & blockThreadIdxX(blockThreadIdx[1u]);
            TSize const & blockThreadIdxY(blockThreadIdx[0u]);

            // The block threads extents.
            auto const blockThreadExtent(alpaka::workdiv::getWorkDiv<alpaka::Block, alpaka::Threads>(acc));
            TSize const & blockThreadExtentX(blockThreadExtent[1u]);
            TSize const & blockThreadExtentY(blockThreadExtent[0u]);
            //assert(blockThreadExtentX == blockThreadExtentY);
            TSize const & blockThreadExtent1d(blockThreadExtentX);

            // Shared memory used to store the current blocks of A and B.
            TElem * const pBlockSharedA(alpaka::block::shared::dyn::getMem<TElem>(acc));
            TElem * const pBlockSharedB(pBlockSharedA + blockThreadExtentX*blockThreadExtentY);

            TSize const sharedBlockIdx1d(blockThreadIdxY*blockThreadExtentX + blockThreadIdxX);

            // If the element corresponding to the current thread is outside of the respective matrix.
            bool const insideA(gridThreadIdxY < m);
            bool const insideB(gridThreadIdxX < n);
            bool const insideC(insideA && insideB);

            TElem dotProduct(0);

            // Loop over all blocks of A and B that are required to compute the C block.
            TSize const blockMulCount(
                static_cast<TSize>(
                    alpaka::math::ceil(
                        acc,
                        static_cast<float>(k)/static_cast<float>(blockThreadExtent1d))));
            for(TSize k2(0); k2<blockMulCount; ++k2)
            {
                // Copy the current blocks of A and B into shared memory in parallel.
                // If the element of the current thread is outside of the matrix, zero is written into the shared memory.
                // This is possible because zero is a result neutral extension of the matrices regarding the dot product.
                TSize const AIdxX(k2*blockThreadExtentX + blockThreadIdxX);
                TSize const AIdx1d(gridThreadIdxY*lda + AIdxX);
                pBlockSharedA[sharedBlockIdx1d] =
                    ((!insideA) || (AIdxX>=k))
                    ? static_cast<TElem>(0)
                    : A[AIdx1d];

                TSize const BIdxY(k2*blockThreadExtentY + blockThreadIdxY);
                TSize const BIdx1d(BIdxY*ldb + gridThreadIdxX);
                pBlockSharedB[sharedBlockIdx1d] =
                    ((!insideB) || (BIdxY>=k))
                    ? static_cast<TElem>(0)
                    : B[BIdx1d];

                // Synchronize to make sure the complete blocks are loaded before starting the computation.
                alpaka::block::sync::syncBlockThreads(acc);

                // Compute the dot products within shared memory.
                for(TSize k3(0); k3<blockThreadExtent1d; ++k3)
                {
                    dotProduct += pBlockSharedA[blockThreadIdxY*blockThreadExtentX + k3]
                        * pBlockSharedB[k3*blockThreadExtentY + blockThreadIdxX];
                }

                // Synchronize to make sure that the preceding computation is done before loading the next blocks of A and B.
                alpaka::block::sync::syncBlockThreads(acc);
            }

            // If the element is outside of the matrix it was only a helper thread that did not calculate any meaningful results.
            if(insideC)
            {
                TSize const CIdx1d(gridThreadIdxY*ldc + gridThreadIdxX);
                C[CIdx1d] = alpha * dotProduct + beta * C[CIdx1d];
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
                struct BlockSharedMemDynSizeBytes<
                    GemmAlpakaSharedKernel,
                    TAcc>
                {
                    //-----------------------------------------------------------------------------
                    //! \return The size of the shared memory allocated for a block.
                    //-----------------------------------------------------------------------------
                    template<
                        typename TElem>
                    ALPAKA_FN_HOST static auto getBlockSharedMemDynSizeBytes(
                        GemmAlpakaSharedKernel const & blockSharedMemDyn,
                        alpaka::Vec<alpaka::dim::Dim<TAcc>, size::Size<TAcc>> const & blockThreadExtent,
                        alpaka::Vec<alpaka::dim::Dim<TAcc>, size::Size<TAcc>> const & threadElemExtent,
                        TSize const & m,
                        TSize const & n,
                        TSize const & k,
                        TElem const & alpha,
                        TElem const * const A,
                        TSize const & lda,
                        TElem const * const B,
                        TSize const & ldb,
                        TElem const & beta,
                        TElem * const C,
                        TSize const & ldc)
                    -> size::Size<TAcc>
                    {
                        static_assert(
                            std::is_same<TSize, size::Size<TAcc>>::value,
                            "TSize and size::Size<TAcc> have to be identical!");

                        boost::ignore_unused(blockSharedMemDyn);
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
                        return 2u * blockThreadExtent.prod() * threadElemExtent.prod() * sizeof(TElem);
                    }
                };
            }
        }
    }

    namespace detail
    {
        //#############################################################################
        //! The stream type trait for the stream that should be used for the given device.
        //#############################################################################
        template<
            typename TDev,
            typename TSfinae = void>
        struct StreamType;

        //#############################################################################
        //! The stream type trait specialization for the CPU device.
        //#############################################################################
        template<>
        struct StreamType<
            alpaka::dev::DevCpu>
        {
//#if (MATMUL_DEBUG >= MATMUL_DEBUG_FULL)
            using type = alpaka::stream::StreamCpuSync;
/*#else
            using type = alpaka::stream::StreamCpuAsync;
#endif*/
        };

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
        //#############################################################################
        //! The stream type trait specialization for the CUDA device.
        //#############################################################################
        template<>
        struct StreamType<
            alpaka::dev::DevCudaRt>
        {
#if (MATMUL_DEBUG >= MATMUL_DEBUG_FULL)
            using type = alpaka::stream::StreamCudaRtSync;
#else
            using type = alpaka::stream::StreamCudaRtAsync;
#endif
        };
#endif
    }
    //#############################################################################
    //! The stream type that should be used for the given device.
    //#############################################################################
    template<
        typename TDev>
    using Stream = typename detail::StreamType<TDev>::type;

    //-----------------------------------------------------------------------------
    //
    //-----------------------------------------------------------------------------
    template<
        typename TAcc,
        typename TKernelFnObj>
    TReturn matmul_gemm_par_alpaka(
        TSize const m, TSize const n, TSize const k,
        TElem const alpha,
        TElem const * const MATMUL_RESTRICT A, TSize const lda,
        TElem const * const MATMUL_RESTRICT B, TSize const ldb,
        TElem const beta,
        TElem * const MATMUL_RESTRICT C, TSize const ldc)
    {

        using Dim2 = alpaka::dim::DimInt<2u>;
        using Vec2 = alpaka::Vec<Dim2, TSize>;

        using DevAcc = alpaka::dev::Dev<TAcc>;
        using PltfAcc = alpaka::pltf::Pltf<DevAcc>;

        if(matmul_mat_gemm_early_out(m, n, k, alpha, beta))
        {
            MATMUL_TIME_RETURN_EARLY_OUT;
        }

        // Select a device to execute on.
        auto devAcc(
            alpaka::pltf::getDevByIdx<PltfAcc>(0));

        // Get a stream on this device.
        Stream<alpaka::dev::Dev<TAcc>> stream(devAcc);

        // Result matrix is MxN. We create one worker per result matrix cell.
        Vec2 const extentC(
            m,
            n);

        Vec2 const elemExtent(
            Vec2::ones());

        // Let alpaka calculate good block and grid sizes given our full problem extents.
        alpaka::workdiv::WorkDivMembers<Dim2, TSize> const workDiv(
            alpaka::workdiv::getValidWorkDiv<TAcc>(
                devAcc,
                extentC,
                elemExtent,
                false,
                alpaka::workdiv::GridBlockExtentSubDivRestrictions::EqualExtent));

        // Create an instance of the kernel functor.
        TKernelFnObj kernel;

        // Create the executor.
        // NOTE: We remove the __restrict__ because alpaka calls std::ref on the arguments and std::ref errors.
        // This is most probably undefined. MSVC compiles it without any warning.
        auto const exec(alpaka::exec::create<TAcc>(
            workDiv,
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
            ldc));

        MATMUL_TIME_START;

        // Execute the kernel.
        alpaka::stream::enqueue(stream, exec);

        // Wait for the stream to finish the operations.
        alpaka::wait::wait(stream);

        MATMUL_TIME_END;
        MATMUL_TIME_RETURN;
    }

    //-----------------------------------------------------------------------------
    //
    //-----------------------------------------------------------------------------
    template<
        typename TAcc,
        typename TKernelFnObj>
    TReturn matmul_gemm_par_alpaka_memcpy(
        TSize const m, TSize const n, TSize const k,
        TElem const alpha,
        TElem const * const MATMUL_RESTRICT A, TSize const lda,
        TElem const * const MATMUL_RESTRICT B, TSize const ldb,
        TElem const beta,
        TElem * const MATMUL_RESTRICT C, TSize const ldc)
    {
        using Dim2 = alpaka::dim::DimInt<2u>;
        using Vec2 = alpaka::Vec<Dim2, TSize>;

        using DevAcc = alpaka::dev::Dev<TAcc>;
        using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
        using PltfHost = alpaka::pltf::PltfCpu;

        if(matmul_mat_gemm_early_out(m, n, k, alpha, beta))
        {
            MATMUL_TIME_RETURN_EARLY_OUT;
        }

        // Get the host device.
        auto devHost(
            alpaka::pltf::getDevByIdx<PltfHost>(0u));

        // Select a device to execute on.
        auto devAcc(
            alpaka::pltf::getDevByIdx<PltfAcc>(0));

        // Get a stream on this device.
        Stream<alpaka::dev::Dev<TAcc>> stream(devAcc);

        Vec2 const v2uiExtentA(
            m,
            k);

        Vec2 const v2uiExtentB(
            k,
            n);

        // Result matrix is MxN. We create one worker per result matrix cell.
        Vec2 const extentC(
            m,
            n);

        Vec2 const elemExtent(
            Vec2::all(static_cast<TSize>(1)));


        // Wrap the Pointers into memory buffer objects.
        using DevHost = std::decay<decltype(devHost)>::type;
        using BufWrapperIn = alpaka::mem::view::ViewPlainPtr<
            DevHost,
            TElem const,
            Dim2,
            TSize>;
        constexpr TSize elemSize(static_cast<TSize>(sizeof(TElem)));
        TSize const pitchBytesXAHost = lda * elemSize;
        Vec2 const pitchBytesAHost(k * pitchBytesXAHost, pitchBytesXAHost);
        BufWrapperIn bufAHost(A, devHost, v2uiExtentA, pitchBytesAHost);
        TSize const pitchBytesXBHost = ldb * elemSize;
        Vec2 const pitchBytesBHost(n * pitchBytesXBHost, pitchBytesXBHost);
        BufWrapperIn bufBHost(B, devHost, v2uiExtentB, pitchBytesBHost);
        using BufWrapperOut = alpaka::mem::view::ViewPlainPtr<
            DevHost,
            TElem,
            Dim2,
            TSize>;
        TSize const pitchBytesXCHost = ldc * elemSize;
        Vec2 const pitchBytesCHost(n * pitchBytesXCHost, pitchBytesXCHost);
        BufWrapperOut bufCHost(C, devHost, extentC, pitchBytesCHost);

        // Allocate the buffers on the accelerator and copy Host -> Acc.
        // TODO: Test if interleaved is better then alloc first, copy later.
        // Because alloc causes a device sync this may hinder the copies.
        auto bufAAcc(alpaka::mem::buf::alloc<TElem, TSize>(devAcc, v2uiExtentA));
        alpaka::mem::view::copy(stream, bufAAcc, bufAHost, v2uiExtentA);
        auto bufBAcc(alpaka::mem::buf::alloc<TElem, TSize>(devAcc, v2uiExtentB));
        alpaka::mem::view::copy(stream, bufBAcc, bufBHost, v2uiExtentB);
        auto bufCAcc(alpaka::mem::buf::alloc<TElem, TSize>(devAcc, extentC));
        alpaka::mem::view::copy(stream, bufCAcc, bufCHost, extentC);

        // Let alpaka calculate good block and grid sizes given our full problem extents.
        alpaka::workdiv::WorkDivMembers<Dim2, TSize> const workDiv(
            alpaka::workdiv::getValidWorkDiv<TAcc>(
                devAcc,
                extentC,
                elemExtent,
                false,
                alpaka::workdiv::GridBlockExtentSubDivRestrictions::EqualExtent));

        // Create an instance of the kernel functor.
        TKernelFnObj kernel;

        // Create the executor.
        // NOTE: We remove the __restrict__ because alpaka calls std::ref on the arguments and std::ref errors.
        // This is most probably undefined. MSVC compiles it without any warning.
        auto const exec(alpaka::exec::create<TAcc>(
            workDiv,
            kernel,
            m,
            n,
            k,
            alpha,
            reinterpret_cast<TElem const *>(alpaka::mem::view::getPtrNative(bufAAcc)),
            alpaka::mem::view::getPitchBytes<1>(bufAAcc) / elemSize,
            reinterpret_cast<TElem const *>(alpaka::mem::view::getPtrNative(bufBAcc)),
            alpaka::mem::view::getPitchBytes<1>(bufBAcc) / elemSize,
            beta,
            reinterpret_cast<TElem *>(alpaka::mem::view::getPtrNative(bufCAcc)),
            alpaka::mem::view::getPitchBytes<1>(bufCAcc) / elemSize));

#ifdef MATMUL_RETURN_COMPUTATION_TIME
        alpaka::wait::wait(stream);
#endif
        MATMUL_TIME_START;

        // Execute the kernel.
        alpaka::stream::enqueue(stream, exec);

#ifdef MATMUL_RETURN_COMPUTATION_TIME
        alpaka::wait::wait(stream);
#endif
        MATMUL_TIME_END;

        // Copy back the result.
        alpaka::mem::view::copy(stream, bufCHost, bufCAcc, extentC);

        // Wait for the stream to finish the operations.
        alpaka::wait::wait(stream);

        MATMUL_TIME_RETURN;
    }
#endif
