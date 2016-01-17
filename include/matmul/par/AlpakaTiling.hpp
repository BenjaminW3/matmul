//-----------------------------------------------------------------------------
//! \file
//! Copyright 2013-2016 Benjamin Worpitz, Rene Widera
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

    #include "matmul/common/AlpakaHelper.hpp"

    #include <matmul/common/Mat.h>  // matmul_mat_gemm_early_out

    #include <alpaka/alpaka.hpp>

    #include <stdio.h>              // printf
    #include <math.h>               // ceil
    #include <type_traits>          // std::is_same


    template<
        typename T_Acc
    >
    struct OptimalVectorSize
    {
        using type = alpaka::dim::DimInt<1u>;
    };

#ifdef MATMUL_BUILD_PAR_ALPAKA_ACC_GPU_CUDA
    template<
        typename... T_Args
    >
    struct OptimalVectorSize<
        alpaka::acc::AccGpuCudaRt<
            T_Args...
        >
    >
    {
        using type = alpaka::dim::DimInt<2u>;
    };
#endif

#ifdef MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_B_SEQ_T_OMP2
    template<
        typename... T_Args
    >
    struct OptimalVectorSize<
        alpaka::acc::AccCpuOmp2Threads<
            T_Args...
        >
    >
    {
        using type = alpaka::dim::DimInt<128u>;
    };
#endif

#ifdef  MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_B_OMP2_T_SEQ
    template<
        typename... T_Args
    >
    struct OptimalVectorSize<
        alpaka::acc::AccCpuOmp2Blocks<
            T_Args...
        >
    >
    {
        using type = alpaka::dim::DimInt<128u>;
    };
#endif

#ifdef MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_BT_OMP4
    template<
        typename... T_Args
    >
    struct OptimalVectorSize<
        alpaka::acc::AccCpuOmp4<
            T_Args...
        >
    >
    {
        using type = alpaka::dim::DimInt<128u>;
    };
#endif

    template<
        typename T_Size
    >
    struct ElementMatMul
    {
        template<
            typename MatA,
            typename MatB,
            typename MatC
        >
        ALPAKA_FN_ACC
        void
        operator()(
            MatA const & matA,
            MatB const & matB,
            MatC & matC
        ) const
        {
            using Vec2 = typename MatA::IndexType;

            constexpr auto numElements = T_Size::value;
            for( TSize i(0); i < numElements; ++i )
                for( TSize k(0); k < numElements; ++k )
                {
                    auto const a = matA[Vec2(i,k)];
                    for( TSize j(0); j < numElements; ++j )
                    {
                            matC[Vec2(i,j)] += a * matB[Vec2(k,j)];
                    }
                }
        }
    };

    //#############################################################################
    //! An alpaka kernel implementing an adaptive tiling scheme.
    //#############################################################################
    class GemmAlpakaTiling
    {
    public:
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TAcc,
            typename TElem,
            typename MatA,
            typename MatB,
            typename MatC>
        ALPAKA_FN_ACC auto operator()(
            TAcc const & acc,
            TSize const & m, TSize const & n, TSize const & k,
            TElem const & alpha,
            MatA const & matA, TSize const & lda,
            MatB const & matB, TSize const & ldb,
            TElem const & beta,
            MatC matC, TSize const & ldc) const
        -> void
        {
            using Dim2 = alpaka::dim::DimInt<2u>;
            using Vec2 = alpaka::Vec<Dim2, TSize>;

            using VecSize = typename OptimalVectorSize<TAcc>::type;

            using Matrix = alpakaHelper::Matrix<
                alpakaHelper2::ConstPtrValue<TElem>,
                Vec2
            >;

            auto const numBlocks(alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc));
            auto const numThreads(alpaka::workdiv::getWorkDiv<alpaka::Block, alpaka::Threads>(acc));

            auto const gridBlockIdx(alpaka::idx::getIdx<alpaka::Grid, alpaka::Blocks>(acc));
            auto const blockThreadIdx(alpaka::idx::getIdx<alpaka::Block, alpaka::Threads>(acc));

            constexpr auto numWorkElemsPerDim = VecSize::value;

            Vec2 const workSize(
                numThreads[ 0 ] * numWorkElemsPerDim,
                numThreads[ 1 ] * numWorkElemsPerDim
            );

            //Shared alpakaHelperory used to store the current blocks of A and B.
            TElem * const sharedBasePointer(alpaka::block::shared::dyn::getMem<TElem>(acc));
            TElem * const sharedBasePointerB(sharedBasePointer + workSize[0] * workSize[1]);
            Matrix sharedMatA(
                sharedBasePointer,
                workSize
            );

            Matrix sharedMatB(
                sharedBasePointerB,
                workSize
            );

            using MVecNN = alpakaHelper::MathVec<
                TElem,
                VecSize
            >;

            MVecNN matDot;

            for( size_t j = 0; j < VecSize::value; ++j )
                for( size_t i = 0; i < VecSize::value; ++i ){
                    matDot[ Vec2(j,i) ] = 0;
                }

            // Loop over all blocks of A and B that are required to compute the C block.
            TSize const nBlocks(
                static_cast<TSize>(
                    alpaka::math::ceil(
                        acc,
                        static_cast<float>(k)/static_cast<float>(
                            workSize[1]
                        )
                    )
                )
            );

            TSize const currentThreadInA_y( blockThreadIdx[ 0 ] * numWorkElemsPerDim);
            TSize const currentThreadInB_x( blockThreadIdx[ 1 ] * numWorkElemsPerDim);
            // needs architecture based mapping
            TSize const offsetInA_y(
                gridBlockIdx[ 0 ] * workSize[ 0 ]

            );
            TSize const offsetInB_x(
                gridBlockIdx[ 1 ] * workSize[ 1 ]

            );

            for(TSize blockA_x = 0; blockA_x < nBlocks; ++blockA_x)
            {

                TSize const offsetA_x = blockA_x * workSize[ 1 ];
                Vec2 const globalBlockOffsetInA(
                    offsetInA_y,
                    offsetA_x
                );
                Vec2 const globalBlockOffsetInB(
                    offsetA_x,
                    offsetInB_x
                );
                //load shared A
                for( TSize i(0); i < numWorkElemsPerDim; ++i )
                {
                    for( TSize j(0); j < numWorkElemsPerDim; ++j )
                    {
                        Vec2 const offsetInTile(
                            currentThreadInA_y + i,
                            currentThreadInB_x + j
                        );
                        Vec2 const globalIdxA(offsetInTile + globalBlockOffsetInA);
                        Vec2 const globalIdxB(offsetInTile + globalBlockOffsetInB);

                        auto const isValidA = (globalIdxA[0]<matA.m_extent[0]) && (globalIdxA[1]<k);

                        auto const isValidB = (globalIdxB[0]<matB.m_extent[0]) && (globalIdxB[1]<n);

                        sharedMatA[ offsetInTile ] = isValidA ? matA[ globalIdxA ] : static_cast<TElem>(0);
                        sharedMatB[ offsetInTile ] = isValidB ? matB[ globalIdxB ] : static_cast<TElem>(0);

                    }
                }


                alpaka::block::sync::syncBlockThreads(acc);


                // move over line in A workSize
                for( TSize k3 = 0; k3 < workSize[ 0 ]; k3 +=numWorkElemsPerDim )
                {

                    Vec2 const globalIdx_A(
                        currentThreadInA_y,
                        k3
                    );
                    Vec2 const globalIdx_B(
                        k3,
                        currentThreadInB_x
                    );

                    Matrix const tmpA(
                        sharedMatA.view(
                            Vec2(
                                globalIdx_A[ 0 ],
                                globalIdx_A[ 1 ]
                            )
                        )
                    );
                    Matrix const tmpB(
                        sharedMatB.view(
                            Vec2(
                                globalIdx_B[ 0 ],
                                globalIdx_B[ 1 ]
                            )
                        )
                    );

                    ElementMatMul<VecSize> const elemMatMul;

                    elemMatMul(tmpA,tmpB,matDot);
                }

                alpaka::block::sync::syncBlockThreads(acc);

            }

            for( TSize i(0); i < numWorkElemsPerDim; ++i )
            {
                for( TSize j(0); j < numWorkElemsPerDim; ++j )
                {
                    Vec2 const offsetC(
                        offsetInA_y + currentThreadInA_y + i,
                        offsetInB_x + currentThreadInB_x + j
                    );
                    auto const isValid = (offsetC[0] < matC.m_extent[0]) && (offsetC[1] <  n);

                    if(isValid)
                        matC[ offsetC ] = alpha * matDot[ Vec2( i, j ) ] + beta * matC[ offsetC ];

                }
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
                    GemmAlpakaTiling,
                    TAcc>
                {
                    //-----------------------------------------------------------------------------
                    //! \return The size of the shared memory allocated for a block.
                    //-----------------------------------------------------------------------------
                    template<
                        typename TElem,
                        typename MatA,
                        typename MatB,
                        typename MatC
                    >
                    ALPAKA_FN_HOST static auto getBlockSharedMemDynSizeBytes(
                        alpaka::Vec<alpaka::dim::Dim<TAcc>, size::Size<TAcc>> const & blockThreadExtent,
                        alpaka::Vec<alpaka::dim::Dim<TAcc>, size::Size<TAcc>> const & threadElemExtent,
                        TSize const & m,
                        TSize const & n,
                        TSize const & k,
                        TElem const & alpha,
                        MatA const & matA,
                        TSize const & lda,
                        MatB const & matB,
                        TSize const & ldb,
                        TElem const & beta,
                        MatC matC,
                        TSize const & ldc)
                    -> size::Size<TAcc>
                    {
                        static_assert(
                            std::is_same<TSize, size::Size<TAcc>>::value,
                            "TSize and size::Size<TAcc> have to be identical!");

                        boost::ignore_unused(m);
                        boost::ignore_unused(n);
                        boost::ignore_unused(k);
                        boost::ignore_unused(alpha);
                        boost::ignore_unused(matA);
                        boost::ignore_unused(lda);
                        boost::ignore_unused(matB);
                        boost::ignore_unused(ldb);
                        boost::ignore_unused(beta);
                        boost::ignore_unused(matC);
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
#if (MATMUL_DEBUG >= MATMUL_DEBUG_FULL)
            using type = alpaka::stream::StreamCpuSync;
#else
            using type = alpaka::stream::StreamCpuAsync;
#endif
        };

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && defined(__CUDACC__)
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
    TReturn matmul_gemm_par_alpaka_tiling(
        TSize const m, TSize const n, TSize const k,
        TElem const alpha,
        TElem const * const MATMUL_RESTRICT A, TSize const lda,
        TElem const * const MATMUL_RESTRICT B, TSize const ldb,
        TElem const beta,
        TElem * const MATMUL_RESTRICT C, TSize const ldc)
    {
        using Dim2 = alpaka::dim::DimInt<2u>;
        using Vec2 = alpaka::Vec<Dim2, TSize>;

        if(matmul_mat_gemm_early_out(m, n, k, alpha, beta))
        {
            MATMUL_TIME_RETURN_EARLY_OUT;
        }

        // Select a device to execute on.
        alpaka::dev::Dev<TAcc> devAcc(
            alpaka::dev::DevMan<TAcc>::getDevByIdx(0));

        // Get a stream on this device.
        Stream<alpaka::dev::Dev<TAcc>> stream(devAcc);

        // Result matrix is MxN. We create one worker per result matrix cell.
        alpaka::Vec<Dim2, TSize> const v2uiExtentsC(
            m,
            n);

        alpaka::Vec<Dim2, TSize> const elemExtent(
            static_cast<TSize>(OptimalVectorSize<TAcc>::type::value),
            static_cast<TSize>(OptimalVectorSize<TAcc>::type::value)
        );

        // Let alpaka calculate good block and grid sizes given our full problem extents.
        alpaka::workdiv::WorkDivMembers<Dim2, TSize> const workDiv(
            alpaka::workdiv::getValidWorkDiv<TAcc>(
                devAcc,
                v2uiExtentsC,
                elemExtent,
                false,
                alpaka::workdiv::GridBlockExtentSubDivRestrictions::EqualExtent));

        // Create an instance of the kernel functor.
        TKernelFnObj kernel;

        using Matrix = alpakaHelper::Matrix<
            alpakaHelper2::ConstPtrValue<TElem>,
            Vec2
        >;

        using ConstMatrix = alpakaHelper::Matrix<
            alpakaHelper2::ConstPtrConstValue<TElem>,
            Vec2
        >;

        ConstMatrix const matA(
            A,
            Vec2(
                m,
                lda
            )
        );

        ConstMatrix const matB(
            B,
            Vec2(
                k,
                ldb
            )
        );
        Matrix matC(
            C,
            Vec2(
                m,
                ldc
            )
        );

        // Create the executor.
        auto const exec(alpaka::exec::create<TAcc>(
            workDiv,
            kernel,
            m,
            n,
            k,
            alpha,
            matA,
            lda,
            matB,
            ldb,
            beta,
            matC,
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
    TReturn matmul_gemm_par_alpaka_memcpy_tiling(
        TSize const m, TSize const n, TSize const k,
        TElem const alpha,
        TElem const * const MATMUL_RESTRICT A, TSize const lda,
        TElem const * const MATMUL_RESTRICT B, TSize const ldb,
        TElem const beta,
        TElem * const MATMUL_RESTRICT C, TSize const ldc)
    {

        using Dim2 = alpaka::dim::DimInt<2u>;
        using Vec2 = alpaka::Vec<Dim2, TSize>;

        if(matmul_mat_gemm_early_out(m, n, k, alpha, beta))
        {
            MATMUL_TIME_RETURN_EARLY_OUT;
        }

        // Get the host device.
        auto devHost(alpaka::dev::DevManCpu::getDevByIdx(0u));

        // Select a device to execute on.
        alpaka::dev::Dev<TAcc> devAcc(
            alpaka::dev::DevMan<TAcc>::getDevByIdx(0));

        // Get a stream on this device.
        Stream<alpaka::dev::Dev<TAcc>> stream(devAcc);

        alpaka::Vec<Dim2, TSize> const v2uiExtentsA(
            m,
            k);

        alpaka::Vec<Dim2, TSize> const v2uiExtentsB(
            k,
            n);

        // Result matrix is MxN. We create one worker per result matrix cell.
        alpaka::Vec<Dim2, TSize> const v2uiExtentsC(
            m,
            n);

        alpaka::Vec<Dim2, TSize> const elemExtent(
            static_cast<TSize>(OptimalVectorSize<TAcc>::type::value),
            static_cast<TSize>(OptimalVectorSize<TAcc>::type::value)
        );


        // Wrap the Pointers into memoryory buffer objects.
        using BufWrapperIn = alpaka::mem::view::ViewPlainPtr<
            std::decay<decltype(devHost)>::type,
            TElem const,
            alpaka::dim::DimInt<2u>,
            TSize>;
        BufWrapperIn bufAHost(A, devHost, v2uiExtentsA, lda * sizeof(TElem));
        BufWrapperIn bufBHost(B, devHost, v2uiExtentsB, ldb * sizeof(TElem));
        using BufWrapperOut = alpaka::mem::view::ViewPlainPtr<
            std::decay<decltype(devHost)>::type,
            TElem,
            alpaka::dim::DimInt<2u>,
            TSize>;
        BufWrapperOut bufCHost(C, devHost, v2uiExtentsC, ldc * sizeof(TElem));

        // Allocate the buffers on the accelerator and copy Host -> Acc.
        // TODO: Test if interleaved is better then alloc first, copy later.
        // Because alloc causes a device sync this may hinder the copies.
        auto bufAAcc(alpaka::mem::buf::alloc<TElem, TSize>(devAcc, v2uiExtentsA));
        alpaka::mem::view::copy(stream, bufAAcc, bufAHost, v2uiExtentsA);
        auto bufBAcc(alpaka::mem::buf::alloc<TElem, TSize>(devAcc, v2uiExtentsB));
        alpaka::mem::view::copy(stream, bufBAcc, bufBHost, v2uiExtentsB);
        auto bufCAcc(alpaka::mem::buf::alloc<TElem, TSize>(devAcc, v2uiExtentsC));
        alpaka::mem::view::copy(stream, bufCAcc, bufCHost, v2uiExtentsC);

        // Let alpaka calculate good block and grid sizes given our full problem extents.
        alpaka::workdiv::WorkDivMembers<Dim2, TSize> const workDiv(
            alpaka::workdiv::getValidWorkDiv<TAcc>(
                devAcc,
                v2uiExtentsC,
                elemExtent,
                false,
                alpaka::workdiv::GridBlockExtentSubDivRestrictions::EqualExtent));

        using Matrix = alpakaHelper::Matrix<
            alpakaHelper2::ConstPtrValue<TElem>,
            Vec2
        >;

        using ConstMatrix = alpakaHelper::Matrix<
            alpakaHelper2::ConstPtrConstValue<TElem>,
            Vec2
        >;

        ConstMatrix const matA(
            alpaka::mem::view::getPtrNative(bufAAcc),
            Vec2(
                m,
                alpaka::mem::view::getPitchBytes<1>(bufAAcc) / sizeof(TElem)
            )
        );

        ConstMatrix const matB(
            alpaka::mem::view::getPtrNative(bufBAcc),
            Vec2(
                k,
                alpaka::mem::view::getPitchBytes<1>(bufBAcc) / sizeof(TElem)
            )
        );
        Matrix matC(
            alpaka::mem::view::getPtrNative(bufCAcc),
            Vec2(
                m,
                alpaka::mem::view::getPitchBytes<1>(bufCAcc) / sizeof(TElem)
            )
        );

        // Create an instance of the kernel functor.
        TKernelFnObj kernel;

        // Create the executor.
        auto const exec(alpaka::exec::create<TAcc>(
            workDiv,
            kernel,
            m,
            n,
            k,
            alpha,
            matA,
            lda,
            matB,
            ldb,
            beta,
            matC,
            ldc));


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
        alpaka::mem::view::copy(stream, bufCHost, bufCAcc, v2uiExtentsC);

        // Wait for the stream to finish the operations.
        alpaka::wait::wait(stream);

        MATMUL_TIME_RETURN;
    }
#endif