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

#if defined(MATMUL_BUILD_PAR_ALPAKA_ACC_GPU_CUDA)

    #include <matmul/par/Alpaka.h>

    #include <matmul/par/Alpaka.hpp>

    //-----------------------------------------------------------------------------
    //
    //-----------------------------------------------------------------------------
    template<
        typename TAcc>
    void matmul_gemm_par_alpaka_gpu(
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

        // Create the kernel functor.
        GemmAlpakaKernel kernel;

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

    //-----------------------------------------------------------------------------
    //
    //-----------------------------------------------------------------------------
    void matmul_gemm_par_alpaka_gpu_cuda(
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

        matmul_gemm_par_alpaka_gpu<alpaka::acc::AccGpuCudaRt<alpaka::dim::DimInt<2u>, TIdx>>(
            m, n, k,
            alpha,
            A, lda,
            B, ldb,
            beta,
            C, ldc);
    }
#endif
