#include "catch.hpp"

#include "HeterogeneousCore/CUDACore/interface/CUDA.h"
#include "HeterogeneousCore/CUDACore/interface/CUDAScopedContext.h"
#include "HeterogeneousCore/CUDACore/interface/CUDAToken.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

#include "TestCUDA.h"
#include "test_CUDAScopedContextKernels.h"

namespace {
  std::unique_ptr<CUDA<int *> > produce(const CUDAToken& token, int *d, int *h) {
    auto ctx = CUDAScopedContext(token);

    cuda::memory::async::copy(d, h, sizeof(int), ctx.stream().id());
    testCUDAScopedContextKernels_single(d, ctx.stream());
    return ctx.wrap(d);
  }
}

TEST_CASE("Use of CUDAScopedContext", "[CUDACore]") {
  int deviceCount = 0;
  auto ret = cudaGetDeviceCount( &deviceCount );
  if( ret != cudaSuccess ) {
    WARN("Unable to query the CUDA capable devices from the CUDA runtime API: ("
         << ret << ") " << cudaGetErrorString( ret ) 
         << ". Ignoring tests requiring device to be present.");
    return;
  }

  constexpr int defaultDevice = 0;
  {
    auto token = CUDAToken(defaultDevice);

    SECTION("From CUDAToken") {
      auto ctx = CUDAScopedContext(token);
      REQUIRE(cuda::device::current::get().id() == token.device());
      REQUIRE(ctx.stream().id() == token.stream().id());
    }

    SECTION("From CUDA<T>") {
      const CUDA<int> data = TestCUDA::create(10, token);

      auto ctx = CUDAScopedContext(data);
      REQUIRE(cuda::device::current::get().id() == data.device());
      REQUIRE(ctx.stream().id() == data.stream().id());
    }

    SECTION("Wrap T to CUDA<T>") {
      auto ctx = CUDAScopedContext(token);

      std::unique_ptr<CUDA<int> > dataPtr = ctx.wrap(10);
      REQUIRE(dataPtr.get() != nullptr);
      REQUIRE(dataPtr->device() == ctx.device());
      REQUIRE(dataPtr->stream().id() == ctx.stream().id());
    }

    SECTION("Storing state as CUDAContextToken") {
      CUDAContextToken ctxtok;
      { // acquire
        auto ctx = CUDAScopedContext(token);
        ctxtok = ctx.toToken();
      }

      { // produce
        auto ctx = CUDAScopedContext(std::move(ctxtok));
        REQUIRE(cuda::device::current::get().id() == token.device());
        REQUIRE(ctx.stream().id() == token.stream().id());
      }
    }

    SECTION("Joining multiple CUDA streams") {
      cuda::device::current::scoped_override_t<> setDeviceForThisScope(defaultDevice);
      auto current_device = cuda::device::current::get();

      // Mimick a producer on the second CUDA stream
      int h_a1 = 1;
      auto d_a1 = cuda::memory::device::make_unique<int>(current_device);
      auto wprod1 = produce(token, d_a1.get(), &h_a1);

      // Mimick a producer on the second CUDA stream
      auto token2 = CUDAToken(defaultDevice);
      REQUIRE(token.stream().id() != token2.stream().id());
      int h_a2 = 2;
      auto d_a2 = cuda::memory::device::make_unique<int>(current_device);
      auto wprod2 = produce(token2, d_a2.get(), &h_a2);

      // Mimick a third producer "joining" the two streams
      auto ctx = CUDAScopedContext(token);

      auto prod1 = ctx.get(*wprod1);
      auto prod2 = ctx.get(*wprod2);

      auto d_a3 = cuda::memory::device::make_unique<int>(current_device);
      testCUDAScopedContextKernels_join(prod1, prod2, d_a3.get(), ctx.stream());
      ctx.stream().synchronize();
      REQUIRE(wprod2->event().has_occurred());

      h_a1 = 0;
      h_a2 = 0;
      int h_a3 = 0;
      cuda::memory::async::copy(&h_a1, d_a1.get(), sizeof(int), ctx.stream().id());
      cuda::memory::async::copy(&h_a2, d_a2.get(), sizeof(int), ctx.stream().id());
      cuda::memory::async::copy(&h_a3, d_a3.get(), sizeof(int), ctx.stream().id());

      REQUIRE(h_a1 == 2);
      REQUIRE(h_a2 == 4);
      REQUIRE(h_a3 == 6);
    }
  }

  // Destroy and clean up all resources so that the next test can
  // assume to start from a clean state.
  cudaCheck(cudaSetDevice(defaultDevice));
  cudaCheck(cudaDeviceSynchronize());
  cudaDeviceReset();
}
