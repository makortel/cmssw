#include "catch.hpp"

#include "HeterogeneousCore/CUDACore/interface/CUDA.h"
#include "HeterogeneousCore/CUDACore/interface/CUDAScopedContext.h"
#include "HeterogeneousCore/CUDACore/interface/CUDAToken.h"

#include "TestCUDA.h"

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
}
