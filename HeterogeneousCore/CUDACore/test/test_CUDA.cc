#include "catch.hpp"

#include "HeterogeneousCore/CUDACore/interface/CUDA.h"
#include "HeterogeneousCore/CUDACore/interface/CUDAToken.h"

#include <cuda_runtime_api.h>

TEST_CASE("Use of CUDA template", "[CUDACore]") {
  SECTION("Default constructed") {
    auto foo = CUDA<int>();

    auto bar = std::move(foo);
  }

  int deviceCount = 0;
  auto ret = cudaGetDeviceCount( &deviceCount );
  if( ret != cudaSuccess ) {
    WARN("Unable to query the CUDA capable devices from the CUDA runtime API: ("
         << ret << ") " << cudaGetErrorString( ret ) 
         << ". Ignoring tests requiring device to be present.");
    return;
  }

  SECTION("Construct from CUDAToken") {
    constexpr int defaultDevice = 0;
    auto token = CUDAToken(defaultDevice);

    auto data = CUDA<int>(10, token);

    REQUIRE(data.device() == defaultDevice);
    REQUIRE(&data.stream() == &token.stream());
    REQUIRE(&data.event() != nullptr);
  }
}
