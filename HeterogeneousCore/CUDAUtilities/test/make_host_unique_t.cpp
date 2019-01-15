#include "catch.hpp"

#include "HeterogeneousCore/CUDAUtilities/interface/make_host_unique.h"

TEST_CASE("make_host_unique", "[cudaMemTools]") {
  int deviceCount = 0;
  auto ret = cudaGetDeviceCount( &deviceCount );
  if( ret != cudaSuccess ) {
    WARN("Unable to query the CUDA capable devices from the CUDA runtime API: ("
         << ret << ") " << cudaGetErrorString( ret )
         << ". Ignoring tests requiring device to be present.");
    return;
  }

  SECTION("Single element") {
    auto ptr1 = cudautils::make_host_unique<int>();
    REQUIRE(ptr1 != nullptr);
    auto ptr2 = cudautils::make_host_unique<int>(cudaHostAllocPortable | cudaHostAllocWriteCombined);
    REQUIRE(ptr2 != nullptr);
  }

  SECTION("Multiple elements") {
    auto ptr1 = cudautils::make_host_unique<int[]>(10);
    REQUIRE(ptr1 != nullptr);
    auto ptr2 = cudautils::make_host_unique<int[]>(10, cudaHostAllocPortable | cudaHostAllocWriteCombined);
    REQUIRE(ptr2 != nullptr);
  }
}
