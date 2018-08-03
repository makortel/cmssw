#include "catch.hpp"
#include "FWCore/TestProcessor/interface/TestProcessor.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "HeterogeneousCore/CUDACore/interface/CUDAToken.h"

#include <cuda_runtime_api.h>

static constexpr auto s_tag = "[CUDADeviceChooserFilter]";

TEST_CASE("Standard checks of CUDADeviceProducer", s_tag) {
  const std::string baseConfig{
R"_(from FWCore.TestProcessor.TestProcess import *
process = TestProcess()
process.load("HeterogeneousCore.CUDAServices.CUDAService_cfi")
process.toTest = cms.EDProducer("CUDADeviceChooserProducer")
process.moduleToTest(process.toTest)
)_"
  };

  int deviceCount = 0;
  auto ret = cudaGetDeviceCount( &deviceCount );
  if( ret != cudaSuccess ) {
    WARN("Unable to query the CUDA capable devices from the CUDA runtime API: ("
         << ret << ") " << cudaGetErrorString( ret ) 
         << ". Ignoring tests requiring device to be present.");
    return;
  }
  
  edm::test::TestProcessor::Config config{ baseConfig };  
  SECTION("base configuration is OK") {
    REQUIRE_NOTHROW(edm::test::TestProcessor(config));
  }
  
  SECTION("No event data") {
    edm::test::TestProcessor tester(config);
    
    REQUIRE_NOTHROW(tester.test());
  }
  
  SECTION("beginJob and endJob only") {
    edm::test::TestProcessor tester(config);
    
    REQUIRE_NOTHROW(tester.testBeginAndEndJobOnly());
  }

  SECTION("Run with no LuminosityBlocks") {
    edm::test::TestProcessor tester(config);
    
    REQUIRE_NOTHROW(tester.testRunWithNoLuminosityBlocks());
  }

  SECTION("LuminosityBlock with no Events") {
    edm::test::TestProcessor tester(config);
    
    REQUIRE_NOTHROW(tester.testLuminosityBlockWithNoEvents());
  }

}

TEST_CASE("CUDAService enabled", s_tag) {
  const std::string config{
R"_(from FWCore.TestProcessor.TestProcess import *
process = TestProcess()
process.load("HeterogeneousCore.CUDAServices.CUDAService_cfi")
process.toTest = cms.EDProducer("CUDADeviceChooserProducer")
process.moduleToTest(process.toTest)
)_"
  };

  int deviceCount = 0;
  auto ret = cudaGetDeviceCount( &deviceCount );
  if( ret != cudaSuccess ) {
    WARN("Unable to query the CUDA capable devices from the CUDA runtime API: ("
         << ret << ") " << cudaGetErrorString( ret ) 
         << ". Ignoring tests requiring device to be present.");
    return;
  }

  SECTION("CUDAToken") {
    edm::test::TestProcessor tester{config};
    auto event = tester.test();
    
    REQUIRE(event.get<CUDAToken>()->device() >= 0);
    REQUIRE(event.get<CUDAToken>()->stream().id() != nullptr);
  }
}

TEST_CASE("CUDAService disabled", s_tag) {
  const std::string config{
R"_(from FWCore.TestProcessor.TestProcess import *
process = TestProcess()
process.load("HeterogeneousCore.CUDAServices.CUDAService_cfi")
process.CUDAService.enabled = False
process.toTest = cms.EDProducer("CUDADeviceChooserProducer")
process.moduleToTest(process.toTest)
)_"
  };

  SECTION("Construction") {
    REQUIRE_THROWS_AS(edm::test::TestProcessor{config}, cms::Exception);
  }
}
