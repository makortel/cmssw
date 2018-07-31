#include "catch.hpp"
#include "FWCore/TestProcessor/interface/TestProcessor.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "HeterogeneousCore/CUDACore/interface/CUDA.h"
#include "HeterogeneousCore/CUDACore/interface/CUDAToken.h"

#include "HeterogeneousCore/CUDACore/test/TestCUDA.h" // ugly...

#include <iostream>

static constexpr auto s_tag = "[TestCUDAProducerGPUFirst]";

TEST_CASE("Standard checks of TestCUDAProducerGPUFirst", s_tag) {
  const std::string baseConfig{
R"_(from FWCore.TestProcessor.TestProcess import *
process = TestProcess()
process.load("HeterogeneousCore.CUDAServices.CUDAService_cfi")
process.toTest = cms.EDProducer("TestCUDAProducerGPUFirst")
process.moduleToTest(process.toTest)
)_"
  };
  
  edm::test::TestProcessor::Config config{ baseConfig };  
  SECTION("base configuration is OK") {
    REQUIRE_NOTHROW(edm::test::TestProcessor(config));
  }
  
  SECTION("No event data") {
    edm::test::TestProcessor tester(config);
    
    REQUIRE_THROWS_AS(tester.test(), cms::Exception);
    //If the module does not throw when given no data, substitute 
    //REQUIRE_NOTHROW for REQUIRE_THROWS_AS
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

TEST_CASE("TestCUDAProducerGPUFirst operation", s_tag) {
  const std::string baseConfig{
R"_(from FWCore.TestProcessor.TestProcess import *
process = TestProcess()
process.load("HeterogeneousCore.CUDAServices.CUDAService_cfi")
process.toTest = cms.EDProducer("TestCUDAProducerGPUFirst",
    src = cms.InputTag("deviceChooser")
)
process.moduleToTest(process.toTest)
)_"
  };
  edm::test::TestProcessor::Config config{ baseConfig };  

  int deviceCount = 0;
  auto ret = cudaGetDeviceCount( &deviceCount );
  if( ret != cudaSuccess ) {
    WARN("Unable to query the CUDA capable devices from the CUDA runtime API: ("
         << ret << ") " << cudaGetErrorString( ret ) 
         << ". Ignoring tests requiring device to be present.");
    return;
  }

  auto putToken = config.produces<CUDAToken>("deviceChooser");

  constexpr int defaultDevice = 0;

  SECTION("Produce") {
    edm::test::TestProcessor tester{config};
    auto tokenPtr = std::make_unique<CUDAToken>(defaultDevice);
    auto event = tester.test(std::make_pair(putToken, std::move(tokenPtr)));
    auto prod = event.get<CUDA<float *> >();
    REQUIRE(prod->device() == defaultDevice);
    const float *data = TestCUDA::get(*prod);
    REQUIRE(data != nullptr);

    float firstElements[10];
    cuda::memory::async::copy(firstElements, data, sizeof(float)*10, prod->stream().id());

    std::cout << "Synchronizing with CUDA stream" << std::endl;
    auto stream = prod->stream();
    stream.synchronize();
    std::cout << "Synchronized" << std::endl;
    REQUIRE(firstElements[0] == 0.f);
    REQUIRE(firstElements[1] == 1.f);
    REQUIRE(firstElements[9] == 9.f);
  }
};
