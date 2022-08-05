#ifndef HeterogeneousCore_AlpakaTest_plugins_alpaka_TestHelperClass_h
#define HeterogeneousCore_AlpakaTest_plugins_alpaka_TestHelperClass_h

#include "DataFormats/PortableTestObjects/interface/TestHostCollection.h"
#include "DataFormats/PortableTestObjects/interface/alpaka/TestDeviceCollection.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/DeviceEvent.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/DeviceEventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDDevicePutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESDeviceGetToken.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaTest/interface/AlpakaESTestRecords.h"
#include "HeterogeneousCore/AlpakaTest/interface/ESTestData.h"
#include "HeterogeneousCore/AlpakaTest/interface/alpaka/AlpakaESTestData.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  class TestHelperClass {
  public:
    TestHelperClass(edm::ParameterSet const& iConfig, edm::ConsumesCollector iC);

    void makeAsync(DeviceEvent const& iEvent, DeviceEventSetup const& iSetup);

    portabletest::TestHostCollection moveFrom() { return std::move(hostProduct_); }

  private:
    const EDDeviceGetToken<portabletest::TestDeviceCollection> getToken_;
    const edm::ESGetToken<cms::alpakatest::ESTestDataA, AlpakaESTestRecordA> esTokenHost_;
    const ESDeviceGetToken<AlpakaESTestDataCDevice, AlpakaESTestRecordC> esTokenDevice_;

    // hold the output product between acquire() and produce()
    portabletest::TestHostCollection hostProduct_;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
