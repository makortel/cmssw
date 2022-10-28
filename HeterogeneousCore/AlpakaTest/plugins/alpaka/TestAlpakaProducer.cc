#include "DataFormats/Portable/interface/Product.h"
#include "DataFormats/PortableTestObjects/interface/alpaka/TestDeviceCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDDevicePutToken.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "TestAlgo.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class TestAlpakaProducer : public stream::EDProducer<> {
  public:
    TestAlpakaProducer(edm::ParameterSet const& config)
        : deviceToken_{produces()}, size_{config.getParameter<int32_t>("size")} {}

    void produce(DeviceEvent& event, DeviceEventSetup const&) override {
      // run the algorithm, potentially asynchronously
      portabletest::TestDeviceCollection deviceProduct{size_, event.queue()};
      algo_.fill(event.queue(), deviceProduct);

      // put the asynchronous product into the event without waiting
      event.emplace(deviceToken_, std::move(deviceProduct));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<int32_t>("size");
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    const EDDevicePutToken<portabletest::TestDeviceCollection> deviceToken_;
    const int32_t size_;

    // implementation of the algorithm
    TestAlgo algo_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(TestAlpakaProducer);
