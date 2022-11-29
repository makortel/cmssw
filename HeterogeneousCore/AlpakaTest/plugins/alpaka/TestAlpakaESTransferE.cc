#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaTest/interface/AlpakaESTestData.h"
#include "HeterogeneousCore/AlpakaTest/interface/AlpakaESTestRecords.h"
#include "HeterogeneousCore/AlpakaTest/interface/ESTestData.h"
#include "HeterogeneousCore/AlpakaTest/interface/alpaka/AlpakaESTestData.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  /**
   * This class demonstrates and ESProducer on the data model 3 that
   * - consumes a standard host ESProduct and converts the data into an Alpaka buffer
   * - transfers the buffer contents to the device of the backend
   */
  class TestAlpakaESTransferE : public ESProducer {
  public:
    TestAlpakaESTransferE(edm::ParameterSet const& iConfig) {
      auto cc = setWhatProduced(this);
      token_ = cc.consumes();
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      descriptions.addWithDefaultLabel(desc);
    }

    // TODO: in principle in this model the transfer to device could be automated
    std::optional<AlpakaESTestDataEDevice> produce(device::Record<AlpakaESTestRecordE> const& iRecord) {
      auto hostHandle = iRecord.getTransientHandle(token_);
      auto const& hostProduct = *hostHandle;
      AlpakaESTestDataEDevice deviceProduct(hostProduct->metadata().size(), iRecord.queue());
      alpaka::memcpy(iRecord.queue(), deviceProduct.buffer(), hostProduct.buffer());

      return deviceProduct;
    }

  private:
    edm::ESGetToken<AlpakaESTestDataEHost, AlpakaESTestRecordE> token_;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(TestAlpakaESTransferE);
