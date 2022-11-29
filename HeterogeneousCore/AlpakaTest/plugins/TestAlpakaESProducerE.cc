#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "HeterogeneousCore/AlpakaTest/interface/AlpakaESTestData.h"
#include "HeterogeneousCore/AlpakaTest/interface/AlpakaESTestRecords.h"
#include "HeterogeneousCore/AlpakaTest/interface/ESTestData.h"

/**
 * This class demonstrates and ESProducer on the data model 4 that
 * - consumes a standard host ESProduct and converts the data into an Alpaka buffer
 */
class TestAlpakaESProducerE : public edm::ESProducer {
public:
  TestAlpakaESProducerE(edm::ParameterSet const& iConfig) {
    auto cc = setWhatProduced(this);
    token_ = cc.consumes();
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    descriptions.addWithDefaultLabel(desc);
  }

  std::optional<cms::alpakatest::AlpakaESTestDataEHost> produce(AlpakaESTestRecordE const& iRecord) {
    auto const& input = iRecord.get(token_);

    int const size = 10;
    // Note: can not do pinned or cached allocation here, unless we
    // craft (e.g.) a polymorphic allocator (via plugin mechanism)
    // that could provide pinned(+cached) or managed memory
    // allocations for code that is link-wise independent of any device backend.
    cms::alpakatest::AlpakaESTestDataEHost product(size, cms::alpakatools::host());
    for (int i = 0; i < size; ++i) {
      product.view()[i].z() = input.value() + i;
    }
    return product;
  }

private:
  edm::ESGetToken<cms::alpakatest::ESTestDataE, AlpakaESTestRecordE> token_;
};

DEFINE_FWK_EVENTSETUP_MODULE(TestAlpakaESProducerE);
