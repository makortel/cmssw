#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

namespace edm {
  class SwitchProducer: public global::EDProducer<> {
  public:
    explicit SwitchProducer(ParameterSet const& iConfig);
    ~SwitchProducer() override = default;

    static void fillDescriptions(ConfigurationDescriptions& descriptions);
    void produce(StreamID, Event& e, EventSetup const& c) const final {}
  };

  SwitchProducer::SwitchProducer(ParameterSet const& iConfig) {
    auto const& chosenLabel = iConfig.getUntrackedParameter<std::string>("@chosen_case");
    callWhenNewProductsRegistered([=](edm::BranchDescription const& iBranch) {
        if(iBranch.moduleLabel() == chosenLabel) {
          // With consumes, create the connection to the chosen case EDProducer for prefetching
          this->consumes(edm::TypeToGet{iBranch.unwrappedTypeID(),PRODUCT_TYPE},
                         edm::InputTag{iBranch.moduleLabel(), iBranch.productInstanceName(), iBranch.processName()});
          // With produces, create a producer-like BranchDescription
          // early-enough for it to be flagged as non-OnDemand in case
          // the SwithcProducer is on a Path
          this->produces(iBranch.unwrappedTypeID(), iBranch.productInstanceName()).setSwitchAlias(iBranch.moduleLabel());
        }
      });
  }

  void SwitchProducer::fillDescriptions(ConfigurationDescriptions& descriptions) {
    ParameterSetDescription desc;
    desc.add<std::vector<std::string>>("@all_cases");
    desc.addUntracked<std::string>("@chosen_case");
    descriptions.addDefault(desc);
  }
}

using edm::SwitchProducer;
DEFINE_FWK_MODULE(SwitchProducer);
