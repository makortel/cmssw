#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

namespace edm {
  class SwitchProducer: public global::EDProducer<> {
  public:
    explicit SwitchProducer(ParameterSet const& iConfig);
    ~SwitchProducer() override = default;

    static void fillDescriptions(ConfigurationDescriptions& descriptions);
    void produce(StreamID, Event& e, EventSetup const& c) const final;
  private:
  };

  SwitchProducer::SwitchProducer(ParameterSet const& iConfig) {
    auto const& chosenLabel = iConfig.getUntrackedParameter<std::string>("@chosen_case");
    callWhenNewProductsRegistered([=](edm::BranchDescription const& iBranch) {
        if(iBranch.moduleLabel() == chosenLabel) {
          this->consumes(edm::TypeToGet{iBranch.unwrappedTypeID(),PRODUCT_TYPE},
                         edm::InputTag{iBranch.moduleLabel(), iBranch.productInstanceName(), iBranch.processName()});
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

  void SwitchProducer::produce(StreamID, Event& e, EventSetup const& c) const {
    //throw cms::Exception("LogicError") << "SwitchProcucer::produce() should never get called.\nPlese contact a Framework developer";
    // Ok, it gets called for scheduled by Path::runNextWorkerAsync() -> WorkerInPath::runWorkerAsync() -> Worker::doWorkAsync()
    // What should we actually do then? Call the produce of the chosen EDProducer or just leave that to prefetching?
  }
}

using edm::SwitchProducer;
DEFINE_FWK_MODULE(SwitchProducer);
