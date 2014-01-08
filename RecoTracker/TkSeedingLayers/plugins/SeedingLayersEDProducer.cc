#include "RecoTracker/SeedingLayerSet/interface/SeedingLayerSetNew.h"

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class SeedingLayersEDProducer: public edm::EDProducer {
public:
  SeedingLayersEDProducer(const edm::ParameterSet& iConfig);
  ~SeedingLayersEDProducer();

  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;
};

SeedingLayersEDProducer::SeedingLayersEDProducer(const edm::ParameterSet& iConfig) {
  produces<SeedingLayerSetNew>();
}
SeedingLayersEDProducer::~SeedingLayersEDProducer() {}

void SeedingLayersEDProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  std::auto_ptr<SeedingLayerSetNew> prod(new SeedingLayerSetNew());
  iEvent.put(prod);
}

DEFINE_FWK_MODULE(SeedingLayersEDProducer);
