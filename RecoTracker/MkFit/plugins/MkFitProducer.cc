#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Event.h"

class MkFitProducer: public edm::global::EDProducer<> {
public:
  explicit MkFitProducer(edm::ParameterSet const& iConfig);
  ~MkFitProducer() = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
private:
  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;
};

MkFitProducer::MkFitProducer(edm::ParameterSet const& iConfig) {
}

void MkFitProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  descriptions.add("mkFitProducer", desc);
}

void MkFitProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
}
