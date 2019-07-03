#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

class TestCUDAProducerOverhead: public edm::global::EDProducer<> {
public:
  explicit TestCUDAProducerOverhead(const edm::ParameterSet& iConfig);
  ~TestCUDAProducerOverhead() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::StreamID id, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;
private:
  edm::EDPutTokenT<int> dstToken_;
};

TestCUDAProducerOverhead::TestCUDAProducerOverhead(const edm::ParameterSet& iConfig):
  dstToken_{produces<int>()}
{}

void TestCUDAProducerOverhead::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  descriptions.addWithDefaultLabel(desc);
}

void TestCUDAProducerOverhead::produce(edm::StreamID id, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  iEvent.emplace(dstToken_, 42);
}

DEFINE_FWK_MODULE(TestCUDAProducerOverhead);
