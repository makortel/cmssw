#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "TimeCruncher.h"

class TestCUDAProducerCPUCrunch: public edm::global::EDProducer<> {
public:
  explicit TestCUDAProducerCPUCrunch(const edm::ParameterSet& iConfig);
  ~TestCUDAProducerCPUCrunch() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::StreamID id, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;
private:
  std::vector<edm::EDGetTokenT<int>> srcTokens_;
  const edm::EDPutTokenT<int> dstToken_;
  const std::chrono::microseconds crunchForMicroSeconds_;
};

TestCUDAProducerCPUCrunch::TestCUDAProducerCPUCrunch(const edm::ParameterSet& iConfig):
  dstToken_{produces<int>()},
  crunchForMicroSeconds_{static_cast<long unsigned int>(iConfig.getParameter<double>("crunchForSeconds")*1e6)}
{
  for(const auto& src: iConfig.getParameter<std::vector<edm::InputTag>>("srcs")) {
    srcTokens_.emplace_back(consumes<int>(src));
  }
  cudatest::getTimeCruncher();
}

void TestCUDAProducerCPUCrunch::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::vector<edm::InputTag>> ("srcs", std::vector<edm::InputTag>{});
  desc.add<double>("crunchForSeconds", 0);
  descriptions.addWithDefaultLabel(desc);
}

void TestCUDAProducerCPUCrunch::produce(edm::StreamID id, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  // to make sure the dependencies are set correctly
  for(const auto& t: srcTokens_) {
    iEvent.get(t);
  }

  if(crunchForMicroSeconds_.count() > 0) {
    cudatest::getTimeCruncher().crunch_for(crunchForMicroSeconds_);
  }

  iEvent.emplace(dstToken_, 42);
}

DEFINE_FWK_MODULE(TestCUDAProducerCPUCrunch);
