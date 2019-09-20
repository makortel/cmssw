#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "SimOperations.h"

class TestCUDAProducerSimCPU: public edm::stream::EDProducer<> {
public:
  explicit TestCUDAProducerSimCPU(const edm::ParameterSet& iConfig);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;
private:
  
  std::vector<edm::EDGetTokenT<int>> srcTokens_;
  edm::EDPutTokenT<int> dstToken_;

  cudatest::SimOperations produceOps_;
};

TestCUDAProducerSimCPU::TestCUDAProducerSimCPU(const edm::ParameterSet& iConfig):
  produceOps_{iConfig.getParameter<edm::FileInPath>("config").fullPath(),
              "",
              "moduleDefinitions."+iConfig.getParameter<std::string>("@module_label")+".produce",
              1, 1.0, true}
{
  if(produceOps_.events() == 0) {
    throw cms::Exception("Configuration") << "Got 0 events, which makes this module useless";
  }

  for(const auto& src: iConfig.getParameter<std::vector<edm::InputTag>>("srcs")) {
    srcTokens_.emplace_back(consumes<int>(src));
  }

  if(iConfig.getParameter<bool>("produce")) {
    dstToken_ = produces<int>();
  }
}

void TestCUDAProducerSimCPU::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::vector<edm::InputTag>>("srcs", std::vector<edm::InputTag>{});
  desc.add<bool>("produce", false);

  desc.add<edm::FileInPath>("config", edm::FileInPath())->setComment("Path to a JSON configuration file of the simulation");

  //desc.add<bool>("useCachingAllocator", true);
  descriptions.addWithDefaultLabel(desc);
}

void TestCUDAProducerSimCPU::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // to make sure the dependencies are set correctly
  for(const auto& t: srcTokens_) {
    iEvent.get(t);
  }

  produceOps_.operate(std::vector<size_t>{iEvent.id().event() % produceOps_.events()}, nullptr);

  if(not dstToken_.isUninitialized()) {
    iEvent.emplace(dstToken_, 42);
  }
}

DEFINE_FWK_MODULE(TestCUDAProducerSimCPU);
