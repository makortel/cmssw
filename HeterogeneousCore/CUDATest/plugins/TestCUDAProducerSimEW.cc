#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "HeterogeneousCore/CUDACore/interface/CUDAScopedContext.h"

#include "SimOperations.h"

class TestCUDAProducerSimEW: public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit TestCUDAProducerSimEW(const edm::ParameterSet& iConfig);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void acquire(const edm::Event& iEvent, const edm::EventSetup& iSetup, edm::WaitingTaskWithArenaHolder) override;
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;
private:
  
  std::vector<edm::EDGetTokenT<int>> srcTokens_;
  const edm::EDPutTokenT<int> dstToken_;
  CUDAContextState ctxState_;

  cudatest::SimOperations acquireOps_;
  cudatest::SimOperations produceOps_;
};

TestCUDAProducerSimEW::TestCUDAProducerSimEW(const edm::ParameterSet& iConfig):
  dstToken_{produces<int>()},
  acquireOps_{iConfig.getParameter<edm::FileInPath>("config").fullPath(), "moduleDefinitions."+iConfig.getParameter<std::string>("@module_label")+".acquire"},
  produceOps_{iConfig.getParameter<edm::FileInPath>("config").fullPath(), "moduleDefinitions."+iConfig.getParameter<std::string>("@module_label")+".produce"}
{
  if(acquireOps_.events() != produceOps_.events()) {
    throw cms::Exception("Configuration") << "Got " << acquireOps_.events() << " events for acquire and " << produceOps_.events() << " for produce";
  }
  if(acquireOps_.events() == 0) {
    throw cms::Exception("Configuration") << "Got 0 events, which makes this module useless";
  }
}

void TestCUDAProducerSimEW::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::vector<edm::InputTag>>("srcs", std::vector<edm::InputTag>{});

  desc.add<edm::FileInPath>("config", edm::FileInPath())->setComment("Path to a JSON configuration file of the simulation");

  //desc.add<bool>("useCachingAllocator", true);
  descriptions.addWithDefaultLabel(desc);
}

void TestCUDAProducerSimEW::acquire(const edm::Event& iEvent, const edm::EventSetup& iSetup, edm::WaitingTaskWithArenaHolder h) {
  // to make sure the dependencies are set correctly
  for(const auto& t: srcTokens_) {
    iEvent.get(t);
  }

  CUDAScopedContextAcquire ctx{iEvent.streamID(), std::move(h), ctxState_};

  acquireOps_.operate(std::vector<size_t>{iEvent.id().event() % acquireOps_.events()}, &ctx.stream());
}

void TestCUDAProducerSimEW::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  CUDAScopedContextProduce ctx{ctxState_};

  produceOps_.operate(std::vector<size_t>{iEvent.id().event() % produceOps_.events()}, &ctx.stream());

  iEvent.emplace(dstToken_, 42);
}

DEFINE_FWK_MODULE(TestCUDAProducerSimEW);
