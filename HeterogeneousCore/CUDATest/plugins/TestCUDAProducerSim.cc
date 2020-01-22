#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CUDADataFormats/Common/interface/Product.h"
#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"

#include "SimOperationsService.h"

class TestCUDAProducerSim : public edm::stream::EDProducer<> {
public:
  explicit TestCUDAProducerSim(const edm::ParameterSet& iConfig);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

private:
  std::vector<edm::EDGetTokenT<int>> srcTokens_;
  std::vector<edm::EDGetTokenT<cms::cuda::Product<int>>> cudaSrcTokens_;
  edm::EDPutTokenT<int> dstToken_;
  edm::EDPutTokenT<cms::cuda::Product<int>> cudaDstToken_;

  SimOperationsService::ProduceCPUProcessor produceOpsCPU_;
  SimOperationsService::ProduceGPUProcessor produceOpsGPU_;
};

TestCUDAProducerSim::TestCUDAProducerSim(const edm::ParameterSet& iConfig) {
  const auto moduleLabel = iConfig.getParameter<std::string>("@module_label");
  edm::Service<SimOperationsService> sos;

  produceOpsCPU_ = sos->produceCPUProcessor(moduleLabel);
  produceOpsGPU_ = sos->produceGPUProcessor(moduleLabel);

  if (produceOpsCPU_.events() == 0 and produceOpsGPU_.events() == 0) {
    throw cms::Exception("Configuration") << "Got 0 events, which makes this module useless";
  }

  for (const auto& src : iConfig.getParameter<std::vector<edm::InputTag>>("srcs")) {
    srcTokens_.emplace_back(consumes<int>(src));
  }
  for (const auto& src : iConfig.getParameter<std::vector<edm::InputTag>>("cudaSrcs")) {
    cudaSrcTokens_.emplace_back(consumes<cms::cuda::Product<int>>(src));
  }

  if (iConfig.getParameter<bool>("produce")) {
    dstToken_ = produces<int>();
  }
  if (iConfig.getParameter<bool>("produceCUDA")) {
    cudaDstToken_ = produces<cms::cuda::Product<int>>();
  }
}

void TestCUDAProducerSim::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::vector<edm::InputTag>>("srcs", std::vector<edm::InputTag>{});
  desc.add<std::vector<edm::InputTag>>("cudaSrcs", std::vector<edm::InputTag>{});
  desc.add<bool>("produce", false);
  desc.add<bool>("produceCUDA", false);

  //desc.add<bool>("useCachingAllocator", true);
  descriptions.addWithDefaultLabel(desc);
}

void TestCUDAProducerSim::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // to make sure the dependencies are set correctly
  for (const auto& t : srcTokens_) {
    iEvent.get(t);
  }

  std::vector<const cms::cuda::Product<int>*> cudaProducts(cudaSrcTokens_.size(), nullptr);
  std::transform(cudaSrcTokens_.begin(), cudaSrcTokens_.end(), cudaProducts.begin(), [&iEvent](const auto& tok) {
    return &iEvent.get(tok);
  });

  auto ctx = cudaProducts.empty() ? cms::cuda::ScopedContextProduce(iEvent.streamID())
                                  : cms::cuda::ScopedContextProduce(*cudaProducts[0]);

  for (const auto ptr : cudaProducts) {
    ctx.get(*ptr);
  }

  if (produceOpsCPU_.events() > 0) {
    produceOpsCPU_.process(std::vector<size_t>{iEvent.id().event() % produceOpsCPU_.events()});
  }
  if (produceOpsGPU_.events() > 0) {
    produceOpsGPU_.process(std::vector<size_t>{iEvent.id().event() % produceOpsGPU_.events()}, ctx.stream());
  }

  if (not dstToken_.isUninitialized()) {
    iEvent.emplace(dstToken_, 42);
  }
  if (not cudaDstToken_.isUninitialized()) {
    ctx.emplace(iEvent, cudaDstToken_, 42);
  }
}

DEFINE_FWK_MODULE(TestCUDAProducerSim);
