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

class TestCUDAProducerSimEW : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit TestCUDAProducerSimEW(const edm::ParameterSet& iConfig);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void acquire(const edm::Event& iEvent, const edm::EventSetup& iSetup, edm::WaitingTaskWithArenaHolder) override;
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

private:
  std::vector<edm::EDGetTokenT<int>> srcTokens_;
  std::vector<edm::EDGetTokenT<cms::cuda::Product<int>>> cudaSrcTokens_;
  edm::EDPutTokenT<int> dstToken_;
  edm::EDPutTokenT<cms::cuda::Product<int>> cudaDstToken_;
  cms::cuda::ContextState ctxState_;

  SimOperationsService::AcquireCPUProcessor acquireOpsCPU_;
  SimOperationsService::AcquireGPUProcessor acquireOpsGPU_;
  SimOperationsService::ProduceCPUProcessor produceOpsCPU_;
  SimOperationsService::ProduceGPUProcessor produceOpsGPU_;
};

TestCUDAProducerSimEW::TestCUDAProducerSimEW(const edm::ParameterSet& iConfig) {
  const auto moduleLabel = iConfig.getParameter<std::string>("@module_label");
  edm::Service<SimOperationsService> sos;
  acquireOpsCPU_ = sos->acquireCPUProcessor(moduleLabel);
  acquireOpsGPU_ = sos->acquireGPUProcessor(moduleLabel);
  produceOpsCPU_ = sos->produceCPUProcessor(moduleLabel);
  produceOpsGPU_ = sos->produceGPUProcessor(moduleLabel);

  if (acquireOpsCPU_.events() == 0 && acquireOpsGPU_.events() == 0) {
    throw cms::Exception("Configuration") << "Got 0 events, which makes this module useless";
  }
  const auto nevents = std::max(acquireOpsCPU_.events(), acquireOpsGPU_.events());

  if (nevents != produceOpsCPU_.events() and produceOpsCPU_.events() > 0) {
    throw cms::Exception("Configuration")
        << "Got " << nevents << " events for acquire and " << produceOpsCPU_.events() << " for produce CPU";
  }
  if (nevents != produceOpsGPU_.events() and produceOpsGPU_.events() > 0) {
    throw cms::Exception("Configuration")
        << "Got " << nevents << " events for acquire and " << produceOpsGPU_.events() << " for produce GPU";
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

void TestCUDAProducerSimEW::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::vector<edm::InputTag>>("srcs", std::vector<edm::InputTag>{});
  desc.add<std::vector<edm::InputTag>>("cudaSrcs", std::vector<edm::InputTag>{});
  desc.add<bool>("produce", false);
  desc.add<bool>("produceCUDA", false);

  //desc.add<bool>("useCachingAllocator", true);
  descriptions.addWithDefaultLabel(desc);
}

void TestCUDAProducerSimEW::acquire(const edm::Event& iEvent,
                                    const edm::EventSetup& iSetup,
                                    edm::WaitingTaskWithArenaHolder h) {
  // to make sure the dependencies are set correctly
  for (const auto& t : srcTokens_) {
    iEvent.get(t);
  }

  std::vector<const cms::cuda::Product<int>*> cudaProducts(cudaSrcTokens_.size(), nullptr);
  std::transform(cudaSrcTokens_.begin(), cudaSrcTokens_.end(), cudaProducts.begin(), [&iEvent](const auto& tok) {
    return &iEvent.get(tok);
  });

  auto ctx = cudaProducts.empty() ? cms::cuda::ScopedContextAcquire(iEvent.streamID(), std::move(h), ctxState_)
                                  : cms::cuda::ScopedContextAcquire(*cudaProducts[0], std::move(h), ctxState_);

  for (const auto ptr : cudaProducts) {
    ctx.get(*ptr);
  }

  if (acquireOpsCPU_.events() > 0) {
    acquireOpsCPU_.process(std::vector<size_t>{iEvent.id().event() % acquireOpsCPU_.events()});
  }
  if (acquireOpsGPU_.events() > 0) {
    acquireOpsGPU_.process(std::vector<size_t>{iEvent.id().event() % acquireOpsGPU_.events()}, ctx.stream());
  }
}

void TestCUDAProducerSimEW::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  cms::cuda::ScopedContextProduce ctx{ctxState_};

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

DEFINE_FWK_MODULE(TestCUDAProducerSimEW);
