#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Concurrency/interface/LimitedTaskQueue.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CUDADataFormats/Common/interface/CUDAProduct.h"
#include "HeterogeneousCore/CUDACore/interface/CUDAScopedContext.h"

#include "SimOperationsService.h"

#include <atomic>

namespace {
  std::once_flag taskQueueFlag;
  std::unique_ptr<edm::LimitedTaskQueue> taskQueue;
}

class TestCUDAProducerSimEWLimitedTaskQueue: public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit TestCUDAProducerSimEWLimitedTaskQueue(const edm::ParameterSet& iConfig);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void acquire(const edm::Event& iEvent, const edm::EventSetup& iSetup, edm::WaitingTaskWithArenaHolder) override;
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

private:
  std::vector<edm::EDGetTokenT<int>> srcTokens_;
  std::vector<edm::EDGetTokenT<CUDAProduct<int>>> cudaSrcTokens_;
  edm::EDPutTokenT<int> dstToken_;
  edm::EDPutTokenT<CUDAProduct<int>> cudaDstToken_;
  CUDAContextState ctxState_;
  std::atomic<bool> queueingFinished_ = true;

  SimOperationsService::AcquireCPUProcessor acquireOpsCPU_;
  SimOperationsService::AcquireGPUProcessor acquireOpsGPU_;
  SimOperationsService::ProduceCPUProcessor produceOpsCPU_;
  SimOperationsService::ProduceGPUProcessor produceOpsGPU_;
};

TestCUDAProducerSimEWLimitedTaskQueue::TestCUDAProducerSimEWLimitedTaskQueue(const edm::ParameterSet& iConfig) {
  std::call_once(taskQueueFlag, [limit=iConfig.getParameter<unsigned int>("limit")](){
      taskQueue = std::make_unique<edm::LimitedTaskQueue>(limit);
    });

  const auto moduleLabel = iConfig.getParameter<std::string>("@module_label");
  edm::Service<SimOperationsService> sos;
  acquireOpsCPU_ = sos->acquireCPUProcessor(moduleLabel);
  acquireOpsGPU_ = sos->acquireGPUProcessor(moduleLabel);
  produceOpsCPU_ = sos->produceCPUProcessor(moduleLabel);
  produceOpsGPU_ = sos->produceGPUProcessor(moduleLabel);

  if(acquireOpsCPU_.events() == 0 && acquireOpsGPU_.events() == 0) {
    throw cms::Exception("Configuration") << "Got 0 events, which makes this module useless";
  }
  const auto nevents = std::max(acquireOpsCPU_.events(), acquireOpsGPU_.events());

  if(nevents != produceOpsCPU_.events() and produceOpsCPU_.events() > 0) {
    throw cms::Exception("Configuration") << "Got " << nevents << " events for acquire and " << produceOpsCPU_.events() << " for produce CPU";
  }
  if(nevents != produceOpsGPU_.events() and produceOpsGPU_.events() > 0) {
    throw cms::Exception("Configuration") << "Got " << nevents << " events for acquire and " << produceOpsGPU_.events() << " for produce GPU";
  }

  for(const auto& src: iConfig.getParameter<std::vector<edm::InputTag>>("srcs")) {
    srcTokens_.emplace_back(consumes<int>(src));
  }
  for(const auto& src: iConfig.getParameter<std::vector<edm::InputTag>>("cudaSrcs")) {
    cudaSrcTokens_.emplace_back(consumes<CUDAProduct<int>>(src));
  }

  if(iConfig.getParameter<bool>("produce")) {
    dstToken_ = produces<int>();
  }
  if(iConfig.getParameter<bool>("produceCUDA")) {
    cudaDstToken_ = produces<CUDAProduct<int>>();
  }
}

void TestCUDAProducerSimEWLimitedTaskQueue::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::vector<edm::InputTag>>("srcs", std::vector<edm::InputTag>{});
  desc.add<std::vector<edm::InputTag>>("cudaSrcs", std::vector<edm::InputTag>{});
  desc.add<bool>("produce", false);
  desc.add<bool>("produceCUDA", false);
  desc.add<unsigned int>("limit", 1);

  //desc.add<bool>("useCachingAllocator", true);
  descriptions.addWithDefaultLabel(desc);
}

void TestCUDAProducerSimEWLimitedTaskQueue::acquire(const edm::Event& iEvent, const edm::EventSetup& iSetup, edm::WaitingTaskWithArenaHolder h) {
  // to make sure the dependencies are set correctly
  for(const auto& t: srcTokens_) {
    iEvent.get(t);
  }

  std::vector<const CUDAProduct<int> *> cudaProducts(cudaSrcTokens_.size(), nullptr);
  std::transform(cudaSrcTokens_.begin(), cudaSrcTokens_.end(), cudaProducts.begin(), [&iEvent](const auto& tok) {
      return &iEvent.get(tok);
    });

  // This is now a bit stupid, but I need the ctxState_ to be fully set before calling taskQueue::push()
  {
    auto ctx = cudaProducts.empty() ? CUDAScopedContextAcquire(iEvent.streamID(), h, ctxState_) :
      CUDAScopedContextAcquire(*cudaProducts[0], h, ctxState_);

    for(const auto ptr: cudaProducts) {
      ctx.get(*ptr);
    }
  }

  if(acquireOpsCPU_.events() > 0) {
    acquireOpsCPU_.process(std::vector<size_t>{iEvent.id().event() % acquireOpsCPU_.events()});
  }
  if(acquireOpsGPU_.events() > 0) {
    queueingFinished_.store(false);
    taskQueue->push(edm::make_lambda_with_holder(std::move(h), [this,
                                                                eventId=iEvent.id()](edm::WaitingTaskWithArenaHolder h) {
                                                   CUDAScopedContextTask ctx{&ctxState_, std::move(h)};
                                                   acquireOpsGPU_.process(std::vector<size_t>{eventId.event() % acquireOpsGPU_.events()}, ctx.stream());
                                                   queueingFinished_.store(true);
                                                 }));
  }
}

void TestCUDAProducerSimEWLimitedTaskQueue::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  CUDAScopedContextProduce ctx{ctxState_};
  if(not queueingFinished_.load()) {
    throw cms::Exception("Assert") << "Work was not yet fully queued in acquire!";
  }
  if(produceOpsCPU_.events() > 0) {
    produceOpsCPU_.process(std::vector<size_t>{iEvent.id().event() % produceOpsCPU_.events()});
  }
  if(produceOpsGPU_.events() > 0) {
    produceOpsGPU_.process(std::vector<size_t>{iEvent.id().event() % produceOpsGPU_.events()}, ctx.stream());
  }

  if(not dstToken_.isUninitialized()) {
    iEvent.emplace(dstToken_, 42);
  }
  if(not cudaDstToken_.isUninitialized()) {
    ctx.emplace(iEvent, cudaDstToken_, 42);
  }
}

DEFINE_FWK_MODULE(TestCUDAProducerSimEWLimitedTaskQueue);
