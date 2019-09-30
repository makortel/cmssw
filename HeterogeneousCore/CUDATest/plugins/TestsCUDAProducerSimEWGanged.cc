#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/ReusableObjectHolder.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CUDADataFormats/Common/interface/CUDAProduct.h"
#include "HeterogeneousCore/CUDACore/interface/CUDAScopedContext.h"

#include "SimOperationsService.h"

namespace {
  class Ganger {
  public:
    Ganger(const edm::ParameterSet& iConfig) {
      edm::Service<SimOperationsService> sos;
      gangSize_ = sos->gangSize();
      const auto gangNum = sos->numberOfGangs();

      const auto moduleLabel = iConfig.getParameter<std::string>("@module_label");
      for(unsigned int i=0; i<gangNum; ++i) {
        auto tmp = std::make_unique<SimOperationsService::AcquireGPUProcessor>(sos->acquireGPUProcessor(moduleLabel));
        events_ = tmp->events();
        acquireOpsGPU_.add(std::move(tmp));
      }

      reserve();
    }

    size_t events() const { return events_; }

    void enqueue(unsigned int eventIndex, std::vector<const CUDAProduct<int>*> inputData, edm::WaitingTaskWithArenaHolder holder, CUDAScopedContextAcquire& ctx) const {
      std::vector<size_t> indicesToLaunch;
      std::vector<edm::WaitingTaskWithArenaHolder> holdersToLaunch;
      std::vector<std::vector<const CUDAProduct<int>*>> inputsToLaunch;
      {
        std::lock_guard<std::mutex> guard{mutex_};
        workIndices_.push_back(eventIndex % events());
        workHolders_.emplace_back(std::move(holder));
        workInputs_.emplace_back(std::move(inputData));
        LogTrace("Foo") << "Enqueued work for event " << eventIndex << ", queue size is " << workIndices_.size() << " last index " << workIndices_.back() << ", has info for events " << events();
        if(workIndices_.size() == gangSize_) {
          std::swap(workIndices_, indicesToLaunch);
          std::swap(workHolders_, holdersToLaunch);
          std::swap(workInputs_, inputsToLaunch);
          reserve();
        }
      }
      if(not indicesToLaunch.empty()) {
        LogTrace("Foo").log([&](auto& l) {
            l << "Launching work for indices ";
            for(auto i: indicesToLaunch) {
              l << i << " ";
            }
            l << "in CUDA stream " << ctx.stream().id();
          });
        // need to synchronize the input data only wrt. the CUDA stream the work will be executed in
        for(auto& inputsForEvent: inputsToLaunch) {
          for(auto* input: inputsForEvent) {
            ctx.get(*input);
          }
        }
        auto opsPtr = acquireOpsGPU_.tryToGet();
        if(not opsPtr) {
          throw cms::Exception("LogicError") << "Tried to get acquire operations in Ganger::enqueue, but got none. Are gangSize and gangNum compatible with numberOfStreams?";
        }

        opsPtr->process(indicesToLaunch, ctx.stream());
        ctx.replaceWaitingTaskHolder(edm::WaitingTaskWithArenaHolder{edm::make_waiting_task(tbb::task::allocate_root(),
                                                                                            [holders=std::move(holdersToLaunch),
                                                                                             opsPtr=std::move(opsPtr)](std::exception_ptr const* excptr) mutable {
                                                                                              LogTrace("Foo") << "Joint callback task to reset shared_ptr and release contained WaitingTaskWithArenaHolders";
                                                                                              opsPtr.reset();
                                                                                              for(auto& h: holders) {
                                                                                                if(excptr) {
                                                                                                  h.doneWaiting(*excptr);
                                                                                                }
                                                                                              }
                                                                                            })});
      }
    }

  private:
    void reserve() const {
      workIndices_.reserve(gangSize_);
      workHolders_.reserve(gangSize_);
      workInputs_.reserve(gangSize_);
    }

    mutable std::mutex mutex_;
    // These three are protected with the mutex
    mutable std::vector<size_t> workIndices_;
    mutable std::vector<edm::WaitingTaskWithArenaHolder> workHolders_;
    mutable std::vector<std::vector<const CUDAProduct<int>*>> workInputs_;

    // one for each gang (i.e. N(streams) / gangSize)
    mutable edm::ReusableObjectHolder<SimOperationsService::AcquireGPUProcessor> acquireOpsGPU_;

    size_t events_ = 0;
    unsigned int gangSize_ = 0;
  };
}

class TestCUDAProducerSimEWGanged: public edm::stream::EDProducer<edm::ExternalWork, edm::GlobalCache<Ganger>> {
public:
  explicit TestCUDAProducerSimEWGanged(const edm::ParameterSet& iConfig, const Ganger* ganger);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  static std::unique_ptr<Ganger> initializeGlobalCache(const edm::ParameterSet& iConfig) {
    return std::make_unique<Ganger>(iConfig);
  }

  void acquire(const edm::Event& iEvent, const edm::EventSetup& iSetup, edm::WaitingTaskWithArenaHolder) override;
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  static void globalEndJob(const Ganger* ganger) {}
private:
  
  std::vector<edm::EDGetTokenT<int>> srcTokens_;
  std::vector<edm::EDGetTokenT<CUDAProduct<int>>> cudaSrcTokens_;
  edm::EDPutTokenT<int> dstToken_;
  edm::EDPutTokenT<CUDAProduct<int>> cudaDstToken_;
  CUDAContextState ctxState_;

  SimOperationsService::AcquireCPUProcessor acquireOpsCPU_;
  SimOperationsService::ProduceGPUProcessor produceOpsGPU_;
  SimOperationsService::ProduceCPUProcessor produceOpsCPU_;
};

TestCUDAProducerSimEWGanged::TestCUDAProducerSimEWGanged(const edm::ParameterSet& iConfig, const Ganger* ganger) {
  const auto moduleLabel = iConfig.getParameter<std::string>("@module_label");
  edm::Service<SimOperationsService> sos;
  acquireOpsCPU_ = sos->acquireCPUProcessor(moduleLabel);
  produceOpsCPU_ = sos->produceCPUProcessor(moduleLabel);
  produceOpsGPU_ = sos->produceGPUProcessor(moduleLabel);

  const auto acquireEvents = ganger->events();
  if(acquireEvents == 0) {
    throw cms::Exception("Configuration") << "Got 0 events for GPU ops, which makes this module useless";
  }
  if(acquireEvents != acquireOpsCPU_.events() and acquireOpsCPU_.events() > 0) {
    throw cms::Exception("LogicError") << "Got " << acquireEvents << " from GPU acquire ops, but " << acquireOpsCPU_.events() << " from CPU ops";
  }
  if(acquireEvents != produceOpsCPU_.events() and produceOpsCPU_.events() > 0) {
    throw cms::Exception("Configuration") << "Got " << acquireEvents << " events for acquire and " << produceOpsCPU_.events() << " for produce CPU";
  }
  if(acquireEvents != produceOpsGPU_.events() and produceOpsGPU_.events() > 0) {
    throw cms::Exception("Configuration") << "Got " << acquireEvents << " events for acquire and " << produceOpsGPU_.events() << " for produce CPU";
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

void TestCUDAProducerSimEWGanged::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::vector<edm::InputTag>>("srcs", std::vector<edm::InputTag>{});
  desc.add<std::vector<edm::InputTag>>("cudaSrcs", std::vector<edm::InputTag>{});
  desc.add<bool>("produce", false);
  desc.add<bool>("produceCUDA", false);

  //desc.add<bool>("useCachingAllocator", true);
  descriptions.addWithDefaultLabel(desc);
}

void TestCUDAProducerSimEWGanged::acquire(const edm::Event& iEvent, const edm::EventSetup& iSetup, edm::WaitingTaskWithArenaHolder h) {
  // to make sure the dependencies are set correctly
  for(const auto& t: srcTokens_) {
    iEvent.get(t);
  }

  std::vector<const CUDAProduct<int> *> cudaProducts(cudaSrcTokens_.size(), nullptr);
  std::transform(cudaSrcTokens_.begin(), cudaSrcTokens_.end(), cudaProducts.begin(), [&iEvent](const auto& tok) {
      return &iEvent.get(tok);
    });

  // In principle the CUDAScopedContext is not needed in acquire() in
  // those streams that do not process the data, but it is needed in
  // produce() in all streams, so let's just create it here to leave
  // ctxState_ in valid state in all stream.
  auto ctx = cudaProducts.empty() ? CUDAScopedContextAcquire(iEvent.streamID(), h, ctxState_) :
    CUDAScopedContextAcquire(*cudaProducts[0], h, ctxState_);

  if(acquireOpsCPU_.events() > 0) {
    acquireOpsCPU_.process(std::vector<size_t>{iEvent.id().event() % acquireOpsCPU_.events()});
  }
  globalCache()->enqueue(iEvent.id().event(), std::move(cudaProducts), std::move(h), ctx);
}

void TestCUDAProducerSimEWGanged::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  CUDAScopedContextProduce ctx{ctxState_};

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

DEFINE_FWK_MODULE(TestCUDAProducerSimEWGanged);
