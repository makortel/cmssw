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
  std::once_flag taskQueueFlag;
  std::unique_ptr<edm::LimitedTaskQueue> taskQueue;

  class Ganger {
  public:
    Ganger(const edm::ParameterSet& iConfig) {
      edm::Service<SimOperationsService> sos;
      maxGangSize_ = sos->gangSize();
      maxEvents_ = sos->maxEvents();
      if(maxEvents_ < 1) {
        throw cms::Exception("Configuration") << "This module needs SimOperationsService.maxEvents to be set, now it was " << maxEvents_;
      }
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

    void enqueue(unsigned int eventIndex, std::vector<const CUDAProduct<int>*> inputData, edm::WaitingTaskWithArenaHolder holder, CUDAScopedContextAcquire& ctx, CUDAContextState* ctxState) const {
      bool queueTask = false;
      {
        std::lock_guard<std::mutex> guard{mutex_};
        processedEvents_ += 1;
        queueTask = workIndices_.empty() or processedEvents_ == maxEvents_;; // queue processing task if we're first to add work for it, or at the last event
        workIndices_.push_back(eventIndex % events());
        workHolders_.emplace_back(std::move(holder));
        if(queueTask) {
          workInputs_.emplace_back(inputData);
        }
        else {
          workInputs_.emplace_back(std::move(inputData));
        }
        LogTrace("Foo") << "Enqueued work for event " << eventIndex << ", queue size is " << workIndices_.size() << " last index " << workIndices_.back() << ", has info for events " << events();
      }

      if(queueTask) {
        // need to synchronize the input data only wrt. the CUDA stream the work will be executed in
        for(auto* input: inputData) {
          ctx.get(*input);
        }

        auto queueTaskHolder = edm::WaitingTaskWithArenaHolder{
          edm::make_waiting_task(tbb::task::allocate_root(),
                                 [this, ctxState](std::exception_ptr const* excptr) mutable {
                                   std::vector<size_t> indicesToLaunch;
                                   std::vector<edm::WaitingTaskWithArenaHolder> holdersToLaunch;
                                   std::vector<std::vector<const CUDAProduct<int>*>> inputsToLaunch;
                                   {
                                     std::lock_guard<std::mutex> guard{mutex_};
                                     std::swap(workIndices_, indicesToLaunch);
                                     std::swap(workHolders_, holdersToLaunch);
                                     std::swap(workInputs_, inputsToLaunch);
                                     reserve();
                                   }
                                   if(excptr) {
                                     for(auto& h: holdersToLaunch) {
                                       h.doneWaiting(*excptr);
                                     }
                                   }

                                   // nothing to do?
                                   if(indicesToLaunch.empty()) {
                                     return;
                                   }
                                   try {
                                     auto opsPtr = acquireOpsGPU_.tryToGet();
                                     if(not opsPtr) {
                                       throw cms::Exception("LogicError") << "Tried to get acquire operations in Ganger::enqueue, but got none. Has gangNum been set to numberOfStreams?";
                                     }

                                     CUDAScopedContextTask ctx{ctxState,
                                                               edm::WaitingTaskWithArenaHolder{
                                         edm::make_waiting_task(tbb::task::allocate_root(),
                                                                [holders=holdersToLaunch,
                                                                 opsPtr=opsPtr](std::exception_ptr const* excptr) mutable {
                                                                  LogTrace("Foo") << "Joint callback task to reset shared_ptr and release contained WaitingTaskWithArenaHolders";
                                                                  opsPtr.reset();
                                                                  for(auto& h: holders) {
                                                                    if(excptr) {
                                                                      h.doneWaiting(*excptr);
                                                                    }
                                                                  }
                                                                })}};
              
                                     LogTrace("Foo").log([&](auto& l) {
                                         l << "Launching work for indices ";
                                         for(auto i: indicesToLaunch) {
                                           l << i << " ";
                                         }
                                         l << "in CUDA stream " << ctx.stream().id();
                                       });
                                     opsPtr->process(indicesToLaunch, ctx.stream());
                                   } catch(...) {
                                     for(auto& h: holdersToLaunch) {
                                       h.doneWaiting(std::current_exception());
                                     }
                                   }
                                 })};
        taskQueue->push([queueTaskHolder]() mutable {
            queueTaskHolder.doneWaiting(nullptr);
          });
        ctx.replaceWaitingTaskHolder(std::move(queueTaskHolder));
      }
    }

  private:
    void reserve() const {
      workIndices_.reserve(maxGangSize_);
      workHolders_.reserve(maxGangSize_);
      workInputs_.reserve(maxGangSize_);
    }

    mutable std::mutex mutex_;
    // These three are protected with the mutex
    mutable std::vector<size_t> workIndices_;
    mutable std::vector<edm::WaitingTaskWithArenaHolder> workHolders_;
    mutable std::vector<std::vector<const CUDAProduct<int>*>> workInputs_;

    // one for each gang (i.e. N(streams) / gangSize)
    mutable edm::ReusableObjectHolder<SimOperationsService::AcquireGPUProcessor> acquireOpsGPU_;

    mutable int processedEvents_ = 0; // protected by the mutex
    size_t events_ = 0;
    int maxEvents_ = 0;
    unsigned int maxGangSize_ = 0;
  };
}

class TestCUDAProducerSimEWGangedLimitedTaskQueue: public edm::stream::EDProducer<edm::ExternalWork, edm::GlobalCache<Ganger>> {
public:
  explicit TestCUDAProducerSimEWGangedLimitedTaskQueue(const edm::ParameterSet& iConfig, const Ganger* ganger);

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

TestCUDAProducerSimEWGangedLimitedTaskQueue::TestCUDAProducerSimEWGangedLimitedTaskQueue(const edm::ParameterSet& iConfig, const Ganger* ganger) {
  std::call_once(taskQueueFlag, [limit=iConfig.getParameter<unsigned int>("limit")](){
      taskQueue = std::make_unique<edm::LimitedTaskQueue>(limit);
    });

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

void TestCUDAProducerSimEWGangedLimitedTaskQueue::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::vector<edm::InputTag>>("srcs", std::vector<edm::InputTag>{});
  desc.add<std::vector<edm::InputTag>>("cudaSrcs", std::vector<edm::InputTag>{});
  desc.add<bool>("produce", false);
  desc.add<bool>("produceCUDA", false);
  desc.add<unsigned int>("limit", 1);

  //desc.add<bool>("useCachingAllocator", true);
  descriptions.addWithDefaultLabel(desc);
}

void TestCUDAProducerSimEWGangedLimitedTaskQueue::acquire(const edm::Event& iEvent, const edm::EventSetup& iSetup, edm::WaitingTaskWithArenaHolder h) {
  //edm::LogWarning("foo") << "TestCUDAProducerSimEWGangedLimitedTaskQueue::acquire() begin";
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
  globalCache()->enqueue(iEvent.id().event(), std::move(cudaProducts), h, ctx, &ctxState_);
  // must have h alive here to set ctxState_ correctly for the task in LimitedTaskQueue
  //edm::LogWarning("foo") << "TestCUDAProducerSimEWGangedLimitedTaskQueue::acquire() end";
}

void TestCUDAProducerSimEWGangedLimitedTaskQueue::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  //edm::LogWarning("foo") << "TestCUDAProducerSimEWGangedLimitedTaskQueue::produce()";
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

DEFINE_FWK_MODULE(TestCUDAProducerSimEWGangedLimitedTaskQueue);
