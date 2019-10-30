#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/ReusableObjectHolder.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"

#include "CUDADataFormats/Common/interface/CUDAProduct.h"
#include "HeterogeneousCore/CUDACore/interface/CUDAScopedContext.h"

#include "SimOperationsService.h"

#include <fstream>

namespace {
  std::once_flag taskQueueFlag;
  std::unique_ptr<edm::LimitedTaskQueue> taskQueue;
  std::mutex histoMutex;

  unsigned int pow2(unsigned int exp) {
    return 1<<exp;
  }

  unsigned int ilog2(unsigned int val) {
    unsigned int ret = 0;
    while(val >>= 1) ++ret;
    return ret;
  }

  unsigned int bit_length(unsigned int val) {
    if(val == 0) {
      return 0;
    }
    return ilog2(val)+1;
  }

  class Ganger {
  public:
    Ganger(const edm::ParameterSet& iConfig):
      histoOutput_{iConfig.getParameter<std::string>("histoOutput")},
      modLabel_{iConfig.getParameter<std::string>("@module_label")}
    {
      edm::Service<SimOperationsService> sos;
      maxGangSize_ = sos->gangSize();
      maxEvents_ = sos->maxEvents();
      if(maxEvents_ < 1) {
        throw cms::Exception("Configuration") << "This module needs SimOperationsService.maxEvents to be set, now it was " << maxEvents_;
      }
      const auto maxGangNum = sos->numberOfGangs();

      const auto moduleLabel = iConfig.getParameter<std::string>("@module_label");
      acquireOpsGPU_.resize(maxGangNum);

      for(unsigned int bits=0, end=bit_length(maxGangSize_)+1; bits<end; ++bits) {
        const unsigned int gangSize = pow2(bits);
        const unsigned int gangNum = bits > 0 ? maxGangNum/(pow2(bits-1)+1) :  maxGangNum;
        LogTrace("foo") << "Index " << bits << " gang size " << gangSize << " max number of gangs " << gangNum;
        for(unsigned int i=0; i<gangNum; ++i) {
          auto tmp = std::make_unique<SimOperationsService::AcquireGPUProcessor>(sos->acquireGPUProcessor(moduleLabel, gangSize));
          events_ = tmp->events();
          acquireOpsGPU_[bits].add(std::move(tmp));
        }
      }
      histoGangSize_.resize(maxGangSize_);
      for(auto& bin: histoGangSize_) {
        bin = std::make_unique<std::atomic<unsigned long long> >();
      }

      reserve();
    }

    size_t events() const { return events_; }

    void enqueue(unsigned int eventIndex, std::vector<const CUDAProduct<int>*> inputData, edm::WaitingTaskWithArenaHolder& holder, CUDAContextState* ctxState) const {
      bool queueTask = false;
      {
        std::lock_guard<std::mutex> guard{mutex_};
        processedEvents_ += 1;
        queueTask = workIndices_.empty() or processedEvents_ == maxEvents_;; // queue processing task if we're first to add work for it, or at the last event
        workIndices_.push_back(eventIndex % events());
        workHolders_.emplace_back(holder.makeWaitingTaskHolderAndRelease());
        if(queueTask) {
          workInputs_.emplace_back(inputData);
        }
        else {
          workInputs_.emplace_back(std::move(inputData));
        }
        LogTrace("Foo") << "Enqueued work for event " << eventIndex << ", queue size is " << workIndices_.size() << " last index " << workIndices_.back() << ", has info for events " << events();
      }

      if(queueTask) {
        auto queueTaskHolder = edm::WaitingTaskWithArenaHolder{
          edm::make_waiting_task(tbb::task::allocate_root(),
                                 [this, ctxState](std::exception_ptr const* excptr) mutable {
                                   std::vector<size_t> indicesToLaunch;
                                   std::vector<edm::WaitingTaskHolder> holdersToLaunch;
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
                                     ++(*(histoGangSize_[indicesToLaunch.size()-1]));
                                     const auto ind = bit_length(indicesToLaunch.size()-1);
                                     LogTrace("foo") << "Gang size " << indicesToLaunch.size() << " getting work from index " << ind;
                                     auto opsPtr = acquireOpsGPU_.at(ind).tryToGet();
                                     if(not opsPtr) {
                                       throw cms::Exception("LogicError") << "Tried to get acquire operations in Ganger::enqueue, but got none. Likely there is an internal logic error";
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
              
                                     // need to synchronize the input data only wrt. the CUDA stream the work will be executed in
                                     for(auto& inputsForEvent: inputsToLaunch) {
                                       for(auto* input: inputsForEvent) {
                                         ctx.get(*input);
                                       }
                                     }

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
      }
    }

    void saveHisto() const {
      if(histoOutput_.empty()) {
        return;
      }

      std::lock_guard<std::mutex> guard{histoMutex};
      std::ofstream out{histoOutput_.c_str(), std::ios_base::out | std::ios_base::app};
      out << modLabel_;
      for(const auto& bin: histoGangSize_) {
        out << " " << *bin;
      }
      out << std::endl;
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
    mutable std::vector<edm::WaitingTaskHolder> workHolders_;
    mutable std::vector<std::vector<const CUDAProduct<int>*>> workInputs_;

    // vector has one entry per possible gang size, ReusableObjectHolder entries for the possible number of members of that gang size
    mutable std::vector<edm::ReusableObjectHolder<SimOperationsService::AcquireGPUProcessor>> acquireOpsGPU_;

    mutable int processedEvents_ = 0; // protected by the mutex
    size_t events_ = 0;
    int maxEvents_ = 0;
    unsigned int maxGangSize_ = 0;

    // gather information
    mutable std::vector<std::unique_ptr<std::atomic<unsigned long long> > > histoGangSize_;
    const std::string histoOutput_;
    const std::string modLabel_;
  };
}

class TestCUDAProducerSimEWGangedLimitedTaskQueueV3: public edm::stream::EDProducer<edm::ExternalWork, edm::GlobalCache<Ganger>> {
public:
  explicit TestCUDAProducerSimEWGangedLimitedTaskQueueV3(const edm::ParameterSet& iConfig, const Ganger* ganger);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  static std::unique_ptr<Ganger> initializeGlobalCache(const edm::ParameterSet& iConfig) {
    return std::make_unique<Ganger>(iConfig);
  }

  void acquire(const edm::Event& iEvent, const edm::EventSetup& iSetup, edm::WaitingTaskWithArenaHolder) override;
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  static void globalEndJob(const Ganger* ganger) {
    ganger->saveHisto();
  }
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

TestCUDAProducerSimEWGangedLimitedTaskQueueV3::TestCUDAProducerSimEWGangedLimitedTaskQueueV3(const edm::ParameterSet& iConfig, const Ganger* ganger) {
  std::call_once(taskQueueFlag, [limit=iConfig.getParameter<unsigned int>("limit")](){
      LogTrace("Foo") << "Creating LimitedTaskQueue with limit " << limit;
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

void TestCUDAProducerSimEWGangedLimitedTaskQueueV3::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::vector<edm::InputTag>>("srcs", std::vector<edm::InputTag>{});
  desc.add<std::vector<edm::InputTag>>("cudaSrcs", std::vector<edm::InputTag>{});
  desc.add<bool>("produce", false);
  desc.add<bool>("produceCUDA", false);
  desc.add<unsigned int>("limit", 1);
  desc.add<std::string>("histoOutput", "")->setComment("If empty, output disabled");

  //desc.add<bool>("useCachingAllocator", true);
  descriptions.addWithDefaultLabel(desc);
}

void TestCUDAProducerSimEWGangedLimitedTaskQueueV3::acquire(const edm::Event& iEvent, const edm::EventSetup& iSetup, edm::WaitingTaskWithArenaHolder h) {
  //edm::LogWarning("foo") << "TestCUDAProducerSimEWGangedLimitedTaskQueueV3::acquire() begin";
  // to make sure the dependencies are set correctly
  for(const auto& t: srcTokens_) {
    iEvent.get(t);
  }

  std::vector<const CUDAProduct<int> *> cudaProducts(cudaSrcTokens_.size(), nullptr);
  std::transform(cudaSrcTokens_.begin(), cudaSrcTokens_.end(), cudaProducts.begin(), [&iEvent](const auto& tok) {
      return &iEvent.get(tok);
    });

  {
    // In principle the CUDAScopedContext is not needed in acquire() in
    // those streams that do not process the data, but it is needed in
    // produce() in all streams, so let's just create it here to leave
    // ctxState_ in valid state in all stream.
    auto ctx = cudaProducts.empty() ? CUDAScopedContextAcquire(iEvent.streamID(), ctxState_) :
      CUDAScopedContextAcquire(*cudaProducts[0], ctxState_);
  }

  if(acquireOpsCPU_.events() > 0) {
    acquireOpsCPU_.process(std::vector<size_t>{iEvent.id().event() % acquireOpsCPU_.events()});
  }
  globalCache()->enqueue(iEvent.id().event(), std::move(cudaProducts), h, &ctxState_);
  // must have h alive here to set ctxState_ correctly for the task in LimitedTaskQueue
  //edm::LogWarning("foo") << "TestCUDAProducerSimEWGangedLimitedTaskQueueV3::acquire() end";
}

void TestCUDAProducerSimEWGangedLimitedTaskQueueV3::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  //edm::LogWarning("foo") << "TestCUDAProducerSimEWGangedLimitedTaskQueueV3::produce()";
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

DEFINE_FWK_MODULE(TestCUDAProducerSimEWGangedLimitedTaskQueueV3);
