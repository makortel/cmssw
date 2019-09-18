#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "CUDADataFormats/Common/interface/CUDAProduct.h"
#include "HeterogeneousCore/CUDACore/interface/CUDAScopedContext.h"

#include "SimOperations.h"

namespace {
  class Ganger {
  public:
    Ganger(cudatest::SimOperations* acquireOps, unsigned int gangSize):
      acquireOps_{acquireOps},
      gangSize_{gangSize} {

      if(gangSize_ == 0) {
        throw cms::Exception("Configuration") << "gangSize must be larger than 0";
      }
      reserve();
    }

    void enqueue(unsigned int workIndex, edm::WaitingTaskWithArenaHolder holder, CUDAScopedContextAcquire& ctx) {
      std::vector<size_t> indicesToLaunch;
      std::vector<edm::WaitingTaskWithArenaHolder> holdersToLaunch;
      {
        std::lock_guard<std::mutex> guard{mutex_};
        workIndices_.push_back(workIndex);
        workHolders_.emplace_back(std::move(holder));
        if(workIndices_.size() == gangSize_) {
          std::swap(workIndices_, indicesToLaunch);
          std::swap(workHolders_, holdersToLaunch);
          reserve();
        }
      }
      if(not indicesToLaunch.empty()) {
        acquireOps_->operate(indicesToLaunch, &ctx.stream());
        ctx.replaceWaitingTaskHolder(edm::WaitingTaskWithArenaHolder{edm::make_waiting_task(tbb::task::allocate_root(),
                                                                                            [holders=std::move(workHolders_)](std::exception_ptr const* excptr) mutable {
                                                                                              for(auto& h: holders) {
                                                                                                if(excptr) {
                                                                                                  h.doneWaiting(*excptr);
                                                                                                }
                                                                                              }
                                                                                            })});
      }
    }

  private:
    void reserve() {
      workIndices_.reserve(gangSize_);
      workHolders_.reserve(gangSize_);
    }

    std::mutex mutex_;
    std::vector<size_t> workIndices_;
    std::vector<edm::WaitingTaskWithArenaHolder> workHolders_;
    cudatest::SimOperations* acquireOps_;
    const unsigned int gangSize_;
  };
}

class TestCUDAProducerSimEWGanged: public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit TestCUDAProducerSimEWGanged(const edm::ParameterSet& iConfig);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void acquire(const edm::Event& iEvent, const edm::EventSetup& iSetup, edm::WaitingTaskWithArenaHolder) override;
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;
private:
  
  std::vector<edm::EDGetTokenT<int>> srcTokens_;
  std::vector<edm::EDGetTokenT<CUDAProduct<int>>> cudaSrcTokens_;
  edm::EDPutTokenT<int> dstToken_;
  edm::EDPutTokenT<CUDAProduct<int>> cudaDstToken_;
  CUDAContextState ctxState_;

  cudatest::SimOperations acquireOps_;
  cudatest::SimOperations produceOps_;
  Ganger ganger_;
};

TestCUDAProducerSimEWGanged::TestCUDAProducerSimEWGanged(const edm::ParameterSet& iConfig):
  acquireOps_{iConfig.getParameter<edm::FileInPath>("config").fullPath(),
              iConfig.getParameter<edm::FileInPath>("cudaCalibration").fullPath(),
              "moduleDefinitions."+iConfig.getParameter<std::string>("@module_label")+".acquire"},
  produceOps_{iConfig.getParameter<edm::FileInPath>("config").fullPath(),
              iConfig.getParameter<edm::FileInPath>("cudaCalibration").fullPath(),
              "moduleDefinitions."+iConfig.getParameter<std::string>("@module_label")+".produce"},
  ganger_{&acquireOps_, iConfig.getParameter<unsigned int>("gangSize")}
{
  if(acquireOps_.events() == 0) {
    throw cms::Exception("Configuration") << "Got 0 events, which makes this module useless";
  }
  if(acquireOps_.events() != produceOps_.events() and produceOps_.events() > 0) {
    throw cms::Exception("Configuration") << "Got " << acquireOps_.events() << " events for acquire and " << produceOps_.events() << " for produce";
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
  desc.add<unsigned int>("gangSize", 1);

  desc.add<edm::FileInPath>("config", edm::FileInPath())->setComment("Path to a JSON configuration file of the simulation");
  desc.add<edm::FileInPath>("cudaCalibration", edm::FileInPath())->setComment("Path to a JSON file for the CUDA calibration");

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

  auto ctx = cudaProducts.empty() ? CUDAScopedContextAcquire(iEvent.streamID(), h, ctxState_) :
    CUDAScopedContextAcquire(*cudaProducts[0], h, ctxState_);

  // In principle this introduces the wrong synchronization
  for(const auto ptr: cudaProducts) {
    ctx.get(*ptr);
  }

  ganger_.enqueue(iEvent.id().event() % acquireOps_.events(), std::move(h), ctx);
}

void TestCUDAProducerSimEWGanged::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  CUDAScopedContextProduce ctx{ctxState_};

  if(produceOps_.events() > 0) {
    produceOps_.operate(std::vector<size_t>{iEvent.id().event() % produceOps_.events()}, &ctx.stream());
  }

  if(not dstToken_.isUninitialized()) {
    iEvent.emplace(dstToken_, 42);
  }
  if(not cudaDstToken_.isUninitialized()) {
    ctx.emplace(iEvent, cudaDstToken_, 42);
  }
}

DEFINE_FWK_MODULE(TestCUDAProducerSimEWGanged);
