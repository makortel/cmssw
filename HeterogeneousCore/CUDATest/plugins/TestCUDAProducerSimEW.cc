#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"

#include "HeterogeneousCore/CUDACore/interface/CUDAScopedContext.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_noncached_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"

#include "TimeCruncher.h"
#include "TestCUDAProducerSimEWGPUKernel.h"

#include <random>


class TestCUDAProducerSimEW: public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit TestCUDAProducerSimEW(const edm::ParameterSet& iConfig);
  ~TestCUDAProducerSimEW() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void acquire(const edm::Event& iEvent, const edm::EventSetup& iSetup, edm::WaitingTaskWithArenaHolder) override;
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;
private:
  
  std::vector<edm::EDGetTokenT<int>> srcTokens_;
  const edm::EDPutTokenT<int> dstToken_;
  std::vector<unsigned int> numberOfElementsToDevice_;
  std::vector<unsigned int> kernelLoops_;
  std::vector<unsigned int> numberOfElementsToHost_;
  std::vector<unsigned int> numberOfElementsAlloc_;
  const bool useCachingAllocator_;
  const std::chrono::microseconds crunchForMicroSeconds_;

  std::vector<float *> data_d_;
  std::vector<cudautils::host::noncached::unique_ptr<float[]>> data_h_;
};

TestCUDAProducerSimEW::TestCUDAProducerSimEW(const edm::ParameterSet& iConfig):
  dstToken_{produces<int>()},
  numberOfElementsToDevice_{iConfig.getParameter<std::vector<unsigned int>>("numberOfElementsToDevice")},
  kernelLoops_{iConfig.getParameter<std::vector<unsigned int>>("kernelLoops")},
  numberOfElementsToHost_{iConfig.getParameter<std::vector<unsigned int>>("numberOfElementsToHost")},
  useCachingAllocator_{iConfig.getParameter<bool>("useCachingAllocator")},
  crunchForMicroSeconds_{static_cast<long unsigned int>(iConfig.getParameter<double>("crunchForSeconds")*1e6)}
{
  edm::Service<CUDAService> cs;
  if(cs->enabled()) {
    numberOfElementsAlloc_.resize(std::max(numberOfElementsToDevice_.size(), numberOfElementsToHost_.size()));
    data_h_.resize(numberOfElementsAlloc_.size());
    for(size_t i=0; i<data_h_.size(); ++i) {
      unsigned int elem = 0;
      if(i < numberOfElementsToDevice_.size()) {
        elem = numberOfElementsToDevice_[i];
      }
      if(i < numberOfElementsToHost_.size()) {
        elem = std::max(elem, numberOfElementsToDevice_[i]);
      }

      data_h_[i] = cudautils::make_host_noncached_unique<float[]>(elem /*, cudaHostAllocWriteCombined*/);
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(1e-5, 100.);
    for(size_t i=0; i<numberOfElementsAlloc_.size(); ++i) {
      for(size_t j=0; j<numberOfElementsAlloc_[i]; ++j) {
        data_h_[i][j] = dis(gen);
      }
    }

    if(not kernelLoops_.empty()) {
      if(numberOfElementsAlloc_.empty()) {
        numberOfElementsAlloc_.resize(32);
      }
      else if(numberOfElementsAlloc_[0] < 32) {
        numberOfElementsAlloc_[0] = 32;
      }
    }

    if(not useCachingAllocator_) {
      data_d_.resize(numberOfElementsAlloc_.size());
      for(size_t i=0; i<data_d_.size(); ++i) {
        cuda::throw_if_error(cudaMalloc(&data_d_[i], numberOfElementsAlloc_[i]*sizeof(float)));
      }
    }
  }

  for(const auto& src: iConfig.getParameter<std::vector<edm::InputTag>>("srcs")) {
    srcTokens_.emplace_back(consumes<int>(src));
  }
  cudatest::getTimeCruncher();
}

TestCUDAProducerSimEW::~TestCUDAProducerSimEW() {
  for(auto& ptr: data_d_) {
    cuda::throw_if_error(cudaFree(ptr));
  }
}

void TestCUDAProducerSimEW::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::vector<edm::InputTag>> ("srcs", std::vector<edm::InputTag>{});
  desc.add<std::vector<unsigned int>>("numberOfElementsToDevice", std::vector<unsigned int>{});
  desc.add<std::vector<unsigned int>>("kernelLoops", std::vector<unsigned int>{});
  desc.add<std::vector<unsigned int>>("numberOfElementsToHost", std::vector<unsigned int>{});
  desc.add<bool>("useCachingAllocator", true);
  desc.add<double>("crunchForSeconds", 0);
  descriptions.addWithDefaultLabel(desc);
}

void TestCUDAProducerSimEW::acquire(const edm::Event& iEvent, const edm::EventSetup& iSetup, edm::WaitingTaskWithArenaHolder h) {
  // to make sure the dependencies are set correctly
  for(const auto& t: srcTokens_) {
    iEvent.get(t);
  }

  if(crunchForMicroSeconds_.count() > 0) {
    cudatest::getTimeCruncher().crunch_for(crunchForMicroSeconds_);
  }

  CUDAScopedContextAcquire ctx{iEvent.streamID(), std::move(h)};
  std::vector<float *> data_d = data_d;
  std::vector<cudautils::device::unique_ptr<float[]>> data_d_own;
  if(useCachingAllocator_) {
    data_d.resize(numberOfElementsAlloc_.size());
    data_d_own.resize(numberOfElementsAlloc_.size());
    for(size_t i=0; i<numberOfElementsAlloc_.size(); ++i) {
      data_d_own[i] = cudautils::make_device_unique<float[]>(numberOfElementsAlloc_[i], ctx.stream());
      data_d[i] = data_d_own[i].get();
    }
  }
  for(size_t i=0; i<numberOfElementsToDevice_.size(); ++i) {
    cuda::memory::async::copy(data_d[i], data_h_[i].get(), numberOfElementsToDevice_[i]*sizeof(float), ctx.stream().id());
  }
  for(size_t i=0; i<kernelLoops_.size(); ++i) {
    TestCUDAProducerSimEWGPUKernel::kernel(data_d[0], numberOfElementsToDevice_[0], kernelLoops_[i], ctx.stream());
  }
  for(size_t i=0; i<numberOfElementsToHost_.size(); ++i) {
    cuda::memory::async::copy(data_h_[i].get(), data_d[i], numberOfElementsToHost_[i]*sizeof(float), ctx.stream().id());
  }
}

void TestCUDAProducerSimEW::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  iEvent.emplace(dstToken_, 42);
}

DEFINE_FWK_MODULE(TestCUDAProducerSimEW);
