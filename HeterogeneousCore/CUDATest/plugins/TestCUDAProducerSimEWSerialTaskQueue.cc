#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Concurrency/interface/SerialTaskQueue.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"

#include "HeterogeneousCore/CUDACore/interface/CUDAScopedContext.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_noncached_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"

#include "TestCUDAProducerSimEWGPUKernel.h"

#include <random>

namespace {
  edm::SerialTaskQueue taskQueue;
}

class TestCUDAProducerSimEWSerialTaskQueue: public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit TestCUDAProducerSimEWSerialTaskQueue(const edm::ParameterSet& iConfig);
  ~TestCUDAProducerSimEWSerialTaskQueue() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void acquire(const edm::Event& iEvent, const edm::EventSetup& iSetup, edm::WaitingTaskWithArenaHolder) override;
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;
private:
  
  const edm::EDPutTokenT<int> dstToken_;
  const size_t numberOfElements_;
  const int kernels_;
  const int kernelLoops_;
  const bool useCachingAllocator_;
  const bool transferDevice_;
  const bool transferHost_;

  float *data_d_ = nullptr;
  cudautils::host::noncached::unique_ptr<float[]> data_h_;
};

TestCUDAProducerSimEWSerialTaskQueue::TestCUDAProducerSimEWSerialTaskQueue(const edm::ParameterSet& iConfig):
  dstToken_{produces<int>()},
  numberOfElements_{iConfig.getParameter<unsigned int>("numberOfElements")},
  kernels_{iConfig.getParameter<int>("kernels")},
  kernelLoops_{iConfig.getParameter<int>("kernelLoops")},
  useCachingAllocator_{iConfig.getParameter<bool>("useCachingAllocator")},
  transferDevice_{iConfig.getParameter<bool>("transferDevice")},
  transferHost_{iConfig.getParameter<bool>("transferHost")}
{
  edm::Service<CUDAService> cs;
  if(cs->enabled()) {
    data_h_ = cudautils::make_host_noncached_unique<float[]>(numberOfElements_ /*, cudaHostAllocWriteCombined*/);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(1e-5, 100.);
    for(size_t i=0; i<numberOfElements_; ++i) {
      data_h_[i] = dis(gen);
    }

    if(not useCachingAllocator_) {
      cuda::throw_if_error(cudaMalloc(&data_d_, numberOfElements_*sizeof(float)));
    }
  }
}

TestCUDAProducerSimEWSerialTaskQueue::~TestCUDAProducerSimEWSerialTaskQueue() {
  if(not useCachingAllocator_) {
    cuda::throw_if_error(cudaFree(data_d_));
  }
}
void TestCUDAProducerSimEWSerialTaskQueue::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<unsigned int>("numberOfElements", 1);
  desc.add<bool>("useCachingAllocator", true);
  desc.add<bool>("transferDevice", false);
  desc.add<int>("kernels", 1);
  desc.add<int>("kernelLoops", -1);
  desc.add<bool>("transferHost", false);
  descriptions.addWithDefaultLabel(desc);
}

void TestCUDAProducerSimEWSerialTaskQueue::acquire(const edm::Event& iEvent, const edm::EventSetup& iSetup, edm::WaitingTaskWithArenaHolder h) {
  taskQueue.push([this,h,streamID=iEvent.streamID()](){
    CUDAScopedContextAcquire ctx{streamID, std::move(h)};
    float *data_d = data_d_;
    cudautils::device::unique_ptr<float[]> data_d_own;

    if(useCachingAllocator_) {
      data_d_own = cudautils::make_device_unique<float[]>(numberOfElements_, ctx.stream());
      data_d = data_d_own.get();
    }
    if(transferDevice_) {
      cuda::memory::async::copy(data_d, data_h_.get(), numberOfElements_*sizeof(float), ctx.stream().id());
      if(kernelLoops_ > 0) {
        for(int i=0; i<kernels_; ++i) {
          TestCUDAProducerSimEWGPUKernel::kernel(data_d, numberOfElements_, kernelLoops_, ctx.stream());
        }
      }
      if(transferHost_) {
        cuda::memory::async::copy(data_h_.get(), data_d, numberOfElements_*sizeof(float), ctx.stream().id());
      }
    }

  });
}

void TestCUDAProducerSimEWSerialTaskQueue::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  iEvent.emplace(dstToken_, 42);
}

DEFINE_FWK_MODULE(TestCUDAProducerSimEWSerialTaskQueue);
