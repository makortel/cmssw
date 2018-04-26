#ifndef HeterogeneousCore_CUDAServices_GPUCuda_h
#define HeterogeneousCore_CUDAServices_GPUCuda_h

#include "FWCore/Concurrency/interface/WaitingTaskWithArenaHolder.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "HeterogeneousCore/HeterogeneousEDProducer/interface/DeviceWrapper.h"
#include "HeterogeneousCore/HeterogeneousEDProducer/interface/HeterogeneousEvent.h"

#include <cuda/api_wrappers.h>

namespace heterogeneous {
  class GPUCuda {
  public:
    using CallbackType = std::function<void(cuda::device::id_t, cuda::stream::id_t, cuda::status_t)>;

    void call_beginStreamGPUCuda(edm::StreamID id);
    bool call_acquireGPUCuda(DeviceBitSet inputLocation, edm::HeterogeneousEvent& iEvent, const edm::EventSetup& iSetup, edm::WaitingTaskWithArenaHolder waitingTaskHolder);
    void call_produceGPUCuda(edm::HeterogeneousEvent& iEvent, const edm::EventSetup& iSetup);

  private:
    virtual void beginStreamGPUCuda(edm::StreamID id) {};
    virtual void acquireGPUCuda(const edm::HeterogeneousEvent& iEvent, const edm::EventSetup& iSetup, CallbackType callback) = 0;
    virtual void produceGPUCuda(edm::HeterogeneousEvent& iEvent, const edm::EventSetup& iSetup) = 0;
  };
  DEFINE_DEVICE_WRAPPER(GPUCuda, HeterogeneousDevice::kGPUCuda);
}

#endif
