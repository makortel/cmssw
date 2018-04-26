#include "HeterogeneousCore/CUDACore/interface/GPUCuda.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"

#include <cuda.h>

#include <exception>

namespace heterogeneous {
  void GPUCuda::call_beginStreamGPUCuda(edm::StreamID id) {
    beginStreamGPUCuda(id);
  }

  bool GPUCuda::call_acquireGPUCuda(DeviceBitSet inputLocation, edm::HeterogeneousEvent& iEvent, const edm::EventSetup& iSetup, edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
    edm::Service<CUDAService> cudaService;
    if(!cudaService->enabled()) {
      return false;
    }

    // set device
    // if there is input, use that
    // if there is no input, use device with most free memory (simple&stupid strategy, to be improved)
    // eventually in case of certain failures (e.g. no free memory) try another device
    int device = -1;
    if(inputLocation.any()) {
      for(unsigned int i=0; i<inputLocation.size(); ++i) {
        // TODO: how to deal if input is on multiple CUDA devices?
        // For now just pick the first one
        if(inputLocation[i]) {
          device = i;
          break;
        }
      }
    }
    if(device < 0) {
      device = cudaService->deviceWithMostFreeMemory();
    }
    // In this case we can't do anything with the GPU
    if(device < 0) {
      return false;
    }

    // TODO: Consider using cuda::device::current::scoped_override_t<>?
    cudaService->setCurrentDevice(device);

    try {
      iEvent.setInputLocation(HeterogeneousDeviceId(HeterogeneousDevice::kGPUCuda, 0));
      acquireGPUCuda(iEvent, iSetup,
                     [waitingTaskHolder, // copy needed for the catch block
                      locationSetter = iEvent.locationSetter()
                      ](cuda::device::id_t deviceId, cuda::stream::id_t streamId, cuda::status_t status) mutable {
                       if(status == cudaSuccess) {
                         locationSetter(HeterogeneousDeviceId(HeterogeneousDevice::kGPUCuda, deviceId));
                         waitingTaskHolder.doneWaiting(nullptr);
                       }
                       else {
                         auto error = cudaGetErrorName(status);
                         auto message = cudaGetErrorString(status);
                         waitingTaskHolder.doneWaiting(std::make_exception_ptr(cms::Exception("CUDAError") << "Callback of CUDA stream " << streamId << " in device " << deviceId << " error " << error << ": " << message));
                       }
                     });
    } catch(...) {
      waitingTaskHolder.doneWaiting(std::current_exception());
    }
    return true;
  }

  void GPUCuda::call_produceGPUCuda(edm::HeterogeneousEvent& iEvent, const edm::EventSetup& iSetup) {
    // I guess we have to assume that produce() may be called from a different thread than acquire() was run
    // The current CUDA device is a thread-local property, so have to set it here
    edm::Service<CUDAService> cudaService;
    // TODO: Consider using cuda::device::current::scoped_override_t<>?
    cudaService->setCurrentDevice(iEvent.location().deviceId());

    produceGPUCuda(iEvent, iSetup);
  }
}
