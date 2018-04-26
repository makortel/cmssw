#include "HeterogeneousCore/CUDACore/interface/GPUCuda.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"

#include <cuda.h>

#include <exception>

namespace heterogeneous {
  void GPUCuda::call_beginStreamGPUCuda(edm::StreamID id) {
    edm::Service<CUDAService> cudaService;
    if(!cudaService->enabled()) {
      return;
    }

    // For startes we "statically" assign the device based on
    // edm::Stream number. This is suboptimal if the number of
    // edm::Streams is not a multiple of the number of CUDA devices
    // (and even then there is no load balancing).
    //
    // TODO: improve. Possible ideas include
    // - allocate M (< N(edm::Streams)) buffers per device per module, choose dynamically which (buffer, device) to use
    //   * the first module of a chain dictates the device for the rest of the chain
    // - our own CUDA memory allocator
    //   * being able to cheaply allocate+deallocate scratch memory allows to make the execution fully dynamic e.g. based on current load
    //   * would probably still need some buffer space/device to hold e.g. conditions data
    //     - for conditions, how to handle multiple lumis per job?
    deviceId_ = id % cudaService->numberOfDevices();
    // TODO: Consider using cuda::device::current::scoped_override_t<>?
    cudaService->setCurrentDevice(deviceId_);

    beginStreamGPUCuda(id);
  }

  bool GPUCuda::call_acquireGPUCuda(DeviceBitSet inputLocation, edm::HeterogeneousEvent& iEvent, const edm::EventSetup& iSetup, edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
    edm::Service<CUDAService> cudaService;
    if(!cudaService->enabled()) {
      return false;
    }

    // TODO: Consider using cuda::device::current::scoped_override_t<>?
    cudaService->setCurrentDevice(deviceId_);

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
