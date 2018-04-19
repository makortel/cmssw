#include "HeterogeneousCore/AcceleratorService/interface/HeterogeneousEDProducer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"

#include <cuda.h>

#include <exception>
#include <thread>
#include <random>
#include <chrono>

namespace heterogeneous {
  bool CPU::call_launchCPU(HeterogeneousDeviceId *algoExecutionLocation, edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
    std::exception_ptr exc;
    try {
      launchCPU();
      *algoExecutionLocation = HeterogeneousDeviceId(HeterogeneousDevice::kCPU);
    } catch(...) {
      exc = std::current_exception();
    }
    waitingTaskHolder.doneWaiting(exc);
    return true;
  }

  bool GPUMock::call_launchGPUMock(DeviceBitSet inputLocation, HeterogeneousDeviceId *algoExecutionLocation, edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
    // Decide randomly whether to run on GPU or CPU to simulate scheduler decisions
    std::random_device r;
    std::mt19937 gen(r());
    auto dist1 = std::uniform_int_distribution<>(0, 3); // simulate GPU (in)availability
    if(dist1(gen) == 0) {
      edm::LogPrint("HeterogeneousEDProducer") << "Mock GPU is not available (by chance)";
      return false;
    }

    try {
      launchGPUMock([waitingTaskHolder, // copy needed for the catch block
                     &algoExecutionLocation = *algoExecutionLocation
                     ]() mutable {
                      algoExecutionLocation = HeterogeneousDeviceId(HeterogeneousDevice::kGPUMock, 0);
                      waitingTaskHolder.doneWaiting(nullptr);
                    });
    } catch(...) {
      waitingTaskHolder.doneWaiting(std::current_exception());
    }
    return true;
  }

  bool GPUCuda::call_launchGPUCuda(DeviceBitSet inputLocation, HeterogeneousDeviceId *algoExecutionLocation, edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
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
      launchGPUCuda([waitingTaskHolder, // copy needed for the catch block
                     &algoExecutionLocation = *algoExecutionLocation
                     ](cuda::device::id_t deviceId, cuda::stream::id_t streamId, cuda::status_t status) mutable {
                      if(status == cudaSuccess) {
                        algoExecutionLocation = HeterogeneousDeviceId(HeterogeneousDevice::kGPUCuda, deviceId);
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

  void GPUCuda::call_produceGPUCuda(const HeterogeneousDeviceId& algoExecutionLocation, edm::Event& iEvent, const edm::EventSetup& iSetup) {
    // I guess we have to assume that produce() may be called from a different thread than acquire() was run
    // The current CUDA device is a thread-local property, so have to set it here
    edm::Service<CUDAService> cudaService;
    // TODO: Consider using cuda::device::current::scoped_override_t<>?
    cudaService->setCurrentDevice(algoExecutionLocation.deviceId());

    produceGPUCuda(algoExecutionLocation, iEvent, iSetup);
  }
}
