#include "HeterogeneousCore/AcceleratorService/interface/HeterogeneousEDProducer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"

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

  bool GPUMock::call_launchGPUMock(HeterogeneousDeviceId *algoExecutionLocation, edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
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

  bool GPUCuda::call_launchGPUCuda(HeterogeneousDeviceId *algoExecutionLocation, edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
    edm::Service<CUDAService> cudaService;
    if(!cudaService->enabled()) {
      return false;
    }

    try {
      launchGPUCuda([waitingTaskHolder, // copy needed for the catch block
                     &algoExecutionLocation = *algoExecutionLocation
                     ]() mutable {
                      algoExecutionLocation = HeterogeneousDeviceId(HeterogeneousDevice::kGPUCuda, 0);
                      waitingTaskHolder.doneWaiting(nullptr);
                    });
    } catch(...) {
      waitingTaskHolder.doneWaiting(std::current_exception());
    }
    return true;
  }
}
