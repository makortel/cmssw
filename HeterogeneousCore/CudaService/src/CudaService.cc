#include "HeterogeneousCore/CudaService/interface/CudaService.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include <cuda.h>

#include <dlfcn.h>

CudaService::CudaService(edm::ParameterSet const& iConfig, edm::ActivityRegistry& iRegistry) {
  bool configEnabled = iConfig.getUntrackedParameter<bool>("enabled");
  if(!configEnabled) {
    edm::LogInfo("CudaService") << "CudaService disabled by configuration";
  }

  // First check if we can load the cuda runtime library
  void *cudaLib = dlopen("libcuda.so", RTLD_NOW);
  if(cudaLib == nullptr) {
    edm::LogWarning("CudaService") << "Failed to dlopen libcuda.so, disabling CudaService";
    return;
  }
  edm::LogInfo("CudaService") << "Found libcuda.so";

  // Find functions
  auto cuInit = reinterpret_cast<CUresult (*)(unsigned int Flags)>(dlsym(cudaLib, "cuInit"));
  if(cuInit == nullptr) {
    edm::LogWarning("CudaService") << "Failed to find cuInit from libcuda.so, disabling CudaService";
    return;
  }
  edm::LogInfo("CudaService") << "Found cuInit from libcuda";

  auto cuDeviceGetCount = reinterpret_cast<CUresult (*)(int *count)>(dlsym(cudaLib, "cuDeviceGetCount"));
  if(cuDeviceGetCount == nullptr) {
    edm::LogWarning("CudaService") << "Failed to find cuDeviceGetCount from libcuda.so, disabling CudaService";
    return;
  }
  edm::LogInfo("CudaService") << "Found cuDeviceGetCount from libcuda";

  auto cuDeviceComputeCapability = reinterpret_cast<CUresult (*)(int *major, int *minor, CUdevice dev)>(dlsym(cudaLib, "cuDeviceComputeCapability"));
  if(cuDeviceComputeCapability == nullptr) {
    edm::LogWarning("CudaService") << "Failed to find cuDeviceComputeCapability from libcuda.so, disabling CudaService";
    return;
  }
  edm::LogInfo("CudaService") << "Found cuDeviceComputeCapability";

  // Then call functions
  auto ret = cuInit(0);
  if(CUDA_SUCCESS != ret) {
    edm::LogWarning("CudaService") << "cuInit failed, return value " << ret << ", disabling CudaService";
    return;
  }
  edm::LogInfo("CudaService") << "cuInit succeeded";

  ret = cuDeviceGetCount(&numberOfDevices_);
  if(CUDA_SUCCESS != ret) {
    edm::LogWarning("CudaService") << "cuDeviceGetCount failed, return value " << ret << ", disabling CudaService";
    return;
  }
  edm::LogInfo("CudaService") << "cuDeviceGetCount succeeded, found " << numberOfDevices_ << " devices";
  if(numberOfDevices_ < 1) {
    edm::LogWarning("CudaService") << "Number of devices < 1, disabling CudaService";
    return;
  }

  computeCapabilities_.reserve(numberOfDevices_);
  for(int i=0; i<numberOfDevices_; ++i) {
    int major, minor;
    ret = cuDeviceComputeCapability(&major, &minor, i);
    if(CUDA_SUCCESS != ret) {
      edm::LogWarning("CudaService") << "cuDeviceComputeCapability failed for device " << i << ", return value " << ret << " disabling CudaService";
      return;
    }
    edm::LogInfo("CudaService") << "Device " << i << " compute capability major " << major << " minor " << minor;
    computeCapabilities_.emplace_back(major, minor);
  }

  edm::LogInfo("CudaService") << "CudaService fully initialized";
  enabled_ = true;
}

void CudaService::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<bool>("enabled", true);

  descriptions.add("CudaService", desc);
}
