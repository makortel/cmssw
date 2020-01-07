#ifndef HeterogenousCore_CUDAUtilities_deviceCount_h
#define HeterogenousCore_CUDAUtilities_deviceCount_h

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

#include <cuda_runtime.h>

namespace cudautils {
  inline int deviceCount() {
    int ndevices;
    cudaCheck(cudaGetDeviceCount(&ndevices));
    return ndevices;
  }
}  // namespace cudautils

#endif
