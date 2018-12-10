#ifndef HeterogeneousCore_CUDACore_TestCUDAProducerGPUKernel_h
#define HeterogeneousCore_CUDACore_TestCUDAProducerGPUKernel_h

#include "CUDADataFormats/Common/interface/device_unique_ptr.h"

#include <cuda/api_wrappers.h>

/**
 * This class models the actual CUDA implementation of an algorithm.
 *
 * Memory is allocated dynamically with the allocator in CUDAService
 *
 * The algorithm is intended to waste time with large matrix
 * operations so that the asynchronous nature of the CUDA integration
 * becomes visible with debug prints.
 */
class TestCUDAProducerGPUKernel {
public:
  static constexpr int NUM_VALUES = 4000;

  TestCUDAProducerGPUKernel() = default;
  ~TestCUDAProducerGPUKernel() = default;

  // returns (owning) pointer to device memory
  edm::cuda::device::unique_ptr<float[]> runAlgo(const std::string& label, cuda::stream_t<>& stream) const {
    return runAlgo(label, nullptr, stream);
  }
  edm::cuda::device::unique_ptr<float[]> runAlgo(const std::string& label, const float *d_input, cuda::stream_t<>& stream) const;
};

#endif
