#ifndef HeterogeneousCore_CUDATest_TestCUDAProducerSimEWGPUKernel_h
#define HeterogeneousCore_CUDATest_TestCUDAProducerSimEWGPUKernel_h

#include <cuda/api_wrappers.h>

struct TestCUDAProducerSimEWGPUKernel {
  static void kernel(float *data, size_t elements, size_t loops, cuda::stream_t<>& stream);
};

#endif
