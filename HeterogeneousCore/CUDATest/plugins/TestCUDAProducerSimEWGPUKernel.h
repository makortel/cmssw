#ifndef HeterogeneousCore_CUDATest_TestCUDAProducerSimEWGPUKernel_h
#define HeterogeneousCore_CUDATest_TestCUDAProducerSimEWGPUKernel_h

#include <cuda_runtime.h>

struct TestCUDAProducerSimEWGPUKernel {
  static void kernel(float *data, size_t elements, size_t loops, cudaStream_t stream);
};

#endif
