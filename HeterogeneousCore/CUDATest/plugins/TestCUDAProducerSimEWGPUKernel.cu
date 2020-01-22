#include "TestCUDAProducerSimEWGPUKernel.h"

namespace {
  __global__ void kernel_looping(float *a, size_t size, size_t loops) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    a[idx] = 0.f;

    for (size_t iloop = 0; iloop < loops; ++iloop) {
      a[idx] = (a[idx] + 4.0f) * 0.5f - 1.0f;
    }
  }
}  // namespace

void TestCUDAProducerSimEWGPUKernel::kernel(float *data, size_t elements, size_t loops, cudaStream_t stream) {
  kernel_looping<<<1, 32, 0, stream>>>(data, elements, loops);
}
