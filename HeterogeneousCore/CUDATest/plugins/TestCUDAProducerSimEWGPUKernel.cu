#include "TestCUDAProducerSimEWGPUKernel.h"

namespace {
  __global__ void kernel_looping(float *a, size_t size, size_t loops) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    for(size_t iloop=0; iloop<loops; ++iloop) {
      size_t ind = iloop*gridDim.x+idx;
      if(ind < size) {
        a[ind] = a[ind] + 4.0f;
      }
    }
  }
}

void TestCUDAProducerSimEWGPUKernel::kernel(float *data, size_t elements, size_t loops, cuda::stream_t<>& stream) {
  kernel_looping<<<1,32,0, stream.id()>>>(data, elements, loops);
}
