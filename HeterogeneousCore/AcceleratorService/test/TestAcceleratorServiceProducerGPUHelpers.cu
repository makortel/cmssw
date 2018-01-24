#include "TestAcceleratorServiceProducerGPUHelpers.h"

#include <cuda/api_wrappers.h>
#include <cuda.h>
#include <cuda_runtime.h>

//
// Vector Addition Kernel
//
namespace {
template<typename T>
__global__
void vectorAdd(const T *a, const T *b, T *c, int numElements) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < numElements) { c[i] = a[i] + b[i]; }
}
}

int TestAcceleratorServiceProducerGPUHelpers_simple_kernel(int input) {
  // Example from Viktor/cuda-api-wrappers
  constexpr int NUM_VALUES = 10000;

  //cudaStream_t stream;
  //cudaStreamCreate(&stream);

  auto h_a = std::make_unique<int[]>(NUM_VALUES);
  auto h_b = std::make_unique<int[]>(NUM_VALUES);
  auto h_c = std::make_unique<int[]>(NUM_VALUES);

  for (auto i=0; i<NUM_VALUES; i++) {
    h_a[i] = input + i;
    h_b[i] = i*i;
  }

  auto current_device = cuda::device::current::get();
  auto d_a = cuda::memory::device::make_unique<int[]>(current_device, NUM_VALUES);
  auto d_b = cuda::memory::device::make_unique<int[]>(current_device, NUM_VALUES);
  auto d_c = cuda::memory::device::make_unique<int[]>(current_device, NUM_VALUES);

  cuda::memory::copy(d_a.get(), h_a.get(), NUM_VALUES*sizeof(int));
  cuda::memory::copy(d_b.get(), h_b.get(), NUM_VALUES*sizeof(int));

  int threadsPerBlock {256};
  int blocksPerGrid = (NUM_VALUES + threadsPerBlock - 1) / threadsPerBlock;

  vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a.get(), d_b.get(), d_c.get(), NUM_VALUES);
  //vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_a, d_b, d_c, NUM_VALUES);
  /*
  cuda::launch(vectorAdd, {blocksPerGrid, threadsPerBlock},
               d_a.get(), d_b.get(), d_c.get(), NUM_VALUES);
  */

  cuda::memory::copy(h_c.get(), d_c.get(), NUM_VALUES*sizeof(int));

  //cudaStreamSynchronize(stream);

  //cudaStreamDestroy(stream);

  int ret = 0;
  for (auto i=0; i<10; i++) {
    ret += h_c[i];
  }

  return ret;
}

