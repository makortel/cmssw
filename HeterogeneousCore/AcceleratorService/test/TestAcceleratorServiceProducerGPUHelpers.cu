#include "TestAcceleratorServiceProducerGPUHelpers.h"

#include <cuda.h>
#include <cuda_runtime.h>

//
// Vector Addition Kernel
//
namespace {
template<typename T>
__global__
void vectorAdd(T *a, T *b, T *c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = a[i] + b[i];
}
}

int TestAcceleratorServiceProducerGPUHelpers_simple_kernel(int input) {
  // Example from Viktor
  constexpr int NUM_VALUES = 10000;

  cudaStream_t stream;

  cudaStreamCreate(&stream);
  
  int h_a[NUM_VALUES], h_b[NUM_VALUES], h_c[NUM_VALUES];
  for (auto i=0; i<NUM_VALUES; i++) {
    h_a[i] = input + i;
    h_b[i] = i*i;
  }

  int *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, NUM_VALUES*sizeof(int));
  cudaMalloc(&d_b, NUM_VALUES*sizeof(int));
  cudaMalloc(&d_c, NUM_VALUES*sizeof(int));

  cudaMemcpyAsync(d_a, h_a, NUM_VALUES*sizeof(int), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_b, h_b, NUM_VALUES*sizeof(int), cudaMemcpyHostToDevice, stream);

  int threadsPerBlock {256};
  int blocksPerGrid = (NUM_VALUES + threadsPerBlock - 1) / threadsPerBlock;
  vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_a, d_b, d_c);

  cudaMemcpyAsync(h_c, d_c, NUM_VALUES*sizeof(int), cudaMemcpyDeviceToHost, stream);

  cudaStreamSynchronize(stream);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  cudaStreamDestroy(stream);

  int ret = 0;
  for (auto i=0; i<10; i++) {
    ret += h_c[i];
  }

  return ret;
}

