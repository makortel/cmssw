// -*- C++ -*-

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#include <cuda.h>

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

namespace {
  constexpr size_t kernel_elements = 32;

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

int main(int argc, char **argv) {
  if(argc < 2) {
    std::cout << "Need at least 1 argument for the numbers of iterations" << std::endl;
    return 1;
  }

  // Read input
  std::vector<size_t> iters(argc-1, 0);
  char *tmp;
  std::transform(argv+1, argv+argc, iters.begin(), [&tmp](const char *str) {
      auto val = std::strtol(str, &tmp, 10);
      if(val < 0) {
        std::cout << "Got a negative number " << val << std::endl;
        abort();
      }
      return val;
    });

  // Make sure they're increasing
  for(size_t i=1; i<iters.size(); ++i) {
    if(iters[i-1] >= iters[i]) {
      std::cout << "Number of iterations for i " << i-1 << " (" << iters[i-1] << ") is larger than or equal for i " << i << " (" << iters[i] << ")" << std::endl;
      return 1;
    }
  }

  cudaStream_t stream;
  cudaCheck(cudaStreamCreate(&stream));

  float* data_h;
  float* data_d;
  cudaCheck(cudaMallocHost(&data_h, kernel_elements*sizeof(float)));
  cudaCheck(cudaMalloc(&data_d, kernel_elements*sizeof(float)));

  // Data for kernel
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(1e-5, 100.);

  for(size_t i=0; i!=kernel_elements; ++i) {
    data_h[i] = dis(gen);
  }
  cudaCheck(cudaMemcpy(data_d, data_h, kernel_elements*sizeof(float), cudaMemcpyDefault));
  
  // do 4 warmups
  for(size_t i=0; i<4; ++i) {
    kernel_looping<<<1, kernel_elements, 0, stream>>>(data_d, kernel_elements, iters.back());
  }

  // Then repeat all 4 times
  for(size_t i=0; i<4; ++i) {
    for(int n: iters) {
      kernel_looping<<<1, kernel_elements, 0, stream>>>(data_d, kernel_elements, n);
    }
  }

  cudaCheck(cudaFree(data_d));
  cudaCheck(cudaFreeHost(data_h));
  cudaCheck(cudaStreamDestroy(stream));

  return 0;
}
