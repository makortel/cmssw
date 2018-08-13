#include <iostream>

#include <cuda_runtime.h>

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

namespace {
  constexpr size_t WIDTH = 256;
  constexpr size_t HEIGHT = 3;

  __global__ void kernel_test(int *ptr, size_t pitch) {
    auto index = threadIdx.x + blockIdx.x * blockDim.x;
    auto i2 = threadIdx.x + pitch*blockIdx.x;
    ptr[i2] = index;
  }

}

int main() {
  int *h_ptr = nullptr;
  int *d_ptr = nullptr;

  std::cout << "2D array, width " << WIDTH << " height " << HEIGHT << " size of element " << sizeof(int) << std::endl;

  size_t cpuPitch = WIDTH*sizeof(int);
  cudaCheck(cudaMallocHost(&h_ptr, cpuPitch*HEIGHT));
  std::cout << "Allocated host memory for " << cpuPitch*HEIGHT << " bytes" << std::endl;
  size_t gpuPitch = 0;
  cudaCheck(cudaMallocPitch(&d_ptr, &gpuPitch, WIDTH*sizeof(int), HEIGHT));
  std::cout << "Allocated device memory, pitch is " << gpuPitch << " so total allocated memory is " << gpuPitch*HEIGHT << std::endl;

  constexpr size_t ELEM = 256;
  std::cout << "Running kernel for " << ELEM << " elements, device pitch corresponds to " << gpuPitch/sizeof(int) << " elements" << std::endl;

  kernel_test<<<HEIGHT, ELEM>>>(d_ptr, gpuPitch/sizeof(int));
  cudaDeviceSynchronize();
  cudaCheck(cudaGetLastError());

  cudaCheck(cudaMemcpy2D(h_ptr, cpuPitch,
                         d_ptr, gpuPitch,
                         ELEM*sizeof(int), HEIGHT,
                         cudaMemcpyDefault));
  std::cout << "Copied data back with cudaMemcpy2D" << std::endl;
  for(size_t i=0; i<HEIGHT; ++i) {
    std::cout << "Row " << i << std::endl;
    for(size_t j=0; j<ELEM; ++j) {
      std::cout << " " << h_ptr[i*WIDTH + j];
    }
    std::cout << std::endl;
  }

  return 0;
}
