#ifndef HeterogeneousCore_CUDAUtilities_CUDAHostAllocator_h
#define HeterogeneousCore_CUDAUtilities_CUDAHostAllocator_h

#include <memory>
#include <new>
#include <cuda_runtime.h>

#include "HeterogeneousCore/CUDAUtilities/interface/cuda_bad_alloc.h"

template <typename T, unsigned int FLAGS = cudaHostAllocDefault> class CUDAHostAllocator {
public:
  using value_type = T;

  template <typename U>
  struct rebind { 
    using other = CUDAHostAllocator<U, FLAGS>;
  };

  T* allocate(std::size_t n) const __attribute__((warn_unused_result)) __attribute__((malloc)) __attribute__((returns_nonnull))
  {
    void* ptr = nullptr;
    cudaError_t status = cudaMallocHost(&ptr, n * sizeof(T), FLAGS);
    if (status != cudaSuccess) {
      throw cuda_bad_alloc(status);
    }
    if (ptr == nullptr) {
      throw std::bad_alloc();
    }
    return static_cast<T*>(ptr);
  }

  void deallocate(T* p, std::size_t n) const
  {
    cudaError_t status = cudaFreeHost(p);
    if (status != cudaSuccess) {
      throw cuda_bad_alloc(status);
    }
  }

};

#endif // HeterogeneousCore_CUDAUtilities_CUDAHostAllocator_h
