#ifndef HeterogeneousCore_CUDAUtilities_memsetAsync_h
#define HeterogeneousCore_CUDAUtilities_memsetAsync_h

#include "CUDADataFormats/Common/interface/device_unique_ptr.h"

#include <cuda/api_wrappers.h>

#include <type_traits>

namespace cudautils {
  template <typename T>
  inline
  void memsetAsync(edm::cuda::device::unique_ptr<T>& ptr, T value, cuda::stream_t<>& stream) {
    // Shouldn't compile for array types because of sizeof(T), but
    // let's add an assert with a more helpful message
    static_assert(std::is_array<T>::value == false, "For array types, use the other overload with the size parameter");
    cuda::memory::device::async::set(ptr.get(), value, sizeof(T), stream.id());
  }

  template <typename T>
  inline
  void memsetAsync(edm::cuda::device::unique_ptr<T[]>& ptr, T value, size_t nelements, cuda::stream_t<>& stream) {
    cuda::memory::device::async::set(ptr.get(), value, nelements*sizeof(T), stream.id());
  }
}

#endif
