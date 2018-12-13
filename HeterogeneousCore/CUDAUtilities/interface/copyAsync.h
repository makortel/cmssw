#ifndef HeterogeneousCore_CUDAUtilities_copyAsync_h
#define HeterogeneousCore_CUDAUtilities_copyAsync_h

#include "CUDADataFormats/Common/interface/device_unique_ptr.h"
#include "CUDADataFormats/Common/interface/host_unique_ptr.h"

#include <cuda/api_wrappers.h>

#include <type_traits>

namespace cudautils {
  // Single element
  template <typename T>
  inline
  void copyAsync(edm::cuda::device::unique_ptr<T>& dst, edm::cuda::host::unique_ptr<T>& src, cuda::stream_t<>& stream) {
    // Shouldn't compile for array types because of sizeof(T), but
    // let's add an assert with a more helpful message
    static_assert(std::is_array<T>::value == false, "For array types, use the other overload with the size parameter");
    cuda::memory::async::copy(dst.get(), src.get(), sizeof(T), stream.id());
  }

  template <typename T>
  inline
  void copyAsync(edm::cuda::host::unique_ptr<T>& dst, edm::cuda::device::unique_ptr<T>& src, cuda::stream_t<>& stream) {
    static_assert(std::is_array<T>::value == false, "For array types, use the other overload with the size parameter");
    cuda::memory::async::copy(dst.get(), src.get(), sizeof(T), stream.id());
  }

  // Multiple elements
  template <typename T>
  inline
  void copyAsync(edm::cuda::device::unique_ptr<T[]>& dst, edm::cuda::host::unique_ptr<T[]>& src, size_t nelements, cuda::stream_t<>& stream) {
    cuda::memory::async::copy(dst.get(), src.get(), nelements*sizeof(T), stream.id());
  }

  template <typename T>
  inline
  void copyAsync(edm::cuda::host::unique_ptr<T[]>& dst, edm::cuda::device::unique_ptr<T[]>& src, size_t nelements, cuda::stream_t<>& stream) {
    cuda::memory::async::copy(dst.get(), src.get(), nelements*sizeof(T), stream.id());
  }
}

#endif
