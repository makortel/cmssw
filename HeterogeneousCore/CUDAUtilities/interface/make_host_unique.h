#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"

#include <cuda/api_wrappers.h>
#include <cuda_runtime.h>

namespace cudautils {
  /**
   * The difference wrt. CUDAService::make_host_unique is that these
   * do not cache, so they should not be called per-event.
   */
  template <typename T>
  typename host::impl::make_host_unique_selector<T>::non_array
  make_host_unique(unsigned int flags = cudaHostAllocDefault) {
    static_assert(std::is_trivially_constructible<T>::value, "Allocating with non-trivial constructor on the pinned host memory is not supported");
    void *mem;
    cuda::throw_if_error(cudaHostAlloc(&mem, sizeof(T), flags));
    return typename cudautils::host::impl::make_host_unique_selector<T>::non_array(reinterpret_cast<T *>(mem),
                                                                                   cudautils::host::impl::HostDeleter([](void *ptr) {
                                                                                       cuda::throw_if_error(cudaFreeHost(ptr));
                                                                                     }));
  }

  template <typename T>
  typename host::impl::make_host_unique_selector<T>::unbounded_array
  make_host_unique(size_t n, unsigned int flags = cudaHostAllocDefault) {
    using element_type = typename std::remove_extent<T>::type;
    static_assert(std::is_trivially_constructible<element_type>::value, "Allocating with non-trivial constructor on the pinned host memory is not supported");
    void *mem;
    cuda::throw_if_error(cudaHostAlloc(&mem, n*sizeof(element_type), flags));
    return typename cudautils::host::impl::make_host_unique_selector<T>::unbounded_array(reinterpret_cast<element_type *>(mem),
                                                                                         cudautils::host::impl::HostDeleter([](void *ptr) {
                                                                                             cuda::throw_if_error(cudaFreeHost(ptr));
                                                                                           }));
  }

  template <typename T, typename ...Args>
  typename cudautils::host::impl::make_host_unique_selector<T>::bounded_array
  make_host_unique(Args&&...) = delete;
}
