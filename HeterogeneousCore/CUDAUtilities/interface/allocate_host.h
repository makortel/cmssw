#ifndef HeterogeneousCore_CUDAUtilities_allocate_host_h
#define HeterogeneousCore_CUDAUtilities_allocate_host_h

#include <cuda/api_wrappers.h>

namespace cudautils {
  // Allocate pinned host memory (to be called from unique_ptr)
  // This variant does not create device-side ownership
  void *allocate_host(size_t nbytes);

  // Allocate pinned host memory (to be called from unique_ptr)
  // This variant creates device-side ownership. When freed, all work
  // in the stream up to the freeing point must be finished for the
  // memory block to be considered free (except for new allocation in
  // the same stream)
  void *allocate_host(size_t nbytes, cuda::stream_t<> &stream);

  // Free pinned host memory (to be called from unique_ptr)
  void free_host(void *ptr);
}  // namespace cudautils

#endif
