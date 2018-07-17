#ifndef HeterogeneousCore_CUDACore_CUDAToken_h
#define HeterogeneousCore_CUDACore_CUDAToken_h

#include <cuda_runtime.h>

/**
 * The purpose of this class is to deliver the device and CUDA stream
 * information from CUDADeviceChooser to the EDModules with CUDA
 * implementation.
 */
class CUDAToken {
public:
  CUDAToken() = default;
  explicit CUDAToken(int device, cudaStream_t stream): stream_(stream), device_(device) {}

  int device() const { return device_; }
  cudaStream_t stream() const { return stream_; }
  
private:
  mutable cudaStream_t stream_ = nullptr; // it is a pointer, we don't own it, but we need to be able to pass it around as non-const
  int device_ = -1;
};

#endif
