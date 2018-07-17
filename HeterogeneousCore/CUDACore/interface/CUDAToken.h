#ifndef HeterogeneousCore_CUDACore_CUDAToken_h
#define HeterogeneousCore_CUDACore_CUDAToken_h

#include <cuda/api_wrappers.h>

#include <memory>

/**
 * The purpose of this class is to deliver the device and CUDA stream
 * information from CUDADeviceChooser to the EDModules with CUDA
 * implementation.
 *
 * Currently the class is declared as transient in the dictionary, but
 * in principle (for debugging purposes) it could be possible to
 * persist it by marking only the CUDA stream as transient.
 *
 * Note that the CUDA stream is returned only as a const reference.
 * Various methods (e.g. cuda::stream_t<>::synchronize()) are
 * non-const, but on the other hand cuda:stream_t is just a handle
 * wrapping the real CUDA stream, and can thus be cheaply copied as a
 * non-owning non-const handle.
 */
class CUDAToken {
public:
  CUDAToken() = default;
  explicit CUDAToken(int device);

  ~CUDAToken();

  CUDAToken(const CUDAToken&) = delete;
  CUDAToken& operator=(const CUDAToken&) = delete;
  CUDAToken(CUDAToken&&) = default;
  CUDAToken& operator=(CUDAToken&&) = default;

  int device() const { return device_; }
  const cuda::stream_t<>& stream() const { return *stream_; }
  
private:
  std::unique_ptr<cuda::stream_t<>> stream_;
  int device_ = -1;
};

#endif
