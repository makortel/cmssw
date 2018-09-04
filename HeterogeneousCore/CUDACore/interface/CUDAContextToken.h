#ifndef HeterogeneousCore_CUDACore_CUDAContextToken_h
#define HeterogeneousCore_CUDACore_CUDAContextToken_h

#include <memory>

/**
 * The purpose of this class is to deliver the device and CUDA stream
 * information from ExternalWork's acquire() to producer() via a
 * member/StreamCache variable.
 */
class CUDAContextToken {
public:
  CUDAContextToken() = default;
  ~CUDAContextToken() = default;

  CUDAContextToken(const CUDAContextToken&) = delete;
  CUDAContextToken& operator=(const CUDAToken&) = delete;
  CUDAContextToken(CUDAContextToken&&) = default;
  CUDAContextToken& operator=(CUDAContextToken&& other) = default;

private:
  friend class CUDAScopedContext;

  explicit CUDAContextToken(int device, cuda::stream_t<>&& stream):
    stream_(std::make_unique<cuda::stream_t<>>(std::move(stream))),
    device_(device)
  {}

  int device() { return device_; }
  cuda::stream_t<>&& stream() {
    auto ret = std::move(*stream_);
    stream_.reset();
    return std::move(ret);
  }

  std::unique_ptr<cuda::stream_t<>> stream_;
  int device_;
};

#endif
