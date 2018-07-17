#ifndef HeterogeneousCore_CUDACore_CUDA_h
#define HeterogeneousCore_CUDACore_CUDA_h

#include <optional>

#include <cuda/api_wrappers.h>

/**
 * The purpose of this class is to wrap CUDA data to edm::Event in a
 * way which forces correct use of various utilities.
 *
 *
 * The CUDA event is in practice needed only for stream-stream
 * synchronization, but someone with long-enough lifetime has to own
 * it. Here is a somewhat natural place. If overhead is too much, we
 * can e.g. make CUDAService own them (creating them on demand) and
 * use them only where synchronization between streams is needed.
 */
template <typename T>
class CUDA {
public:
  CUDA() = default;

  template <typename TokenOrContext>
  explicit CUDA(T data, const TokenOrContext& token):
    stream_(&token.stream()),
    event_(cuda::event::create(token.device(),
                               cuda::event::sync_by_busy_waiting,  // default; we should try to avoid explicit synchronization, so maybe the value doesn't matter much?
                               cuda::event::dont_record_timings)), // it should be a bit faster to ignore timings
    data_(std::move(data)),
    device_(token.device())
  {}

  CUDA(const CUDA&) = delete;
  CUDA& operator=(const CUDA&) = delete;
  CUDA(CUDA&&) = default;
  CUDA& operator=(CUDA&&) = default;

  int device() const { return device_; }
  const cuda::stream_t<>& stream() const { return *stream_; }
  const cuda::event_t& event() const { return *event_; }

private:
  friend class CUDAScopedContext;

  const cuda::stream_t<> *stream_ = nullptr; // owned elsewhere with long-enough life time
  std::optional<cuda::event_t> event_; // std::optional used only to support default constructor (which is needed for ROOT dictionaries)
  T data_;
  int device_ = -1;
};

#endif
