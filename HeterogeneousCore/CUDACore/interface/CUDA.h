#ifndef HeterogeneousCore_CUDACore_CUDA_h
#define HeterogeneousCore_CUDACore_CUDA_h

#include <optional>

#include <cuda/api_wrappers.h>

/**
 * The purpose of this class is to wrap CUDA data to edm::Event in a
 * way which forces correct use of various utilities.
 *
 * The non-default construction has to be done with CUDAScopedContext
 * (in order to properly register the CUDA event).
 *
 * The default constructor is needed only for the ROOT dictionary generation.
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
  CUDA() = default; // Needed only for ROOT dictionary generation

  CUDA(const CUDA&) = delete;
  CUDA& operator=(const CUDA&) = delete;
  CUDA(CUDA&&) = default;
  CUDA& operator=(CUDA&&) = default;

  bool isValid() const { return streamEvent_.get() != nullptr; }

  int device() const { return device_; }

  const cuda::stream_t<>& stream() const { return streamEvent_->stream; }
  cuda::stream_t<>& stream() { return streamEvent_->stream; }

  const cuda::event_t& event() const { return streamEvent_->event; }
  cuda::event_t& event() { return streamEvent_->event; }

private:
  friend class CUDAScopedContext;
  friend class TestCUDA;

  template <typename TokenOrContext>
  explicit CUDA(T data, const TokenOrContext& token):
    streamEvent_(std::make_unique<StreamEvent>(token)),
    data_(std::move(data)),
    device_(token.device())
  {}

  // Using unique_ptr to support the default constructor. Tried
  // std::optional, but cuda::stream_t and cuda::event_t have their
  // move assignment operators deleted. Use a struct to save one
  // memory allocation.
public: // need to be public for ROOT dicrionary generation?
  struct StreamEvent {
    template <typename TokenOrContext>
    explicit StreamEvent(const TokenOrContext& token):
      stream(token.stream()),
      event(cuda::event::create(token.device(),
                                cuda::event::sync_by_busy_waiting, // default; we should try to avoid explicit synchronization, so maybe the value doesn't matter much?
                                cuda::event::dont_record_timings)) // it should be a bit faster to ignore timings
    {}

    cuda::stream_t<> stream; // stream_t is just a handle, the real CUDA stream is owned by CUDAToken (with long-enough life time)
    cuda::event_t event;
  };
private:
  std::unique_ptr<StreamEvent> streamEvent_;
  T data_;
  int device_ = -1;
};

#endif
