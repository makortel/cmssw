#ifndef CUDADataFormats_Common_CUDA_h
#define CUDADataFormats_Common_CUDA_h

#include <optional>

#include <cuda/api_wrappers.h>

namespace edm {
  template <typename T> class Wrapper;
}

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

  bool isValid() const { return stream_.get() != nullptr; }

  int device() const { return device_; }

  const cuda::stream_t<>& stream() const { return *stream_; }
  cuda::stream_t<>& stream() { return *stream_; }
  const std::shared_ptr<cuda::stream_t<>>& streamPtr() const { return stream_; }

  const cuda::event_t& event() const { return *event_; }
  cuda::event_t& event() { return *event_; }

private:
  friend class CUDAScopedContext;
  friend class edm::Wrapper<CUDA<T>>;

  // Using template to break circular dependency
  template <typename Context>
  explicit CUDA(const Context& ctx, T data):
    stream_(ctx.streamPtr()),
    event_(std::make_unique<cuda::event_t>(cuda::event::create(ctx.device(),
                                                               cuda::event::sync_by_busy_waiting,   // default; we should try to avoid explicit synchronization, so maybe the value doesn't matter much?
                                                               cuda::event::dont_record_timings))), // it should be a bit faster to ignore timings
    data_(std::move(data)),
    device_(ctx.device())
  {
    // Record CUDA event to the CUDA stream. The event will become
    // "occurred" after all work queued to the stream before this
    // point has been finished.
    event_->record(stream_->id());
  }

private:
  // The cuda::stream_t is really shared among edm::Event products, so
  // using shared_ptr also here
  std::shared_ptr<cuda::stream_t<>> stream_; //!
  // Using unique_ptr to support the default constructor. Tried
  // std::optional, but cuda::event_t has its move assignment
  // operators deleted.
  std::unique_ptr<cuda::event_t> event_; //!

  T data_; //!
  int device_ = -1; //!
};

#endif
