#ifndef HeterogeneousCore_CUDACore_CUDAScopedContext_h
#define HeterogeneousCore_CUDACore_CUDAScopedContext_h

#include "FWCore/Concurrency/interface/WaitingTaskWithArenaHolder.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "HeterogeneousCore/CUDACore/interface/CUDA.h"
#include "HeterogeneousCore/CUDACore/interface/CUDAContextToken.h"

#include <optional>

/**
 * The aim of this class is to do necessary per-event "initialization":
 * - setting the current device
 * - calling edm::WaitingTaskWithArenaHolder::doneWaiting() when necessary
 * - synchronizing between CUDA streams if necessary
 * and enforce that those get done in a proper way in RAII fashion.
 */
class CUDAScopedContext {
public:
  explicit CUDAScopedContext(edm::StreamID streamID);

  // This constructor takes the device as a parameter. It is mainly
  // inteded for testing, but can be used for special cases if you
  // really know what you're doing. Please use the StreamID overload
  // if at all possible.
  explicit CUDAScopedContext(int device);

  explicit CUDAScopedContext(CUDAContextToken&& token):
    currentDevice_(token.device()),
    setDeviceForThisScope_(currentDevice_),
    stream_(std::move(token.streamPtr()))
  {}

  template<typename T>
  explicit CUDAScopedContext(const CUDA<T>& data):
    currentDevice_(data.device()),
    setDeviceForThisScope_(currentDevice_),
    stream_(data.streamPtr())
  {}

  explicit CUDAScopedContext(edm::StreamID streamID, edm::WaitingTaskWithArenaHolder waitingTaskHolder):
    CUDAScopedContext(streamID)
  {
    waitingTaskHolder_ = waitingTaskHolder;
  }

  template <typename T>
  explicit CUDAScopedContext(const CUDA<T>& data, edm::WaitingTaskWithArenaHolder waitingTaskHolder):
    CUDAScopedContext(data)
  {
    waitingTaskHolder_ = waitingTaskHolder;
  }

  ~CUDAScopedContext();

  int device() const { return currentDevice_; }

  cuda::stream_t<>& stream() { return *stream_; }
  const cuda::stream_t<>& stream() const { return *stream_; }
  const std::shared_ptr<cuda::stream_t<>> streamPtr() const { return stream_; }

  CUDAContextToken toToken() {
    return CUDAContextToken(currentDevice_, stream_);
  }

  template <typename T>
  const T& get(const CUDA<T>& data) {
    if(data.device() != currentDevice_) {
      // Eventually replace with prefetch to current device (assuming unified memory works)
      // If we won't go to unified memory, need to figure out something else...
      throw cms::Exception("LogicError") << "Handling data from multiple devices is not yet supported";
    }

    if(data.stream().id() != stream_->id()) {
      // Different streams, need to synchronize
      if(!data.event().has_occurred()) {
        // Event not yet occurred, so need to add synchronization
        // here. Sychronization is done by making the CUDA stream to
        // wait for an event, so all subsequent work in the stream
        // will run only after the event has "occurred" (i.e. data
        // product became available).
        auto ret = cudaStreamWaitEvent(stream_->id(), data.event().id(), 0);
        cuda::throw_if_error(ret, "Failed to make a stream to wait for an event");
      }
    }

    return data.data_;
  }

  template <typename T>
  std::unique_ptr<CUDA<T> > wrap(T data) {
    // make_unique doesn't work because of private constructor
    auto ret = std::unique_ptr<CUDA<T> >(new CUDA<T>(std::move(data), *this));
    // Record CUDA event to the CUDA stream. The event will become
    // "occurred" after all work queued to the stream before this
    // point has been finished.
    ret->event().record(stream_->id());
    return ret;
  }

private:
  int currentDevice_;
  std::optional<edm::WaitingTaskWithArenaHolder> waitingTaskHolder_;
  cuda::device::current::scoped_override_t<> setDeviceForThisScope_;
  std::shared_ptr<cuda::stream_t<>> stream_;
};

#endif
