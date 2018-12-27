#ifndef HeterogeneousCore_CUDACore_CUDAScopedContext_h
#define HeterogeneousCore_CUDACore_CUDAScopedContext_h

#include "FWCore/Concurrency/interface/WaitingTaskWithArenaHolder.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "CUDADataFormats/Common/interface/CUDA.h"
#include "HeterogeneousCore/CUDACore/interface/CUDAContextToken.h"

#include <cuda/api_wrappers.h>

#include <optional>

namespace cudatest {
  class TestCUDAScopedContext;
}

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
    //
    // CUDA<T> constructor records CUDA event to the CUDA stream. The
    // event will become "occurred" after all work queued to the
    // stream before this point has been finished.
    return std::unique_ptr<CUDA<T> >(new CUDA<T>(*this, std::move(data)));
  }

  template <typename T, typename... Args>
  auto emplace(edm::Event& iEvent, edm::EDPutTokenT<T> token, Args&&... args) {
    return iEvent.emplace(token, *this, std::forward<Args>(args)...);
  }

private:
  friend class cudatest::TestCUDAScopedContext;

  // This construcor is only meant for testing
  explicit CUDAScopedContext(int device, std::unique_ptr<cuda::stream_t<>> stream);

  int currentDevice_;
  std::optional<edm::WaitingTaskWithArenaHolder> waitingTaskHolder_;
  cuda::device::current::scoped_override_t<> setDeviceForThisScope_;
  std::shared_ptr<cuda::stream_t<>> stream_;
};

#endif
