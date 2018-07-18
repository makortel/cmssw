#ifndef HeterogeneousCore_CUDACore_CUDAScopedContext_h
#define HeterogeneousCore_CUDACore_CUDAScopedContext_h

#include "FWCore/Concurrency/interface/WaitingTaskWithArenaHolder.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "HeterogeneousCore/CUDACore/interface/CUDA.h"
#include "HeterogeneousCore/CUDACore/interface/CUDAToken.h"

#include <optional>

/**
 * The aim of this class is to do necessary per-event "initialization"
 * (like setting the current device, synchronizing between CUDA
 * streams etc), and enforce that those get done in a proper way in RAII fashion.
 */
class CUDAScopedContext {
public:
  explicit CUDAScopedContext(const CUDAToken& token):
    currentDevice_(token.device()),
    setDeviceForThisScope_(currentDevice_),
    stream_(token.stream())
  {}

  template<typename T>
  explicit CUDAScopedContext(const CUDA<T>& data):
    currentDevice_(data.device()),
    setDeviceForThisScope_(currentDevice_),
    stream_(data.stream())
  {}

  explicit CUDAScopedContext(const CUDAToken& token, edm::WaitingTaskWithArenaHolder waitingTaskHolder):
    CUDAScopedContext(token)
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

  cuda::stream_t<>& stream() { return stream_; }

  template <typename T>
  const T& get(const CUDA<T>& data) {
    if(data.device() != currentDevice_) {
      // Eventually replace with prefetch to current device (assuming unified memory works)
      // If we won't go to unified memory, need to figure out something else...
      throw cms::Exception("LogicError") << "Handling data from multiple devices is not yet supported";
    }

    if(data.stream().id() != stream_.id()) {
      // Different streams, need to synchronize
      if(!data.event().has_occurred()) {
        // Event not yet occurred, so need to add synchronization
        // here. Sychronization is done by making the CUDA stream to
        // wait for an event, so all subsequent work in the stream
        // will run only after the event has occurred (i.e. data
        // product became available).
        auto ret = cudaStreamWaitEvent(stream_.id(), data.event().id(), 0);
        cuda::throw_if_error(ret, "Failed to make a stream to wait for an event");
      }
    }

    return data.data_;
  }

  template <typename T>
  std::unique_ptr<CUDA<T> > wrap(T data) {
    auto ret = std::make_unique<CUDA<T> >(std::move(data), *this);
    ret->event().record(stream_.id());
    return ret;
  }

private:
  int currentDevice_;
  std::optional<edm::WaitingTaskWithArenaHolder> waitingTaskHolder_;
  cuda::device::current::scoped_override_t<> setDeviceForThisScope_;
  cuda::stream_t<> stream_;
};

#endif
