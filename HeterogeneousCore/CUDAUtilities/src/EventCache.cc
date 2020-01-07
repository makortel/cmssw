#include "FWCore/Utilities/interface/thread_safety_macros.h"
#include "HeterogeneousCore/CUDAUtilities/interface/EventCache.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/currentDevice.h"
#include "HeterogeneousCore/CUDAUtilities/interface/deviceCount.h"
#include "HeterogeneousCore/CUDAUtilities/interface/ScopedSetDevice.h"

namespace cudautils {
  void EventCache::Deleter::operator()(cudaEvent_t event) const {
    if (device_ != -1) {
      ScopedSetDevice deviceGuard{device_};
      cudaCheck(cudaEventDestroy(event));
    }
  }

  // EventCache should be constructed by the first call to
  // getEventCache() only if we have CUDA devices present
  EventCache::EventCache() : cache_(cudautils::deviceCount()) {}

  SharedEventPtr EventCache::get() {
    const auto dev = cudautils::currentDevice();
    auto event = makeOrGet(dev);
    auto ret = cudaEventQuery(event.get());
    // event is occurred, return immediately
    if (ret == cudaSuccess) {
      return event;
    }
    // return code is something else than "recorded", throw exception
    if (ret != cudaErrorNotReady) {
      cudaCheck(ret);
    }

    // Got recorded, but not yet occurred event. Try until we get an
    // occurred event. Need to keep all recorded events until an
    // occurred event is found in order to avoid ping-pong with a
    // recorded event.
    std::vector<SharedEventPtr> ptrs{std::move(event)};
    do {
      event = makeOrGet(dev);
      ret = cudaEventQuery(event.get());
      if (ret == cudaErrorNotReady) {
        ptrs.emplace_back(std::move(event));
      } else if (ret != cudaSuccess) {
        cudaCheck(ret);
      }
    } while (ret != cudaSuccess);
    return event;
  }

  SharedEventPtr EventCache::makeOrGet(int dev) {
    return cache_[dev].makeOrGet([dev]() {
      cudaEvent_t event;
      // it should be a bit faster to ignore timings
      cudaCheck(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
      return std::unique_ptr<BareEvent, Deleter>(event, Deleter{dev});
    });
  }

  void EventCache::clear() {
    // Reset the contents of the caches, but leave an
    // edm::ReusableObjectHolder alive for each device. This is needed
    // mostly for the unit tests, where the function-static
    // EventCache lives through multiple tests (and go through
    // multiple shutdowns of the framework).
    cache_.clear();
    cache_.resize(cudautils::deviceCount());
  }

  EventCache& getEventCache() {
    // the public interface is thread safe
    CMS_THREAD_SAFE static EventCache cache;
    return cache;
  }
}  // namespace cudautils
