#include "FWCore/Utilities/interface/EDMException.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDMetadata.h"

template <typename TQueue>
using enable_if_async_t =
    std::enable_if_t<cms::alpakatools::is_queue_v<TQueue> and not cms::alpakatools::is_queue_blocking_v<TQueue>>;

namespace cms::alpakatools {
  template <typename TQueue>
  EDMetadataImpl<TQueue, enable_if_async_t<TQueue>>::~EDMetadataImpl() {
    // Make sure that the production of the product in the GPU is
    // complete before destructing the product. This is to make sure
    // that the EDM stream does not move to the next event before all
    // asynchronous processing of the current is complete.

    // TODO: a callback notifying a WaitingTaskHolder (or similar)
    // would avoid blocking the CPU, but would also require more work.

    if (event_) {
      // Must not throw in a destructor, and if there were an
      // exception could not really propagate it anyway.
      CMS_SA_ALLOW try { alpaka::wait(*event_); } catch (...) {
      }
    }
  }

  template <typename TQueue>
  void EDMetadataImpl<TQueue, enable_if_async_t<TQueue>>::enqueueCallback(edm::WaitingTaskWithArenaHolder holder) {
    alpaka::enqueue(*queue_, alpaka::HostOnlyTask([holder = std::move(holder)]() {
      // The functor is required to be const, but the original waitingTaskHolder_
      // needs to be notified...
      const_cast<edm::WaitingTaskWithArenaHolder&>(holder).doneWaiting(nullptr);
    }));
  }

  template <typename TQueue>
  void EDMetadataImpl<TQueue, enable_if_async_t<TQueue>>::synchronize(EDMetadataImpl& consumer,
                                                                      bool tryReuseQueue) const {
    if (*queue_ == *consumer.queue_) {
      return;
    }

    if (tryReuseQueue) {
      if (auto queue = tryReuseQueue_()) {
        consumer.queue_ = queue_;
        return;
      }
    }

    // TODO: how necessary this check is?
    if (alpaka::getDev(*queue_) != alpaka::getDev(*consumer.queue_)) {
      throw edm::Exception(edm::errors::LogicError) << "Handling data from multiple devices is not yet supported";
    }

    if (not alpaka::isComplete(*event_)) {
      // Event not yet occurred, so need to add synchronization
      // here. Sychronization is done by making the queue to wait
      // for an event, so all subsequent work in the queue will run
      // only after the event has "occurred" (i.e. data product
      // became available).
      alpaka::wait(*consumer.queue_, *event_);
    }
  }

  template <typename TQueue>
  std::shared_ptr<TQueue> EDMetadataImpl<TQueue, enable_if_async_t<TQueue>>::tryReuseQueue_() const {
    bool expected = true;
    if (mayReuseQueue_.compare_exchange_strong(expected, false)) {
      // If the current thread is the one flipping the flag, it may
      // reuse the queue.
      return queue_;
    }
    return nullptr;
  }

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
  template class cms::alpakatools::EDMetadataImpl<alpaka_cuda_async::Queue>;
#endif
}  // namespace cms::alpakatools
