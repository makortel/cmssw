#ifndef HeterogeneousCore_AlpakaCore_interface_alpaka_EDMetadata_h
#define HeterogeneousCore_AlpakaCore_interface_alpaka_EDMetadata_h

#include <atomic>
#include <memory>
#include <type_traits>

#include <alpaka/alpaka.hpp>

#include "FWCore/Concurrency/interface/WaitingTaskWithArenaHolder.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/traits.h"
#include "HeterogeneousCore/AlpakaInterface/interface/HostOnlyTask.h"

namespace cms::alpakatools {
  template <typename TQueue, typename TSfinae = void>
  class EDMetadataImpl;

  // Host backends with a synchronous queue
  template <typename TQueue>
  class EDMetadataImpl<TQueue,
                       std::enable_if_t<is_queue_v<TQueue> and is_queue_blocking_v<TQueue> and
                                        std::is_same_v<alpaka::Dev<TQueue>, alpaka_common::DevHost>>> {
  public:
    using Queue = TQueue;

    EDMetadataImpl(std::shared_ptr<Queue> queue) : queue_(std::move(queue)) {}

    // Alpaka operations do not accept a temporary as an argument
    // TODO: Returning non-const reference here is BAD
    Queue& queue() const { return *queue_; }

    void recordEvent() {}

  private:
    std::shared_ptr<Queue> queue_;
  };

  // TODO: device backends with a synchronous queue

  // All backends with an asynchronous queue
  template <typename TQueue>
  class EDMetadataImpl<TQueue, std::enable_if_t<is_queue_v<TQueue> and not is_queue_blocking_v<TQueue>>> {
  public:
    using Queue = TQueue;
    using Event = alpaka::Event<Queue>;

    EDMetadataImpl(std::shared_ptr<Queue> queue, std::shared_ptr<Event> event)
        : queue_(std::move(queue)), event_(std::move(event)) {}
    ~EDMetadataImpl();

    // Alpaka operations do not accept a temporary as an argument
    // TODO: Returning non-const reference here is BAD
    Queue& queue() const { return *queue_; }

    void enqueueCallback(edm::WaitingTaskWithArenaHolder holder);

    void recordEvent() { alpaka::enqueue(*queue_, *event_); }

    /**
     * Synchronizes 'consumer' metadata wrt. 'this' in the event product
     */
    void synchronize(EDMetadataImpl& consumer, bool tryReuseQueue) const;

  private:
    /**
     * Returns a shared_ptr to the Queue if it can be reused, or a
     * null shared_ptr if not
     */
    std::shared_ptr<Queue> tryReuseQueue_() const;

    std::shared_ptr<Queue> queue_;
    std::shared_ptr<Event> event_;
    // This flag tells whether the Queue may be reused by a
    // consumer or not. The goal is to have a "chain" of modules to
    // queue their work to the same queue.
    mutable std::atomic<bool> mayReuseQueue_ = true;
  };

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
  extern template class EDMetadataImpl<alpaka_cuda_async::Queue>;
#endif
}  // namespace cms::alpakatools

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  /**
   * The EDMetadata class provides the exact synchronization
   * mechanisms for Event data products for backends with asynchronous
   * Queue. These include
   * - adding a notification for edm::WaitingTaskWithArenaHolder
   * - recording an Event
   * - synchronizing an Event data product and a consuming EDModule
   *
   * For synchronous backends the EDMetadata acts as an owner of the
   * Queue object, as no further synchronization is needed.
   *
   * EDMetadata is used as the Metadata class for
   * edm::DeviceProduct<T>, and is an implementation detail (not
   * visible to user code).
   *
   * TODO: What to do with device-synchronous backends? The data
   * product needs to be wrapped into the edm::DeviceProduct, but the
   * EDMetadata class used there does not need anything except "dummy"
   * implementation of synchronize(). The question is clearly
   * solvable, so maybe leave it to the time we would actually need
   * one?
   */

  using EDMetadata = cms::alpakatools::EDMetadataImpl<Queue>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
