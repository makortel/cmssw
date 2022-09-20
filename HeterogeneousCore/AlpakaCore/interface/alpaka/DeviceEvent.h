#ifndef HeterogeneousCore_AlpakaCore_interface_alpaka_DeviceEvent_h
#define HeterogeneousCore_AlpakaCore_interface_alpaka_DeviceEvent_h

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/DeviceProductType.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDDeviceGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDDevicePutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDMetadata.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  /**
   * The DeviceEvent mimics edm::Event, and provides access to
   * EDProducts in the host memory space, and in the device memory
   * space defined by the backend (i.e. ALPAKA_ACCELERATOR_NAMESPACE).
   * The DeviceEvent also gives access to the Queue object the
   * EDModule code should use to queue all the device operations.
   *
   * Access to device memory space products is synchronized properly.
   * For backends with synchronous Queue this is trivial. For
   * asynchronous Queue, either the Queue of the EDModule is taken
   * from the first data product, or a wait is inserted into the
   * EDModule's Queue to wait for the product's asynchronous
   * production to finish.
   *
   * Note that not full interface of edm::Event is replicated here. If
   * something important is missing, that can be added.
   */
  class DeviceEvent {
  public:
    // To be called in produce()
    explicit DeviceEvent(edm::Event& ev, std::shared_ptr<EDMetadata> metadata)
        : constEvent_(ev), event_(&ev), metadata_(std::move(metadata)) {}

    // To be called in acquire()
    explicit DeviceEvent(edm::Event const& ev, std::shared_ptr<EDMetadata> metadata)
        : constEvent_(ev), metadata_(std::move(metadata)) {}

    DeviceEvent(DeviceEvent const&) = delete;
    DeviceEvent& operator=(DeviceEvent const&) = delete;
    DeviceEvent(DeviceEvent&&) = delete;
    DeviceEvent& operator=(DeviceEvent&&) = delete;

    auto streamID() const { return constEvent_.streamID(); }
    auto id() const { return constEvent_.id(); }

    // Alpaka operations do not accept a temporary as an argument
    // TODO: Returning non-const reference here is BAD
    Queue& queue() const {
      queueUsed_ = true;
      return metadata_->queue();
    }

    // Note: this function is planned to be removed after a more
    // proper treatment of multiple devices per backend is implemented
    // in EventSetup
    Device device() const { return alpaka::getDev(metadata_->queue()); }

    // get()

    template <typename T>
    T const& get(edm::EDGetTokenT<T> const& token) const {
      return constEvent_.get(token);
    }

    template <typename T>
    T const& get(EDDeviceGetToken<T> const& token) const {
      auto const& deviceProduct = constEvent_.get(token.underlyingToken());
      if constexpr (std::is_same_v<typename detail::DeviceProductType<T>::type, T>) {
        return deviceProduct;
      } else {
        // try to re-use queue from deviceProduct if our queue has not yet been used
        T const& product = deviceProduct.template getSynchronized<EDMetadata>(*metadata_, not queueUsed_);
        queueUsed_ = true;
        return product;
      }
    }

    // getHandle()

    template <typename T>
    edm::Handle<T> getHandle(edm::EDGetTokenT<T> const& token) const {
      return constEvent_.getHandle(token);
    }

    template <typename T>
    edm::Handle<T> getHandle(EDDeviceGetToken<T> const& token) const {
      auto deviceProductHandle = constEvent_.getHandle(token.underlyingToken());
      if constexpr (std::is_same_v<typename detail::DeviceProductType<T>::type, T>) {
        return deviceProductHandle;
      } else {
        if (not deviceProductHandle) {
          return edm::Handle<T>(deviceProductHandle.whyFailedFactory());
        }
        // try to re-use queue from deviceProduct if our queue has not yet been used
        T const& product = deviceProductHandle->getSynchronized(*metadata_, not queueUsed_);
        queueUsed_ = true;
        return edm::Handle<T>(&product, deviceProductHandle.provenance());
      }
    }

    // emplace()

    template <typename T, typename... Args>
    edm::OrphanHandle<T> emplace(edm::EDPutTokenT<T> const& token, Args&&... args) {
      return event_->emplace(token, std::forward<Args>(args)...);
    }

    // TODO: what to do about the returned OrphanHandle object?
    // The idea for Ref-like things in this domain differs from earlier Refs anyway
    template <typename T, typename... Args>
    void emplace(EDDevicePutToken<T> const& token, Args&&... args) {
      if constexpr (std::is_same_v<typename detail::DeviceProductType<T>::type, T>) {
        event_->emplace(token.underlyingToken(), std::forward<Args>(args)...);
      } else {
        event_->emplace(token.underlyingToken(), metadata_, std::forward<Args>(args)...);
      }
    }

    // put()

    template <typename T>
    edm::OrphanHandle<T> put(edm::EDPutTokenT<T> const& token, std::unique_ptr<T> product) {
      return event_->put(token, std::move(product));
    }

    template <typename T>
    void put(EDDevicePutToken<T> const& token, std::unique_ptr<T> product) {
      if constexpr (std::is_same_v<typename detail::DeviceProductType<T>::type, T>) {
        event_->emplace(token.underlyingToken(), std::move(*product));
      } else {
        event_->emplace(token.underlyingToken(), metadata_, std::move(*product));
      }
    }

  private:
    // Having both const and non-const here in order to serve the
    // clients with one DeviceEvent class
    edm::Event const& constEvent_;
    edm::Event* event_ = nullptr;

    std::shared_ptr<EDMetadata> metadata_;
    // DeviceEvent is not supposed to be const-thread-safe, so no
    // additional protection is needed.
    mutable bool queueUsed_ = false;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
