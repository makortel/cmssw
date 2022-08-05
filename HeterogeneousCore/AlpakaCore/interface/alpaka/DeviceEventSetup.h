#ifndef HeterogeneousCore_AlpakaCore_interface_alpaka_DeviceEventSetup_h
#define HeterogeneousCore_AlpakaCore_interface_alpaka_DeviceEventSetup_h

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESDeviceGetToken.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  /**
   * The DeviceEventSetup mimics edm::EventSetup, and provides access
   * to ESProducts in the host memory space, and in the device memory
   * space defined by the backend (i.e. ALPAKA_ACCELERATOR_NAMESPACE).
   *
   * Access to device memory space products is synchronized properly.
   *
   * Note that not full interface of edm::EventSetup is replicated
   * here. If something important is missing, that can be added.
   */
  class DeviceEventSetup {
  public:
    DeviceEventSetup(edm::EventSetup const& iSetup) : setup_(iSetup) {}

    // getData()

    template <typename T, typename R>
    T const& getData(edm::ESGetToken<T, R> const& iToken) const {
      return setup_.getData(iToken);
    }

    template <typename T, typename R>
    T const& getData(ESDeviceGetToken<T, R> const& iToken) const {
      auto const& product = setup_.getData(iToken.underlyingToken());
      return product.get();
    }

    // getHandle()

    template <typename T, typename R>
    edm::ESHandle<T> getHandle(edm::ESGetToken<T, R> const& iToken) const {
      return setup_.getHandle(iToken);
    }

    template <typename T, typename R>
    edm::ESHandle<T> getHandle(ESDeviceGetToken<T, R> const& iToken) const {
      auto handle = setup_.getHandle(iToken.underlyingToken());
      if (not handle) {
        return edm::ESHandle<T>(handle.whyFailedFactory());
      }
      return edm::ESHandle<T>(&handle->get(), handle.description());
    }

    // getTransientHandle() is intentionally omitted for now. It makes
    // little sense for event transitions, and for now
    // DeviceEventSetup is available only for those. If
    // DeviceEventSetup ever gets added for run or lumi transitions,
    // getTransientHandle() will be straightforward to add

  private:
    edm::EventSetup const& setup_;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
