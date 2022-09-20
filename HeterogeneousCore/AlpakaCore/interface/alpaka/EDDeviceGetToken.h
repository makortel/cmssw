#ifndef HeterogeneousCore_AlpakaCore_interface_alpaka_EDDeviceGetToken_h
#define HeterogeneousCore_AlpakaCore_interface_alpaka_EDDeviceGetToken_h

#include "DataFormats/Common/interface/DeviceProduct.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/DeviceProductType.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  class DeviceEvent;

  /**
   * The EDDeviceGetToken is similar to edm::EDGetTokenT, but is
   * intended for Event data products in the device memory space
   * defined by the backend (i.e. ALPAKA_ACCELERATOR_NAMESPACE). It
   * can be used only to get data from a DeviceEvent.
   *
   * A specific token class is motivated with
   * - enforce stronger the type-deducing consumes(). Consumes() with
   *   explicit type will fail anyway in general, but succeeds on one
   *   of the backends. With a specific token type the explicit-type
   *   consumes() would fail always.
   *- to avoid using EDDeviceGetToken with edm::Event
   */
  template <typename TProduct>
  class EDDeviceGetToken {
    using ProductType = typename detail::DeviceProductType<TProduct>::type;

  public:
    constexpr EDDeviceGetToken() = default;

    template <typename TAdapter>
    constexpr EDDeviceGetToken(TAdapter&& iAdapter) : token_(std::forward<TAdapter>(iAdapter)) {}

  private:
    friend class DeviceEvent;

    auto const& underlyingToken() const { return token_; }

    edm::EDGetTokenT<ProductType> token_;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
