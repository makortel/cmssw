#ifndef HeterogeneousCore_AlpakaCore_interface_alpaka_ESDeviceGetToken_h
#define HeterogeneousCore_AlpakaCore_interface_alpaka_ESDeviceGetToken_h

#include "FWCore/Utilities/interface/ESGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESDeviceProduct.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  class DeviceEventSetup;
  template <typename T>
  class DeviceRecord;

  /**
   * The ESDeviceGetToken is similar to edm::ESGetToken, but is
   * intended for EventSetup data products in the device memory space
   * defined by the backend (i.e. ALPAKA_ACCELERATOR_NAMESPACE). It
   * can be used only to get data from a DeviceEventSetup and
   * DeviceRecord<T>.
   */
  template <typename ESProduct, typename ESRecord>
  class ESDeviceGetToken {
  public:
    constexpr ESDeviceGetToken() noexcept = default;

    template <typename TAdapter>
    constexpr ESDeviceGetToken(TAdapter&& iAdapter) : token_(std::forward<TAdapter>(iAdapter)) {}

  private:
    friend class DeviceEventSetup;
    template <typename T>
    friend class DeviceRecord;

    auto const& underlyingToken() const { return token_; }

    edm::ESGetToken<ESDeviceProduct<ESProduct>, ESRecord> token_;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
