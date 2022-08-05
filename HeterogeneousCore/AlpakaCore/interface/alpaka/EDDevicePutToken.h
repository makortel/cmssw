#ifndef HeterogeneousCore_AlpakaCore_interface_alpaka_EDDevicePutToken_h
#define HeterogeneousCore_AlpakaCore_interface_alpaka_EDDevicePutToken_h

#include "DataFormats/Common/interface/DeviceProduct.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  class DeviceEvent;

  /**
   * The EDDevicePutToken is similar to edm::EDPutTokenT, but is
   * intended for Event data products in the device memory space
   * defined by the backend (i.e. ALPAKA_ACCELERATOR_NAMESPACE). It
   * can be used only to put data into a DeviceEvent
   */
  template <typename TProduct>
  class EDDevicePutToken {
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
    // host synchronous backends can use TProduct directly
    using ProductType = TProduct;
#else
    // all device and asynchronous backends need to be wrapped
    using ProductType = edm::DeviceProduct<TProduct>;
#endif

  public:
    constexpr EDDevicePutToken() noexcept = default;

    template <typename TAdapter>
    explicit EDDevicePutToken(TAdapter&& adapter) : token_(adapter.template deviceProduces<TProduct, ProductType>()) {}

    template <typename TAdapter>
    EDDevicePutToken& operator=(TAdapter&& adapter) {
      edm::EDPutTokenT<ProductType> tmp(adapter.template deviceProduces<TProduct, ProductType>());
      token_ = tmp;
      return *this;
    }

  private:
    friend class DeviceEvent;

    auto const& underlyingToken() const { return token_; }

    edm::EDPutTokenT<ProductType> token_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
