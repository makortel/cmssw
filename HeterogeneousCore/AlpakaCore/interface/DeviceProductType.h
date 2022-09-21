#ifndef HeterogeneousCore_AlpakaCore_interface_DeviceProductType_h
#define HeterogeneousCore_AlpakaCore_interface_DeviceProductType_h

#include "DataFormats/Common/interface/DeviceProduct.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/traits.h"

namespace cms::alpakatools {
  /**
   * This "trait" class abstracts the actual product type put in the
   * edm::Event.
   */
  template <typename TProduct, bool wrapProduct = true>
  struct DeviceProductTypeImpl {
    // all device and asynchronous backends need to be wrapped
    using type = edm::DeviceProduct<TProduct>;
  };

  template <typename TProduct>
  struct DeviceProductTypeImpl<TProduct, false> {
    // host synchronous backends can use TProduct directly
    using type = TProduct;
  };

  template <typename TProduct, typename TQueue, typename = std::enable_if_t<is_queue_v<TQueue> > >
  using DeviceProductType =
      typename DeviceProductTypeImpl<TProduct,
                                     not(std::is_same_v<alpaka::Dev<TQueue>, alpaka_common::DevHost> and
                                         is_queue_blocking_v<TQueue>)>::type;
}  // namespace cms::alpakatools

#endif
