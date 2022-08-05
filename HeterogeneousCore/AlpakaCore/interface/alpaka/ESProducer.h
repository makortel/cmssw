#ifndef HeterogeneousCore_AlpakaCore_interface_alpaka_ESProducer_h
#define HeterogeneousCore_AlpakaCore_interface_alpaka_ESProducer_h

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/produce_helpers.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/DeviceRecord.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESDeviceProduct.h"

#include <functional>

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  /**
   * The ESProducer is a base class for modules producing data into
   * the host memory space and/or the device memory space defined by
   * the backend (i.e. ALPAKA_ACCELERATOR_NAMESPACE). The interface
   * looks similar to the normal edm::ESProducer.
   *
   * When producing a host product, the produce function should have
   * the the usual Record argument. For producing a device product,
   * the produce funtion should have DeviceRecord<Record> argument.
   */
  class ESProducer : public edm::ESProducer {
    using Base = edm::ESProducer;

  protected:
    template <typename T>
    auto setWhatProduced(T* iThis, edm::es::Label const& label = {}) {
      return setWhatProduced(iThis, &T::produce, label);
    }

    template <typename T, typename TReturn, typename TRecord>
    auto setWhatProduced(T* iThis, TReturn (T ::*iMethod)(TRecord const&), edm::es::Label const& label = {}) {
      return Base::setWhatProduced(iThis, iMethod, label);
    }

    template <typename T, typename TReturn, typename TRecord>
    auto setWhatProduced(T* iThis,
                         TReturn (T ::*iMethod)(DeviceRecord<TRecord> const&),
                         edm::es::Label const& label = {}) {
      using TProduct = typename edm::eventsetup::produce::smart_pointer_traits<TReturn>::type;
      using ProductType = ESDeviceProduct<TProduct>;
      using ReturnType = detail::ESDeviceProductWithStorage<TProduct, TReturn>;
      return Base::setWhatProduced(
          [iThis, iMethod](TRecord const& record) -> std::unique_ptr<ProductType> {
            DeviceRecord<TRecord> const deviceRecord(record);
            auto ret = std::invoke(iMethod, iThis, deviceRecord);
            // TODO: to be changed asynchronous later
            alpaka::wait(deviceRecord.queue());
            if (ret) {
              return std::make_unique<ReturnType>(std::move(ret));
            }
            return nullptr;
          },
          label);
    }
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
