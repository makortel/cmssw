#ifndef HeterogeneousCore_Producer_HeterogeneousEvent_h
#define HeterogeneousCore_Producer_HeterogeneousEvent_h

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"

#include "HeterogeneousCore/Product/interface/HeterogeneousProduct.h"

namespace edm {
  class HeterogeneousEvent {
  public:
    HeterogeneousEvent(edm::Event *event, HeterogeneousDeviceId location): event_(event), location_(location) {}

    edm::Event& event() { return *event_; }
    const edm::Event& event() const { return *event_; }

    template <typename Product, typename Token, typename Type>
    void getByToken(const Token& token, edm::Handle<Type>& handle) const {
      edm::Handle<HeterogeneousProduct> tmp;
      event_->getByToken(token, tmp);
      if(tmp.failedToGet()) {
        handle = edm::Handle<Type>(tmp.whyFailedFactory());
        return;
      }
      if(tmp.isValid()) {
        const auto& concrete = tmp->get<Product>();
        const auto& provenance = tmp.provenance();
#define CASE(ENUM) case ENUM: handle = edm::Handle<Type>(&(concrete.template getProduct<ENUM>()), provenance); break
        switch(location_.deviceType()) {
        CASE(HeterogeneousDevice::kCPU);
        CASE(HeterogeneousDevice::kGPUMock);
        CASE(HeterogeneousDevice::kGPUCuda);
        default:
          throw cms::Exception("LogicError") << "edm::HeterogeneousEvent::getByToken(): no case statement for device " << static_cast<unsigned int>(location_.deviceType());
        }
#undef CASE
      }
    }

    template <typename Product, typename Type>
    void put(std::unique_ptr<Type> product) {
      assert(location_.deviceType() == HeterogeneousDevice::kCPU);
      event_->put(std::make_unique<HeterogeneousProduct>(Type(heterogeneous::HeterogeneousDeviceTag<HeterogeneousDevice::kCPU>(), std::move(*product))));
    }

    template <typename Product, typename Type, typename F>
    void put(std::unique_ptr<Type> product, F transferToCPU) {
      std::unique_ptr<HeterogeneousProduct> prod;
#define CASE(ENUM) case ENUM: this->template make<ENUM, Product>(prod, std::move(product), std::move(transferToCPU), 0); break;
      switch(location_.deviceType()) {
      CASE(HeterogeneousDevice::kGPUMock);
      CASE(HeterogeneousDevice::kGPUCuda);
      default:
        throw cms::Exception("LogicError") << "edm::HeterogeneousEvent::put(): no case statement for device " << static_cast<unsigned int>(location_.deviceType());
      }
#undef CASE
      event_->put(std::move(prod));
    }

  private:
    template<HeterogeneousDevice Device, typename Product, typename Type, typename F>
    typename std::enable_if_t<Product::template IsAssignable<Device, Type>::value, void>
    make(std::unique_ptr<HeterogeneousProduct>& ret, std::unique_ptr<Type> product, F transferToCPU, int) {
      ret = std::make_unique<HeterogeneousProduct>(Product(heterogeneous::HeterogeneousDeviceTag<Device>(),
                                                           std::move(*product), location_, std::move(transferToCPU)));
    }
    template<HeterogeneousDevice Device, typename Product, typename Type, typename F>
    void make(std::unique_ptr<HeterogeneousProduct>& ret, std::unique_ptr<Type> product, F transferToCPU, long) {
      assert(false);
    }

    edm::Event *event_;
    HeterogeneousDeviceId location_;
  };
} // end namespace edm

#endif
