#ifndef HeterogeneousCore_AcceleratorService_HeterogeneousEDProducer_h
#define HeterogeneousCore_AcceleratorService_HeterogeneousEDProducer_h

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "HeterogeneousCore/Product/interface/HeterogeneousProduct.h"

#include "HeterogeneousCore/AcceleratorService/interface/AcceleratorService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

namespace heterogeneous {
  template <typename T> struct Mapping;
}
#define MAKE_MAPPING(DEVICE, ENUM) \
  template <> \
  struct Mapping<DEVICE> { \
    template <typename ...Args> \
    static bool launch(DEVICE& algo, Args&&... args) { algo.launch##DEVICE(std::forward<Args>(args)...); } \
    template <typename ...Args> \
    static void produce(DEVICE& algo, Args&&... args) { algo.produce#DEVICE(std::forward<Args>(args)...); } \
    static constexpr HeterogeneousDevice deviceEnum = ENUM; \
  }


namespace heterogeneous {
  class CPU {
  public:
    virtual bool launchCPU(const edm::Event& iEvent, const edm::EventSetup& iSetup) = 0;
    virtual void produceCPU(edm::Event& iEvent, const edm::EventSetup& iSetup) = 0;
  };
  MAKE_MAPPING(CPU, HeterogeneousDevice::kCPU);
  /*
  template <>
  struct Mapping<CPU> {
    template <typename ...Args>
    static void launch(CPU& algo, Args&&... args) { algo.launchCPU(std::forward<Args>(args)...); }
    template <typename ...Args>
    static void produce(CPU& algo, Args&&... args) { algo.produceCPU(std::forward<Args>(args)...); }
  };
  */

  class GPUMock {
  public:
    virtual bool launchGPUMock(const edm::Event& iEvent, const edm::EventSetup& iSetup, std::function<void()> callback) = 0;
    virtual void produceGPUMock(edm::Event& iEvent, const edm::EventSetup& iSetup) = 0;
  };
  MAKE_MAPPING(GPUMock, HeterogeneousDevice::kGPUMock);
  /*
  template <>
  struct Mapping<GPUMock> {
    template <typename ...Args>
    static void launch(GPUMock& algo, Args&&... args) { algo.launchGPUMock(std::forward<Args>(args)...); }
    template <typename ...Args>
    static void produce(GPUMock& algo, Args&&... args) { algo.produceCPU(std::forward<Args>(args)...); }
  };
  */
}

#undef MAKE_MAPPING

namespace heterogeneous {
  ////////////////////
  template <typename ...Args>
  struct CallLaunch;
  template <typename T, typename D, typename ...Devices>
  struct CallLaunch<T, D, Devices...> {
    template <typename ...Args>
    static void call(T& ref, const HeterogeneousProduct *input, Args&&... args) {
      bool succeeded = true;
      if(input) {
        succeeded = input->isProductOn(Mapping<D>::deviceEnum);
      }
      if(succeeded) {
        succeeded = Mapping<D>::launch(ref, std::forward<Args>(args)...);
      }
      if(!succeeded) {
        CallLaunch<T, Devices...>::call(ref, input, std:forward<Args>(args)...);
      }
    }
  };
  template <typename T>
  struct CallLaunch<T> {
    template <typename ...Args>
    static void call(T& ref, HeterogeneousProduct *input, Args&&... args) {}
  };

  ////////////////////
  template <typename ...Args>
  struct CallProduce;
  template <typename T, typename D, typename ...Devices>
  struct CallProduce<T, D, Devices...> {
    template <typename ...Args>
    static void call(T& ref, Args&&... args) {
      Mapping<D>::produce(ref, std::forward<Args>(args)...);
      CallProduce<T, Devices...>::call(ref, std::forward<Args>(args)...);
    }
  };
  template <typename T>
  struct CallProduce<T> {
    template <typename ...Args>
    static void call(T& ref, Args&&... args) {}
  };


  template <typename ...Devices>
  class HeterogeneousDevices: public Devices... {
  public:
    void acquire(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
      CallLaunch<HeterogeneousDevices, Devices...>::call(*this, iEvent, iSetup);
    }

    void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
      CallProduce<HeterogeneousDevices, Devices...>::call(*this, iEvent, iSetup);
    }
  };
}

template <typename Devices, typename ...Capabilities>
class HeterogeneousEDProducer: public Devices, edm::stream::EDProducer<edm::ExternalWork, Capabilities...> {
public:
  HeterogeneousEDProducer():
    accToken_(edm::Service<AcceleratorService>()->book())
  {}

  ~HeterogeneousEDProducer() = default override;

protected:
  /*
  template <typename ...Args>
  void schedule(Args&&... args) {
    std::initializer_list<const HeterogeneousProduct&> inputs{std::forward<Args>(args)...};
  */

  void schedule(const HeterogeneousProduct *input) {
    input_ = input;
    wasScheduleCalled = true;
  }

private:



  void acquire(const edm::Event& iEvent, const edm::EventSetup& iSetup) = 0;

  void acquire(const edm::Event& iEvent, const edm::EventSetup& iSetup, edm::WaitingTaskWithArenaHolder waitingTask) override final {
    input_ = nullptr;
    wasScheduleCalled_ = false;
    acquire(iEvent, iSetup);
    if(!wasScheduleCalled_) {
      throw cms::Exception("LogicError") << "Call to schedule() is missing from acquire(), please add it.";
    }
    CallLaunch();
    input_ = nullptr;
  }

  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override final {
  }

  HeterogeneousProduct *input_;
  //HeterogeneousDeviceId inputLocation_;
  bool wasScheduleCalled_;

  //AcceleratorService::Token accToken_;
};

#endif



