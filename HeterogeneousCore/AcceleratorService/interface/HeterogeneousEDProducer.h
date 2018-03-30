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
    static bool launch(DEVICE& algo, Args&&... args) { return algo.call_launch##DEVICE(std::forward<Args>(args)...); } \
    template <typename ...Args> \
    static void produce(DEVICE& algo, Args&&... args) { algo.call_produce##DEVICE(std::forward<Args>(args)...); } \
    static constexpr HeterogeneousDevice deviceEnum = ENUM; \
  }


namespace heterogeneous {
  class CPU {
  public:
    bool call_launchCPU(HeterogeneousDeviceId *algoExecutionLocation, edm::WaitingTaskWithArenaHolder waitingTaskHolder);
    void call_produceCPU(edm::Event& iEvent, const edm::EventSetup& iSetup) {
      produceCPU(iEvent, iSetup);
    }

  private:
    virtual void launchCPU() = 0;
    virtual void produceCPU(edm::Event& iEvent, const edm::EventSetup& iSetup) = 0;
  };
  MAKE_MAPPING(CPU, HeterogeneousDevice::kCPU);

  class GPUMock {
  public:
    bool call_launchGPUMock(HeterogeneousDeviceId *algoExecutionLocation, edm::WaitingTaskWithArenaHolder waitingTaskHolder);
    void call_produceGPUMock(edm::Event& iEvent, const edm::EventSetup& iSetup) {
      produceGPUMock(iEvent, iSetup);
    }

  private:
    virtual void launchGPUMock(std::function<void()> callback) = 0;
    virtual void produceGPUMock(edm::Event& iEvent, const edm::EventSetup& iSetup) = 0;
  };
  MAKE_MAPPING(GPUMock, HeterogeneousDevice::kGPUMock);

  class GPUCuda {
  public:
    bool call_launchGPUCuda(HeterogeneousDeviceId *algoExecutionLocation, edm::WaitingTaskWithArenaHolder waitingTaskHolder);
    void call_produceGPUCuda(edm::Event& iEvent, const edm::EventSetup& iSetup) {
      produceGPUCuda(iEvent, iSetup);
    }

  private:
    virtual void launchGPUCuda(std::function<void()> callback) = 0;
    virtual void produceGPUCuda(edm::Event& iEvent, const edm::EventSetup& iSetup) = 0;
  };
  MAKE_MAPPING(GPUCuda, HeterogeneousDevice::kGPUCuda);
}

#undef MAKE_MAPPING

namespace heterogeneous {
  ////////////////////
  template <typename ...Args>
  struct CallLaunch;
  template <typename T, typename D, typename ...Devices>
  struct CallLaunch<T, D, Devices...> {
    template <typename ...Args>
    static void call(T& ref, const HeterogeneousProductBase *input, Args&&... args) {
      bool succeeded = true;
      if(input) {
        succeeded = input->isProductOn(Mapping<D>::deviceEnum);
      }
      if(succeeded) {
        succeeded = Mapping<D>::launch(ref, std::forward<Args>(args)...);
      }
      if(!succeeded) {
        CallLaunch<T, Devices...>::call(ref, input, std::forward<Args>(args)...);
      }
    }
  };
  template <typename T>
  struct CallLaunch<T> {
    template <typename ...Args>
    static void call(T& ref, const HeterogeneousProductBase *input, Args&&... args) {}
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
    void call_launch(const HeterogeneousProductBase *input,
                     HeterogeneousDeviceId *algoExecutionLocation,
                     edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
      CallLaunch<HeterogeneousDevices, Devices...>::call(*this, input, algoExecutionLocation, std::move(waitingTaskHolder));
    }

    void call_produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
      CallProduce<HeterogeneousDevices, Devices...>::call(*this, iEvent, iSetup);
    }
  };
}

template <typename Devices, typename ...Capabilities>
class HeterogeneousEDProducer: public Devices, public edm::stream::EDProducer<edm::ExternalWork, Capabilities...> {
public:
  HeterogeneousEDProducer() {}

  ~HeterogeneousEDProducer() = default;

protected:
  void schedule(const HeterogeneousProductBase *input) {
    input_ = input;
    wasScheduleCalled_ = true;
  }

private:
  virtual void acquire(const edm::Event& iEvent, const edm::EventSetup& iSetup) = 0;

  void acquire(const edm::Event& iEvent, const edm::EventSetup& iSetup, edm::WaitingTaskWithArenaHolder waitingTaskHolder) override final {
    input_ = nullptr;
    algoExecutionLocation_ = HeterogeneousDeviceId();
    wasScheduleCalled_ = false;
    acquire(iEvent, iSetup);
    if(!wasScheduleCalled_) {
      throw cms::Exception("LogicError") << "Call to schedule() is missing from acquire(), please add it.";
    }
    Devices::call_launch(input_, &algoExecutionLocation_, std::move(waitingTaskHolder));

    input_ = nullptr;
  }

  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override final {
  }

  const HeterogeneousProductBase *input_;
  HeterogeneousDeviceId algoExecutionLocation_;
  bool wasScheduleCalled_;
};

#endif



