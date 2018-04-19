#ifndef HeterogeneousCore_AcceleratorService_HeterogeneousEDProducer_h
#define HeterogeneousCore_AcceleratorService_HeterogeneousEDProducer_h

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "HeterogeneousCore/Product/interface/HeterogeneousProduct.h"

#include "HeterogeneousCore/AcceleratorService/interface/AcceleratorService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <cuda/api_wrappers.h> // TODO: we need to split this file to minimize unnecessary dependencies

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
    void call_produceCPU(const HeterogeneousDeviceId& algoExecutionLocation, edm::Event& iEvent, const edm::EventSetup& iSetup) {
      produceCPU(iEvent, iSetup);
    }

  private:
    virtual void launchCPU() = 0;
    virtual void produceCPU(edm::Event& iEvent, const edm::EventSetup& iSetup) = 0;
  };
  MAKE_MAPPING(CPU, HeterogeneousDevice::kCPU);

  class GPUMock {
  public:
    bool call_launchGPUMock(DeviceBitSet inputLocation, HeterogeneousDeviceId *algoExecutionLocation, edm::WaitingTaskWithArenaHolder waitingTaskHolder);
    void call_produceGPUMock(const HeterogeneousDeviceId& algoExecutionLocation, edm::Event& iEvent, const edm::EventSetup& iSetup) {
      produceGPUMock(algoExecutionLocation, iEvent, iSetup);
    }

  private:
    virtual void launchGPUMock(std::function<void()> callback) = 0;
    virtual void produceGPUMock(const HeterogeneousDeviceId& location, edm::Event& iEvent, const edm::EventSetup& iSetup) = 0;
  };
  MAKE_MAPPING(GPUMock, HeterogeneousDevice::kGPUMock);

  class GPUCuda {
  public:
    using CallbackType = std::function<void(cuda::device::id_t, cuda::stream::id_t, cuda::status_t)>;

    bool call_launchGPUCuda(DeviceBitSet inputLocation, HeterogeneousDeviceId *algoExecutionLocation, edm::WaitingTaskWithArenaHolder waitingTaskHolder);
    void call_produceGPUCuda(const HeterogeneousDeviceId& algoExecutionLocation, edm::Event& iEvent, const edm::EventSetup& iSetup);

  private:
    virtual void launchGPUCuda(CallbackType callback) = 0;
    virtual void produceGPUCuda(const HeterogeneousDeviceId& location, edm::Event& iEvent, const edm::EventSetup& iSetup) = 0;
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
      DeviceBitSet inputLocation;
      if(input) {
        succeeded = input->isProductOn(Mapping<D>::deviceEnum);
        if(succeeded) {
          inputLocation = input->onDevices(Mapping<D>::deviceEnum);
        }
      }
      if(succeeded) {
        // may not perfect-forward here in order to be able to forward arguments to next CallLaunch.
        succeeded = Mapping<D>::launch(ref, inputLocation, args...);
      }
      if(!succeeded) {
        CallLaunch<T, Devices...>::call(ref, input, std::forward<Args>(args)...);
      }
    }
  };
  // break recursion and require CPU to be the last
  template <typename T>
  struct CallLaunch<T, CPU> {
    template <typename ...Args>
    static void call(T& ref, const HeterogeneousProductBase *input, Args&&... args) {
      Mapping<CPU>::launch(ref, std::forward<Args>(args)...);
    }
  };

  ////////////////////
  template <typename ...Args>
  struct CallProduce;
  template <typename T, typename D, typename ...Devices>
  struct CallProduce<T, D, Devices...> {
    template <typename ...Args>
    static void call(T& ref, const HeterogeneousDeviceId& algoExecutionLocation, Args&&... args) {
      if(algoExecutionLocation.deviceType() == Mapping<D>::deviceEnum) {
        Mapping<D>::produce(ref, algoExecutionLocation, std::forward<Args>(args)...);
      }
      else {
        CallProduce<T, Devices...>::call(ref, algoExecutionLocation, std::forward<Args>(args)...);
      }
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

    void call_produce(const HeterogeneousDeviceId &algoExecutionLocation, edm::Event& iEvent, const edm::EventSetup& iSetup) {
      CallProduce<HeterogeneousDevices, Devices...>::call(*this, algoExecutionLocation, iEvent, iSetup);
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
    if(algoExecutionLocation_.deviceType() == HeterogeneousDeviceId::kInvalidDevice) {
      // TODO: eventually fall back to CPU
      throw cms::Exception("LogicError") << "Trying to produce(), but algorithm was not executed successfully anywhere?";
    }
    Devices::call_produce(algoExecutionLocation_, iEvent, iSetup);
  }

  const HeterogeneousProductBase *input_;
  HeterogeneousDeviceId algoExecutionLocation_;
  bool wasScheduleCalled_;
};

#endif



