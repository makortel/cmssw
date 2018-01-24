#ifndef HeterogeneousCore_AcceleratorService_AcceleratorTask_h
#define HeterogeneousCore_AcceleratorService_AcceleratorTask_h

#include <functional>

namespace accelerator {
  // Below we could assume that the CPU would be always present. For
  // the sake of demonstration I'll keep it separate entity.

  // similar to FWCore/Framework/interface/moduleAbilityEnums.h
  enum class Capabilities {
    kCPU,
    kGPUMock,
    kGPUCuda
  };
  
  namespace CapabilityBits {
    enum Bits {
      kCPU = 1,
      kGPUMock = 2,
      kGPUCuda = 3
    };
  }
}

// Task base class
class AcceleratorTaskBase {
public:
  AcceleratorTaskBase() = default;
  virtual ~AcceleratorTaskBase() = 0;

  virtual accelerator::Capabilities preferredDevice() const = 0;

  // CPU functions
  virtual bool runnable_CPU() const { return false; }
  virtual void call_run_CPU() {}

  // GPU mock functions to allow testing
  virtual bool runnable_GPUMock() const { return false; }
  virtual void call_run_GPUMock(std::function<void()> callback) {}

  // GPU functions
  virtual bool runnable_GPUCuda() const { return false; }
  virtual void call_run_GPUCuda(std::function<void()> callback) {} // do not call the callback without implementation
};

namespace accelerator {
  // similar to e.g. FWCore/Framework/interface/one/moduleAbilities.h
  struct CPU {
    static constexpr Capabilities kCapability = Capabilities::kCPU;
  };

  struct GPUMock {
    static constexpr Capabilities kCapability = Capabilities::kGPUMock;
  };

  struct GPUCuda {
    static constexpr Capabilities kCapability = Capabilities::kGPUCuda;
  };

  // similar to e.g. FWCore/Framework/interface/one/implementors.h
  namespace impl {
    class CPU: public virtual AcceleratorTaskBase {
    public:
      CPU() = default;
      bool runnable_CPU() const override { return true; }

    private:
      void call_run_CPU() override {
        run_CPU();
      };

      virtual void run_CPU() = 0;
    };

    class GPUMock: public virtual AcceleratorTaskBase {
    public:
      GPUMock() = default;
      bool runnable_GPUMock() const override { return true; }

    private:
      void call_run_GPUMock(std::function<void()> callback) override {
        run_GPUMock(std::move(callback));
      };

      virtual void run_GPUMock(std::function<void()> callback) = 0;
    };

    class GPUCuda: public virtual AcceleratorTaskBase {
    public:
      GPUCuda() = default;
      bool runnable_GPUCuda() const override { return true; }

    private:
      void call_run_GPUCuda(std::function<void()> callback) override {
        run_GPUCuda(std::move(callback));
      };

      virtual void run_GPUCuda(std::function<void()> callback) = 0;
    };
  }

  // similar to e.g. FWCore/Framework/interface/one/producerAbilityToImplementor.h
  template <typename T> struct CapabilityToImplementor;
  
  template<>
  struct CapabilityToImplementor<CPU> {
    using Type = impl::CPU;
  };

  template<>
  struct CapabilityToImplementor<GPUMock> {
    using Type = impl::GPUMock;
  };

  template<>
  struct CapabilityToImplementor<GPUCuda> {
    using Type = impl::GPUCuda;
  };
}

template <typename ... T>
class AcceleratorTask:
  public virtual AcceleratorTaskBase,
  public accelerator::CapabilityToImplementor<T>::Type... {

 public:
  AcceleratorTask() = default;
  
};


#endif
