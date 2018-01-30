#ifndef HeterogeneousCore_AcceleratorService_AcceleratorService_h
#define HeterogeneousCore_AcceleratorService_AcceleratorService_h

#include "FWCore/Concurrency/interface/WaitingTaskWithArenaHolder.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/Utilities/interface/propagate_const.h"

#include "HeterogeneousCore/Product/interface/HeterogeneousProduct.h"

#include <memory>
#include <mutex>
#include <vector>

namespace edm {
  class Event;
  class ParameterSet;
  class ActivityRegistry;
  class ModuleDescription;
  namespace service {
    class SystemBounds;
  }
}

namespace accelerator {
  // Inheritance vs. type erasure? I'm now going with the latter even
  // if it is more complex to setup and maintain in order to support a
  // case where a single class implements multiple CPU/GPU/etc
  // interfaces, in which case via inheritance we can't separate the
  // cases in the scheduling interface.
  //
  // Want a base class in order to have the per-device calls to be
  // made non-inlined (how necessary is this?)
  class AlgoBase {
  public:
    AlgoBase() {}
    virtual ~AlgoBase() = 0;
    HeterogeneousDeviceId executionLocation() const = 0;
  };

  class AlgoCPUBase: public AlgoBase {
  public:
    AlgoCPUBase() {}
    virtual ~AlgoCPUBase() = default;
    virtual void runCPU() = 0;
  };
  template <typename T> class AlgoCPU: public AlgoCPUBase {
  public:
    AlgoCPU(T *algo): algo_(algo) {}
    void runCPU() override { algo_->runCPU(); }
    HeterogeneousDeviceId executionLocation() const { return HeterogeneousDeviceId(HeterogeneousDevice::kCPU, 0); }
  private:
    T *algo_;
  };
  template <typename T> AlgoCPU<T> algoCPU(T *algo) { return AlgoCPU<T>(algo); }

  //
  class AlgoGPUMockBase: public AlgoBase {
  public:
    AlgoGPUMockBase() {}
    virtual ~AlgoGPUMockBase() = default;
    virtual void runGPUMock(std::function<void()> callback) = 0;
  };
  template <typename T> class AlgoGPUMock: public AlgoGPUMockBase {
  public:
    AlgoGPUMock(T *algo): algo_(algo) {}
    void runGPUMock(std::function<void()> callback) override { algo_->runGPUMock(std::move(callback)); }
    HeterogeneousDeviceId executionLocation() const { return HeterogeneousDeviceId(HeterogeneousDevice::kGPUMock, 0); }
  private:
    T *algo_;
  };
  template <typename T> AlgoGPUMock<T> algoGPUMock(T *algo) { return AlgoGPUMock<T>(algo); }

  //
  class AlgoGPUCudaBase: public AlgoBase {
  public:
    AlgoGPUCudaBase() {}
    virtual ~AlgoGPUCudaBase() = default;
    virtual void runGPUCuda(std::function<void()> callback) = 0;
  };
  template <typename T> class AlgoGPUCuda: public AlgoGPUCudaBase {
  public:
    AlgoGPUCuda(T *algo): algo_(algo) {}
    void runGPUCuda(std::function<void()> callback) override { algo_->runGPUCuda(std::move(callback)); }
    HeterogeneousDeviceId executionLocation() const { return HeterogeneousDeviceId(HeterogeneousDevice::kGPUCuda, 0); }
  private:
    T *algo_;
  };
  template <typename T> AlgoGPUCuda<T> algoGPUCuda(T *algo) { return AlgoGPUCuda<T>(algo); }
}

class AcceleratorService {
public:
  class Token {
  public:
    explicit Token(unsigned int id): id_(id) {}

    unsigned int id() const { return id_; }
  private:
    unsigned int id_;
  };

  AcceleratorService(edm::ParameterSet const& iConfig, edm::ActivityRegistry& iRegistry);

  Token book(); // TODO: better name, unfortunately 'register' is a reserved keyword...

  /**
   * Schedule the various versions of the algorithm to the available
   * heterogeneous devices.
   *
   * The parameter pack is an ordered list of accelerator::Algo*
   * pointers (that are owned by the calling code). The lifetime of
   * the objects must be (at least) at the end of the
   * EDModule<ExternalWork> produce() call.
   *
   * The order of the algorithms is taken as the preferred order to be
   * tried. I.e. the code tries to schedule the first algorithm, if
   * that fails (no device, to be extended) try the next one etc. The
   * CPU version has to be the last one.
   *
   *
   * TODO: passing the "input" parameter here is a bit boring, but
   * somehow we have to schedule according to the input. Try to think
   * something better.
   */
  template <typename I, typename... Args>
  void schedule(Token token, edm::StreamID streamID, edm::WaitingTaskWithArenaHolder waitingTaskHolder, const I *input, Args&&... args) {
    scheduleImpl(token, streamID, std::move(waitingTaskHolder), input, std::forward<Args>(args)...);
  }
  HeterogeneousDeviceId algoExecutionLocation(Token token, edm::StreamID streamID) const {
    return algoExecutionLocation_[tokenStreamIdsToDataIndex(token.id(), streamID)];
  }

private:
  // signals
  void preallocate(edm::service::SystemBounds const& bounds);
  void preModuleConstruction(edm::ModuleDescription const& desc);
  void postModuleConstruction(edm::ModuleDescription const& desc);


  // other helpers
  unsigned int tokenStreamIdsToDataIndex(unsigned int tokenId, edm::StreamID streamId) const;

  // experimenting new interface
  template <typename I, typename A, typename... Args>
  void scheduleImpl(Token token, edm::StreamID streamID, edm::WaitingTaskWithArenaHolder waitingTaskHolder, const I *input,
                    accelerator::AlgoGPUMock<A> gpuMockAlgo, Args&&... args) {
    bool succeeded = true;
    if(input) {
      succeeded = input->isProductOn(HeterogeneousLocation::kGPU);
    }
    if(succeeded) {
      succeeded = scheduleGPUMock(token, streamID, waitingTaskHolder, gpuMockAlgo);
    }
    if(!succeeded) {
      scheduleImpl(token, streamID, std::move(waitingTaskHolder), input, std::forward<Args>(args)...);
    }
  }
  template <typename I, typename A, typename... Args>
  void scheduleImpl(Token token, edm::StreamID streamID, edm::WaitingTaskWithArenaHolder waitingTaskHolder, const I *input,
                    accelerator::AlgoGPUCuda<A> gpuCudaAlgo, Args&&... args) {
    bool succeeded = true;
    if(input) {
      succeeded = input->isProductOn(HeterogeneousLocation::kGPU);
    }
    if(succeeded) {
      succeeded = scheduleGPUCuda(token, streamID, waitingTaskHolder, gpuCudaAlgo);
    }
    if(!succeeded)
      scheduleImpl(token, streamID, std::move(waitingTaskHolder), input, std::forward<Args>(args)...);
  }
  // Break recursion, require CPU to be the last
  template <typename I, typename A>
  void scheduleImpl(Token token, edm::StreamID streamID, edm::WaitingTaskWithArenaHolder waitingTaskHolder, const I *input,
                    accelerator::AlgoCPU<A> cpuAlgo) {
    scheduleCPU(token, streamID, std::move(waitingTaskHolder), cpuAlgo);
  }
  bool scheduleGPUMock(Token token, edm::StreamID streamID, edm::WaitingTaskWithArenaHolder waitingTaskHolder, accelerator::AlgoGPUMockBase& gpuMockAlgo);
  bool scheduleGPUCuda(Token token, edm::StreamID streamID, edm::WaitingTaskWithArenaHolder waitingTaskHolder, accelerator::AlgoGPUCudaBase& gpuCudaAlgo);
  void scheduleCPU(Token token, edm::StreamID streamID, edm::WaitingTaskWithArenaHolder waitingTaskHolder, accelerator::AlgoCPUBase& cpuAlgo);

  
  unsigned int numberOfStreams_ = 0;

  // nearly (if not all) happens multi-threaded, so we need some
  // thread-locals to keep track in which module we are
  static thread_local unsigned int currentModuleId_;
  static thread_local std::string currentModuleLabel_; // only for printouts

  // TODO: how to treat subprocesses?
  std::mutex moduleMutex_;
  std::vector<unsigned int> moduleIds_;                      // list of module ids that have registered something on the service
  std::vector<HeterogeneousDeviceId> algoExecutionLocation_;
};

#endif
