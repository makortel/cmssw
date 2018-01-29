#ifndef HeterogeneousCore_AcceleratorService_AcceleratorService_h
#define HeterogeneousCore_AcceleratorService_AcceleratorService_h

#include "FWCore/Concurrency/interface/WaitingTaskWithArenaHolder.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/Utilities/interface/propagate_const.h"

#include "HeterogeneousCore/AcceleratorService/interface/AcceleratorTask.h"
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
  /*
#define GENERATE(NAME) \
  template <typename T> struct Algo##NAME { T *algo; }; \
  template <typename T> Algo##Name<T> algoCPU(T *algo) { return Algo##Name<T>{algo}; }

  GENERATE(CPU);
  GENERATE(GPUMock);
  GENERATE(GPU);
#undef GENERATE
  */

  // Inheritance vs. type erasure? I'm now going with the former as it
  // is simpler to set up.
  //
  // Need a base class in order to have the per-device calls to be
  // made non-inlined.
  class AlgoCPU {
  public:
    AlgoCPU() {}
    virtual ~AlgoCPU() = default;
    virtual void runCPU() = 0;
  };
  class AlgoGPUMock {
  public:
    AlgoGPUMock() {}
    virtual ~AlgoGPUMock() = default;
    virtual void runGPUMock(std::function<void()> callback) = 0;
  };
  class AlgoGPUCuda {
  public:
    AlgoGPUCuda() {}
    virtual ~AlgoGPUCuda() = default;
    virtual void runGPUCuda(std::function<void()> callback) = 0;
  };
}

class AcceleratorService {
  // experiment just storing the output
  /*
  class AsyncWrapperBase {
  public:
    virtual std::type_info const& dynamicTypeInfo() = 0;
    virtual accelerator::Capabilities const whichRun() = 0;
    virtual void putInEvent(edm::Event& iEvent) = 0;
  };
  template <typename T>
  class AsyncWrapper {
  public:
    AsyncWrapper(T const *ptr, accelerator::Capabilities whichRun): ptr_(ptr), whichRun_(whichRun) {}
    std::type_info const& dynamicTypeInfo() override { return typeid(T); }
    void putInEvent(edm::Event& iEvent) { ptr_->putInEvent(iEvent); }
  private:
    T const *ptr_; // does not own
    accelerator::Capabilities whichRun_;
  };
  */

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

  // old interface
  void async(Token token, edm::StreamID streamID, std::unique_ptr<AcceleratorTaskBase> task, edm::WaitingTaskWithArenaHolder waitingTaskHolder);
  const AcceleratorTaskBase& getTask(Token token, edm::StreamID streamID) const;

  /**
   * Schedule the various versions of the algorithm to the available
   * heterogeneous devices.
   *
   * The parameter pack is an ordered list of accelerator::Algo*
   * pointers (that are owned by the calling code). The order of the
   * algorithms is taken as the preferred order to be tried. I.e. the
   * code tries to schedule the first algorithm, if that fails (no
   * device, to be extended) try the next one etc. The CPU version has
   * to be the last one.
   *
   */
  template <typename... Args>
  void schedule(Token token, edm::StreamID streamID, edm::WaitingTaskWithArenaHolder waitingTaskHolder, Args&&... args) {
    scheduleImpl(token, streamID, std::move(waitingTaskHolder), std::forward<Args>(args)...);
  }
  HeterogeneousDeviceId algoExecutionLocation(Token token, edm::StreamID streamID) const {
    return algoExecutionLocation_[tokenStreamIdsToDataIndex(token.id(), streamID)];
  }

  void print();

private:
  // signals
  void preallocate(edm::service::SystemBounds const& bounds);
  void preModuleConstruction(edm::ModuleDescription const& desc);
  void postModuleConstruction(edm::ModuleDescription const& desc);


  // other helpers
  unsigned int tokenStreamIdsToDataIndex(unsigned int tokenId, edm::StreamID streamId) const;

  // experimenting new interface
  template <typename... Args>
  void scheduleImpl(Token token, edm::StreamID streamID, edm::WaitingTaskWithArenaHolder waitingTaskHolder, accelerator::AlgoGPUMock *gpuMockAlgo, Args&&... args) {
    auto ret = scheduleGPUMock(token, streamID, waitingTaskHolder, gpuMockAlgo);
    if(!ret)
      scheduleImpl(token, streamID, std::move(waitingTaskHolder), std::forward<Args>(args)...);
  }
  template <typename... Args>
  void scheduleImpl(Token token, edm::StreamID streamID, edm::WaitingTaskWithArenaHolder waitingTaskHolder, accelerator::AlgoGPUCuda *gpuCudaAlgo, Args&&... args) {
    auto ret = scheduleGPUCuda(token, streamID, waitingTaskHolder, gpuCudaAlgo);
    if(!ret)
      scheduleImpl(token, streamID, std::move(waitingTaskHolder), std::forward<Args>(args)...);
  }
  // Break recursion, require CPU to be the last
  void scheduleImpl(Token token, edm::StreamID streamID, edm::WaitingTaskWithArenaHolder waitingTaskHolder, accelerator::AlgoCPU *cpuAlgo) {
    scheduleCPU(token, streamID, std::move(waitingTaskHolder), cpuAlgo);
  }
  bool scheduleGPUMock(Token token, edm::StreamID streamID, edm::WaitingTaskWithArenaHolder waitingTaskHolder, accelerator::AlgoGPUMock *gpuMockAlgo);
  bool scheduleGPUCuda(Token token, edm::StreamID streamID, edm::WaitingTaskWithArenaHolder waitingTaskHolder, accelerator::AlgoGPUCuda *gpuCudaAlgo);
  void scheduleCPU(Token token, edm::StreamID streamID, edm::WaitingTaskWithArenaHolder waitingTaskHolder, accelerator::AlgoCPU *cpuAlgo);

  
  unsigned int numberOfStreams_ = 0;

  // nearly (if not all) happens multi-threaded, so we need some
  // thread-locals to keep track in which module we are
  static thread_local unsigned int currentModuleId_;
  static thread_local std::string currentModuleLabel_; // only for printouts

  // TODO: how to treat subprocesses?
  std::mutex moduleMutex_;
  std::vector<unsigned int> moduleIds_;                      // list of module ids that have registered something on the service
  std::vector<std::unique_ptr<AcceleratorTaskBase> > tasks_; // numberOfStreams x moduleIds_.size(), indexing defined by moduleStreamIdsToDataIndex

  // experimenting new interface
  std::vector<HeterogeneousDeviceId> algoExecutionLocation_;
};

#endif
