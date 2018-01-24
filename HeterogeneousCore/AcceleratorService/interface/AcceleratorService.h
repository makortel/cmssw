#ifndef HeterogeneousCore_AcceleratorService_AcceleratorService_h
#define HeterogeneousCore_AcceleratorService_AcceleratorService_h

#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/Concurrency/interface/WaitingTaskWithArenaHolder.h"

#include "HeterogeneousCore/AcceleratorService/interface/AcceleratorTask.h"

#include <memory>
#include <mutex>
#include <vector>

namespace edm {
  class ParameterSet;
  class ActivityRegistry;
  class ModuleDescription;
  namespace service {
    class SystemBounds;
  }
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

  void async(Token token, edm::StreamID streamID, std::unique_ptr<AcceleratorTaskBase> task, edm::WaitingTaskWithArenaHolder waitingTaskHolder);

  const AcceleratorTaskBase& getTask(Token token, edm::StreamID streamID) const;

  void print();

private:
  // signals
  void preallocate(edm::service::SystemBounds const& bounds);
  void preModuleConstruction(edm::ModuleDescription const& desc);
  void postModuleConstruction(edm::ModuleDescription const& desc);


  // other helpers
  unsigned int tokenStreamIdsToDataIndex(unsigned int tokenId, edm::StreamID streamId) const;
  bool isGPUAvailable() const;

  unsigned int numberOfStreams_ = 0;

  // nearly (if not all) happens multi-threaded, so we need some
  // thread-locals to keep track in which module we are
  static thread_local unsigned int currentModuleId_;
  static thread_local std::string currentModuleLabel_; // only for printouts

  // TODO: how to treat subprocesses?
  std::mutex moduleMutex_;
  std::vector<unsigned int> moduleIds_;                      // list of module ids that have registered something on the service
  std::vector<std::unique_ptr<AcceleratorTaskBase> > tasks_; // numberOfStreams x moduleIds_.size(), indexing defined by moduleStreamIdsToDataIndex
};

#endif
