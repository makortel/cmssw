#include "HeterogeneousCore/AcceleratorService/interface/AcceleratorService.h"

#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/SystemBounds.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"

#include <limits>
#include <algorithm>
#include <thread>
#include <random>
#include <chrono>
#include <cassert>

thread_local unsigned int AcceleratorService::currentModuleId_ = std::numeric_limits<unsigned int>::max();
thread_local std::string AcceleratorService::currentModuleLabel_ = "";

AcceleratorService::AcceleratorService(edm::ParameterSet const& iConfig, edm::ActivityRegistry& iRegistry) {
  iRegistry.watchPreallocate(           this, &AcceleratorService::preallocate );
  iRegistry.watchPreModuleConstruction (this, &AcceleratorService::preModuleConstruction );
  iRegistry.watchPostModuleConstruction(this, &AcceleratorService::postModuleConstruction );
}

// signals
void AcceleratorService::preallocate(edm::service::SystemBounds const& bounds) {
  numberOfStreams_ = bounds.maxNumberOfStreams();
  edm::LogPrint("Foo") << "AcceleratorService: number of streams " << numberOfStreams_;
  // called after module construction, so initialize tasks_ here
  tasks_.resize(moduleIds_.size()*numberOfStreams_);
}

void AcceleratorService::preModuleConstruction(edm::ModuleDescription const& desc) {
  currentModuleId_ = desc.id();
  currentModuleLabel_ = desc.moduleLabel();
}
void AcceleratorService::postModuleConstruction(edm::ModuleDescription const& desc) {
  currentModuleId_ = std::numeric_limits<unsigned int>::max();
  currentModuleLabel_ = "";
}


// actual functionality
AcceleratorService::Token AcceleratorService::book() {
  if(currentModuleId_ == std::numeric_limits<unsigned int>::max())
    throw cms::Exception("AcceleratorService") << "Calling AcceleratorService::register() outside of EDModule constructor is forbidden.";

  unsigned int index=0;

  std::lock_guard<std::mutex> guard(moduleMutex_);

  auto found = std::find(moduleIds_.begin(), moduleIds_.end(), currentModuleId_);
  if(found == moduleIds_.end()) {
    index = moduleIds_.size();
    moduleIds_.push_back(currentModuleId_);
  }
  else {
    index = std::distance(moduleIds_.begin(), found);
  }

  edm::LogPrint("Foo") << "AcceleratorService::book for module " << currentModuleId_ << " " << currentModuleLabel_ << " token id " << index << " moduleIds_.size() " << moduleIds_.size();

  return Token(index);
}

void AcceleratorService::async(Token token, edm::StreamID streamID, std::unique_ptr<AcceleratorTaskBase> taskPtr, edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
  const auto index = tokenStreamIdsToDataIndex(token.id(), streamID);
  tasks_[index] = std::move(taskPtr);
  auto task = tasks_[index].get();

  edm::Service<CUDAService> cudaService;
  const auto cudaAvailable = cudaService->enabled();
  if(task->preferredDevice() == accelerator::Capabilities::kGPUCuda && cudaAvailable) {
    edm::LogPrint("Foo") << "  AcceleratorService token " << token.id() << " stream " << streamID << " launching task on GPU";
    task->call_run_GPUCuda([waitingTaskHolder = std::move(waitingTaskHolder),
                            token = token,
                            streamID = streamID,
                            task = task]() mutable {
                             edm::LogPrint("Foo") << "  AcceleratorService token " << token.id() << " stream " << streamID << " task finished on GPU";
                             waitingTaskHolder.doneWaiting(nullptr);
                           });
    edm::LogPrint("Foo") << "  AcceleratorService token " << token.id() << " stream " << streamID << " launched task on GPU asynchronously(?)";
  }
  else if(task->preferredDevice() == accelerator::Capabilities::kGPUMock) { // assume the mock GPU is always available
    edm::LogPrint("Foo") << "  AcceleratorService token " << token.id() << " stream " << streamID << " launching task on GPUMock";
    task->call_run_GPUMock([waitingTaskHolder = std::move(waitingTaskHolder),
                            token = token,
                            streamID = streamID,
                            task = task]() mutable {
                             edm::LogPrint("Foo") << "  AcceleratorService token " << token.id() << " stream " << streamID << " task finished on GPUMock";
                             waitingTaskHolder.doneWaiting(nullptr);
                           });
    edm::LogPrint("Foo") << "  AcceleratorService token " << token.id() << " stream " << streamID << " launched task on GPUMock asynchronously(?)";
  }
  else {
    if(task->preferredDevice() == accelerator::Capabilities::kGPUCuda) {
      edm::LogPrint("Foo") << "  AcceleratorService token " << token.id() << " stream " << streamID << " preferred GPU but it was not available";
    }
    edm::LogPrint("Foo") << "  AcceleratorService token " << token.id() << " stream " << streamID << " launching task on CPU";
    task->call_run_CPU();
    edm::LogPrint("Foo") << "  AcceleratorService token " << token.id() << " stream " << streamID << " task finished on CPU";
  }
}

const AcceleratorTaskBase& AcceleratorService::getTask(Token token, edm::StreamID streamID) const {
  auto& ptr = tasks_[tokenStreamIdsToDataIndex(token.id(), streamID)];
  if(ptr == nullptr) {
    throw cms::Exception("LogicError") << "No task for token " << token.id() << " stream " << streamID;
  }
  return *ptr;
}

void AcceleratorService::print() {
  edm::LogPrint("AcceleratorService") << "Hello world";
}

unsigned int AcceleratorService::tokenStreamIdsToDataIndex(unsigned int tokenId, edm::StreamID streamId) const {
  assert(streamId < numberOfStreams_);
  return tokenId*numberOfStreams_ + streamId;
}
