#include "HeterogeneousCore/CUDACore/interface/CUDAScopedContext.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"

#include "chooseCUDADevice.h"


CUDAScopedContext::CUDAScopedContext(edm::StreamID streamID):
  currentDevice_(cudacore::chooseCUDADevice(streamID)),
  setDeviceForThisScope_(currentDevice_)
{
  edm::Service<CUDAService> cs;
  stream_ = cs->getCUDAStream();
}

CUDAScopedContext::CUDAScopedContext(int device, std::unique_ptr<cuda::stream_t<>> stream):
  currentDevice_(device),
  setDeviceForThisScope_(device),
  stream_(std::move(stream))
{}

CUDAScopedContext::~CUDAScopedContext() {
  if(waitingTaskHolder_.has_value()) {
    stream_->enqueue.callback([device=currentDevice_,
                               waitingTaskHolder=*waitingTaskHolder_]
                              (cuda::stream::id_t streamId, cuda::status_t status) mutable {
                                if(cuda::is_success(status)) {
                                  LogTrace("CUDAScopedContext") << " GPU kernel finished (in callback) device " << device << " CUDA stream " << streamId;
                                  waitingTaskHolder.doneWaiting(nullptr);
                                }
                                else {
                                  // wrap the exception in a try-catch block to let GDB "catch throw" break on it
                                  try {
                                    auto error = cudaGetErrorName(status);
                                    auto message = cudaGetErrorString(status);
                                    throw cms::Exception("CUDAError") << "Callback of CUDA stream " << streamId << " in device " << device << " error " << error << ": " << message;
                                  } catch(cms::Exception&) {
                                    waitingTaskHolder.doneWaiting(std::current_exception());
                                  }
                                }
                              });
  }
}
