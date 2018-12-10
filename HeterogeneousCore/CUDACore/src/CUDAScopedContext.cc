#include "HeterogeneousCore/CUDACore/interface/CUDAScopedContext.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "chooseCUDADevice.h"


CUDAScopedContext::CUDAScopedContext(edm::StreamID streamID): CUDAScopedContext(cudacore::chooseCUDADevice(streamID)) {}
CUDAScopedContext::CUDAScopedContext(int device):
  currentDevice_(device),
  setDeviceForThisScope_(device)
{
  auto current_device = cuda::device::current::get();
  stream_ = std::make_shared<cuda::stream_t<>>(current_device.create_stream(cuda::stream::implicitly_synchronizes_with_default_stream));
}

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
                                  auto error = cudaGetErrorName(status);
                                  auto message = cudaGetErrorString(status);
                                  waitingTaskHolder.doneWaiting(std::make_exception_ptr(cms::Exception("CUDAError") << "Callback of CUDA stream " << streamId << " in device " << device << " error " << error << ": " << message));
                                }
                              });
  }
}
