#ifndef HeterogeneousCore_CUDACore_CUDAStreamEDProducer_h
#define HeterogeneousCore_CUDACore_CUDAStreamEDProducer_h

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"

#include <cuda/api_wrappers.h>

/**
 * This class is a bit hacky but intended only for a transition
 * period. It also duplicates the EDM stream -> CUDA device assignment.
 */
template <typename ...Args>
class CUDAStreamEDProducer: public edm::stream::EDProducer<Args...> {
private:
  void beginStream(edm::StreamID id) override final {
    // The following checks only from CUDAService whether it is
    // enabled or not. Also CUDADeviceChooser can be configured to be
    // disabled, effectively disabling that "CUDA chain".
    // Unfortunately we have no (easy) means here to know whether this
    // EDProducer is part of such a chain. On the other hand,
    // beginStream() is intended only for block memory allocations
    // (and we will likely adjust the strategy), and the
    // CUDADeviceChooser.enabled is intended for debugging/testing
    // purposes, so maybe this solution is good enough (i.e. for
    // debugging it doesn't matter if we allocate "too much")
    edm::Service<CUDAService> cudaService;
    if(cudaService->enabled(id)) {
      // This logic is duplicated from CUDADeviceChooser
      int device = id % cudaService->numberOfDevices();
      cuda::device::current::scoped_override_t<> setDeviceForThisScope(device);
      beginStreamCUDA(id);
    }
  }

  // It's a bit stupid to change the name, but I don't have anything
  // additional to pass down.
  //
  // Note: contrary to HeterogeneousEDProducer+GPUCuda, the CUDA
  // stream is *not* passed to the deriving class (there is no good
  // place for a CUDA stream here in this design).
  virtual void beginStreamCUDA(edm::StreamID id) = 0;
};

#endif
