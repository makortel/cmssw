#include "chooseCUDADevice.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"

namespace cudacore {
  int chooseCUDADevice(edm::StreamID id) {
    edm::Service<CUDAService> cudaService;

    // For startes we "statically" assign the device based on
    // edm::Stream number. This is suboptimal if the number of
    // edm::Streams is not a multiple of the number of CUDA devices
    // (and even then there is no load balancing).
    //
    // TODO: improve. Possible ideas include
    // - allocate M (< N(edm::Streams)) buffers per device per "chain of modules", choose dynamically which (buffer, device) to use
    // - our own CUDA memory allocator
    //   * being able to cheaply allocate+deallocate scratch memory allows to make the execution fully dynamic e.g. based on current load
    //   * would probably still need some buffer space/device to hold e.g. conditions data
    //     - for conditions, how to handle multiple lumis per job?
    return id % cudaService->numberOfDevices();
  }
}
