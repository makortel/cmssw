#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDACore/interface/CUDAToken.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"

#include <cuda/api_wrappers.h>

#include <memory>

namespace {
  struct DeviceCache {
    int device;
    bool enabled;
  };
}

class CUDADeviceChooser: public edm::global::EDProducer<edm::StreamCache<::DeviceCache> > {
public:
  explicit CUDADeviceChooser(const edm::ParameterSet& iConfig);
  ~CUDADeviceChooser() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  std::unique_ptr<::DeviceCache> beginStream(edm::StreamID id) const;

  void produce(edm::StreamID id, edm::Event& iEvent, const edm::EventSetup& iSetup) const;

private:
  bool enabled_;
};

CUDADeviceChooser::CUDADeviceChooser(const edm::ParameterSet& iConfig):
  enabled_(iConfig.getParameter<bool>("enabled"))
{
  produces<CUDAToken>();
}

void CUDADeviceChooser::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<bool>("enabled", true)->setComment("This parameter is intended for debugging purposes only. If disabling some CUDA chains is needed for production, it is better to remove the CUDA modules altogether from the configuration.");
  descriptions.addWithDefaultLabel(desc);
  descriptions.setComment("This EDProducer chooses whether a chain of CUDA EDModules depending on it should run or not. The decision is communicated downstream by the existence of a 'CUDAToken' event product. Intended to be used with CUDADeviceFilter.");
}

std::unique_ptr<::DeviceCache> CUDADeviceChooser::beginStream(edm::StreamID id) const {
  auto ret = std::make_unique<::DeviceCache>();

  edm::Service<CUDAService> cudaService;
  ret->enabled = (enabled_ && cudaService->enabled(id));
  if(!ret->enabled) {
    return ret;
  }

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
  ret->device = id % cudaService->numberOfDevices();

  LogDebug("CUDADeviceChooser") << "EDM stream " << id << " set to CUDA device " << ret->device;

  return ret;
}

void CUDADeviceChooser::produce(edm::StreamID id, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  auto cache = streamCache(id);
  if(!cache->enabled) {
    return;
  }

  auto ret = std::make_unique<CUDAToken>(cache->device);
  LogDebug("CUDADeviceChooser") << "EDM stream " << id << " CUDA device " << ret->device() << " with CUDA stream " << ret->stream().id();
  iEvent.put(std::move(ret));
}


DEFINE_FWK_MODULE(CUDADeviceChooser);
