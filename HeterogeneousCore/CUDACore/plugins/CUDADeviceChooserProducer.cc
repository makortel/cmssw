#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDACore/interface/CUDAToken.h"
#include "HeterogeneousCore/CUDACore/interface/chooseCUDADevice.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"


#include <memory>

namespace {
  struct DeviceCache {
    int device;
  };
}

class CUDADeviceChooserProducer: public edm::global::EDProducer<edm::StreamCache<::DeviceCache>> {
public:
  explicit CUDADeviceChooserProducer(const edm::ParameterSet& iConfig);
  ~CUDADeviceChooserProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  std::unique_ptr<::DeviceCache> beginStream(edm::StreamID id) const;

  void produce(edm::StreamID id, edm::Event& iEvent, const edm::EventSetup& iSetup) const;
};

CUDADeviceChooserProducer::CUDADeviceChooserProducer(const edm::ParameterSet& iConfig) {
  edm::Service<CUDAService> cudaService;
  if(!cudaService->enabled()) {
    throw cms::Exception("Configuration") << "CUDAService is disabled so CUDADeviceChooserProducer is unable to make decisions on which CUDA device to run. If you need to run without CUDA devices, please use CUDADeviceChooserFilter for conditional execution, or remove all CUDA modules from your configuration.";
  }
  produces<CUDAToken>();
}

void CUDADeviceChooserProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  descriptions.addWithDefaultLabel(desc);
  descriptions.setComment("This EDProducer chooses on which CUDA device the chain of CUDA EDModules depending on it should run. The decision is communicated downstream with the 'CUDAToken' event product. It is an error if there are no CUDA devices, or CUDAService is disabled.");
}

std::unique_ptr<::DeviceCache> CUDADeviceChooserProducer::beginStream(edm::StreamID id) const {
  auto ret = std::make_unique<::DeviceCache>();

  edm::Service<CUDAService> cudaService;
  if(!cudaService->enabled(id)) {
    throw cms::Exception("LogicError") << "CUDA is disabled for EDM stream " << id << " in CUDAService, so CUDADeviceChooser is unable to decide the CUDA device for this EDM stream. If you need to dynamically decide whether a chain of CUDA EDModules is run or not, please use CUDADeviceChooserFilter instead.";
  }
  ret->device = cudacore::chooseCUDADevice(id);

  LogDebug("CUDADeviceChooserProducer") << "EDM stream " << id << " set to CUDA device " << ret->device;

  return ret;
}

void CUDADeviceChooserProducer::produce(edm::StreamID id, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  auto ret = std::make_unique<CUDAToken>(streamCache(id)->device);
  LogDebug("CUDADeviceChooserProducer") << "EDM stream " << id << " CUDA device " << ret->device() << " with CUDA stream " << ret->stream().id();
  iEvent.put(std::move(ret));
}


DEFINE_FWK_MODULE(CUDADeviceChooserProducer);
