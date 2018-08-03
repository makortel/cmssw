#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDACore/interface/CUDAToken.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"

#include "chooseCUDADevice.h"

namespace {
  struct DeviceCache {
    int device;
    bool enabled;
  };
}

class CUDADeviceChooserFilter: public edm::global::EDFilter<edm::StreamCache<::DeviceCache>> {
public:
  explicit CUDADeviceChooserFilter(const edm::ParameterSet& iConfig);
  ~CUDADeviceChooserFilter() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  std::unique_ptr<::DeviceCache> beginStream(edm::StreamID id) const;

  bool filter(edm::StreamID id, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

private:
  bool enabled_;
};

CUDADeviceChooserFilter::CUDADeviceChooserFilter(const edm::ParameterSet& iConfig):
  enabled_(iConfig.getParameter<bool>("enabled"))
{
  produces<CUDAToken>();
}

void CUDADeviceChooserFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<bool>("enabled", true)->setComment("This parameter is intended for debugging purposes only. If disabling some CUDA chains is needed for production, it is better to remove the CUDA modules altogether from the configuration.");
  descriptions.addWithDefaultLabel(desc);
  descriptions.setComment("This EDFilter chooses whether a chain of CUDA EDModules depending on it should run or not, and on which CUDA device they should run. The decision is communicated downstream with the filter decision. In addition, if the filter returns true, a 'CUDAToken' is produced into the event (for false nothing is produced).");
}

std::unique_ptr<::DeviceCache> CUDADeviceChooserFilter::beginStream(edm::StreamID id) const {
  auto ret = std::make_unique<::DeviceCache>();

  edm::Service<CUDAService> cudaService;
  ret->enabled = (enabled_ && cudaService->enabled(id));
  if(!ret->enabled) {
    return ret;
  }

  ret->device = cudacore::chooseCUDADevice(id);

  LogDebug("CUDADeviceChooserFilter") << "EDM stream " << id << " set to CUDA device " << ret->device;

  return ret;
}

bool CUDADeviceChooserFilter::filter(edm::StreamID id, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  auto cache = streamCache(id);
  if(!cache->enabled) {
    return false;
  }

  auto ret = std::make_unique<CUDAToken>(cache->device);
  LogDebug("CUDADeviceChooserFilter") << "EDM stream " << id << " CUDA device " << ret->device() << " with CUDA stream " << ret->stream().id();
  iEvent.put(std::move(ret));
  return true;
}

DEFINE_FWK_MODULE(CUDADeviceChooserFilter);
