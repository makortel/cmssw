#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "HeterogeneousCore/CUDACore/interface/CUDAToken.h"

class CUDADeviceFilter: public edm::global::EDFilter<> {
public:
  explicit CUDADeviceFilter(const edm::ParameterSet& iConfig);
  ~CUDADeviceFilter() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  bool filter(edm::StreamID id, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

private:
  edm::EDGetTokenT<CUDAToken> token_;
};

CUDADeviceFilter::CUDADeviceFilter(const edm::ParameterSet& iConfig):
  token_(consumes<CUDAToken>(iConfig.getParameter<edm::InputTag>("src")))
{}

void CUDADeviceFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("cudaDeviceChooser"))->setComment("Source of the 'CUDAToken'.");
  descriptions.addWithDefaultLabel(desc);
  descriptions.setComment("This EDFilter filters based on the existence of a 'CUDAToken' event product. Intended to be used together with CUDADeviceChooser. Returns 'true' if the product exists, and 'false' if not.");
}

bool CUDADeviceFilter::filter(edm::StreamID id, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  edm::Handle<CUDAToken> handle;
  iEvent.getByToken(token_, handle);
  return handle.isValid();
}

DEFINE_FWK_MODULE(CUDADeviceFilter);
